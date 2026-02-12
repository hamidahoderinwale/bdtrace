#!/usr/bin/env python3
"""
Sequence Clustering Script
Clusters event sequences using DTW or other methods
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    HAS_TSLEARN = True
except ImportError:
    HAS_TSLEARN = False
    print("[WARNING] tslearn not available, falling back to simple k-means")

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[ERROR] scikit-learn required but not installed")
    sys.exit(1)


def load_sequences(input_file=None):
    """Load sequences from stdin or file"""
    if input_file:
        with open(input_file, 'r') as f:
            return json.load(f)
    else:
        # Read from stdin
        return json.load(sys.stdin)


def pad_sequences(sequences, max_length=None):
    """Pad sequences to same length"""
    if max_length is None:
        max_length = max(len(s) for s in sequences)
    
    padded = []
    for seq in sequences:
        if len(seq) < max_length:
            # Pad with zeros
            padding = [[0.0] * len(seq[0])] * (max_length - len(seq))
            padded.append(seq + padding)
        else:
            padded.append(seq[:max_length])
    
    return np.array(padded), max_length


def cluster_with_dtw(sequences, n_clusters, min_size=3):
    """Cluster sequences using DTW-based k-means"""
    if not HAS_TSLEARN:
        raise ImportError("tslearn required for DTW clustering")
    
    # Convert to numpy array and pad
    sequences_array, max_length = pad_sequences(sequences)
    
    # Scale sequences
    scaler = TimeSeriesScalerMeanVariance()
    sequences_scaled = scaler.fit_transform(sequences_array)
    
    # Cluster using DTW
    model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="dtw",
        max_iter=10,
        random_state=42
    )
    
    labels = model.fit_predict(sequences_scaled)
    
    return labels, model


def cluster_with_kmeans(sequences, n_clusters, min_size=3):
    """Cluster sequences using standard k-means (flattened)"""
    # Flatten sequences to fixed-size vectors
    max_length = max(len(s) for s in sequences)
    vector_dim = len(sequences[0][0]) if sequences else 0
    
    flattened = []
    for seq in sequences:
        # Flatten sequence
        flat = []
        for vec in seq:
            flat.extend(vec)
        # Pad to max_length
        while len(flat) < max_length * vector_dim:
            flat.append(0.0)
        flattened.append(flat[:max_length * vector_dim])
    
    flattened = np.array(flattened)
    
    # Scale
    scaler = StandardScaler()
    flattened_scaled = scaler.fit_transform(flattened)
    
    # Cluster
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(flattened_scaled)
    
    return labels, model


def find_representative_sequence(cluster_sequences, all_sequences):
    """Find most representative sequence in cluster (median length)"""
    if not cluster_sequences:
        return None
    
    avg_length = sum(len(all_sequences[idx]) for idx in cluster_sequences) / len(cluster_sequences)
    
    # Find sequence closest to average length
    best_idx = cluster_sequences[0]
    best_diff = abs(len(all_sequences[best_idx]) - avg_length)
    
    for idx in cluster_sequences[1:]:
        diff = abs(len(all_sequences[idx]) - avg_length)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx
    
    return best_idx


def extract_common_event_types(cluster_sequences, all_sequences, event_types):
    """Extract common event types in cluster"""
    types_in_cluster = []
    for idx in cluster_sequences:
        seq = all_sequences[idx]
        for event in seq:
            if event.get('event_type'):
                types_in_cluster.append(event['event_type'])
    
    # Count frequencies
    from collections import Counter
    type_counts = Counter(types_in_cluster)
    return [t for t, _ in type_counts.most_common(5)]


def main():
    parser = argparse.ArgumentParser(description='Cluster event sequences')
    parser.add_argument('--method', choices=['dtw', 'kmeans'], default='dtw',
                       help='Clustering method')
    parser.add_argument('--n-clusters', type=int, default=10,
                       help='Number of clusters')
    parser.add_argument('--min-size', type=int, default=3,
                       help='Minimum cluster size')
    parser.add_argument('--input', type=str, default=None,
                       help='Input JSON file (default: stdin)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file (default: stdout)')
    
    args = parser.parse_args()
    
    # Load sequences
    data = load_sequences(args.input)
    
    if 'sequences' in data:
        sequences = data['sequences']
        event_types = data.get('event_types', [])
        metadata = data.get('metadata', {})
    else:
        # Assume data is list of sequences
        sequences = data
        event_types = []
        metadata = {}
    
    # Extract vectors
    sequence_vectors = []
    all_sequences = []
    
    for seq in sequences:
        if isinstance(seq, dict) and 'combined_vector' in seq:
            # Single sequence
            all_sequences.append([seq])
            sequence_vectors.append([seq['combined_vector']])
        elif isinstance(seq, list):
            # Sequence of events
            all_sequences.append(seq)
            vectors = [e.get('combined_vector', []) for e in seq]
            sequence_vectors.append(vectors)
        else:
            # Raw vectors
            all_sequences.append([{'combined_vector': seq}])
            sequence_vectors.append([seq] if isinstance(seq[0], (int, float)) else seq)
    
    # Cluster
    try:
        if args.method == 'dtw' and HAS_TSLEARN:
            labels, model = cluster_with_dtw(sequence_vectors, args.n_clusters, args.min_size)
        else:
            labels, model = cluster_with_kmeans(sequence_vectors, args.n_clusters, args.min_size)
    except Exception as e:
        print(f"[ERROR] Clustering failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Organize clusters
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)
    
    # Build behavioral library
    behavioral_library = []
    for cluster_id, sequence_indices in clusters.items():
        if len(sequence_indices) >= args.min_size:
            representative_idx = find_representative_sequence(sequence_indices, all_sequences)
            representative = all_sequences[representative_idx] if representative_idx is not None else None
            
            behavioral_library.append({
                'cluster_id': int(cluster_id),
                'frequency': len(sequence_indices),
                'representative_sequence': representative,
                'sequence_indices': sequence_indices,
                'metadata': {
                    'avg_length': sum(len(all_sequences[idx]) for idx in sequence_indices) / len(sequence_indices),
                    'event_types': extract_common_event_types(sequence_indices, all_sequences, event_types)
                }
            })
    
    # Output results
    result = {
        'method': args.method,
        'n_clusters': args.n_clusters,
        'min_size': args.min_size,
        'total_sequences': len(sequences),
        'clusters_found': len(behavioral_library),
        'cluster_assignments': labels.tolist(),
        'behavioral_library': behavioral_library,
        'metadata': metadata
    }
    
    output_json = json.dumps(result, indent=2)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
    else:
        print(output_json)


if __name__ == '__main__':
    main()

