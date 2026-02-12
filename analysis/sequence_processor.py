"""
Main Sequence Processor Service
Orchestrates the computational model pipeline
"""

import json
import sys
import numpy as np
from pathlib import Path

from database_connector import DatabaseConnector
from vectorizer import EventVectorizer
import config


class SequenceProcessor:
    """Main service for processing sequences and building behavioral library"""
    
    def __init__(self):
        self.db = DatabaseConnector()
        self.vectorizer = EventVectorizer()
    
    def extract_sequences(self, workspace_path: str | None = None, 
                         session_id: str | None = None,
                         min_sequence_length: int = 3,
                         max_sequence_length: int = 50) -> list[list[dict]]:
        """Extract event sequences from database"""
        # Get events with prompts
        events = self.db.get_events_with_prompts(workspace_path=workspace_path)
        
        # Group by session
        sequences_by_session = {}
        for event in events:
            session = event.get('session_id') or 'default'
            if session not in sequences_by_session:
                sequences_by_session[session] = []
            sequences_by_session[session].append(event)
        
        # Sort each session by timestamp
        sequences = []
        for session_events in sequences_by_session.values():
            session_events.sort(key=lambda e: e.get('timestamp', ''))
            
            # Filter by length
            if min_sequence_length <= len(session_events) <= max_sequence_length:
                sequences.append(session_events)
        
        return sequences
    
    def vectorize_sequences(self, sequences: list[list[dict]]) -> list[dict]:
        """Vectorize all sequences"""
        # Build event type encoder from all events
        all_events = [e for seq in sequences for e in seq]
        self.vectorizer.build_event_type_encoder(all_events)
        
        # Build prompts map
        prompts_map = {}
        for seq in sequences:
            for event in seq:
                prompt_id = event.get('prompt_id')
                if prompt_id and 'prompt_text' in event:
                    prompts_map[prompt_id] = {'text': event.get('prompt_text')}
        
        # Vectorize sequences
        vectorized_sequences = []
        for seq in sequences:
            vectorized = self.vectorizer.vectorize_sequence(seq, prompts_map)
            vectorized_sequences.append({
                'sequence': vectorized,
                'metadata': {
                    'session_id': seq[0].get('session_id') if seq else None,
                    'workspace_path': seq[0].get('workspace_path') if seq else None,
                    'length': len(seq),
                    'timestamp_range': [
                        seq[0].get('timestamp') if seq else None,
                        seq[-1].get('timestamp') if seq else None
                    ]
                }
            })
        
        return vectorized_sequences
    
    def cluster_sequences(self, vectorized_sequences: list[dict], 
                         method: str = None,
                         n_clusters: int = None,
                         min_cluster_size: int = None) -> dict:
        """Cluster sequences using external script"""
        import subprocess
        import tempfile
        
        method = method or config.CLUSTERING_METHOD
        n_clusters = n_clusters or config.MAX_CLUSTERS
        min_cluster_size = min_cluster_size or config.MIN_CLUSTER_SIZE
        
        # Prepare input data
        sequences_data = {
            'sequences': [vs['sequence'] for vs in vectorized_sequences],
            'metadata': {
                'total_sequences': len(vectorized_sequences),
                'method': method
            }
        }
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sequences_data, f)
            input_file = f.name
        
        try:
            # Call clustering script
            script_path = Path(__file__).parent / 'scripts' / 'cluster_sequences.py'
            result = subprocess.run(
                [
                    'python3',
                    str(script_path),
                    '--method', method,
                    '--n-clusters', str(n_clusters),
                    '--min-size', str(min_cluster_size),
                    '--input', input_file
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Clustering failed: {e.stderr}", file=sys.stderr)
            raise
        finally:
            # Clean up temp file
            Path(input_file).unlink()
    
    def build_behavioral_library(self, workspace_path: str | None = None,
                                min_cp: float = None) -> dict:
        """Build high-fidelity behavioral library"""
        min_cp = min_cp or config.MIN_CP_THRESHOLD
        
        # 1. Extract sequences
        sequences = self.extract_sequences(workspace_path=workspace_path)
        
        if not sequences:
            return {
                'error': 'No sequences found',
                'library': []
            }
        
        # 2. Vectorize
        vectorized = self.vectorize_sequences(sequences)
        
        # 3. Cluster
        clustering_result = self.cluster_sequences(vectorized)
        
        # 4. Calculate CP for each cluster and filter
        behavioral_library = []
        for cluster in clustering_result.get('behavioral_library', []):
            # Calculate average CP for sequences in cluster
            cp_scores = []
            for seq_idx in cluster.get('sequence_indices', []):
                seq = vectorized[seq_idx]['sequence']
                for event in seq:
                    prompt_id = event.get('prompt_id')
                    if prompt_id:
                        # Calculate CP (simplified)
                        pass
            
            # For now, include all clusters
            behavioral_library.append(cluster)
        
        return {
            'workspace_path': workspace_path,
            'total_sequences': len(sequences),
            'clusters_found': len(behavioral_library),
            'behavioral_library': behavioral_library,
            'clustering_metadata': clustering_result.get('metadata', {})
        }
    
    def save_library(self, library: Dict, output_path: Optional[Path] = None):
        """Save behavioral library to file"""
        if output_path is None:
            output_path = config.OUTPUT_DIR / 'behavioral_library.json'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(library, f, indent=2)
    
    def close(self):
        """Close database connections"""
        self.db.close()


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process sequences and build behavioral library')
    parser.add_argument('--workspace', type=str, default=None,
                       help='Workspace path to analyze')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path')
    parser.add_argument('--min-cp', type=float, default=None,
                       help='Minimum context precision threshold')
    
    args = parser.parse_args()
    
    processor = SequenceProcessor()
    
    try:
        library = processor.build_behavioral_library(
            workspace_path=args.workspace,
            min_cp=args.min_cp
        )
        
        output_path = Path(args.output) if args.output else None
        processor.save_library(library, output_path)
        
    finally:
        processor.close()


if __name__ == '__main__':
    main()
