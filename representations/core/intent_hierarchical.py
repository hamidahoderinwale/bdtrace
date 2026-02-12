"""
Hierarchical Intent Taxonomy Discovery

Extends EmergentIntentTaxonomy with hierarchical clustering approaches inspired by Clio.

Supports multiple hierarchical strategies:
1. Agglomerative Hierarchical Clustering: Bottom-up tree construction
2. Recursive Clustering: Multi-level summarization (events → sub-clusters → clusters → super-clusters)
3. Hierarchical Annotation: Prescribed top-level + emergent sub-categories

Reference: Clio (2024) - recursive, human-guided clustering for conversation analysis
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .intent import EmergentIntentTaxonomy


class HierarchicalIntentTaxonomy(EmergentIntentTaxonomy):
    """
    Hierarchical extension of EmergentIntentTaxonomy.
    
    Builds multi-level taxonomies where:
    - Level 0: Individual events
    - Level 1: Fine-grained sub-clusters (e.g., "fix null pointer")
    - Level 2: Medium-grained clusters (e.g., "error handling")
    - Level 3: Coarse-grained super-clusters (e.g., "debugging")
    
    This enables multi-granularity analysis: same workflow can be analyzed
    at different abstraction levels depending on the question.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None,
        min_cluster_size: int = 5,
        max_levels: int = 3,
    ):
        super().__init__(embedding_model, cache_dir, min_cluster_size)
        self.max_levels = max_levels
        self._hierarchy = None  # Dict[level -> Dict[cluster_id -> cluster_info]]
        self._cluster_tree = None  # Tree structure: parent -> children mapping
        
    def discover_taxonomy_hierarchical(
        self,
        events: List[Dict],
        strategy: str = "agglomerative",
        n_levels: Optional[int] = None,
        use_llm: bool = True,
        llm_extractor: Optional[callable] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Discover hierarchical intent taxonomy.
        
        Args:
            events: List of event dictionaries
            strategy: "agglomerative" (bottom-up tree) or "recursive" (multi-level summarization)
            n_levels: Number of hierarchy levels (None = auto-determine)
            use_llm: Use LLM for intent extraction
            llm_extractor: Custom LLM extraction function
            
        Returns:
            Dictionary mapping (level, cluster_id) -> cluster_info
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for hierarchical taxonomy")
        
        n_levels = n_levels or self.max_levels
        
        # Extract and embed descriptions (same as flat approach)
        descriptions = []
        for event in events:
            desc = self.extract_intent_description(event, use_llm=use_llm, llm_extractor=llm_extractor)
            descriptions.append(desc)
        
        embeddings = np.array([self.embed_intent(d) for d in descriptions])
        
        if strategy == "agglomerative":
            return self._discover_agglomerative(embeddings, descriptions, n_levels)
        elif strategy == "recursive":
            return self._discover_recursive(embeddings, descriptions, n_levels)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'agglomerative' or 'recursive'")
    
    def _discover_agglomerative(
        self,
        embeddings: np.ndarray,
        descriptions: List[str],
        n_levels: int,
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """
        Build hierarchy using agglomerative clustering.
        
        Creates a dendrogram-like structure by clustering at multiple granularities.
        """
        hierarchy = {}
        cluster_tree = defaultdict(list)  # parent -> [children]
        
        # Level 0: Individual events (leaf nodes)
        level_0 = {}
        for idx, desc in enumerate(descriptions):
            level_0[idx] = {
                'label': desc[:50] + "..." if len(desc) > 50 else desc,
                'description': desc,
                'examples': [desc],
                'centroid': embeddings[idx],
                'size': 1,
                'event_indices': [idx],
            }
        hierarchy[0] = level_0
        
        # Build levels bottom-up
        current_embeddings = embeddings
        current_cluster_ids = list(range(len(descriptions)))
        current_level_data = level_0
        
        for level in range(1, n_levels + 1):
            # Determine number of clusters for this level
            # Coarser at higher levels
            n_clusters = max(2, len(current_cluster_ids) // (2 ** level))
            n_clusters = min(n_clusters, len(current_cluster_ids))
            
            if n_clusters < 2:
                break
            
            # Cluster current level's centroids
            centroids = np.array([current_level_data[cid]['centroid'] for cid in current_cluster_ids])
            
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward',
                metric='euclidean',
            )
            level_labels = clusterer.fit_predict(centroids)
            
            # Build level taxonomy
            level_taxonomy = {}
            for cluster_id in set(level_labels):
                # Get members of this cluster
                member_indices = [i for i, label in enumerate(level_labels) if label == cluster_id]
                member_cids = [current_cluster_ids[i] for i in member_indices]
                
                # Aggregate cluster info
                member_centroids = [current_level_data[cid]['centroid'] for cid in member_cids]
                cluster_centroid = np.mean(member_centroids, axis=0)
                
                # Collect all event indices from children
                all_event_indices = []
                all_examples = []
                total_size = 0
                for cid in member_cids:
                    all_event_indices.extend(current_level_data[cid].get('event_indices', []))
                    all_examples.extend(current_level_data[cid].get('examples', [])[:3])
                    total_size += current_level_data[cid]['size']
                    cluster_tree[(level, cluster_id)].append((level - 1, cid))
                
                # Generate label (summarize children)
                label = self._summarize_cluster_labels(
                    [current_level_data[cid]['label'] for cid in member_cids]
                )
                
                level_taxonomy[cluster_id] = {
                    'label': label,
                    'description': f"Aggregation of {len(member_cids)} sub-clusters",
                    'examples': all_examples[:5],
                    'centroid': cluster_centroid,
                    'size': total_size,
                    'event_indices': all_event_indices,
                    'children': member_cids,
                }
            
            hierarchy[level] = level_taxonomy
            current_cluster_ids = list(level_taxonomy.keys())
            current_level_data = level_taxonomy
        
        self._hierarchy = hierarchy
        self._cluster_tree = dict(cluster_tree)
        
        # Flatten for return (keyed by (level, cluster_id))
        flattened = {}
        for level, level_data in hierarchy.items():
            for cluster_id, cluster_info in level_data.items():
                flattened[(level, cluster_id)] = cluster_info
        
        return flattened
    
    def _discover_recursive(
        self,
        embeddings: np.ndarray,
        descriptions: List[str],
        n_levels: int,
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """
        Build hierarchy using recursive clustering (Clio-style).
        
        At each level:
        1. Cluster embeddings
        2. Summarize clusters bottom-up
        3. Use cluster summaries as input for next level
        """
        hierarchy = {}
        cluster_tree = defaultdict(list)
        
        # Level 0: Fine-grained clusters (similar to flat approach)
        # Use HDBSCAN for natural grouping
        from sklearn.cluster import HDBSCAN
        
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=2,
            metric='cosine',
        )
        level_0_labels = clusterer.fit_predict(embeddings)
        
        # Build level 0 taxonomy
        level_0 = {}
        unique_labels = set(level_0_labels)
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Noise
                continue
            
            mask = level_0_labels == cluster_id
            cluster_embeddings = embeddings[mask]
            cluster_descriptions = [d for d, m in zip(descriptions, mask) if m]
            event_indices = [i for i, m in enumerate(mask) if m]
            
            centroid = cluster_embeddings.mean(axis=0)
            label = self._generate_cluster_label(cluster_descriptions, cluster_descriptions[0])
            
            level_0[cluster_id] = {
                'label': label,
                'description': cluster_descriptions[0],
                'examples': cluster_descriptions[:5],
                'centroid': centroid,
                'size': len(cluster_descriptions),
                'event_indices': event_indices,
            }
        
        hierarchy[0] = level_0
        
        # Recursively build higher levels
        current_level = 0
        current_clusters = level_0
        
        for level in range(1, n_levels + 1):
            if len(current_clusters) < 2:
                break
            
            # Cluster the centroids from previous level
            centroids = np.array([c['centroid'] for c in current_clusters.values()])
            cluster_ids = list(current_clusters.keys())
            
            # Determine number of clusters (coarser at higher levels)
            n_clusters = max(2, len(cluster_ids) // 3)
            n_clusters = min(n_clusters, len(cluster_ids))
            
            if n_clusters < 2:
                break
            
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward',
            )
            level_labels = clusterer.fit_predict(centroids)
            
            # Build level taxonomy
            level_taxonomy = {}
            for cluster_id in set(level_labels):
                member_indices = [i for i, label in enumerate(level_labels) if label == cluster_id]
                member_cids = [cluster_ids[i] for i in member_indices]
                
                # Aggregate
                member_centroids = [current_clusters[cid]['centroid'] for cid in member_cids]
                cluster_centroid = np.mean(member_centroids, axis=0)
                
                all_event_indices = []
                all_examples = []
                total_size = 0
                for cid in member_cids:
                    all_event_indices.extend(current_clusters[cid].get('event_indices', []))
                    all_examples.extend(current_clusters[cid].get('examples', [])[:2])
                    total_size += current_clusters[cid]['size']
                    cluster_tree[(level, cluster_id)].append((level - 1, cid))
                
                # Summarize labels from children
                child_labels = [current_clusters[cid]['label'] for cid in member_cids]
                label = self._summarize_cluster_labels(child_labels)
                
                level_taxonomy[cluster_id] = {
                    'label': label,
                    'description': f"Recursive aggregation: {', '.join(child_labels[:3])}",
                    'examples': all_examples[:5],
                    'centroid': cluster_centroid,
                    'size': total_size,
                    'event_indices': all_event_indices,
                    'children': member_cids,
                }
            
            hierarchy[level] = level_taxonomy
            current_clusters = level_taxonomy
            current_level = level
        
        self._hierarchy = hierarchy
        self._cluster_tree = dict(cluster_tree)
        
        # Flatten for return
        flattened = {}
        for level, level_data in hierarchy.items():
            for cluster_id, cluster_info in level_data.items():
                flattened[(level, cluster_id)] = cluster_info
        
        return flattened
    
    def _summarize_cluster_labels(self, labels: List[str]) -> str:
        """Summarize multiple cluster labels into a higher-level label."""
        # Simple heuristic: find common words or use first few labels
        if len(labels) == 1:
            return labels[0]
        
        # Extract key words from labels
        words = []
        for label in labels:
            words.extend(label.lower().split()[:2])
        
        # Find most common words
        from collections import Counter
        word_counts = Counter(words)
        common_words = [w for w, c in word_counts.most_common(2)]
        
        if common_words:
            return " ".join(common_words).title()
        else:
            return f"{labels[0]} + {len(labels) - 1} more"
    
    def get_hierarchy_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the hierarchy."""
        if self._hierarchy is None:
            return {}
        
        summary = {
            'n_levels': len(self._hierarchy),
            'clusters_per_level': {},
            'total_clusters': 0,
            'avg_cluster_size_per_level': {},
        }
        
        for level, level_data in self._hierarchy.items():
            n_clusters = len(level_data)
            avg_size = np.mean([c['size'] for c in level_data.values()]) if level_data else 0
            
            summary['clusters_per_level'][level] = n_clusters
            summary['avg_cluster_size_per_level'][level] = avg_size
            summary['total_clusters'] += n_clusters
        
        return summary
    
    def assign_intent_hierarchical(
        self,
        event: Dict,
        level: int = 0,
        use_llm: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Assign event to hierarchical taxonomy at specified level.
        
        Args:
            event: Event dictionary
            level: Hierarchy level (0 = finest, higher = coarser)
            use_llm: Use LLM for intent extraction
            
        Returns:
            List of (intent_label, probability) tuples
        """
        if self._hierarchy is None:
            raise ValueError("Hierarchy not discovered. Call discover_taxonomy_hierarchical() first.")
        
        if level not in self._hierarchy:
            raise ValueError(f"Level {level} not found in hierarchy. Available levels: {list(self._hierarchy.keys())}")
        
        # Extract and embed event
        description = self.extract_intent_description(event, use_llm=use_llm)
        embedding = self.embed_intent(description)
        
        # Find closest cluster at specified level
        level_clusters = self._hierarchy[level]
        similarities = []
        
        for cluster_id, cluster_info in level_clusters.items():
            centroid = cluster_info['centroid']
            similarity = np.dot(embedding, centroid) / (np.linalg.norm(embedding) * np.linalg.norm(centroid))
            similarities.append((cluster_info['label'], float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities







