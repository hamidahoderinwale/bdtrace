"""
Intent Extraction and Canonicalization

Privacy-preserving intent detection from prompts and events.

Supports two approaches (inspired by "Values in the Wild" - Huang et al. 2025):
1. KEYWORD-BASED: Fixed 10-category one-hot encoding (fast, deterministic)
2. EMERGENT: Bottom-up discovery via LLM extraction + embedding clustering

The emergent approach discovers intent categories from data rather than
pre-defining them, enabling more expressive and context-dependent intents.
"""

import functools
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Union

import numpy as np

try:
    import requests
except Exception:
    requests = None

# Optional dependencies for emergent intent
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False

try:
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# FIXED CATEGORIES (Backward Compatible)
# =============================================================================

# Intent categories for multi-dimensional encoding (keyword-based)
INTENT_CATEGORIES = [
    "DEBUG",      # Fixing errors, bugs, issues
    "FEATURE",    # Adding new functionality
    "REFACTOR",   # Improving code structure
    "TEST",       # Writing tests
    "DOCUMENT",   # Adding documentation
    "EXPLAIN",    # Understanding/explaining code
    "NAVIGATE",   # Browsing/exploring codebase
    "CONFIGURE",  # Setting up/configuring
    "DEPLOY",     # Deployment-related
    "REVIEW",     # Code review
]


# =============================================================================
# EMERGENT INTENT SYSTEM (Values in the Wild approach)
# =============================================================================

class EmergentIntentTaxonomy:
    """
    Bottom-up intent taxonomy discovery inspired by "Values in the Wild".
    
    Instead of pre-defined categories, this system:
    1. Extracts free-text intent descriptions via LLM
    2. Embeds descriptions into vector space
    3. Clusters embeddings to discover natural intent groupings
    4. Labels clusters to form emergent taxonomy
    
    Reference: Huang et al. "Values in the Wild: Discovering and Analyzing 
    Values in Real-World Language Model Interactions" (arXiv:2504.15236)
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None,
        min_cluster_size: int = 5,
    ):
        self.embedding_model_name = embedding_model
        self.cache_dir = cache_dir or Path(__file__).parent.parent / "cache" / "emergent_intent"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_cluster_size = min_cluster_size
        
        # Lazy-loaded components
        self._encoder = None
        self._taxonomy = None
        self._centroids = None
        self._cluster_labels = None
        
        # Intent extraction cache
        self._extraction_cache: Dict[str, str] = {}
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
    @property
    def encoder(self):
        """Lazy-load sentence transformer."""
        if self._encoder is None:
            if not SENTENCE_TRANSFORMER_AVAILABLE:
                raise ImportError(
                    "sentence-transformers required for emergent intent. "
                    "Install with: pip install sentence-transformers"
                )
            self._encoder = SentenceTransformer(self.embedding_model_name)
        return self._encoder
    
    def extract_intent_description(
        self,
        event: Dict,
        use_llm: bool = True,
        llm_extractor: Optional[callable] = None,
    ) -> str:
        """
        Extract free-text intent description from an event.
        
        Unlike keyword matching, this produces natural language descriptions
        that capture nuance (e.g., "fixing off-by-one error in pagination logic").
        
        Args:
            event: Event dictionary with type, details, etc.
            use_llm: If True, use LLM for extraction; else use heuristics
            llm_extractor: Optional custom LLM extraction function
            
        Returns:
            Natural language intent description
        """
        # Create cache key
        cache_key = hashlib.md5(json.dumps(event, sort_keys=True, default=str).encode()).hexdigest()
        
        if cache_key in self._extraction_cache:
            return self._extraction_cache[cache_key]
        
        # Build context for extraction
        event_type = event.get('type', 'unknown')
        details = event.get('details', {})
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except:
                details = {}
        
        file_path = details.get('file_path') or details.get('file', '')
        diff_summary = details.get('diff_summary', '')
        lines_added = details.get('lines_added', 0) or 0
        lines_removed = details.get('lines_removed', 0) or 0
        prompt_text = details.get('prompt') or details.get('prompt_text') or event.get('prompt', '')
        annotation = event.get('annotation') or event.get('intent', '')
        
        # Use annotation if available (most reliable)
        if annotation and len(str(annotation)) > 5:
            description = str(annotation)
        elif use_llm and llm_extractor:
            # Use custom LLM extractor
            description = llm_extractor(event)
        elif use_llm and OPENROUTER_KEY:
            # Use OpenRouter for extraction
            description = self._llm_extract_intent(event)
        else:
            # Heuristic-based description
            description = self._heuristic_intent_description(
                event_type, file_path, diff_summary, lines_added, lines_removed, prompt_text
            )
        
        self._extraction_cache[cache_key] = description
        return description
    
    def _heuristic_intent_description(
        self,
        event_type: str,
        file_path: str,
        diff_summary: str,
        lines_added: int,
        lines_removed: int,
        prompt_text: str,
    ) -> str:
        """Generate intent description from heuristics when LLM unavailable."""
        parts = []
        
        # Use diff summary if informative
        if diff_summary and len(diff_summary) > 10:
            parts.append(diff_summary[:100])
        
        # Use prompt if available
        if prompt_text and len(prompt_text) > 5:
            # Extract key phrases from prompt
            prompt_lower = prompt_text.lower()[:200]
            parts.append(f"prompted: {prompt_lower}")
        
        # Infer from change characteristics
        if lines_added > 50:
            parts.append("large addition")
        elif lines_removed > lines_added * 2:
            parts.append("major deletion or refactor")
        elif lines_added > 0 and lines_removed > 0:
            parts.append("modification")
        
        # Infer from file path
        if file_path:
            path_lower = str(file_path).lower()
            if 'test' in path_lower:
                parts.append("test-related")
            elif 'config' in path_lower or '.env' in path_lower:
                parts.append("configuration")
            elif 'readme' in path_lower or '.md' in path_lower:
                parts.append("documentation")
        
        # Infer from event type
        if 'create' in event_type:
            parts.append("creating new file")
        elif 'delete' in event_type:
            parts.append("removing file")
        
        return " | ".join(parts) if parts else "general code change"
    
    def _llm_extract_intent(self, event: Dict) -> str:
        """Use LLM to extract natural language intent description."""
        if not OPENROUTER_KEY or not requests:
            return self._heuristic_intent_description(
                event.get('type', ''),
                event.get('details', {}).get('file_path', ''),
                event.get('details', {}).get('diff_summary', ''),
                event.get('details', {}).get('lines_added', 0) or 0,
                event.get('details', {}).get('lines_removed', 0) or 0,
                event.get('details', {}).get('prompt', ''),
            )
        
        details = event.get('details', {})
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except:
                details = {}
        
        context = f"""Event type: {event.get('type', 'unknown')}
File: {details.get('file_path', 'unknown')}
Diff summary: {details.get('diff_summary', 'N/A')[:200]}
Lines added: {details.get('lines_added', 0)}
Lines removed: {details.get('lines_removed', 0)}"""
        
        prompt = f"""What is the developer trying to accomplish with this action?
{context}

Respond in 5-15 words describing the specific intent. Be concrete, not generic.
Examples of good responses:
- "fixing null pointer exception in user authentication"
- "adding pagination to search results"
- "refactoring database queries for performance"
- "exploring codebase to understand data flow"

Your response:"""

        try:
            response = requests.post(
                OPENROUTER_ENDPOINT,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                },
                json={
                    "model": os.getenv("OPENROUTER_INTENT_MODEL", "anthropic/claude-3-haiku"),
                    "temperature": 0.3,
                    "max_tokens": 50,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=10,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            # Fallback to heuristic
            return self._heuristic_intent_description(
                event.get('type', ''),
                details.get('file_path', ''),
                details.get('diff_summary', ''),
                details.get('lines_added', 0) or 0,
                details.get('lines_removed', 0) or 0,
                details.get('prompt', ''),
            )
    
    def embed_intent(self, description: str) -> np.ndarray:
        """Embed intent description into vector space."""
        if description in self._embedding_cache:
            return self._embedding_cache[description]
        
        embedding = self.encoder.encode(description, convert_to_numpy=True)
        self._embedding_cache[description] = embedding
        return embedding
    
    def discover_taxonomy(
        self,
        events: List[Dict],
        n_clusters: Optional[int] = None,
        use_llm: bool = True,
        llm_extractor: Optional[callable] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Discover intent taxonomy from a collection of events.
        
        This is the core "Values in the Wild" approach: bottom-up discovery
        of categories through clustering, rather than top-down definition.
        
        Args:
            events: List of event dictionaries
            n_clusters: Number of clusters (None = auto-detect with HDBSCAN)
            use_llm: Use LLM for intent extraction
            llm_extractor: Custom LLM extraction function
            
        Returns:
            Dictionary mapping cluster_id to {
                'label': str,  # Human-readable cluster label
                'description': str,  # Longer description
                'examples': List[str],  # Example intents in cluster
                'centroid': np.ndarray,  # Cluster centroid
                'size': int,  # Number of events in cluster
            }
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for taxonomy discovery")
        
        # Extract intent descriptions
        descriptions = []
        for event in events:
            desc = self.extract_intent_description(event, use_llm=use_llm, llm_extractor=llm_extractor)
            descriptions.append(desc)
        
        # Embed all descriptions
        embeddings = np.array([self.embed_intent(d) for d in descriptions])
        
        # Cluster embeddings
        if n_clusters is not None:
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        else:
            clusterer = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=2,
                metric='cosine',
            )
        
        labels = clusterer.fit_predict(embeddings)
        self._cluster_labels = labels
        
        # Build taxonomy
        taxonomy = {}
        unique_labels = set(labels)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Noise in HDBSCAN
                continue
                
            # Get cluster members
            mask = labels == cluster_id
            cluster_descriptions = [d for d, m in zip(descriptions, mask) if m]
            cluster_embeddings = embeddings[mask]
            
            # Compute centroid
            centroid = cluster_embeddings.mean(axis=0)
            
            # Find most representative description (closest to centroid)
            distances = cosine_similarity([centroid], cluster_embeddings)[0]
            representative_idx = np.argmax(distances)
            representative = cluster_descriptions[representative_idx]
            
            # Generate cluster label (use LLM if available)
            label = self._generate_cluster_label(cluster_descriptions, representative)
            
            taxonomy[cluster_id] = {
                'label': label,
                'description': representative,
                'examples': cluster_descriptions[:5],  # Top 5 examples
                'centroid': centroid,
                'size': len(cluster_descriptions),
            }
        
        self._taxonomy = taxonomy
        self._centroids = np.array([t['centroid'] for t in taxonomy.values()])
        
        return taxonomy
    
    def _generate_cluster_label(self, descriptions: List[str], representative: str) -> str:
        """Generate a short label for a cluster."""
        # Try LLM summarization
        if OPENROUTER_KEY and requests and len(descriptions) >= 3:
            try:
                examples = "\n".join(f"- {d}" for d in descriptions[:10])
                prompt = f"""These are examples of developer intents in a cluster:
{examples}

Generate a 2-4 word label for this cluster (like "bug fixing", "feature development", "test writing").
Label:"""
                
                response = requests.post(
                    OPENROUTER_ENDPOINT,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {OPENROUTER_KEY}",
                    },
                    json={
                        "model": os.getenv("OPENROUTER_INTENT_MODEL", "anthropic/claude-3-haiku"),
                        "temperature": 0.0,
                        "max_tokens": 20,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=10,
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
            except:
                pass
        
        # Fallback: use first few words of representative
        words = representative.split()[:4]
        return " ".join(words)
    
    def assign_intent(
        self,
        event: Dict,
        use_llm: bool = True,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Assign event to discovered intent categories with soft probabilities.
        
        Unlike one-hot encoding, this returns a distribution over categories,
        capturing blended intents (e.g., "70% bug fix, 30% refactor").
        
        Args:
            event: Event dictionary
            use_llm: Use LLM for intent extraction
            top_k: Return top K most likely intents
            
        Returns:
            List of (intent_label, probability) tuples, sorted by probability
        """
        if self._taxonomy is None or self._centroids is None:
            raise ValueError("Taxonomy not discovered. Call discover_taxonomy() first.")
        
        # Extract and embed intent
        description = self.extract_intent_description(event, use_llm=use_llm)
        embedding = self.embed_intent(description)
        
        # Compute similarity to all centroids
        similarities = cosine_similarity([embedding], self._centroids)[0]
        
        # Softmax to get probabilities
        exp_sim = np.exp(similarities - similarities.max())  # Numerical stability
        probabilities = exp_sim / exp_sim.sum()
        
        # Get top K
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        results = []
        taxonomy_list = list(self._taxonomy.values())
        for idx in top_indices:
            if idx < len(taxonomy_list):
                label = taxonomy_list[idx]['label']
                prob = float(probabilities[idx])
                results.append((label, prob))
        
        return results
    
    def get_soft_intent_vector(
        self,
        event: Dict,
        use_llm: bool = True,
    ) -> np.ndarray:
        """
        Get soft probability distribution over all discovered categories.
        
        This is the emergent equivalent of the one-hot encoding, but:
        - Categories are data-driven, not pre-defined
        - Values are continuous probabilities, not binary
        - Captures nuance and blended intents
        
        Args:
            event: Event dictionary
            use_llm: Use LLM for intent extraction
            
        Returns:
            numpy array of shape (n_categories,) with probabilities
        """
        if self._taxonomy is None or self._centroids is None:
            raise ValueError("Taxonomy not discovered. Call discover_taxonomy() first.")
        
        description = self.extract_intent_description(event, use_llm=use_llm)
        embedding = self.embed_intent(description)
        
        similarities = cosine_similarity([embedding], self._centroids)[0]
        exp_sim = np.exp(similarities - similarities.max())
        probabilities = exp_sim / exp_sim.sum()
        
        return probabilities
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save taxonomy to disk (pickle format for full state)."""
        path = path or (self.cache_dir / "taxonomy.pkl")
        
        data = {
            'taxonomy': self._taxonomy,
            'centroids': self._centroids,
            'cluster_labels': self._cluster_labels,
            'extraction_cache': self._extraction_cache,
            'embedding_model': self.embedding_model_name,
            'min_cluster_size': self.min_cluster_size,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        # Also save JSON version for readability and portability
        json_path = path.with_suffix('.json')
        self.save_json(json_path)
        
        return path
    
    def save_json(self, path: Optional[Path] = None) -> Path:
        """Save taxonomy to JSON (human-readable, portable format).
        
        Centroids are converted to lists for JSON serialization.
        This format is suitable for sharing, version control, and inspection.
        """
        path = path or (self.cache_dir / "taxonomy.json")
        
        if self._taxonomy is None:
            raise ValueError("Taxonomy not discovered. Call discover_taxonomy() first.")
        
        # Convert taxonomy to JSON-serializable format
        json_taxonomy = {}
        for cluster_id, info in self._taxonomy.items():
            json_taxonomy[str(cluster_id)] = {
                'label': info['label'],
                'description': info['description'],
                'examples': info['examples'],
                'centroid': info['centroid'].tolist() if isinstance(info['centroid'], np.ndarray) else info['centroid'],
                'size': info['size'],
            }
        
        json_data = {
            'taxonomy': json_taxonomy,
            'embedding_model': self.embedding_model_name,
            'min_cluster_size': self.min_cluster_size,
            'num_clusters': len(self._taxonomy),
            'total_events_processed': sum(info['size'] for info in self._taxonomy.values()),
        }
        
        with open(path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        return path
    
    def save_taxonomy(self, path: Optional[Path] = None) -> Path:
        """Alias for save() for backward compatibility.
        
        Saves both pickle (full state) and JSON (readable) formats.
        """
        return self.save(path)
    
    def load(self, path: Optional[Path] = None) -> bool:
        """Load taxonomy from disk."""
        path = path or (self.cache_dir / "taxonomy.pkl")
        
        if not path.exists():
            return False
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self._taxonomy = data['taxonomy']
            self._centroids = data['centroids']
            self._cluster_labels = data.get('cluster_labels', None)
            self._extraction_cache = data.get('extraction_cache', {})
            
            return True
        except Exception:
            return False
    
    def get_intent_tokens(
        self,
        event: Dict,
        use_llm: bool = True,
        threshold: float = 0.1,
    ) -> List[str]:
        """
        Get intent tokens for compatibility with existing rung system.
        
        Returns tokens like "EMERGENT_INTENT_bug_fixing" for categories
        above the probability threshold.
        
        Args:
            event: Event dictionary
            use_llm: Use LLM for intent extraction
            threshold: Minimum probability to include intent
            
        Returns:
            List of intent token strings
        """
        if self._taxonomy is None:
            # Fall back to keyword-based
            return extract_event_intent(event)
        
        assignments = self.assign_intent(event, use_llm=use_llm, top_k=5)
        
        tokens = []
        for label, prob in assignments:
            if prob >= threshold:
                # Sanitize label for token
                token_label = label.replace(" ", "_").replace("-", "_").upper()
                tokens.append(f"EMERGENT_INTENT_{token_label}")
        
        return tokens if tokens else ["EMERGENT_INTENT_OTHER"]


# Global taxonomy instance (lazy-loaded)
_emergent_taxonomy: Optional[EmergentIntentTaxonomy] = None


def get_emergent_taxonomy() -> EmergentIntentTaxonomy:
    """Get or create the global emergent taxonomy instance."""
    global _emergent_taxonomy
    if _emergent_taxonomy is None:
        _emergent_taxonomy = EmergentIntentTaxonomy()
        # Try to load cached taxonomy
        _emergent_taxonomy.load()
    return _emergent_taxonomy


def extract_emergent_intent(
    event: Dict,
    use_llm: bool = True,
    threshold: float = 0.1,
) -> List[str]:
    """
    Extract emergent intent tokens from an event.
    
    Drop-in replacement for extract_event_intent() that uses
    discovered categories instead of fixed ones.
    
    Args:
        event: Event dictionary
        use_llm: Use LLM for extraction
        threshold: Minimum probability threshold
        
    Returns:
        List of intent tokens
    """
    taxonomy = get_emergent_taxonomy()
    
    if taxonomy._taxonomy is None:
        # No taxonomy discovered yet, fall back to keyword-based
        return extract_event_intent(event)
    
    return taxonomy.get_intent_tokens(event, use_llm=use_llm, threshold=threshold)

def extract_intent_vector(text: str) -> List[str]:
    """Extract multi-dimensional intent vector (one-hot-like encoding).
    
    Returns multiple intent categories that apply to the text, enabling
    more expressive and informative intent representation.
    
    Args:
        text: Text content (prompt, event description, etc.)
    
    Returns:
        List of intent category strings (e.g., ["INTENT_DEBUG", "INTENT_FEATURE"])
    """
    if not text or not isinstance(text, str):
        return ["INTENT_OTHER"]
    
    text_lower = text.lower()
    intents = []
    
    # Debug/error fixing intent
    if any(word in text_lower for word in ["fix", "error", "bug", "debug", "broken", "issue", "problem", "wrong", "crash", "exception"]):
        intents.append("INTENT_DEBUG")
    
    # Feature creation intent
    if any(word in text_lower for word in ["add", "create", "implement", "new", "build", "make", "feature", "functionality"]):
        intents.append("INTENT_FEATURE")
    
    # Refactoring intent
    if any(word in text_lower for word in ["refactor", "clean", "improve", "optimize", "restructure", "reorganize", "simplify"]):
        intents.append("INTENT_REFACTOR")
    
    # Testing intent
    if any(word in text_lower for word in ["test", "verify", "check", "validate", "ensure", "assert", "spec"]):
        intents.append("INTENT_TEST")
    
    # Documentation intent
    if any(word in text_lower for word in ["document", "comment", "explain", "describe", "docstring", "readme", "docs"]):
        intents.append("INTENT_DOCUMENT")
    
    # Code review/explanation intent
    if any(word in text_lower for word in ["review", "explain", "understand", "what", "how", "why", "analyze", "inspect"]):
        intents.append("INTENT_EXPLAIN")
    
    # Navigation/exploration intent
    if any(word in text_lower for word in ["browse", "explore", "find", "search", "look", "navigate", "goto"]):
        intents.append("INTENT_NAVIGATE")
    
    # Configuration intent
    if any(word in text_lower for word in ["config", "setup", "install", "configure", "settings", "preferences"]):
        intents.append("INTENT_CONFIGURE")
    
    # Deployment intent
    if any(word in text_lower for word in ["deploy", "release", "publish", "ship", "production", "staging"]):
        intents.append("INTENT_DEPLOY")
    
    # Review intent (separate from explain)
    if any(word in text_lower for word in ["review", "pr", "pull request", "code review", "feedback"]):
        intents.append("INTENT_REVIEW")
    
    # If no intents detected, return OTHER
    if not intents:
        intents.append("INTENT_OTHER")
    
    return intents


def canonicalize_prompt(text: str) -> str:
    """Canonicalize prompt text to primary intent category (backward compatibility).
    
    Uses extract_intent_vector and returns the first (primary) intent.
    
    Args:
        text: Prompt text content
    
    Returns:
        Primary intent category string like "INTENT_DEBUG", "INTENT_FEATURE", etc.
    """
    intents = extract_intent_vector(text)
    return intents[0] if intents else "INTENT_OTHER"


def extract_keyword_intent(event: Dict, use_canonicalized_paths: bool = False) -> str:
    """
    Extract single keyword-based intent category from an event (backward compatibility).
    
    This is a convenience wrapper around extract_event_intent() that returns
    the primary (first) intent category as a single string.
    
    Args:
        event: Event dictionary
        use_canonicalized_paths: If True, works with canonicalized paths
        
    Returns:
        Single intent category string (e.g., "INTENT_DEBUG", "INTENT_FEATURE")
    """
    intents = extract_event_intent(event, use_canonicalized_paths=use_canonicalized_paths)
    return intents[0] if intents else "INTENT_OTHER"


def extract_event_intent(event: Dict, use_canonicalized_paths: bool = False) -> List[str]:
    """Extract intent from an event based on event characteristics.
    
    Analyzes event type, operation, file patterns, and code changes to infer intent.
    Works with both canonicalized and non-canonicalized data - canonicalization
    is a policy lever that doesn't affect intent extraction expressiveness.
    
    More expressive than prompt-only intent extraction: captures multi-dimensional
    intent from event characteristics, file patterns, code changes, and explicit annotations.
    
    Args:
        event: Event dictionary
        use_canonicalized_paths: If True, intent extraction works with canonicalized paths
                                (e.g., F001, FN001) - still extracts meaningful intent
    
    Returns:
        List of intent category strings (multi-dimensional, one-hot-like encoding)
    """
    import json
    from pathlib import Path
    
    intents = []
    
    # Check explicit intent annotation (most reliable signal)
    explicit_intent = event.get('intent') or event.get('annotation')
    if explicit_intent:
        if isinstance(explicit_intent, str):
            intents.extend(extract_intent_vector(explicit_intent))
        elif isinstance(explicit_intent, list):
            for intent in explicit_intent:
                if isinstance(intent, str):
                    intents.extend(extract_intent_vector(intent))
    
    # Infer from event type and details
    event_type = (event.get('type') or '').lower()
    details = event.get('details', {})
    
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except:
            details = {}
    
    if not isinstance(details, dict):
        details = {}
    
    # File operations - works with both canonicalized and named paths
    file_path = details.get('file_path') or details.get('file')
    if file_path:
        path_str = str(file_path).lower()
        
        # Pattern matching works even with canonicalized paths if patterns are preserved
        # For canonicalized paths (F001, FN001), we rely more on other signals
        
        if not use_canonicalized_paths:
            # Test files suggest testing intent
            if any(pattern in path_str for pattern in ['test', 'spec', '__tests__', '.test.', '.spec.']):
                intents.append("INTENT_TEST")
            
            # Config files suggest configuration intent
            if any(pattern in path_str for pattern in ['config', 'settings', '.env', 'package.json', 'requirements.txt', 'setup.py', 'pyproject.toml']):
                intents.append("INTENT_CONFIGURE")
            
            # Documentation files
            if any(pattern in path_str for pattern in ['readme', 'docs', '.md', 'docstring', 'changelog']):
                intents.append("INTENT_DOCUMENT")
        else:
            # With canonicalized paths, infer from file extension if available
            file_ext = details.get('file_extension', '').lower()
            if file_ext in ['.test.', '.spec.', '.test.js', '.test.ts', '.test.py']:
                intents.append("INTENT_TEST")
            elif file_ext in ['.md', '.txt', '.rst']:
                intents.append("INTENT_DOCUMENT")
    
    # Code change characteristics (most reliable signal, works regardless of canonicalization)
    lines_added = details.get('lines_added', 0) or 0
    lines_removed = details.get('lines_removed', 0) or 0
    total_changes = lines_added + lines_removed
    
    # Check diff summary FIRST (most expressive signal)
    diff_summary = details.get('diff_summary', '') or ''
    diff_summary_lower = str(diff_summary).lower()
    
    if diff_summary_lower:
        # Debug signals
        if any(word in diff_summary_lower for word in ['fix', 'bug', 'error', 'issue', 'broken', 'crash', 'exception']):
            intents.append("INTENT_DEBUG")
        # Feature signals
        if any(word in diff_summary_lower for word in ['add', 'create', 'implement', 'new', 'feature', 'function', 'method', 'class']):
            intents.append("INTENT_FEATURE")
        # Refactor signals
        if any(word in diff_summary_lower for word in ['refactor', 'clean', 'improve', 'optimize', 'restructure', 'reorganize', 'simplify']):
            intents.append("INTENT_REFACTOR")
        # Test signals
        if any(word in diff_summary_lower for word in ['test', 'spec', 'assert', 'verify', 'validate']):
            intents.append("INTENT_TEST")
        # Document signals
        if any(word in diff_summary_lower for word in ['document', 'comment', 'docstring', 'readme', 'docs']):
            intents.append("INTENT_DOCUMENT")
    
    # Code change size patterns (works even without diff summary)
    # Large additions suggest feature work
    if lines_added > 30 and lines_added > lines_removed * 1.5:
        intents.append("INTENT_FEATURE")
    
    # Large deletions might indicate refactoring
    if lines_removed > lines_added * 2 and lines_removed > 20:
        intents.append("INTENT_REFACTOR")
    
    # Small focused changes might indicate debugging
    if total_changes < 15 and total_changes > 0 and event_type in ('file_change', 'code_change'):
        intents.append("INTENT_DEBUG")
    
    # Balanced changes suggest refactoring
    if total_changes > 20 and abs(lines_added - lines_removed) < total_changes * 0.3:
        intents.append("INTENT_REFACTOR")
    
    # Medium-sized additions (15-30 lines) often indicate feature work
    if 15 <= lines_added <= 30 and lines_added > lines_removed:
        intents.append("INTENT_FEATURE")
    
    # New file creation suggests feature addition
    if event_type in ('file_created', 'entry_created'):
        intents.append("INTENT_FEATURE")
    
    # File deletion might indicate refactoring
    if event_type in ('file_deleted', 'entry_deleted'):
        intents.append("INTENT_REFACTOR")
    
    # Navigation events
    if event_type in ('file_open', 'file_close', 'goto', 'navigate', 'search'):
        intents.append("INTENT_NAVIGATE")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_intents = []
    for intent in intents:
        if intent not in seen:
            seen.add(intent)
            unique_intents.append(intent)
    
    return unique_intents if unique_intents else ["INTENT_OTHER"]


OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_INTENT_MODEL", "google/flan-t5-xl:2024-11-01")
OPENROUTER_ENDPOINT = os.getenv(
    "OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions"
)
LLM_INTENT_PREFIX = "INTENT_LLM_"
LLM_INTENT_OPTIONS = [
    "DEBUG",
    "FEATURE",
    "REFACTOR",
    "TEST",
    "DOCUMENT",
    "EXPLAIN",
    "OTHER",
]


class OpenRouterIntentClassifier:
    """LLM-backed intent classifier using OpenRouter chat completions."""

    def __init__(self):
        self.enabled = bool(OPENROUTER_KEY and requests)
        self.session = requests.Session() if self.enabled else None
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_KEY}",
        } if self.enabled else {}
        self.payload_template = {
            "model": OPENROUTER_MODEL,
            "temperature": 0.0,
            "max_tokens": 10,
            "messages": [],
        }
        self.system_prompt = (
            "You are an intent classifier. Respond with a single intent label from the "
            f"choices: {', '.join(LLM_INTENT_OPTIONS)}. Do not add any explanation."
        )

    @functools.lru_cache(maxsize=2048)
    def classify(self, text: str) -> str:
        if not self.enabled or not text:
            return "OTHER"
        payload = {
            **self.payload_template,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
        }
        try:
            response = self.session.post(OPENROUTER_ENDPOINT, headers=self.headers, json=payload, timeout=8)
            response.raise_for_status()
            data = response.json()
            label = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            label = label.upper().split()[0]
            if label in LLM_INTENT_OPTIONS:
                return label
        except Exception:
            return "OTHER"
        return "OTHER"


llm_intent_classifier = OpenRouterIntentClassifier()


def intent_tokens_for_prompt(prompt: str, include_llm: bool = False) -> list[str]:
    """Return intent tokens for a prompt, optionally including LLM-derived labels.
    
    Now returns multi-dimensional intent vector (one-hot-like encoding).
    """
    tokens = []
    # Get multi-dimensional intent vector
    intent_vector = extract_intent_vector(prompt)
    tokens.extend(intent_vector)
    
    if include_llm:
        llm_label = llm_intent_classifier.classify(prompt)
        tokens.append(f"{LLM_INTENT_PREFIX}{llm_label}")
    return tokens


def intent_tokens_for_event(
    event: Dict,
    include_llm: bool = False,
    use_canonicalized_paths: bool = False,
    use_emergent: bool = True,  # Default to emergent (Values in the Wild approach)
    emergent_threshold: float = 0.1,
) -> list[str]:
    """Extract intent tokens from an event (more expressive than prompt-only).
    
    Analyzes event characteristics to infer intent, making semantic edits
    intent-aware even without explicit prompts. Works with both canonicalized
    and non-canonicalized data - canonicalization is a policy lever.
    
    Args:
        event: Event dictionary
        include_llm: If True, include LLM-derived intent (for keyword approach)
        use_canonicalized_paths: If True, intent extraction adapts to canonicalized paths
        use_emergent: If True (default), use emergent taxonomy; False for fixed categories
        emergent_threshold: Minimum probability for emergent intents
    
    Returns:
        List of intent tokens (multi-dimensional encoding)
    """
    # Use emergent taxonomy if requested and available
    if use_emergent:
        try:
            return extract_emergent_intent(event, use_llm=True, threshold=emergent_threshold)
        except (ValueError, ImportError):
            # Fall back to keyword-based if taxonomy not ready
            pass
    
    tokens = []
    
    # Extract intent from event characteristics (works with canonicalized or named paths)
    intent_vector = extract_event_intent(event, use_canonicalized_paths=use_canonicalized_paths)
    tokens.extend(intent_vector)
    
    # Also check for prompts in event (most expressive signal)
    prompt_text = None
    if 'prompt' in event:
        prompt_text = event.get('prompt')
    elif 'details' in event:
        details = event.get('details', {})
        if isinstance(details, dict):
            prompt_text = details.get('prompt') or details.get('prompt_text')
    
    if prompt_text:
        prompt_intents = extract_intent_vector(str(prompt_text))
        tokens.extend(prompt_intents)
    
    if include_llm and prompt_text:
        llm_label = llm_intent_classifier.classify(str(prompt_text))
        tokens.append(f"{LLM_INTENT_PREFIX}{llm_label}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tokens = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)
    
    return unique_tokens if unique_tokens else ["INTENT_OTHER"]


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

def extract_intent(
    event_or_text: Union[Dict, str],
    method: str = "emergent",  # Default to emergent (Values in the Wild approach)
    use_llm: bool = False,
    threshold: float = 0.1,
) -> List[str]:
    """
    Unified intent extraction interface.
    
    Supports three methods:
    - "emergent": Bottom-up discovered categories (default, requires taxonomy discovery)
    - "keyword": Fixed 10-category keyword matching (fast, deterministic, for comparison)
    - "llm": LLM classification into fixed categories
    
    Args:
        event_or_text: Event dictionary or text string
        method: "emergent" (default), "keyword", or "llm"
        use_llm: For emergent method, whether to use LLM for extraction
        threshold: For emergent method, minimum probability threshold
        
    Returns:
        List of intent tokens
    """
    if method == "keyword":
        if isinstance(event_or_text, str):
            return extract_intent_vector(event_or_text)
        else:
            return extract_event_intent(event_or_text)
    
    elif method == "llm":
        if isinstance(event_or_text, str):
            return intent_tokens_for_prompt(event_or_text, include_llm=True)
        else:
            return intent_tokens_for_event(event_or_text, include_llm=True)
    
    elif method == "emergent":
        if isinstance(event_or_text, str):
            # Wrap text in event-like dict
            event = {"type": "text_input", "details": {"prompt": event_or_text}}
        else:
            event = event_or_text
        return extract_emergent_intent(event, use_llm=use_llm, threshold=threshold)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'keyword', 'llm', or 'emergent'.")


def discover_intent_taxonomy(
    events: List[Dict],
    n_clusters: Optional[int] = None,
    use_llm: bool = True,
    save_path: Optional[Path] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Discover intent taxonomy from events (Values in the Wild approach).
    
    This is the main entry point for bottom-up taxonomy discovery.
    Run this on a representative sample of events to discover natural
    intent categories before using emergent intent extraction.
    
    Args:
        events: List of event dictionaries
        n_clusters: Number of clusters (None = auto-detect)
        use_llm: Use LLM for intent extraction
        save_path: Path to save taxonomy (auto-saved to cache if None)
        
    Returns:
        Taxonomy dictionary mapping cluster_id to cluster info
        
    Example:
        >>> events = load_events_from_db()[:1000]  # Sample
        >>> taxonomy = discover_intent_taxonomy(events)
        >>> print(f"Discovered {len(taxonomy)} intent categories:")
        >>> for cid, info in taxonomy.items():
        ...     print(f"  {info['label']}: {info['size']} events")
    """
    taxonomy_instance = get_emergent_taxonomy()
    taxonomy = taxonomy_instance.discover_taxonomy(
        events,
        n_clusters=n_clusters,
        use_llm=use_llm,
    )
    
    # Save taxonomy
    save_path = save_path or taxonomy_instance.save()
    print(f"Taxonomy saved to: {save_path}")
    
    return taxonomy


def compare_intent_methods(
    events: List[Dict],
    sample_size: int = 100,
) -> Dict[str, Any]:
    """
    Compare keyword vs. emergent intent extraction.
    
    Useful for understanding what the emergent approach discovers
    that keywords miss.
    
    Args:
        events: List of events to analyze
        sample_size: Number of events to compare
        
    Returns:
        Comparison statistics
    """
    sample = events[:sample_size]
    
    results = {
        'keyword_intents': [],
        'emergent_intents': [],
        'agreements': 0,
        'disagreements': 0,
        'emergent_only': [],  # Categories emergent finds that keywords miss
    }
    
    taxonomy = get_emergent_taxonomy()
    has_taxonomy = taxonomy._taxonomy is not None
    
    for event in sample:
        keyword = extract_event_intent(event)
        results['keyword_intents'].extend(keyword)
        
        if has_taxonomy:
            emergent = extract_emergent_intent(event)
            results['emergent_intents'].extend(emergent)
            
            # Check for agreement
            keyword_set = set(k.replace("INTENT_", "").lower() for k in keyword)
            emergent_labels = [e.replace("EMERGENT_INTENT_", "").lower() for e in emergent]
            
            # Simple heuristic: any word overlap?
            overlap = any(
                any(kw in el or el in kw for kw in keyword_set)
                for el in emergent_labels
            )
            
            if overlap:
                results['agreements'] += 1
            else:
                results['disagreements'] += 1
                results['emergent_only'].append({
                    'event_type': event.get('type'),
                    'keyword': keyword,
                    'emergent': emergent,
                })
    
    # Summarize
    from collections import Counter
    results['keyword_distribution'] = dict(Counter(results['keyword_intents']))
    if has_taxonomy:
        results['emergent_distribution'] = dict(Counter(results['emergent_intents']))
        results['agreement_rate'] = results['agreements'] / max(1, results['agreements'] + results['disagreements'])
    
    return results






