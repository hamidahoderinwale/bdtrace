"""
Motif Mining Algorithms

Statistical sequence pattern mining for workflow discovery.
Uses PrefixSpan, Sequitur, transitions, and structural analysis.

Includes MotifRegistry for tracking hash→pattern mappings and generating
natural language descriptions for all discovered motifs.
"""

import hashlib
import json
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple


# =============================================================================
# MOTIF REGISTRY - Track hash→pattern mappings for meaningful descriptions
# =============================================================================

class MotifRegistry:
    """Registry that tracks motif hash → original pattern mappings.
    
    This solves the problem of hash-based motifs (M_xxx) being meaningless.
    The registry maintains a global mapping that can be used to:
    1. Look up what pattern a hash represents
    2. Generate natural language descriptions
    3. Categorize motifs by type
    """
    
    _instance = None
    _registry: Dict[str, str] = {}  # hash → original pattern
    _descriptions: Dict[str, str] = {}  # hash → natural language description
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, original: str, hashed: str) -> None:
        """Register a hash→pattern mapping."""
        cls._registry[hashed] = original
    
    @classmethod
    def get_original(cls, hashed: str) -> Optional[str]:
        """Get the original pattern for a hash."""
        return cls._registry.get(hashed)
    
    @classmethod
    def describe(cls, motif: str) -> str:
        """Get a natural language description for a motif."""
        # Check cache first
        if motif in cls._descriptions:
            return cls._descriptions[motif]
        
        # Generate description
        desc = cls._generate_description(motif)
        cls._descriptions[motif] = desc
        return desc
    
    @classmethod
    def get_category(cls, motif: str) -> str:
        """Get the behavioral category for a motif."""
        return cls._categorize(motif)
    
    @classmethod
    def _generate_description(cls, motif: str) -> str:
        """Generate a natural language description for any motif type."""
        
        # First check if it's a hash we can look up
        if motif.startswith('M_') and motif in cls._registry:
            original = cls._registry[motif]
            return cls._describe_original_pattern(original)
        
        # Handle known pattern types directly
        return cls._describe_pattern_by_type(motif)
    
    @classmethod
    def _describe_original_pattern(cls, pattern: str) -> str:
        """Describe the original (pre-hash) pattern."""
        
        # Transition pattern: T_EV_xxx_EV_yyy
        if pattern.startswith('T_'):
            parts = pattern[2:].split('_')
            if len(parts) >= 4:
                # T_EV_hash1_EV_hash2
                return "Sequential Edit Transition"
            return "Edit Transition"
        
        # PrefixSpan pattern: PS_EV_xxx_EV_yyy...
        if pattern.startswith('PS_'):
            parts = pattern[3:].split('_')
            n_events = sum(1 for p in parts if p == 'EV')
            if n_events >= 3:
                return f"Frequent {n_events}-Step Sequence"
            return "Frequent Edit Sequence"
        
        # Sequitur rule: SQ_xxx
        if pattern.startswith('SQ_'):
            return "Compressed Edit Pattern"
        
        # Cycle pattern: CYCLE_EV_xxx_EV_yyy
        if pattern.startswith('CYCLE_'):
            return "Iterative Edit Cycle"
        
        # Hotspot: HOT_EV_xxx_N
        if pattern.startswith('HOT_'):
            match = re.search(r'_(\d+)$', pattern)
            if match:
                count = match.group(1)
                return f"Edit Hotspot ({count}x)"
            return "Edit Hotspot"
        
        # High switching
        if pattern == 'HIGH_SWITCHING':
            return "High Edit Diversity"
        
        # Intent patterns
        if 'INTENT_' in pattern:
            intent = pattern.split('INTENT_')[-1].split('_')[0]
            return f"{intent.title()} Intent Signal"
        
        # Generic event pattern
        if pattern.startswith('EV_'):
            return "Edit Event"
        
        # Fallback
        return "Workflow Pattern"
    
    @classmethod
    def _describe_pattern_by_type(cls, motif: str) -> str:
        """Describe a motif based on its type prefix."""
        
        # Hash-based motif without lookup
        if motif.startswith('M_'):
            # Use hash to deterministically select a description type
            hash_val = motif[2:6]
            try:
                idx = int(hash_val, 16) % 8
            except ValueError:
                idx = 0
            
            pattern_types = [
                'Edit Sequence',
                'Code Change Flow',
                'Multi-File Update',
                'Refactor Pattern',
                'Navigation Sequence',
                'Development Flow',
                'Modification Chain',
                'Workflow Step',
            ]
            return f"{pattern_types[idx]} #{motif[2:6]}"
        
        # Transition
        if motif.startswith('T_'):
            return "Edit Transition"
        
        # PrefixSpan
        if motif.startswith('PS_'):
            return "Frequent Sequence"
        
        # Sequitur
        if motif.startswith('SQ_'):
            return "Compression Rule"
        
        # Cycle
        if motif.startswith('CYCLE_'):
            return "Edit Cycle"
        
        # Hotspot
        if motif.startswith('HOT') or motif.startswith('HOTSPOT'):
            match = re.search(r'(\d+)', motif)
            if match:
                return f"Edit Hotspot ({match.group(1)}x)"
            return "Edit Hotspot"
        
        # Intent
        if motif.startswith('INTENT_TYPE_'):
            intent = motif.replace('INTENT_TYPE_', '').replace('_', ' ').title()
            return f"{intent} Intent"
        
        if motif.startswith('INTENT_TRANS'):
            return "Intent Transition"
        
        if motif.startswith('INTENT_'):
            intent = motif.replace('INTENT_', '').replace('_', ' ').title()
            return f"{intent} Signal"
        
        # Dependency
        if motif.startswith('DEPENDENCY'):
            return "Dependency Traversal"
        
        # High switching
        if 'SWITCHING' in motif.upper():
            return "High Edit Diversity"
        
        # N-gram
        if motif.startswith('NG_'):
            return "N-gram Pattern"
        
        # Fallback
        if len(motif) <= 20:
            return motif
        return motif[:17] + '...'
    
    @classmethod
    def _categorize(cls, motif: str) -> str:
        """Categorize a motif into behavioral categories."""
        
        # Check original pattern if available
        original = cls._registry.get(motif, motif)
        
        # Transition/sequence patterns
        if any(motif.startswith(p) for p in ['T_', 'PS_', 'TRANS_', 'NG_']):
            return 'Sequential Pattern'
        
        if motif.startswith('M_'):
            # Hash-based - check original
            if original != motif:
                if original.startswith('T_'):
                    return 'Sequential Pattern'
                if original.startswith('PS_'):
                    return 'Frequent Sequence'
                if original.startswith('SQ_'):
                    return 'Compression Pattern'
                if original.startswith('CYCLE_'):
                    return 'Iterative Pattern'
                if original.startswith('HOT_'):
                    return 'Hotspot Pattern'
            # Default for hash
            return 'Mined Pattern'
        
        # Structural patterns
        if motif.startswith('CYCLE_') or 'CYCLE' in motif:
            return 'Iterative Pattern'
        
        if motif.startswith('HOT') or motif.startswith('HOTSPOT'):
            return 'Hotspot Pattern'
        
        if 'SWITCHING' in motif.upper():
            return 'Diversity Pattern'
        
        # Intent patterns
        if motif.startswith('INTENT_'):
            return 'Intent Signal'
        
        # Dependency patterns
        if motif.startswith('DEPENDENCY'):
            return 'Dependency Pattern'
        
        # Compression patterns
        if motif.startswith('SQ_'):
            return 'Compression Pattern'
        
        return 'Other Pattern'
    
    @classmethod
    def clear(cls) -> None:
        """Clear the registry (useful for testing)."""
        cls._registry.clear()
        cls._descriptions.clear()
    
    @classmethod
    def stats(cls) -> Dict[str, int]:
        """Get registry statistics."""
        return {
            'registered_hashes': len(cls._registry),
            'cached_descriptions': len(cls._descriptions),
        }


# Global registry instance
motif_registry = MotifRegistry()


def transition_motifs(seq: List[str], max_count: int = 100) -> List[str]:
    """Extract transition motifs (Markov 1-step transitions).
    
    Captures sequential patterns: A -> B transitions.
    
    Args:
        seq: Sequence of event symbols
        max_count: Maximum number of transitions to return
    
    Returns:
        List of transition motif strings like "T_EV_a13f92_EV_10c99d"
    """
    if len(seq) < 2:
        return []
    
    motifs = []
    for a, b in zip(seq, seq[1:]):
        motifs.append(f"T_{a}_{b}")
    
    return motifs[:max_count]


def structural_motifs(seq: List[str]) -> List[str]:
    """Extract structural motifs (cycles, hotspots, switching patterns).
    
    Args:
        seq: Sequence of event symbols
    
    Returns:
        List of structural motif strings
    """
    motifs = []
    
    if len(seq) < 2:
        return motifs
    
    # Cycles: A -> B -> A pattern
    for i in range(len(seq) - 2):
        if seq[i] == seq[i+2] and seq[i] != seq[i+1]:
            motifs.append(f"CYCLE_{seq[i]}_{seq[i+1]}")
    
    # Hotspots: frequent repeated event types
    cnt = Counter(seq)
    for sym, k in cnt.items():
        if k >= 5:  # Threshold for "hotspot"
            motifs.append(f"HOT_{sym}_{k}")
    
    # High-switching: high diversity (many unique items)
    if len(seq) > 0:
        diversity_ratio = len(set(seq)) / len(seq)
        if diversity_ratio > 0.7:
            motifs.append("HIGH_SWITCHING")
    
    return motifs


def prefixspan(seq: List[str], min_support: int = 2, max_len: int = 4) -> List[List[str]]:
    """PrefixSpan algorithm for frequent sequence pattern mining.
    
    Discovers repeated subsequences without semantic knowledge.
    
    Args:
        seq: Input sequence
        min_support: Minimum support count for a pattern
        max_len: Maximum pattern length
    
    Returns:
        List of frequent patterns (each pattern is a list of symbols)
    """
    patterns = []
    
    def project(prefix: List[str], projected_db: List[List[str]]):
        """Recursive projection step."""
        if len(prefix) >= max_len:
            return
        
        # Count items in projected database
        counts = Counter()
        for s in projected_db:
            visited = set()
            for item in s:
                if item not in visited:
                    counts[item] += 1
                    visited.add(item)
        
        # Extend prefix with frequent items
        for item, count in counts.items():
            if count >= min_support:
                new_prefix = prefix + [item]
                patterns.append(new_prefix)
                
                # Create new projected database
                new_db = []
                for s in projected_db:
                    try:
                        idx = s.index(item)
                        new_db.append(s[idx+1:])
                    except ValueError:
                        continue
                
                # Recursively project
                if len(new_prefix) < max_len:
                    project(new_prefix, new_db)
    
    # Initialize with full sequence as single-item database
    project([], [seq])
    return patterns


def prefixspan_motifs(seq: List[str], min_support: int = 2, max_len: int = 4) -> List[str]:
    """Extract frequent sequence motifs using PrefixSpan.
    
    Args:
        seq: Sequence of event symbols
        min_support: Minimum support count
        max_len: Maximum pattern length
    
    Returns:
        List of prefixspan motif strings like "PS_EV_a13f92_EV_10c99d"
    """
    if len(seq) < 2:
        return []
    
    patterns = prefixspan(seq, min_support=min_support, max_len=max_len)
    return [f"PS_{'_'.join(p)}" for p in patterns if len(p) >= 2]  # Only multi-item patterns


def sequitur_rules(seq: List[str]) -> List[str]:
    """Extract compression motifs using Sequitur algorithm.
    
    Sequitur discovers repeated subsequences without semantics.
    This is a simplified implementation that finds common bigrams/trigrams.
    
    Args:
        seq: Sequence of event symbols
    
    Returns:
        List of sequitur motif strings
    """
    if len(seq) < 2:
        return []
    
    motifs = []
    
    # Find repeated bigrams (simplified Sequitur)
    bigram_counts = Counter()
    for i in range(len(seq) - 1):
        bigram = (seq[i], seq[i+1])
        bigram_counts[bigram] += 1
    
    # Extract bigrams that appear multiple times
    for bigram, count in bigram_counts.items():
        if count >= 2:
            motifs.append(f"SQ_{bigram[0]}_{bigram[1]}")
    
    # Find repeated trigrams
    if len(seq) >= 3:
        trigram_counts = Counter()
        for i in range(len(seq) - 2):
            trigram = (seq[i], seq[i+1], seq[i+2])
            trigram_counts[trigram] += 1
        
        for trigram, count in trigram_counts.items():
            if count >= 2:
                motifs.append(f"SQ_{trigram[0]}_{trigram[1]}_{trigram[2]}")
    
    return motifs


def unify_motifs(*motif_lists: List[str], max_total: int = 300, register: bool = True) -> List[str]:
    """Combine, hash, deduplicate, and bound motif sets.
    
    Now registers hash→pattern mappings in the MotifRegistry so that
    hashed motifs can be decoded into meaningful descriptions.
    
    Args:
        *motif_lists: Variable number of motif lists to combine
        max_total: Maximum total motifs to return
        register: If True, register hash→pattern mappings in MotifRegistry
    
    Returns:
        Unified, deduplicated, bounded list of motifs
    """
    motifs = []
    for mlist in motif_lists:
        motifs.extend(mlist)
    
    # Hash motifs to bound cardinality & protect privacy
    # Use first 10 hex chars for stable but readable hashes
    hashed = []
    seen_hashes = set()
    
    for m in motifs:
        h = hashlib.sha1(m.encode()).hexdigest()[:10]
        hashed_motif = f"M_{h}"
        
        if h not in seen_hashes:
            hashed.append(hashed_motif)
            seen_hashes.add(h)
            
            # Register the hash→pattern mapping for later description
            if register:
                motif_registry.register(m, hashed_motif)
    
    # Bound total
    return hashed[:max_total]


def motifs_from_sequence(seq: List[str], max_total: int = 300) -> List[str]:
    """Extract all motif types from a sequence using statistical methods.
    
    This is the core motif extraction function that combines:
    - Transition motifs (Markov)
    - Structural motifs (cycles, hotspots, switching)
    - PrefixSpan frequent sequences
    - Sequitur compression rules
    
    Args:
        seq: Sequence of event symbols
        max_total: Maximum total motifs to return
    
    Returns:
        Unified list of motif strings
    """
    if not seq or len(seq) < 2:
        return []
    
    return unify_motifs(
        transition_motifs(seq),
        structural_motifs(seq),
        prefixspan_motifs(seq, min_support=2, max_len=4),
        sequitur_rules(seq),
        max_total=max_total
    )


def extract_universal_motifs(
    sequence: List[str],
    include_transitions: bool = True,
    include_ngrams: bool = True,
    include_structural: bool = True,
    ngram_sizes: List[int] = [3, 4],
    use_statistical_mining: bool = True
) -> List[str]:
    """Extract motifs from any sequence using universal programmatic methods.
    
    This function works consistently across all rungs:
    - tokens sequences
    - semantic_edits sequences  
    - functions sequences
    - module_graph sequences
    
    Now uses statistical mining (PrefixSpan, Sequitur) instead of hard-coded rules.
    
    Args:
        sequence: List of items (tokens, edits, functions, etc.)
        include_transitions: Extract transition motifs (bigrams)
        include_ngrams: Extract n-gram motifs (legacy, kept for compatibility)
        include_structural: Extract structural patterns (repetition, cycles)
        ngram_sizes: Sizes of n-grams to extract (legacy, kept for compatibility)
        use_statistical_mining: If True, use PrefixSpan/Sequitur; if False, use legacy n-grams
    
    Returns:
        List of motif strings in unified format
    """
    if not sequence or len(sequence) < 2:
        return []
    
    motifs = []
    
    # Use new statistical mining approach by default
    if use_statistical_mining:
        motifs.extend(motifs_from_sequence(sequence))
    
    # Legacy n-gram approach (for backward compatibility)
    if include_ngrams and not use_statistical_mining:
        for n in ngram_sizes:
            if len(sequence) >= n:
                for i in range(len(sequence) - n + 1):
                    ngram = sequence[i:i+n]
                    motif = f"NG_{'_'.join(str(item) for item in ngram)}"
                    motifs.append(motif)
    
    # Legacy transition approach (for backward compatibility)
    if include_transitions and not use_statistical_mining:
        for i in range(len(sequence) - 1):
            transition = f"TRANS_{sequence[i]}_{sequence[i+1]}"
            motifs.append(transition)
    
    # Legacy structural patterns (for backward compatibility)
    if include_structural and not use_statistical_mining:
        item_counts = Counter(sequence)
        for item, count in item_counts.items():
            if count >= 5:
                motifs.append(f"HOTSPOT_{item}_{count}")
        
        for i in range(len(sequence) - 2):
            if sequence[i] == sequence[i+2] and sequence[i] != sequence[i+1]:
                motifs.append(f"CYCLE_{sequence[i]}_{sequence[i+1]}")
        
        if len(set(sequence)) > len(sequence) * 0.7:
            motifs.append("RAPID_SWITCH")
    
    # Include the sequence itself (for sequence modeling)
    # This preserves the full sequence for LSTM training
    motifs.extend(sequence)
    
    # Deduplicate while preserving order
    return list(dict.fromkeys(motifs))


def extract_structural_motifs(trace: Dict) -> List[str]:
    """Extract structural motifs based on file switching patterns.
    
    Captures workflow patterns like:
    - Hotspot editing (many edits in one file)
    - Dependency chasing (rapid file switches)
    - Iterative refinement (back-and-forth between files)
    """
    motifs = []
    events = trace.get('events', [])
    
    if len(events) < 2:
        return motifs
    
    # Track file switching patterns
    current_file = None
    file_switches = []
    edits_per_file = {}
    
    for event in events:
        details = event.get('details', {})
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except (json.JSONDecodeError, TypeError):
                details = {}
        
        file_path = details.get('file_path') or details.get('file')
        
        if file_path:
            if current_file and file_path != current_file:
                file_switches.append((current_file, file_path))
            current_file = file_path
            edits_per_file[file_path] = edits_per_file.get(file_path, 0) + 1
    
    # Hotspot editing: many edits in one file
    if edits_per_file:
        max_edits = max(edits_per_file.values())
        if max_edits > 5:
            motifs.append(f"HOTSPOT_{max_edits}")
    
    # Dependency chasing: rapid file switches
    if len(file_switches) > 3:
        motifs.append("DEPENDENCY_CHASE")
    
    # Iterative refinement: back-and-forth pattern
    if len(file_switches) >= 2:
        switch_pairs = [(file_switches[i][1], file_switches[i+1][1]) 
                        for i in range(len(file_switches)-1)]
        if any(pair[0] == pair[1] for pair in switch_pairs):
            motifs.append("ITERATIVE_REFINE")
    
    return motifs


def extract_intent_motifs(sequence: List[str]) -> List[str]:
    """Extract intent-specific motifs from a sequence containing INTENT markers.
    
    Discovers patterns like:
    - INTENT_DEBUG → MODIFY → RUN (debug loops)
    - INTENT_FEATURE → CREATE → MODIFY (feature development)
    - INTENT_REFACTOR → MODIFY → MODIFY (refactoring patterns)
    
    Args:
        sequence: Event sequence with INTENT markers
    
    Returns:
        List of intent-aware motif strings
    """
    motifs = []
    
    if not sequence:
        return motifs
    
    # Find intent markers
    intent_indices = [i for i, item in enumerate(sequence) if item.startswith('INTENT_')]
    
    if not intent_indices:
        return motifs
    
    # Extract intent-anchored subsequences
    for intent_idx in intent_indices:
        intent = sequence[intent_idx]
        
        # Look ahead for patterns following this intent
        if intent_idx + 1 < len(sequence):
            next_event = sequence[intent_idx + 1]
            motifs.append(f"{intent}_TO_{next_event}")
        
        # Look for intent → event → event patterns
        if intent_idx + 2 < len(sequence):
            pattern = f"{intent}_{sequence[intent_idx+1]}_{sequence[intent_idx+2]}"
            motifs.append(f"INTENT_PATTERN_{pattern}")
        
        # Extract intent type for clustering
        intent_type = intent.replace('INTENT_', '')
        motifs.append(f"INTENT_TYPE_{intent_type}")
    
    # Find common intent transitions
    intent_transitions = []
    for i in range(len(intent_indices) - 1):
        intent1 = sequence[intent_indices[i]]
        intent2 = sequence[intent_indices[i+1]]
        intent_transitions.append(f"INTENT_TRANS_{intent1}_{intent2}")
    
    motifs.extend(intent_transitions)
    
    return motifs










