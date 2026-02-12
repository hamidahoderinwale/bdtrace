from typing import Dict, List
from ..core.canonicalization import event_sequence
from .motif_mining import (
    extract_intent_motifs,
    extract_structural_motifs,
    motifs_from_sequence,
    extract_universal_motifs,
)

def motifs_repr(
    trace: Dict,
    use_statistical_mining: bool = True,
    include_prompts: bool = True,
    include_llm_intents: bool = False,
) -> List[str]:
    """Extract motif representation using statistical sequence mining."""
    if not trace or not trace.get('events'):
        return []
    
    canonical_seq = event_sequence(
        trace,
        include_prompts=include_prompts,
        include_llm_intents=include_llm_intents,
    )
    
    if not canonical_seq:
        return []
    
    if use_statistical_mining:
        motifs = motifs_from_sequence(canonical_seq, max_total=300)
    else:
        motifs = extract_universal_motifs(
            canonical_seq,
            include_transitions=True,
            include_ngrams=True,
            include_structural=True,
            ngram_sizes=[3, 4],
            use_statistical_mining=False
        )
    
    structural_motifs = extract_structural_motifs(trace)
    motifs.extend(structural_motifs)
    
    if include_prompts:
        intent_motifs = extract_intent_motifs(canonical_seq)
        motifs.extend(intent_motifs)
    
    return list(dict.fromkeys(motifs))

def motifs_repr_str(trace: dict, limit: int = 50, max_length: int = 2000) -> str:
    """Extract motifs as a string representation."""
    motifs = motifs_repr(trace, use_statistical_mining=True, include_prompts=True)
    if not motifs:
        return "EMPTY_WORKFLOW"
    
    unique_motifs = list(dict.fromkeys(motifs))[:limit]
    motif_str = " | ".join(unique_motifs)
    
    if len(motif_str) > max_length:
        motif_str = motif_str[:max_length] + "... [truncated]"
    
    return motif_str

