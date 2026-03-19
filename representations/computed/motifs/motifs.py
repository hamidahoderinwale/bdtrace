"""
Motifs: computed, structural co-occurrence over edit operations.

SWE-bench (diff-based): input = edit certificates; unit = op.type from certificate.
Sequence is flat: [op["type"] for cert in certificates for op in cert["operations"]].
Interpretation: structural co-occurrence patterns (e.g. guard + early return), not process patterns.

Distance: distance.dtw_similarity (DTW soft membership).
"""

from typing import Any

from ...core.intent import extract_prompt_from_event, intent_tokens_for_prompt
from .distance import dtw_similarity
from .motif_mining import (
    extract_intent_motifs,
    extract_structural_motifs,
    extract_universal_motifs,
    motifs_from_sequence,
)


def _operation_sequence_from_certificates(certificates: list[dict[str, Any]]) -> list[str]:
    """Build motif sequence from edit certificates (diff-based, SWE-bench)."""
    if not certificates:
        return []
    sequence = []
    for cert in certificates:
        if not isinstance(cert, dict):
            continue
        for op in cert.get("operations", []):
            if isinstance(op, dict) and op.get("type"):
                sequence.append(str(op["type"]))
    return sequence


def motifs_repr_from_certificates(
    certificates: list[dict[str, Any]],
    use_statistical_mining: bool = True,
    return_structural: bool = True,
) -> dict[str, Any]:
    """
    Extract motifs from edit certificates (diff-based, SWE-bench).
    Structural co-occurrence patterns over op types within a single diff.
    """
    sequence = _operation_sequence_from_certificates(certificates)
    if not sequence or len(sequence) < 2:
        return {"sequence": sequence, "motifs": [], "soft_membership": {}}

    if use_statistical_mining:
        motifs = motifs_from_sequence(sequence, max_total=300)
    else:
        from .motif_mining import extract_universal_motifs

        motifs = extract_universal_motifs(
            sequence,
            include_transitions=True,
            include_ngrams=True,
            include_structural=True,
            ngram_sizes=[3, 4],
            use_statistical_mining=False,
        )

    motifs = list(dict.fromkeys(motifs))[:50]
    from .motif_mining import motif_registry

    soft_membership = {}
    for m in motifs[:20]:
        orig = motif_registry.get_original(m) if m.startswith("M_") else m
        if orig and orig.startswith("PS_"):
            pattern = orig[3:].split("_")
        elif orig and orig.startswith("T_"):
            pattern = orig[2:].split("_")
        elif orig and orig.startswith("SQ_"):
            pattern = orig[3:].split("_")
        else:
            pattern = [m]
        key = orig if orig and len(str(orig)) < 50 else m
        soft_membership[key] = dtw_similarity(sequence, pattern)

    return {
        "sequence": sequence,
        "motifs": [{"pattern": m, "category": "mined"} for m in motifs],
        "soft_membership": soft_membership,
    }


def _event_sequence(
    trace: dict,
    include_prompts: bool = True,
    include_llm_intents: bool = False,
) -> list[str]:
    """Build event sequence for motif mining."""
    if not isinstance(trace, dict):
        return []
    events = trace.get("events", [])
    if not isinstance(events, list):
        return []
    sequence = []
    for event in events:
        if event is None or not isinstance(event, dict):
            continue
        if include_prompts:
            prompt_text = extract_prompt_from_event(event)
            if prompt_text:
                sequence.extend(intent_tokens_for_prompt(prompt_text, include_llm=include_llm_intents))
        ev_type = event.get("type") or event.get("operation") or event.get("verb") or "other"
        sequence.append(str(ev_type).upper().replace(".", "_")[:20])
    if include_prompts and "prompts" in trace:
        for prompt_data in trace.get("prompts", []):
            if isinstance(prompt_data, dict):
                prompt_text = prompt_data.get("text") or prompt_data.get("content") or prompt_data.get("prompt")
                if prompt_text:
                    sequence.extend(intent_tokens_for_prompt(str(prompt_text), include_llm=include_llm_intents))
    return sequence


def motifs_repr(
    trace: dict,
    use_statistical_mining: bool = True,
    include_prompts: bool = True,
    include_llm_intents: bool = False,
    motif_vocabulary: list[tuple[list[str], int]] | None = None,
    return_structural: bool = False,
) -> list[str] | dict[str, Any]:
    """Extract motifs from trace."""
    if not trace or not trace.get("events"):
        return [] if not return_structural else {"sequence": [], "motifs": [], "soft_membership": {}}

    event_seq = _event_sequence(trace, include_prompts, include_llm_intents)
    if not event_seq:
        return [] if not return_structural else {"sequence": [], "motifs": [], "soft_membership": {}}

    if use_statistical_mining:
        motifs = motifs_from_sequence(event_seq, max_total=300)
    else:
        motifs = extract_universal_motifs(
            event_seq,
            include_transitions=True,
            include_ngrams=True,
            include_structural=True,
            ngram_sizes=[3, 4],
            use_statistical_mining=False,
        )

    structural_motifs = extract_structural_motifs(trace)
    motifs = list(dict.fromkeys(motifs + structural_motifs))

    if include_prompts:
        intent_motifs = extract_intent_motifs(event_seq)
        motifs = list(dict.fromkeys(motifs + intent_motifs))

    if return_structural:
        from .motif_mining import motif_registry

        soft_membership = {}
        for m in motifs[:20]:
            orig = motif_registry.get_original(m) if m.startswith("M_") else m
            if orig and orig.startswith("PS_"):
                pattern = orig[3:].split("_")
            elif orig and orig.startswith("T_"):
                pattern = orig[2:].split("_")
            elif orig and orig.startswith("SQ_"):
                pattern = orig[3:].split("_")
            else:
                pattern = [m]
            key = orig if orig and len(str(orig)) < 50 else m
            soft_membership[key] = dtw_similarity(event_seq, pattern)
        return {
            "sequence": event_seq,
            "motifs": [{"pattern": m, "category": "mined"} for m in motifs[:50]],
            "soft_membership": soft_membership,
        }

    return motifs


def motifs_repr_str(trace: dict, limit: int = 50, max_length: int = 2000) -> str:
    """Extract motifs as a string."""
    motifs = motifs_repr(trace, use_statistical_mining=True, include_prompts=True)
    if not motifs:
        return "EMPTY_WORKFLOW"
    unique = list(dict.fromkeys(motifs))[:limit]
    s = " | ".join(unique)
    if len(s) > max_length:
        s = s[:max_length] + "... [truncated]"
    return s


def motifs_repr_structural(
    trace: dict,
    motif_vocabulary: list[tuple[list[str], int]] | None = None,
) -> dict[str, Any]:
    """Return structural format: {sequence, motifs, soft_membership}."""
    return motifs_repr(
        trace,
        use_statistical_mining=True,
        include_prompts=True,
        motif_vocabulary=motif_vocabulary,
        return_structural=True,
    )
