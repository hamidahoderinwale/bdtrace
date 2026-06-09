"""BPE retrain on the extended (8-submission) corpus.

Pipeline:
  1. Walk output/trajectories/.cache/<submission>/<iid>.json
  2. Skip submissions with no procedural data (SWE-Fixer + Qwen)
  3. Apply canonicalize_envelope to every cached trajectory
  4. Train BPE at V=200 on the concatenated atom sequences
  5. Run V-sweep at V in {100, 150, 200, 300, 500} and compute pairwise JSDs
     between submissions; ranking should be stable across V (R6 finding)

Outputs (all under output/paper2_pilot/):
  bpe_model_extended.json          - merges + final vocabulary
  bpe_sequences_extended.jsonl     - per-trajectory canonical + bpe sequences
  bpe_vocab_sweep_extended.json    - V-sweep results with pairwise JSD matrix

Usage:
  python -m analysis.preferences.build_extended_bpe
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.preferences.bpe import train_bpe
from analysis.preferences.canonicalize_extended import canonicalize_envelope

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE = PROJECT_ROOT / "output" / "trajectories" / ".cache"
OUT = PROJECT_ROOT / "output" / "paper2_pilot"

# Submission -> short label used downstream.
# SWE-Fixer is excluded (final-patch only, no procedural trace).
SUBMISSION_LABEL = {
    "20240402_sweagent_claude3opus":       "Claude-3",
    "20240402_sweagent_gpt4":              "GPT-4",
    "20240620_sweagent_claude3.5sonnet":   "Claude-3.5",
    "20240728_sweagent_gpt4o":             "GPT-4o",
    "20250226_sweagent_claude-3-7-sonnet-20250219":         "Claude-3.7-thinking",
    "20250205_dars_agent_claude_3.5_sonnet_deepseek_r1":    "DARS+R1",
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022":    "Agentless+Claude-3.5",
    "20250111_moatless_deepseek_v3":                        "Moatless+V3",
    # 2026-05-07 extension
    "20250526_sweagent_claude-4-sonnet-20250514":           "Claude-4",
    # KGCompass + entroPO fetched but deferred — need custom canonicalizers:
    #   KGCompass: text knowledge-graph log (not dars_traj_list as auto-detected)
    #   entroPO: chat-message list with embedded tool calls
}

EXCLUDE = {
    "20241128_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor_20241128",
}

TARGET_VOCAB = 200
SWEEP_V = [100, 150, 200, 300, 500]


def load_corpus() -> list[dict]:
    """Load every cached trajectory, canonicalize, return list of records."""
    records: list[dict] = []
    for submission, label in SUBMISSION_LABEL.items():
        sub_dir = CACHE / submission
        if not sub_dir.is_dir():
            print(f"  [skip] {submission}: directory missing")
            continue
        for traj_file in sorted(sub_dir.glob("*.json")):
            envelope = json.loads(traj_file.read_text())
            atoms = canonicalize_envelope(envelope)
            if not atoms:
                continue
            records.append({
                "submission": submission,
                "agent":      label,
                "instance_id": traj_file.stem,
                "canonical":   atoms,
            })
    return records


def jsd(a: np.ndarray, b: np.ndarray) -> float:
    """Squared Jensen-Shannon distance (i.e. JS divergence) in bits."""
    d = float(jensenshannon(a, b, base=2))
    return d * d


def per_agent_distribution(records: list[dict], vocab: list[str]) -> dict[str, np.ndarray]:
    """Per-agent probability distribution over the BPE vocabulary."""
    agent_counts: dict[str, Counter] = {}
    for r in records:
        agent_counts.setdefault(r["agent"], Counter()).update(r["bpe"])
    out: dict[str, np.ndarray] = {}
    for agent, counter in agent_counts.items():
        total = sum(counter[v] for v in vocab)
        if total == 0:
            out[agent] = np.zeros(len(vocab))
        else:
            out[agent] = np.array([counter.get(v, 0) / total for v in vocab])
    return out


def pairwise_jsd_matrix(records: list[dict], vocab: list[str]) -> dict[str, float]:
    dist = per_agent_distribution(records, vocab)
    out: dict[str, float] = {}
    for a, b in combinations(sorted(dist), 2):
        out[f"{a}__{b}"] = jsd(dist[a], dist[b])
    return out


def run_sweep(records: list[dict], sweep_V: list[int]) -> list[dict]:
    canonical = [r["canonical"] for r in records]
    n_canonical_tokens = sum(len(s) for s in canonical)

    results: list[dict] = []
    for V in sweep_V:
        print(f"\n  === V = {V} ===")
        model, expressed = train_bpe(canonical, target_size=V, verbose=False)

        # Build per-record records_bpe view for distribution computation.
        records_bpe = [{"agent": r["agent"], "bpe": e} for r, e in zip(records, expressed)]

        # Two distributions: full-vocab and motifs-only.
        jsd_full = pairwise_jsd_matrix(records_bpe, model.vocab)
        motif_vocab = [v for v in model.vocab if "+" in v]
        if motif_vocab:
            jsd_motifs = pairwise_jsd_matrix(records_bpe, motif_vocab)
        else:
            jsd_motifs = {}

        n_bpe_tokens = sum(len(s) for s in expressed)
        compression = n_bpe_tokens / max(n_canonical_tokens, 1)

        # Stability check: which pair has min JSD at this V?
        if jsd_full:
            min_pair_full = min(jsd_full, key=jsd_full.get)
        else:
            min_pair_full = None

        print(f"    merges={len(model.merges)}, vocab={len(model.vocab)}, compression={compression:.3f}")
        print(f"    min-JSD pair (full)   : {min_pair_full}")

        results.append({
            "V": V,
            "n_merges": len(model.merges),
            "actual_vocab": len(model.vocab),
            "n_canonical_tokens": n_canonical_tokens,
            "n_bpe_tokens": n_bpe_tokens,
            "compression_ratio": compression,
            "length_distribution": model.summary()["length_distribution"],
            "jsd_full": jsd_full,
            "jsd_motifs": jsd_motifs,
            "min_pair_full": min_pair_full,
        })
    return results


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Step 1: load + canonicalize")
    records = load_corpus()
    n_per_agent = Counter(r["agent"] for r in records)
    print(f"  total trajectories: {len(records)}")
    for agent, n in sorted(n_per_agent.items()):
        print(f"    {agent:25s}  n={n}")

    if len(records) < 100:
        print("  ERROR: corpus too small; aborting")
        return 1

    canonical = [r["canonical"] for r in records]
    atom_vocab = sorted({t for s in canonical for t in s})
    print(f"  atomic vocabulary: {len(atom_vocab)} tokens")
    print(f"  total canonical tokens: {sum(len(s) for s in canonical):,}")

    if len(atom_vocab) < 5:
        print("  ERROR: atomic vocabulary collapsed (< 5)")
        return 1

    print("\nStep 2: train BPE at V=200")
    model, expressed = train_bpe(canonical, target_size=TARGET_VOCAB, verbose=False)
    if len(model.vocab) < 50:
        print(f"  ERROR: vocabulary collapse (V={len(model.vocab)} < 50)")
        return 1
    for r, e in zip(records, expressed):
        r["bpe"] = e
    n_bpe_tokens = sum(len(s) for s in expressed)
    n_canonical_tokens = sum(len(s) for s in canonical)
    compression = n_bpe_tokens / max(n_canonical_tokens, 1)
    print(f"  vocab={len(model.vocab)}  merges={len(model.merges)}  compression={compression:.3f}")
    print(f"  mean BPE length: {n_bpe_tokens / max(len(expressed), 1):.1f} tokens")
    print(f"  length distribution: {model.summary()['length_distribution']}")

    print("\nStep 3: save model + sequences")
    (OUT / "bpe_model_extended.json").write_text(json.dumps({
        "target_vocab_size": TARGET_VOCAB,
        "final_vocab_size":  len(model.vocab),
        "n_merges":          len(model.merges),
        "atomic_vocab":      atom_vocab,
        "merges":            [[a, b, new] for a, b, new in model.merges],
        "vocab":             list(model.vocab),
        "length_distribution": model.summary()["length_distribution"],
        "submission_label":  SUBMISSION_LABEL,
    }, indent=2))

    with (OUT / "bpe_sequences_extended.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps({
                "submission":      r["submission"],
                "agent":           r["agent"],
                "instance_id":     r["instance_id"],
                "canonical":       r["canonical"],
                "bpe":             r["bpe"],
                "canonical_length": len(r["canonical"]),
                "bpe_length":      len(r["bpe"]),
                "compression":     len(r["bpe"]) / max(len(r["canonical"]), 1),
            }) + "\n")

    print("\nStep 4: V-sweep + pairwise JSD")
    sweep = run_sweep(records, SWEEP_V)
    (OUT / "bpe_vocab_sweep_extended.json").write_text(json.dumps({
        "sweep_V": SWEEP_V,
        "n_records": len(records),
        "agents": sorted(SUBMISSION_LABEL.values()),
        "results": sweep,
    }, indent=2, default=str))

    # Stability check across V
    print("\n=== JSD ranking stability across V ===")
    pairs = sorted(sweep[0]["jsd_full"].keys())
    print(f"{'pair':40s}  " + "  ".join(f"V={V:>3}" for V in SWEEP_V))
    for pair in pairs:
        row = [f"{r['jsd_full'][pair]:.3f}" for r in sweep]
        print(f"  {pair:38s}  " + "  ".join(f"{v:>5}" for v in row))

    # Identify whether the min-JSD pair flips
    min_pairs = [r["min_pair_full"] for r in sweep]
    if len(set(min_pairs)) > 1:
        print(f"\n  NOTE: min-JSD pair flips across V: {min_pairs}")
    else:
        print(f"\n  min-JSD pair stable across V: {min_pairs[0]}")

    print(f"\nDone. Artifacts:")
    for name in ["bpe_model_extended.json", "bpe_sequences_extended.jsonl",
                 "bpe_vocab_sweep_extended.json"]:
        print(f"  {OUT / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
