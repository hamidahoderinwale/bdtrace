"""Append Claude-3 Opus to bpe_sequences.jsonl using the existing BPE model.

Does NOT retrain BPE -- applies the learned merges from bpe_model.json
to Claude-3 Opus canonical sequences so the motif vocabulary is unchanged.

Reads:
    output/paper2_pilot/bpe_model.json
    output/trajectories/.cache/20240402_sweagent_claude3opus/*.json
Appends:
    output/paper2_pilot/bpe_sequences.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.preferences.bpe import BPEModel, apply_bpe
from analysis.preferences.canonicalize import canonicalize_trajectory

AGENT_ID = "20240402_sweagent_claude3opus"
AGENT_SHORT = "Claude-3"
CACHE = ROOT / "output" / "trajectories" / ".cache" / AGENT_ID
OUT = ROOT / "output" / "paper2_pilot"
BPE_SEQUENCES = OUT / "bpe_sequences.jsonl"
BPE_MODEL_PATH = OUT / "bpe_model.json"


def load_bpe_model() -> BPEModel:
    data = json.loads(BPE_MODEL_PATH.read_text())
    merges = [(a, b, new) for a, b, new in data["merges"]]
    return BPEModel(merges=merges, vocab=data["vocab"])


def load_existing_agent_instance_pairs() -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    with open(BPE_SEQUENCES) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                pairs.add((r["agent"], r["instance_id"]))
    return pairs


def main() -> int:
    if not CACHE.exists():
        print(f"Cache dir not found: {CACHE}")
        return 1

    model = load_bpe_model()
    print(f"Loaded BPE model: {len(model.vocab)} vocab, {len(model.merges)} merges")

    existing_pairs = load_existing_agent_instance_pairs()
    print(f"Existing records: {len(existing_pairs)} (agent, instance_id) pairs")

    traj_files = sorted(CACHE.glob("*.json"))
    print(f"Found {len(traj_files)} Claude-3 Opus trajectories")

    records = []
    skipped = 0
    already = 0
    for traj_file in traj_files:
        instance_id = traj_file.stem
        if (AGENT_SHORT, instance_id) in existing_pairs:
            already += 1
            continue
        with open(traj_file) as f:
            raw = json.load(f)
        seq = canonicalize_trajectory(raw.get("trajectory", []))
        if not seq:
            skipped += 1
            continue
        records.append({
            "instance_id": instance_id,
            "canonical": seq,
        })
    if already:
        print(f"  Skipped {already} already-present instance IDs")

    print(f"Canonicalized {len(records)} sequences ({skipped} empty/skipped)")

    # Apply existing BPE model
    canonical_seqs = [r["canonical"] for r in records]
    bpe_seqs = apply_bpe(canonical_seqs, model)

    # Append to bpe_sequences.jsonl
    appended = 0
    with open(BPE_SEQUENCES, "a") as f:
        for r, bpe in zip(records, bpe_seqs):
            f.write(json.dumps({
                "agent": AGENT_SHORT,
                "instance_id": r["instance_id"],
                "canonical": r["canonical"],
                "bpe": bpe,
                "canonical_length": len(r["canonical"]),
                "bpe_length": len(bpe),
                "compression": len(bpe) / max(len(r["canonical"]), 1),
            }) + "\n")
            appended += 1

    print(f"Appended {appended} Claude-3 Opus records to {BPE_SEQUENCES}")

    # Sanity check
    from collections import Counter
    agents: Counter = Counter()
    with open(BPE_SEQUENCES) as f:
        for line in f:
            if line.strip():
                agents[json.loads(line)["agent"]] += 1
    print("Agent counts in bpe_sequences.jsonl:")
    for agent, n in agents.most_common():
        print(f"  {agent}: {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
