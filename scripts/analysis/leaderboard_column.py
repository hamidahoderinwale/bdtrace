"""Procgrep-derived leaderboard column on the nine-agent corpus.

Validates the "benchmark leaderboard" integration claim on the procgrep
landing page (Integrations #4): per submission, emit resolve%, effective
vocabulary size (exp of Shannon entropy of motif distribution), top-3
canonical-atom shares, and an auto-generated descriptor.

Reads:
    output/paper2_pilot/bpe_sequences_extended.jsonl
    output/paper2_pilot/extended_pass_fail.json

Writes:
    output/paper2_pilot/leaderboard_column.json
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SEQ_PATH = ROOT / "output" / "paper2_pilot" / "bpe_sequences_extended.jsonl"
PASS_PATH = ROOT / "output" / "paper2_pilot" / "extended_pass_fail.json"
OUT_JSON = ROOT / "output" / "paper2_pilot" / "leaderboard_column.json"

SUBMISSION_TO_AGENT = {
    "20240402_sweagent_claude3opus":                       "Claude-3",
    "20240402_sweagent_gpt4":                              "GPT-4",
    "20240620_sweagent_claude3.5sonnet":                   "Claude-3.5",
    "20240728_sweagent_gpt4o":                             "GPT-4o",
    "20250226_sweagent_claude-3-7-sonnet-20250219":        "Claude-3.7-thinking",
    "20250526_sweagent_claude-4-sonnet-20250514":          "Claude-4",
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022":   "Agentless+Claude-3.5",
    "20250205_dars_agent_claude_3.5_sonnet_deepseek_r1":   "DARS+R1",
    "20250111_moatless_deepseek_v3":                       "Moatless+V3",
}
AGENT_TO_SUBMISSION = {v: k for k, v in SUBMISSION_TO_AGENT.items()}


def shannon_entropy_bits(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((n / total) * math.log2(n / total) for n in counts.values() if n > 0)


def effective_vocab_size(counts: Counter) -> float:
    """exp_2(Shannon entropy): the number of equally-likely categories
    that would produce the observed entropy. Standard "perplexity"-style
    summary of a distribution's spread."""
    return 2 ** shannon_entropy_bits(counts)


def auto_descriptor(atom_shares: dict[str, float], eff_vocab_motif: float) -> str:
    """Two-tag descriptor from atom shares and motif spread."""
    tags: list[str] = []
    edit = atom_shares.get("EDIT_SRC_PY", 0.0)
    search = atom_shares.get("SEARCH", 0.0)
    test = (
        atom_shares.get("RUN_PYTHON_TEST_PY", 0.0)
        + atom_shares.get("RUN_PYTHON_REPRO_PY", 0.0)
        + atom_shares.get("RUN_PYTEST_SRC_PY", 0.0)
    )
    shell_cd = atom_shares.get("SHELL_CD", 0.0)

    if eff_vocab_motif < 2:
        tags.append("pipeline-template")
    elif edit > 0.30:
        tags.append("edit-dominated")
    elif edit > 0.18:
        tags.append("edit-heavy")
    elif edit > 0.10:
        tags.append("edit-moderate")

    if shell_cd > 0.10:
        tags.append("shell-heavy")
    elif test < 0.005:
        tags.append("no-test")
    elif search > 0.12:
        tags.append("search-heavy")
    elif search < 0.05 and "edit-dominated" in tags:
        tags.append("low-search")

    if eff_vocab_motif < 1.5:
        tags.append("deterministic")
    elif eff_vocab_motif > 15:
        tags.append("high-variety")

    if not tags:
        tags.append("balanced")
    return ", ".join(tags[:2])


def main() -> int:
    print(f"Loading {SEQ_PATH} ...")
    records = [json.loads(line) for line in SEQ_PATH.open() if line.strip()]
    print(f"  {len(records)} trajectories")

    print(f"Loading {PASS_PATH} ...")
    pass_data = json.loads(PASS_PATH.read_text())

    by_agent: dict[str, list[dict]] = {}
    for r in records:
        by_agent.setdefault(r["agent"], []).append(r)

    rows: list[dict] = []
    for agent, agent_records in by_agent.items():
        atom_counts: Counter = Counter()
        motif_counts: Counter = Counter()
        for r in agent_records:
            atom_counts.update(r["canonical"])
            motif_counts.update(r["bpe"])
        total_atoms = sum(atom_counts.values())
        atom_shares = {a: n / total_atoms for a, n in atom_counts.items()}
        eff_vocab_motif = effective_vocab_size(motif_counts)
        eff_vocab_atom = effective_vocab_size(atom_counts)
        top3 = atom_counts.most_common(3)
        top3_shares = [(a, atom_shares[a]) for a, _ in top3]

        sub = AGENT_TO_SUBMISSION.get(agent)
        resolved = pass_data.get(sub, {}).get("resolved", []) if sub else []
        n_traj = len(agent_records)
        resolve_pct = 100.0 * len(resolved) / n_traj if n_traj else 0.0

        descriptor = auto_descriptor(atom_shares, eff_vocab_motif)

        rows.append({
            "agent": agent,
            "submission": sub,
            "n_trajectories": n_traj,
            "resolved": len(resolved),
            "resolve_pct": round(resolve_pct, 1),
            "effective_vocab_atom": round(eff_vocab_atom, 1),
            "effective_vocab_motif": round(eff_vocab_motif, 1),
            "top3_atoms": [
                {"atom": a, "share_pct": round(s * 100, 1)} for a, s in top3_shares
            ],
            "descriptor": descriptor,
        })

    # Sort by resolve% descending
    rows.sort(key=lambda r: r["resolve_pct"], reverse=True)

    payload = {
        "source_corpus": "SWE-bench Verified, 9-agent paper2_pilot extended corpus",
        "metric_notes": {
            "effective_vocab_atom": (
                "exp_2(Shannon entropy of per-agent canonical-atom distribution)."
            ),
            "effective_vocab_motif": (
                "exp_2(Shannon entropy of per-agent BPE motif distribution). "
                "Smaller motif-eff than atom-eff means a few BPE motifs dominate "
                "the trajectory (pipeline-template behavior)."
            ),
            "descriptor": (
                "Two-tag auto-descriptor from atom shares + motif-eff; rule "
                "list in scripts/analysis/leaderboard_column.py."
            ),
        },
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {OUT_JSON}")

    # Human-readable table
    print()
    print(f"{'Agent':<22} {'N':>4} {'Resolve%':>8} {'AtomEff':>7} {'MotifEff':>8}  Descriptor")
    print("-" * 86)
    for r in rows:
        print(
            f"{r['agent']:<22} {r['n_trajectories']:>4} "
            f"{r['resolve_pct']:>7.1f}% "
            f"{r['effective_vocab_atom']:>7.1f} {r['effective_vocab_motif']:>8.1f}  "
            f"{r['descriptor']}"
        )
    print()
    print("Top-3 atom shares (sanity-check the descriptors):")
    print("-" * 86)
    for r in rows:
        atoms_str = "  ".join(
            f"{a['atom']}={a['share_pct']:.1f}%" for a in r["top3_atoms"]
        )
        print(f"  {r['agent']:<22} {atoms_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
