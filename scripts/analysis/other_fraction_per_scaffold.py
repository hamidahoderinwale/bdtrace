"""OTHER fraction per scaffold and per agent.

For each of the 9 agents in the extended corpus, compute what fraction of
canonical atoms in their trajectories fall into the OTHER / UNKNOWN / EMPTY
bucket. These are atoms the canonicalizer could not place into a typed
verb-times-file-type cell — they are the construct-validity surface area
of the cross-scaffold mapping.

Reads:
    output/paper2_pilot/bpe_sequences_extended.jsonl
Writes:
    output/paper2_pilot/other_fraction_per_scaffold.json
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEQS = ROOT / "output" / "paper2_pilot" / "bpe_sequences_extended.jsonl"
OUT = ROOT / "output" / "paper2_pilot" / "other_fraction_per_scaffold.json"

# What we count as "information-loss" atoms — anything that did not place into
# a typed verb-times-file-type cell.
LOSS_ATOMS = {"OTHER", "EMPTY", "COMMENT"}
LOSS_PREFIXES = ("UNKNOWN_",)

# Mapping from agent label to scaffold class
SCAFFOLD_OF = {
    "Claude-3":             "SWE-agent",
    "Claude-3.5":           "SWE-agent",
    "Claude-3.7-thinking":  "SWE-agent",
    "Claude-4":             "SWE-agent",
    "GPT-4":                "SWE-agent",
    "GPT-4o":               "SWE-agent",
    "DARS+R1":              "DARS",
    "Agentless+Claude-3.5": "Agentless",
    "Moatless+V3":          "Moatless",
}


def is_loss(atom: str) -> bool:
    if atom in LOSS_ATOMS:
        return True
    return any(atom.startswith(p) for p in LOSS_PREFIXES)


def main() -> None:
    per_agent: dict[str, dict] = {}
    for line in SEQS.open():
        rec = json.loads(line)
        agent = rec["agent"]
        atoms = rec["canonical"]
        agg = per_agent.setdefault(agent, {"total_atoms": 0, "loss_atoms": 0, "n_trajectories": 0, "loss_counter": Counter()})
        agg["n_trajectories"] += 1
        for a in atoms:
            agg["total_atoms"] += 1
            if is_loss(a):
                agg["loss_atoms"] += 1
                agg["loss_counter"][a] += 1

    out = {"per_agent": {}, "per_scaffold": {}}
    scaffold_agg: dict[str, dict] = {}
    for agent, agg in per_agent.items():
        frac = agg["loss_atoms"] / agg["total_atoms"] if agg["total_atoms"] else 0.0
        out["per_agent"][agent] = {
            "scaffold": SCAFFOLD_OF.get(agent, "?"),
            "n_trajectories": agg["n_trajectories"],
            "total_atoms": agg["total_atoms"],
            "loss_atoms": agg["loss_atoms"],
            "loss_fraction": round(frac, 4),
            "top_loss_atoms": agg["loss_counter"].most_common(5),
        }
        scaf = SCAFFOLD_OF.get(agent, "?")
        s = scaffold_agg.setdefault(scaf, {"total_atoms": 0, "loss_atoms": 0, "agents": []})
        s["total_atoms"] += agg["total_atoms"]
        s["loss_atoms"] += agg["loss_atoms"]
        s["agents"].append(agent)

    for scaf, agg in scaffold_agg.items():
        frac = agg["loss_atoms"] / agg["total_atoms"] if agg["total_atoms"] else 0.0
        out["per_scaffold"][scaf] = {
            "agents": agg["agents"],
            "total_atoms": agg["total_atoms"],
            "loss_atoms": agg["loss_atoms"],
            "loss_fraction": round(frac, 4),
        }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
