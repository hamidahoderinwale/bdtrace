"""Per-agent budget breakdown by action type.

Attributes each trajectory's token cost proportionally across its action
steps, then rolls up by (agent, action_type). Reports budget share —
fraction of each agent's total tokens spent on each action type — so
agents with very different trajectory lengths can be compared fairly.

Saves:
    output/paper2_pilot/cost_by_action.json
    output/paper2_pilot/cost_by_action.png

Usage:
    uv run python scripts/paper_cost_by_action_figure.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, ORANGE, GREEN
register()

CACHE   = ROOT / "output" / "trajectories" / ".cache"
SEQ_PATH = ROOT / "output" / "paper2_pilot" / "bpe_sequences.jsonl"
OUT     = ROOT / "output" / "paper2_pilot"

AGENT_SHORT = {
    "20240402_sweagent_gpt4":           "GPT-4",
    "20240620_sweagent_claude3.5sonnet": "Claude-3.5",
    "20240728_sweagent_gpt4o":           "GPT-4o",
}
AGENT_ORDER  = ["Claude-3.5", "GPT-4", "GPT-4o"]
AGENT_COLORS = [BLUE, "#009E73", ORANGE]
TOP_N = 12


def load_stats() -> dict[tuple[str, str], dict]:
    stats = {}
    for agent_dir in sorted(CACHE.iterdir()):
        if not agent_dir.is_dir():
            continue
        short = AGENT_SHORT.get(agent_dir.name, agent_dir.name)
        for tf in sorted(agent_dir.glob("*.json")):
            with open(tf) as f:
                d = json.load(f)
            ms = (d.get("info") or {}).get("model_stats") or {}
            stats[(short, tf.stem)] = int(ms.get("tokens_sent", 0))
    return stats


def main() -> None:
    stats = load_stats()
    seqs  = [json.loads(l) for l in open(SEQ_PATH) if l.strip()]

    agent_atom_tokens: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    agent_total:       dict[str, float]            = defaultdict(float)

    for seq in seqs:
        agent = seq["agent"]
        inst  = seq["instance_id"]
        atoms = seq["canonical"]
        if not atoms:
            continue
        tokens = stats.get((agent, inst), 0)
        if tokens == 0:
            continue
        tpa = tokens / len(atoms)
        agent_total[agent] += tokens
        for a in atoms:
            agent_atom_tokens[agent][a] += tpa

    # Top N action types by total attributed tokens across all agents
    all_totals: dict[str, float] = defaultdict(float)
    for atom_map in agent_atom_tokens.values():
        for atom, t in atom_map.items():
            all_totals[atom] += t
    top_atoms = [a for a, _ in sorted(all_totals.items(), key=lambda x: -x[1])[:TOP_N]]

    # Build long-form dataframe: budget share = (agent, atom, share)
    rows = []
    for agent in AGENT_ORDER:
        total = agent_total.get(agent, 0)
        for atom in top_atoms:
            share = agent_atom_tokens[agent].get(atom, 0) / total if total else 0
            rows.append({"agent": agent, "action": atom, "share": share})

    df = pd.DataFrame(rows)

    # Save JSON summary
    summary = {
        "agent_total_tokens": {a: int(agent_total[a]) for a in AGENT_ORDER},
        "top_atoms": top_atoms,
        "budget_shares": {
            agent: {
                atom: float(agent_atom_tokens[agent].get(atom, 0) / agent_total[agent])
                for atom in top_atoms
            }
            for agent in AGENT_ORDER
        },
        "note": (
            "Budget share = fraction of each agent's total attributed tokens spent on each action type. "
            "Tokens are attributed uniformly per step (tokens_sent / n_atoms). "
            "Normalizing by agent total enables fair comparison across agents of different trajectory length."
        ),
    }
    (OUT / "cost_by_action.json").write_text(json.dumps(summary, indent=2))

    cscale = alt.Scale(domain=AGENT_ORDER, range=AGENT_COLORS)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("action:N",
                    sort=top_atoms,
                    axis=alt.Axis(title=None, domain=False, ticks=False,
                                  labelAngle=-30, labelFontSize=10)),
            y=alt.Y("share:Q",
                    axis=alt.Axis(title="Share of agent's total token budget",
                                  domain=False, ticks=False,
                                  format=".0%")),
            xOffset=alt.XOffset("agent:N", sort=AGENT_ORDER),
            color=alt.Color("agent:N", sort=AGENT_ORDER, scale=cscale,
                            legend=alt.Legend(orient="bottom", title=None,
                                              symbolSize=80)),
        )
        .properties(
            width=580, height=280,
            title=alt.TitleParams(
                text="Token budget share by action type",
                fontSize=13, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
    )

    out_path = OUT / "cost_by_action.png"
    chart.save(str(out_path), scale_factor=2)
    print(f"Saved {out_path}")

    # Print table for inspection
    print(f"\n{'Action':<28}", end="")
    for a in AGENT_ORDER:
        print(f"  {a:>12}", end="")
    print()
    for atom in top_atoms:
        print(f"{atom:<28}", end="")
        for a in AGENT_ORDER:
            sh = summary["budget_shares"][a][atom]
            print(f"  {sh:>11.1%}", end="")
        print()


if __name__ == "__main__":
    main()
