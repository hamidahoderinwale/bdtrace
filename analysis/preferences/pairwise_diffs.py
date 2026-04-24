"""Pairwise motif-level visual diffs for selected tied-outcome tasks.

For a handful of 3/3-resolved tasks (all three agents solved), render each
agent's BPE-expressed motif sequence side by side, with each motif colored
by whether it appears in the other two agents' sequences on the same task
(shared) or only in this agent's (unique).

Produces an HTML block that can be embedded directly into the dashboard.

Selection criterion:
    tasks where all three agents resolved (3/3) and the total motif length
    across agents falls in a reasonable range.

Inputs:
    output/paper2_pilot/bpe_sequences.jsonl
    output/paper2_pilot/task_diversity.csv

Outputs:
    output/paper2_pilot/pairwise_diffs.html  (HTML snippet; section-level)

Usage:
    python -m analysis.preferences.pairwise_diffs
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "output" / "paper2_pilot"
SEQ_PATH = OUT / "bpe_sequences.jsonl"
DIVERSITY_PATH = OUT / "task_diversity.csv"

AGENT_ORDER = ["GPT-4", "Claude-3.5", "GPT-4o"]

SELECTED_TASKS = [
    "django__django-13447",   # short (4, 10, 12)
    "django__django-11039",   # medium (9, 9, 21)
    "django__django-14855",   # long (5, 12, 34)
]


def load_records() -> list[dict]:
    out = []
    with open(SEQ_PATH) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_difficulty() -> dict[str, int]:
    out = {}
    with open(DIVERSITY_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["instance_id"]] = int(row["n_resolved"])
    return out


def abbrev(motif: str, max_len: int = 28) -> str:
    parts = motif.split("+")
    if len(parts) == 1:
        return motif
    if len(parts) == 2:
        joined = f"{parts[0]}→{parts[1]}"
    else:
        joined = f"{parts[0]}→…→{parts[-1]} ({len(parts)})"
    if len(joined) > max_len:
        return joined[: max_len - 1] + "…"
    return joined


def color_for(motif: str, present_in_other_agents: int) -> tuple[str, str]:
    """Return (bg, border) for a motif given how many other agents use it on this task.

    - shared with 2 others: green (universal for this task)
    - shared with 1 other: yellow (partial)
    - unique to this agent: white (distinctive)
    - atom (not a motif): gray (neutral, since motif-level is the story)
    """
    if "+" not in motif:
        return "#f3f3f3", "#bbb"
    if present_in_other_agents == 2:
        return "#c8e6c9", "#2e7d32"
    if present_in_other_agents == 1:
        return "#fff3c4", "#b58900"
    return "#ffffff", "#c03030"


def render_one_task(
    instance_id: str,
    by_agent: dict[str, list[str]],
) -> str:
    sets = {a: set(s) for a, s in by_agent.items()}

    rows = []
    agents_in_task = [a for a in AGENT_ORDER if a in by_agent]
    for a in agents_in_task:
        seq = by_agent[a]
        other_agents = [b for b in agents_in_task if b != a]
        chips = []
        for motif in seq:
            n_other = sum(1 for b in other_agents if motif in sets[b])
            bg, border = color_for(motif, n_other)
            title = motif
            text = abbrev(motif)
            chips.append(
                f'<span style="display:inline-block; padding:2px 6px; margin:2px; '
                f'background:{bg}; border:1px solid {border}; border-radius:3px; '
                f'font-family:SF Mono,Menlo,monospace; font-size:11px;" title="{title}">'
                f'{text}</span>'
            )
        row_html = f"""
<div style="display: flex; align-items: center; margin: 6px 0;">
  <div style="min-width: 100px; font-weight: 600; font-size: 13px; color: #333;">{a}</div>
  <div style="flex: 1;">{''.join(chips)}</div>
  <div style="min-width: 70px; text-align: right; color: #666; font-size: 12px;">{len(seq)} motifs</div>
</div>
"""
        rows.append(row_html)

    shared_all = set.intersection(*sets.values()) if sets else set()
    shared_all_motifs = sorted([m for m in shared_all if "+" in m])

    summary_chips = []
    for m in shared_all_motifs[:12]:
        summary_chips.append(
            f'<span style="display:inline-block; padding:2px 6px; margin:2px; '
            f'background:#c8e6c9; border:1px solid #2e7d32; border-radius:3px; '
            f'font-family:SF Mono,Menlo,monospace; font-size:11px;" '
            f'title="{m}">{abbrev(m)}</span>'
        )

    return f"""
<div style="margin: 22px 0; padding: 14px 18px; background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;">
  <div style="font-size:14px; font-weight:600; margin-bottom: 4px;">Task: <code>{instance_id}</code> (all 3 agents resolved)</div>
  <div style="font-size:12px; color:#666; margin-bottom: 10px;">
    {len(shared_all_motifs)} motifs used by all three agents on this task.
  </div>
  {''.join(rows)}
  {"<div style='margin-top:10px; padding:6px 8px; background:#f8fdf8; border-radius:4px;'><span style='font-size:12px; color:#2e7d32;'>shared by all three:</span> " + ''.join(summary_chips) + "</div>" if summary_chips else ""}
</div>
"""


def main() -> int:
    records = load_records()
    difficulty = load_difficulty()

    by_task: dict[str, dict[str, list[str]]] = {}
    for r in records:
        by_task.setdefault(r["instance_id"], {})[r["agent"]] = r["bpe"]

    pieces = []
    for inst in SELECTED_TASKS:
        if inst not in by_task:
            print(f"SKIP {inst}: not in corpus")
            continue
        n_res = difficulty.get(inst)
        if n_res != 3:
            print(f"SKIP {inst}: resolved by {n_res}/3 (want 3/3)")
            continue
        if len(by_task[inst]) != 3:
            print(f"SKIP {inst}: only {len(by_task[inst])} agents have data")
            continue
        pieces.append(render_one_task(inst, by_task[inst]))
        print(f"OK   {inst}: rendered")

    header = """
<section id="pairwise-diffs">
<h2>Pairwise motif diffs on shared tasks</h2>

<p>Three 3/3-resolved tasks at short / medium / long procedural regimes, with each agent's BPE-expressed motif sequence rendered side-by-side.
A motif is <span style="padding:1px 5px; background:#c8e6c9; border:1px solid #2e7d32; border-radius:3px; font-size:11px;">shared by all three</span>,
<span style="padding:1px 5px; background:#fff3c4; border:1px solid #b58900; border-radius:3px; font-size:11px;">shared by two agents</span>,
<span style="padding:1px 5px; background:#ffffff; border:1px solid #c03030; border-radius:3px; font-size:11px;">unique to this agent</span>,
or <span style="padding:1px 5px; background:#f3f3f3; border:1px solid #bbb; border-radius:3px; font-size:11px;">an atom</span> (length 1, neutral).</p>
"""
    footer = "</section>\n"

    (OUT / "pairwise_diffs.html").write_text(header + "\n".join(pieces) + footer)
    print(f"\nSaved: {OUT / 'pairwise_diffs.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
