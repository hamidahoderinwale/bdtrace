"""Per-task procedural diversity across the nine-agent extended corpus.

Mirrors task_diversity.py but operates on canonical-atom sequences from
bpe_sequences_extended.jsonl (procgrep's notion of a step) rather than
raw-action sequences from per-agent parquets. The five newer agents
(Claude-3.7-thinking, Claude-4, Moatless+V3, DARS+R1, Agentless+
Claude-3.5) have no per-step parquet but they DO have canonical atom
sequences in the extended corpus, so the diversity comparison extends
to all nine agents at the procgrep grain.

For each task (SWE-bench Verified instance) with >=2 agent trajectories:
  - n_steps_mean / range / cv on canonical_length
  - mean pairwise normalized Levenshtein on canonical atom strings
  - max pairwise normalized Levenshtein
  - n_resolved across the nine agents

The Levenshtein numbers are on the same scale as task_diversity.csv's,
modulo the granularity (canonical-atom tokens are coarser than raw
action tokens, so absolute values may shift slightly; the ordering of
"task diversity" across instances is preserved).

Reads:
    output/paper2_pilot/bpe_sequences_extended.jsonl
    output/paper2_pilot/extended_pass_fail.json
Writes:
    output/paper2_pilot/task_diversity_extended.jsonl
    output/paper2_pilot/task_diversity_distribution.png
    output/paper2_pilot/task_diversity_by_resolved.png

Usage:
    python -m analysis.preferences.task_diversity_extended
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, COPPER, GREEN, MAGENTA, OLIVE
register()

OUT = ROOT / "output" / "paper2_pilot"
SEQ_PATH = OUT / "bpe_sequences_extended.jsonl"
PASS_PATH = OUT / "extended_pass_fail.json"
OUT_JSONL = OUT / "task_diversity_extended.jsonl"

SUBMISSION_TO_AGENT = {
    "20240402_sweagent_claude3opus":                "Claude-3",
    "20240402_sweagent_gpt4":                       "GPT-4",
    "20240620_sweagent_claude3.5sonnet":            "Claude-3.5",
    "20240728_sweagent_gpt4o":                      "GPT-4o",
    "20250226_sweagent_claude-3-7-sonnet-20250219": "Claude-3.7-thinking",
    "20250526_sweagent_claude-4-sonnet-20250514":   "Claude-4",
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022": "Agentless+Claude-3.5",
    "20250205_dars_agent_claude_3.5_sonnet_deepseek_r1": "DARS+R1",
    "20250111_moatless_deepseek_v3":                "Moatless+V3",
}
AGENT_TO_SUBMISSION = {v: k for k, v in SUBMISSION_TO_AGENT.items()}
N_AGENTS_FULL = len(SUBMISSION_TO_AGENT)


def normalized_levenshtein_tokens(a: list[str], b: list[str]) -> float:
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1.0
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m] / max(n, m, 1)


def load_records() -> list[dict]:
    return [json.loads(line) for line in SEQ_PATH.open() if line.strip()]


def load_resolved_sets() -> dict[str, set[str]]:
    pf = json.loads(PASS_PATH.read_text())
    return {
        agent: set(pf.get(sub, {}).get("resolved", []))
        for sub, agent in SUBMISSION_TO_AGENT.items()
    }


def per_task_diversity(
    records: list[dict],
    resolved_by_agent: dict[str, set[str]],
) -> list[dict]:
    by_instance: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_instance[r["instance_id"]].append(r)

    rows: list[dict] = []
    for iid, traj_list in by_instance.items():
        if len(traj_list) < 2:
            continue
        steps = np.array([t["canonical_length"] for t in traj_list], dtype=float)
        seqs = [t["canonical"] for t in traj_list]
        agents_here = [t["agent"] for t in traj_list]

        pair_lev = [
            normalized_levenshtein_tokens(a, b)
            for a, b in combinations(seqs, 2)
        ]
        n_resolved = sum(
            1 for a in agents_here if iid in resolved_by_agent.get(a, set())
        )
        n_agents_on_task = len(traj_list)

        rows.append({
            "instance_id": iid,
            "n_agents": n_agents_on_task,
            "n_resolved": n_resolved,
            "fraction_resolved": n_resolved / n_agents_on_task,
            "all_resolved": n_resolved == n_agents_on_task,
            "none_resolved": n_resolved == 0,
            "n_steps_mean": float(np.mean(steps)),
            "n_steps_range": float(np.max(steps) - np.min(steps)),
            "n_steps_cv": (
                float(np.std(steps) / np.mean(steps))
                if np.mean(steps) > 0 else 0.0
            ),
            "mean_pairwise_levenshtein": (
                float(np.mean(pair_lev)) if pair_lev else 0.0
            ),
            "max_pairwise_levenshtein": (
                float(np.max(pair_lev)) if pair_lev else 0.0
            ),
        })
    return rows


def plot_distribution(rows: list[dict], out_path: Path) -> None:
    df = pd.DataFrame(rows)
    median_lev = df["mean_pairwise_levenshtein"].median()
    median_cv = df["n_steps_cv"].median()

    panel_a = (
        alt.Chart(df)
        .mark_bar(color=BLUE, opacity=0.85, stroke="white", strokeWidth=0.5)
        .encode(
            x=alt.X(
                "mean_pairwise_levenshtein:Q",
                bin=alt.Bin(maxbins=24, extent=[0, 1]),
                axis=alt.Axis(
                    title="Mean pairwise Levenshtein across agents",
                    domain=False, ticks=False, labelFontSize=10,
                ),
            ),
            y=alt.Y(
                "count():Q",
                axis=alt.Axis(title="Number of tasks", domain=False,
                              ticks=False, labelFontSize=10),
            ),
        )
        .properties(width=320, height=200)
    )
    rule_a = (
        alt.Chart(pd.DataFrame({"x": [median_lev]}))
        .mark_rule(color=COPPER, strokeDash=[3, 3], strokeWidth=1.5)
        .encode(x="x:Q")
    )
    label_a = (
        alt.Chart(pd.DataFrame({"x": [median_lev], "y": [0],
                                "label": [f"median = {median_lev:.2f}"]}))
        .mark_text(align="left", dx=6, dy=-6, fontSize=10, color=COPPER)
        .encode(x="x:Q", y=alt.value(12), text="label:N")
    )
    a = (
        (panel_a + rule_a + label_a)
        .properties(title=alt.TitleParams(
            text="How much do agents diverge per task?",
            subtitle="Procedural permissiveness across 9 agents",
            fontSize=11, subtitleFontSize=9,
            anchor="start", color="#111", subtitleColor="#777",
        ))
    )

    panel_b = (
        alt.Chart(df.assign(n_steps_cv_clipped=df["n_steps_cv"].clip(upper=2.0)))
        .mark_bar(color=GREEN, opacity=0.85, stroke="white", strokeWidth=0.5)
        .encode(
            x=alt.X(
                "n_steps_cv_clipped:Q",
                bin=alt.Bin(maxbins=24),
                axis=alt.Axis(
                    title="Coefficient of variation on n_steps",
                    domain=False, ticks=False, labelFontSize=10,
                ),
            ),
            y=alt.Y(
                "count():Q",
                axis=alt.Axis(title="Number of tasks", domain=False,
                              ticks=False, labelFontSize=10),
            ),
        )
        .properties(width=320, height=200)
    )
    rule_b = (
        alt.Chart(pd.DataFrame({"x": [median_cv]}))
        .mark_rule(color=COPPER, strokeDash=[3, 3], strokeWidth=1.5)
        .encode(x="x:Q")
    )
    label_b = (
        alt.Chart(pd.DataFrame({"x": [median_cv], "label": [f"median = {median_cv:.2f}"]}))
        .mark_text(align="left", dx=6, dy=-6, fontSize=10, color=COPPER)
        .encode(x="x:Q", y=alt.value(12), text="label:N")
    )
    b = (
        (panel_b + rule_b + label_b)
        .properties(title=alt.TitleParams(
            text="Step-count diversity per task",
            subtitle="Higher = more agent disagreement on effort",
            fontSize=11, subtitleFontSize=9,
            anchor="start", color="#111", subtitleColor="#777",
        ))
    )

    chart = (
        alt.hconcat(a, b, spacing=40)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    chart.save(str(out_path), scale_factor=2)


def plot_by_resolved(rows: list[dict], out_path: Path) -> None:
    df = pd.DataFrame(rows)

    def bucket(row):
        if row["all_resolved"]:
            return "All resolved"
        if row["none_resolved"]:
            return "None resolved"
        return "Mixed outcome"

    df = df.assign(bucket=df.apply(bucket, axis=1))
    bucket_order = ["All resolved", "Mixed outcome", "None resolved"]
    color_range = [GREEN, OLIVE, MAGENTA]

    chart = (
        alt.Chart(df)
        .mark_boxplot(size=40, extent="min-max", opacity=0.9)
        .encode(
            x=alt.X(
                "bucket:N",
                sort=bucket_order,
                axis=alt.Axis(title=None, domain=False, ticks=False,
                              labelFontSize=11),
            ),
            y=alt.Y(
                "mean_pairwise_levenshtein:Q",
                axis=alt.Axis(
                    title="Mean pairwise Levenshtein across agents",
                    domain=False, ticks=False, labelFontSize=10,
                ),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "bucket:N",
                scale=alt.Scale(domain=bucket_order, range=color_range),
                legend=None,
            ),
        )
        .properties(
            width=420, height=240,
            title=alt.TitleParams(
                text="Per-task diversity by outcome bucket",
                subtitle="Across 9 agents on each instance",
                fontSize=11, subtitleFontSize=9,
                anchor="start", color="#111", subtitleColor="#777",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    chart.save(str(out_path), scale_factor=2)


def main() -> int:
    print(f"Loading {SEQ_PATH} ...")
    records = load_records()
    print(f"  {len(records)} trajectories")

    print(f"Loading {PASS_PATH} ...")
    resolved_by_agent = load_resolved_sets()
    for a, s in resolved_by_agent.items():
        print(f"  {a:22s} resolved {len(s)} instances")

    print("Computing per-task diversity ...")
    rows = per_task_diversity(records, resolved_by_agent)
    print(f"  {len(rows)} instances with >=2 agents")

    OUT_JSONL.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"Wrote {OUT_JSONL}")

    print("Plotting distribution ...")
    plot_distribution(rows, OUT / "task_diversity_distribution.png")
    print(f"Saved {OUT / 'task_diversity_distribution.png'}")

    print("Plotting by outcome bucket ...")
    plot_by_resolved(rows, OUT / "task_diversity_by_resolved.png")
    print(f"Saved {OUT / 'task_diversity_by_resolved.png'}")

    df = pd.DataFrame(rows)
    print()
    print("Summary:")
    print(f"  mean_pairwise_levenshtein: median={df['mean_pairwise_levenshtein'].median():.3f} "
          f"mean={df['mean_pairwise_levenshtein'].mean():.3f}")
    print(f"  n_steps_cv:                median={df['n_steps_cv'].median():.3f} "
          f"mean={df['n_steps_cv'].mean():.3f}")
    print(f"  n_resolved distribution (out of {N_AGENTS_FULL} agents):")
    for n in range(N_AGENTS_FULL + 1):
        c = int((df["n_resolved"] == n).sum())
        if c > 0:
            print(f"    {n}/{N_AGENTS_FULL}: {c} instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
