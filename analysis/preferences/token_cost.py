"""Token and cost analysis from SWE-agent raw trajectories.

Each .traj file has info.model_stats: {total_cost, instance_cost, tokens_sent,
tokens_received, api_calls}. Join with bpe_sequences + task_diversity for
difficulty-bucketed and per-agent infra costs.

Asks:
  - How many tokens does each agent burn per task, on average?
  - Do same-family agents cost similarly?
  - Does cost scale with task difficulty (failing a hard task vs solving an easy one)?
  - Is procedure length (motif count) correlated with tokens (context size)?

Outputs:
    output/paper2_pilot/token_cost.json
    output/paper2_pilot/token_cost_per_agent_{tokens_sent,tokens_received,api_calls,cost}.png
    output/paper2_pilot/token_cost_by_difficulty.png
    output/paper2_pilot/token_cost_by_difficulty_{tokens,api_calls,cost}.png

Usage:
    python -m analysis.preferences.token_cost
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sys
import altair as alt
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.theme import (
    register, BLUE, ORANGE, GREEN, NEAR_BLACK, GRAY,
    AGENT_COLORS as CANONICAL_AGENT_COLORS,
)
register()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE = PROJECT_ROOT / "output" / "trajectories" / ".cache"
OUT = PROJECT_ROOT / "output" / "paper2_pilot"
SEQ_PATH = OUT / "bpe_sequences.jsonl"
DIVERSITY_PATH = OUT / "task_diversity.csv"

# All SWE-agent-based submissions, which share the `info.model_stats`
# extraction path. The three cross-scaffold submissions (Agentless,
# DARS, Moatless) use different schemas and are handled by separate
# extractors; their values land as zero here and are filtered out
# before plotting.
AGENT_SHORT = {
    "20240402_sweagent_claude3opus":                "Claude-3",
    "20240402_sweagent_gpt4":                       "GPT-4",
    "20240620_sweagent_claude3.5sonnet":            "Claude-3.5",
    "20240728_sweagent_gpt4o":                      "GPT-4o",
    "20250226_sweagent_claude-3-7-sonnet-20250219": "Claude-3.7-thinking",
    "20250526_sweagent_claude-4-sonnet-20250514":   "Claude-4",
    "20250111_moatless_deepseek_v3":                "Moatless+V3",
}

# Submissions for which the trace shape carries no token/cost data
# (verified by exhaustive key search; see commit message). They stay
# absent from the figures rather than render as zero rows.
NO_TOKEN_DATA = {
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022",  # agentless_log_text
    "20250205_dars_agent_claude_3.5_sonnet_deepseek_r1",  # dars_traj_list
}
# Per-agent colors fall back to the canonical theme; if an agent is
# missing from the canonical map we use a neutral grey so the plot
# still renders.
_FALLBACK_COLOR = "#888888"
AGENT_COLORS = {
    a: CANONICAL_AGENT_COLORS.get(a, _FALLBACK_COLOR)
    for a in AGENT_SHORT.values()
}


def load_sequences() -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    with open(SEQ_PATH) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(r["agent"], r["instance_id"])] = r
    return out


def load_difficulty() -> dict[str, int]:
    out = {}
    with open(DIVERSITY_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["instance_id"]] = int(row["n_resolved"])
    return out


def extract_model_stats(d: dict) -> dict:
    """Return the model_stats dict, handling both SWE-agent trajectory shapes.

    Old format (Claude-3 / Claude-3.5 / GPT-4 / GPT-4o):
        {environment, trajectory, history, info: {..., model_stats: {...}}}
    New format (Claude-3.7-thinking / Claude-4, post v1.1):
        {submission, instance_id, format, content: {..., info: {..., model_stats}}}

    Moatless and the other cross-scaffold submissions don't use this
    shape; Moatless is handled by extract_moatless_stats (per-step
    `usage` + `response_cost` walker). Agentless and DARS do not carry
    token data in their traces and stay at zero.
    """
    info = d.get("info")
    if not isinstance(info, dict):
        content = d.get("content")
        if isinstance(content, dict):
            info = content.get("info")
    if not isinstance(info, dict):
        return {}
    stats = info.get("model_stats")
    return stats if isinstance(stats, dict) else {}


def extract_moatless_stats(d: dict) -> dict:
    """Sum per-step LLM-call usage and cost across a Moatless trace.

    Moatless serializes each LLM call's response as a nested object with
    `usage = {prompt_tokens, completion_tokens, total_tokens}` and
    `response_cost = <float USD>`. There can be multiple calls per step
    (planner + identifier + editor) at varying depth, so we walk the
    whole structure and aggregate.

    Returns a dict matching the model_stats schema (tokens_sent,
    tokens_received, api_calls, instance_cost), 0 if the trace is not
    Moatless-shaped or has no usage data.
    """
    acc = {"tokens_sent": 0, "tokens_received": 0, "api_calls": 0, "instance_cost": 0.0}

    def walk(node):
        if isinstance(node, dict):
            usage = node.get("usage")
            if isinstance(usage, dict) and (
                "prompt_tokens" in usage or "completion_tokens" in usage
            ):
                acc["api_calls"] += 1
                acc["tokens_sent"] += int(usage.get("prompt_tokens") or 0)
                acc["tokens_received"] += int(usage.get("completion_tokens") or 0)
            cost = node.get("response_cost")
            if isinstance(cost, (int, float)):
                acc["instance_cost"] += float(cost)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(d)
    return acc


def load_model_stats() -> list[dict]:
    records = []
    for agent_dir in sorted(CACHE.iterdir()):
        if not agent_dir.is_dir():
            continue
        if agent_dir.name not in AGENT_SHORT:
            # Skip submissions we don't have an extractor for; they would
            # land as zero rows and be filtered out anyway, but skipping
            # avoids carrying the long submission ID through downstream
            # iteration as if it were an agent.
            continue
        short = AGENT_SHORT[agent_dir.name]
        is_moatless = agent_dir.name == "20250111_moatless_deepseek_v3"
        for traj_file in sorted(agent_dir.glob("*.json")):
            if traj_file.name == "manifest.json":
                continue
            with open(traj_file) as f:
                d = json.load(f)
            stats = (
                extract_moatless_stats(d) if is_moatless
                else extract_model_stats(d)
            )
            records.append({
                "agent": short,
                "instance_id": traj_file.stem,
                "tokens_sent": int(stats.get("tokens_sent", 0)),
                "tokens_received": int(stats.get("tokens_received", 0)),
                "api_calls": int(stats.get("api_calls", 0)),
                "instance_cost_usd": float(stats.get("instance_cost", 0)),
            })
    return records


def summarize_per_agent(records: list[dict]) -> dict:
    per_agent = defaultdict(list)
    for r in records:
        per_agent[r["agent"]].append(r)

    out = {}
    for agent, rs in per_agent.items():
        tokens_sent = np.array([r["tokens_sent"] for r in rs])
        tokens_received = np.array([r["tokens_received"] for r in rs])
        api_calls = np.array([r["api_calls"] for r in rs])
        costs = np.array([r["instance_cost_usd"] for r in rs])
        out[agent] = {
            "n_trajectories": len(rs),
            "tokens_sent_mean": float(tokens_sent.mean()),
            "tokens_sent_median": float(np.median(tokens_sent)),
            "tokens_received_mean": float(tokens_received.mean()),
            "api_calls_mean": float(api_calls.mean()),
            "api_calls_median": float(np.median(api_calls)),
            "cost_mean_usd": float(costs.mean()),
            "cost_median_usd": float(np.median(costs)),
            "cost_total_usd": float(costs.sum()),
        }
    return out


def summarize_by_difficulty(
    records: list[dict], difficulty: dict[str, int]
) -> dict:
    buckets: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        d = difficulty.get(r["instance_id"])
        if d is None:
            continue
        buckets[d][r["agent"]].append(r)

    out = {}
    for d in sorted(buckets):
        per_agent = {}
        for a, rs in buckets[d].items():
            per_agent[a] = {
                "n": len(rs),
                "tokens_sent_mean": float(np.mean([r["tokens_sent"] for r in rs])),
                "api_calls_mean": float(np.mean([r["api_calls"] for r in rs])),
                "cost_mean_usd": float(np.mean([r["instance_cost_usd"] for r in rs])),
            }
        out[str(d)] = per_agent
    return out


def correlate_with_length(
    records: list[dict], seqs: dict[tuple[str, str], dict]
) -> dict:
    out = {}
    for a in ["Claude-3.5", "GPT-4", "GPT-4o"]:
        pts = []
        for r in records:
            if r["agent"] != a:
                continue
            seq = seqs.get((a, r["instance_id"]))
            if not seq:
                continue
            pts.append((seq["canonical_length"], r["tokens_sent"]))
        if len(pts) < 3:
            continue
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        corr = float(np.corrcoef(xs, ys)[0, 1])
        slope, intercept = np.polyfit(xs, ys, 1)
        out[a] = {
            "n": len(pts),
            "pearson_r": corr,
            "slope_tokens_per_atom": float(slope),
            "intercept": float(intercept),
        }
    return out


def plot_per_agent(per_agent: dict, out_path: Path) -> None:
    # Keep agents whose cost extraction succeeded (mean > 0); the
    # zero-valued rows are from scaffolds that need separate extraction.
    agents = [
        a for a in AGENT_SHORT.values()
        if a in per_agent and per_agent[a].get("cost_mean_usd", 0) > 0
    ]
    agent_order = list(reversed(agents))  # top-to-bottom: Claude-3.5 at top

    metric_specs = [
        ("tokens_sent_mean",     "Mean input tokens per task (K)",  1000, ".0f"),
        ("tokens_received_mean", "Mean output tokens per task (K)", 1000, ".1f"),
        ("api_calls_mean",       "Mean API calls per task",         1,    ".1f"),
        ("cost_mean_usd",        "Mean cost per task (USD)",        1,    ".3f"),
    ]

    color_scale = alt.Scale(
        domain=list(AGENT_COLORS.keys()),
        range=list(AGENT_COLORS.values()),
    )

    for key, x_title, scale, fmt in metric_specs:
        rows = [
            {"agent": a, "value": per_agent[a][key] / scale, "zero": 0.0}
            for a in agents
        ]
        df = pd.DataFrame(rows)
        max_val = df["value"].max()
        x_max = max_val * 1.35  # room for value labels

        rule = (
            alt.Chart(df)
            .mark_rule(strokeWidth=2.5, opacity=0.55)
            .encode(
                y=alt.Y("agent:N", sort=agent_order,
                        axis=alt.Axis(title=None, domain=False, ticks=False, labelFontSize=12)),
                x=alt.X("zero:Q",
                        scale=alt.Scale(domain=[0, x_max]),
                        axis=alt.Axis(title=x_title, domain=False, ticks=False,
                                      labelFontSize=10)),
                x2="value:Q",
                color=alt.Color("agent:N", scale=color_scale, legend=None),
            )
        )

        pts = (
            alt.Chart(df)
            .mark_point(size=120, filled=True, strokeWidth=1.5)
            .encode(
                y=alt.Y("agent:N", sort=agent_order),
                x=alt.X("value:Q", scale=alt.Scale(domain=[0, x_max])),
                color=alt.Color("agent:N", scale=color_scale, legend=None),
                stroke=alt.value("white"),
            )
        )

        labels = (
            alt.Chart(df)
            .mark_text(align="left", dx=10, fontSize=11, color="#444444")
            .encode(
                y=alt.Y("agent:N", sort=agent_order),
                x=alt.X("value:Q", scale=alt.Scale(domain=[0, x_max])),
                text=alt.Text("value:Q", format=fmt),
            )
        )

        stem = out_path.stem.replace("_per_agent", "")
        metric_slug = key.replace("_mean", "").replace("_usd", "")
        panel_path = out_path.parent / f"{stem}_per_agent_{metric_slug}.png"

        chart = (
            (rule + pts + labels)
            .properties(
                title=alt.TitleParams(
                    text=x_title,
                    fontSize=13, color="#111111", anchor="start",
                ),
                width=320, height=130,
            )
            .configure_view(strokeWidth=0)
        )

        chart.save(str(panel_path), scale_factor=2)
        print(f"  Saved: {panel_path.name}")


def plot_by_difficulty(by_diff: dict, out_path: Path) -> None:
    """Render one combined 3-panel PNG (token_cost_by_difficulty.png) plus
    three per-metric standalone PNGs (one per axis), per the
    "split into multiple plots" style preference.

    Includes every agent for which token-cost extraction succeeded.
    Agentless and DARS are omitted from the figure because their trace
    formats carry no token/cost data; the subtitle says so.
    """
    metric_specs = [
        ("tokens_sent_mean", "Input tokens per task (thousands)", 1000, "tokens"),
        ("api_calls_mean", "API calls per task", 1, "api_calls"),
        ("cost_mean_usd", "Cost per task (USD)", 1, "cost"),
    ]
    diff_label = {"0": "0", "1": "1", "2": "2", "3": "3"}

    # Agents that appear at all in the by_diff table with non-zero cost.
    # Drops agents whose extraction produced only zeros.
    agents_with_data = sorted({
        a
        for d_str in by_diff
        for a, stats in by_diff[d_str].items()
        if stats.get("cost_mean_usd", 0) > 0
    })

    rows = []
    for key, label, scale, slug in metric_specs:
        for d_str in ["0", "1", "2", "3"]:
            for a in agents_with_data:
                if d_str in by_diff and a in by_diff[d_str]:
                    rows.append({
                        "difficulty": int(d_str),
                        "difficulty_label": diff_label[d_str],
                        "agent": a,
                        "metric_label": label,
                        "metric_slug": slug,
                        "value": by_diff[d_str][a][key] / scale,
                    })
    df = pd.DataFrame(rows)

    color_scale = alt.Scale(
        domain=list(AGENT_COLORS.keys()),
        range=list(AGENT_COLORS.values()),
    )
    agent_order = agents_with_data
    diff_order = ["0", "1", "2", "3"]
    subtitle_text = (
        f"{len(agents_with_data)} agents; Agentless and DARS "
        "omitted (trace shapes carry no token/cost data)"
    )

    def _per_metric_panel(metric_label: str, df_panel: pd.DataFrame, width: int = 320, height: int = 220):
        base = alt.Chart(df_panel).encode(
            x=alt.X("difficulty_label:O", sort=diff_order,
                    axis=alt.Axis(title="Number of agents that solved the task",
                                  domain=False, ticks=False, labelFontSize=10)),
            y=alt.Y("value:Q",
                    axis=alt.Axis(title=metric_label, domain=False, ticks=False,
                                  labelFontSize=10)),
            color=alt.Color("agent:N", scale=color_scale,
                            legend=alt.Legend(orient="bottom", title=None, symbolSize=80)),
            detail="agent:N",
        )
        lines = base.mark_line(strokeWidth=2)
        points = base.mark_point(size=60, filled=True)
        return alt.layer(lines, points).properties(width=width, height=height)

    # 1) Per-metric standalone PNGs (split-into-multiple-plots preference).
    for key, label, scale, slug in metric_specs:
        df_m = df[df["metric_slug"] == slug]
        panel = _per_metric_panel(label, df_m, width=320, height=220)
        chart_m = (
            panel
            .properties(
                title=alt.TitleParams(
                    text=label,
                    subtitle=subtitle_text,
                    fontSize=13, subtitleFontSize=10,
                    color="#111111", subtitleColor="#888888", anchor="start",
                )
            )
            .configure_view(strokeWidth=0)
            .configure_axis(grid=False)
        )
        per_metric_path = out_path.parent / f"token_cost_by_difficulty_{slug}.png"
        chart_m.save(str(per_metric_path), scale_factor=2)
        print(f"  Saved: {per_metric_path.name}")

    # 2) Combined 3-panel PNG (kept for the existing dashboard reference).
    panels = []
    for key, label, scale, slug in metric_specs:
        df_m = df[df["metric_slug"] == slug]
        panels.append(_per_metric_panel(label, df_m, width=200, height=200))

    combined = (
        alt.hconcat(*panels, spacing=40)
        .properties(
            title=alt.TitleParams(
                text="Token cost by task difficulty",
                subtitle="3-agent baseline (cost data unavailable for new submissions)",
                fontSize=13, subtitleFontSize=10,
                color="#111111", subtitleColor="#888888", anchor="start",
            )
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    combined.save(str(out_path), scale_factor=2)


def main() -> int:
    print("Loading model_stats from cache...")
    records = load_model_stats()
    seqs = load_sequences()
    difficulty = load_difficulty()
    print(f"Loaded {len(records)} trajectories")

    per_agent = summarize_per_agent(records)
    by_difficulty = summarize_by_difficulty(records, difficulty)
    length_corr = correlate_with_length(records, seqs)

    print("\nPer-agent summary:")
    for a, s in per_agent.items():
        print(f"  {a:<12s}  n={s['n_trajectories']:>3d}  "
              f"tokens_sent~{s['tokens_sent_mean']/1000:>5.0f}k  "
              f"tokens_recv~{s['tokens_received_mean']/1000:>4.1f}k  "
              f"api_calls~{s['api_calls_mean']:>5.1f}  "
              f"cost~${s['cost_mean_usd']:>5.2f}  "
              f"total=${s['cost_total_usd']:>7.2f}")

    print("\nBy difficulty (tokens_sent_mean / 1000):")
    print(f"  {'difficulty':<12s}  " + "  ".join(f"{a:<12s}" for a in sorted(per_agent.keys())))
    for d in ["0", "1", "2", "3"]:
        if d not in by_difficulty:
            continue
        row = by_difficulty[d]
        vals = [f"{row[a]['tokens_sent_mean']/1000:>10.1f}k" if a in row else " "*11 for a in sorted(per_agent.keys())]
        print(f"  {d+' resolved':<12s}  " + "  ".join(vals))

    print("\nTokens-vs-length correlation (Pearson r):")
    for a, s in length_corr.items():
        print(f"  {a:<12s}  r={s['pearson_r']:.3f}  "
              f"slope ~{s['slope_tokens_per_atom']:.0f} tokens/atom  (n={s['n']})")

    (OUT / "token_cost.json").write_text(json.dumps({
        "per_agent": per_agent,
        "by_difficulty": by_difficulty,
        "length_correlation": length_corr,
    }, indent=2, default=str))
    # plot_per_agent treats the second arg as a filename stem only; it
    # writes one PNG per metric (tokens_sent / tokens_received /
    # api_calls / cost) and does not write the stem path itself.
    plot_per_agent(per_agent, OUT / "token_cost_per_agent.png")
    plot_by_difficulty(by_difficulty, OUT / "token_cost_by_difficulty.png")

    print(f"\nSaved:")
    print(f"  {OUT / 'token_cost.json'}")
    print(f"  {OUT / 'token_cost_by_difficulty.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
