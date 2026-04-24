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
    output/paper2_pilot/token_cost_per_agent.png
    output/paper2_pilot/token_cost_by_difficulty.png

Usage:
    python -m analysis.preferences.token_cost
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE = PROJECT_ROOT / "output" / "trajectories" / ".cache"
OUT = PROJECT_ROOT / "output" / "paper2_pilot"
SEQ_PATH = OUT / "bpe_sequences.jsonl"
DIVERSITY_PATH = OUT / "task_diversity.csv"

AGENT_SHORT = {
    "20240402_sweagent_gpt4": "GPT-4",
    "20240620_sweagent_claude3.5sonnet": "Claude-3.5",
    "20240728_sweagent_gpt4o": "GPT-4o",
}
AGENT_COLORS = {
    "Claude-3.5": "#009E73",
    "GPT-4": "#0072B2",
    "GPT-4o": "#E69F00",
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


def load_model_stats() -> list[dict]:
    records = []
    for agent_dir in sorted(CACHE.iterdir()):
        if not agent_dir.is_dir():
            continue
        short = AGENT_SHORT.get(agent_dir.name, agent_dir.name)
        for traj_file in sorted(agent_dir.glob("*.json")):
            with open(traj_file) as f:
                d = json.load(f)
            stats = (d.get("info") or {}).get("model_stats") or {}
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
    agents = [a for a in ["Claude-3.5", "GPT-4", "GPT-4o"] if a in per_agent]
    x = np.arange(len(agents))
    colors = [AGENT_COLORS[a] for a in agents]

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.8))

    ax = axes[0]
    vals = [per_agent[a]["tokens_sent_mean"] / 1000 for a in agents]
    ax.bar(x, vals, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=9)
    ax.set_ylabel("mean tokens sent per task (thousands)")
    ax.set_title("Input tokens (prompt + history)\nhigher = more context burned per task", fontsize=10)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals) * 0.02, f"{v:.0f}k", ha="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    vals = [per_agent[a]["tokens_received_mean"] / 1000 for a in agents]
    ax.bar(x, vals, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=9)
    ax.set_ylabel("mean tokens received per task (thousands)")
    ax.set_title("Output tokens\n(model's generated actions + thoughts)", fontsize=10)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals) * 0.02, f"{v:.1f}k", ha="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    vals = [per_agent[a]["api_calls_mean"] for a in agents]
    ax.bar(x, vals, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=9)
    ax.set_ylabel("mean API calls per task")
    ax.set_title("API calls per task\n(one per decision step)", fontsize=10)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals) * 0.02, f"{v:.0f}", ha="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[3]
    vals = [per_agent[a]["cost_mean_usd"] for a in agents]
    ax.bar(x, vals, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=9)
    ax.set_ylabel("mean cost per task (USD)")
    ax.set_title(
        "Cost per task (USD)\n(what the agent burned solving or giving up)",
        fontsize=10,
    )
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals) * 0.02, f"${v:.2f}", ha="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Per-agent infrastructure costs (867 trajectories total)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_by_difficulty(by_diff: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)

    metric_specs = [
        ("tokens_sent_mean", "mean tokens sent per task", 1000, "thousands"),
        ("api_calls_mean", "mean API calls per task", 1, ""),
        ("cost_mean_usd", "mean cost per task (USD)", 1, ""),
    ]

    for ax, (key, label, scale, unit) in zip(axes, metric_specs):
        for a in ["Claude-3.5", "GPT-4", "GPT-4o"]:
            xs, ys = [], []
            for d in ["0", "1", "2", "3"]:
                if d in by_diff and a in by_diff[d]:
                    xs.append(int(d))
                    ys.append(by_diff[d][a][key] / scale)
            if xs:
                ax.plot(xs, ys, marker="o", color=AGENT_COLORS[a], label=a,
                        linewidth=2, markersize=7)
        ax.set_xlabel("number of agents that solved the task")
        ylab = f"{label} ({unit})" if unit else label
        ax.set_ylabel(ylab)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["0 (nobody)", "1", "2", "3 (everyone)"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, frameon=False, loc="best")

    fig.suptitle(
        "Cost by task difficulty: do failing tasks burn more than solving tasks?\n"
        "All three agents: tokens and cost rise on harder tasks (all agents flail longer when nobody solves).",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
    plot_per_agent(per_agent, OUT / "token_cost_per_agent.png")
    plot_by_difficulty(by_difficulty, OUT / "token_cost_by_difficulty.png")

    print(f"\nSaved:")
    print(f"  {OUT / 'token_cost.json'}")
    print(f"  {OUT / 'token_cost_per_agent.png'}")
    print(f"  {OUT / 'token_cost_by_difficulty.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
