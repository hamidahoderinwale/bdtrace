"""Step-resource analysis: per-atom and per-motif cost profiles, efficiency frontier.

Attributes trajectory-level model_stats (tokens_sent, instance_cost) back to
per-atom and per-motif estimates. Joins with outcome (resolved yes/no) for
per-motif success association. Identifies wasteful motifs (high cost, low
success rate) and efficient trajectories (low cost per resolved task).

Cost attribution (approximation): per-trajectory tokens_per_atom ratio is
applied uniformly, so atom-level cost is estimated rather than exact. The
attribution assumes each atom contributes its proportional share of total
tokens_sent. Real per-step costs grow over the trajectory (context
accumulates) but the average-per-atom approximation is sufficient for
relative comparisons.

Outputs:
    output/paper2_pilot/step_resources.json
    output/paper2_pilot/step_resources_atoms.png          (atom cost-vs-usage)
    output/paper2_pilot/step_resources_motifs.png         (motif cost-vs-success quadrant)
    output/paper2_pilot/step_resources_wasteful.png       (top wasteful motifs)
    output/paper2_pilot/step_resources_efficiency.png     (per-agent efficiency frontier)

Usage:
    python -m analysis.preferences.step_resource_analysis
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
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
PAIRS_PATH = OUT / "tied_outcome_pairs.csv"

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
    out = {}
    with open(SEQ_PATH) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(r["agent"], r["instance_id"])] = r
    return out


def load_model_stats() -> dict[tuple[str, str], dict]:
    out = {}
    for agent_dir in sorted(CACHE.iterdir()):
        if not agent_dir.is_dir():
            continue
        short = AGENT_SHORT.get(agent_dir.name, agent_dir.name)
        for traj_file in sorted(agent_dir.glob("*.json")):
            with open(traj_file) as f:
                d = json.load(f)
            stats = (d.get("info") or {}).get("model_stats") or {}
            out[(short, traj_file.stem)] = {
                "tokens_sent": int(stats.get("tokens_sent", 0)),
                "tokens_received": int(stats.get("tokens_received", 0)),
                "api_calls": int(stats.get("api_calls", 0)),
                "instance_cost_usd": float(stats.get("instance_cost", 0)),
            }
    return out


def load_resolved_set() -> set[tuple[str, str]]:
    """Return set of (agent, instance_id) pairs where the agent resolved the task."""
    resolved = set()
    with open(PAIRS_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            agent_map = {
                "Claude 3.5 Sonnet (SWE-agent)": "Claude-3.5",
                "GPT-4 (SWE-agent)": "GPT-4",
                "GPT-4o (SWE-agent)": "GPT-4o",
            }
            resolved.add((agent_map.get(row["agent_a"], row["agent_a"]), row["instance_id"]))
            resolved.add((agent_map.get(row["agent_b"], row["agent_b"]), row["instance_id"]))
    return resolved


def per_atom_profile(seqs: dict, stats: dict) -> dict[str, dict]:
    """For each canonical atom, aggregate usage + estimated token cost across corpus."""
    profile: dict[str, dict] = defaultdict(lambda: {
        "occurrences": 0,
        "tokens_attributed": 0.0,
        "n_trajectories": 0,
        "by_agent": defaultdict(int),
    })
    for (agent, inst), seq_rec in seqs.items():
        st = stats.get((agent, inst))
        if not st:
            continue
        atoms = seq_rec["canonical"]
        if not atoms:
            continue
        tokens_per_atom = st["tokens_sent"] / len(atoms) if len(atoms) > 0 else 0
        seen_here = set()
        for a in atoms:
            profile[a]["occurrences"] += 1
            profile[a]["tokens_attributed"] += tokens_per_atom
            profile[a]["by_agent"][agent] += 1
            seen_here.add(a)
        for a in seen_here:
            profile[a]["n_trajectories"] += 1

    out = {}
    for a, p in profile.items():
        out[a] = {
            "occurrences": p["occurrences"],
            "n_trajectories": p["n_trajectories"],
            "mean_tokens_per_use": p["tokens_attributed"] / p["occurrences"] if p["occurrences"] else 0,
            "total_tokens_attributed": p["tokens_attributed"],
            "by_agent": dict(p["by_agent"]),
        }
    return out


def per_motif_profile(seqs: dict, stats: dict, resolved: set) -> dict[str, dict]:
    """For each motif, aggregate usage + cost + success association."""
    profile: dict[str, dict] = defaultdict(lambda: {
        "occurrences": 0,
        "tokens_attributed": 0.0,
        "n_trajectories_with": 0,
        "n_trajectories_with_resolved": 0,
        "by_agent": defaultdict(int),
    })
    total_resolved = sum(1 for key in stats if key in resolved)
    total_trajectories = len(stats)

    for (agent, inst), seq_rec in seqs.items():
        st = stats.get((agent, inst))
        if not st:
            continue
        atoms = seq_rec["canonical"]
        motifs = seq_rec["bpe"]
        if not atoms or not motifs:
            continue
        tokens_per_atom = st["tokens_sent"] / len(atoms) if len(atoms) > 0 else 0
        was_resolved = (agent, inst) in resolved

        seen_motifs = set()
        for m in motifs:
            n_atoms = m.count("+") + 1
            profile[m]["occurrences"] += 1
            profile[m]["tokens_attributed"] += n_atoms * tokens_per_atom
            profile[m]["by_agent"][agent] += 1
            seen_motifs.add(m)
        for m in seen_motifs:
            profile[m]["n_trajectories_with"] += 1
            if was_resolved:
                profile[m]["n_trajectories_with_resolved"] += 1

    base_resolve_rate = total_resolved / total_trajectories if total_trajectories else 0
    out = {}
    for m, p in profile.items():
        success_rate = p["n_trajectories_with_resolved"] / p["n_trajectories_with"] if p["n_trajectories_with"] else 0
        out[m] = {
            "motif": m,
            "n_atoms": m.count("+") + 1,
            "occurrences": p["occurrences"],
            "n_trajectories_with": p["n_trajectories_with"],
            "mean_tokens_per_use": p["tokens_attributed"] / p["occurrences"] if p["occurrences"] else 0,
            "success_rate_when_used": success_rate,
            "success_rate_vs_base": success_rate - base_resolve_rate,
            "by_agent": dict(p["by_agent"]),
        }
    out["__base_rate__"] = {"base_resolve_rate": base_resolve_rate, "n_resolved": total_resolved, "n_total": total_trajectories}
    return out


def efficiency_frontier(seqs: dict, stats: dict, resolved: set) -> dict:
    """Per-agent: trajectories, cost distribution, cost per resolved."""
    by_agent = defaultdict(lambda: {"trajectories": [], "resolved": 0, "costs": [], "costs_resolved": []})
    for (agent, inst), st in stats.items():
        was_resolved = (agent, inst) in resolved
        by_agent[agent]["trajectories"].append((inst, st["instance_cost_usd"], was_resolved))
        by_agent[agent]["costs"].append(st["instance_cost_usd"])
        if was_resolved:
            by_agent[agent]["costs_resolved"].append(st["instance_cost_usd"])
            by_agent[agent]["resolved"] += 1

    out = {}
    for a, d in by_agent.items():
        n_total = len(d["trajectories"])
        total_cost = sum(d["costs"])
        out[a] = {
            "n_total": n_total,
            "n_resolved": d["resolved"],
            "resolve_rate": d["resolved"] / n_total if n_total else 0,
            "total_cost_usd": total_cost,
            "mean_cost_per_task_usd": total_cost / n_total if n_total else 0,
            "mean_cost_per_resolved_usd": total_cost / d["resolved"] if d["resolved"] else float("inf"),
            "median_cost_per_task_usd": float(np.median(d["costs"])) if d["costs"] else 0,
        }
    return out


def plot_atoms(atoms: dict, out_path: Path) -> None:
    # scatter: x = mean tokens per use, y = occurrences (log)
    items = [(a, p) for a, p in atoms.items() if p["occurrences"] >= 5]
    xs = np.array([p["mean_tokens_per_use"] for _, p in items])
    ys = np.array([p["occurrences"] for _, p in items])
    names = [a for a, _ in items]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xs, ys, s=50, color="#5d90e0", edgecolor="white", alpha=0.85)
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlabel("mean tokens attributed per use (estimated)")
    ax.set_ylabel("total occurrences across corpus (log scale)")
    ax.set_title(
        "Canonical atom cost-vs-usage profile\n"
        "Top-right = expensive and common (dominant cost drivers). Bottom-left = rare and cheap.",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25)

    # annotate top-6 by occurrences and top-3 by cost
    top_by_count = sorted(items, key=lambda kv: -kv[1]["occurrences"])[:6]
    top_by_cost = sorted(items, key=lambda kv: -kv[1]["mean_tokens_per_use"])[:3]
    seen = set()
    for a, p in top_by_count + top_by_cost:
        if a in seen:
            continue
        seen.add(a)
        ax.annotate(
            a, xy=(p["mean_tokens_per_use"], p["occurrences"]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=8, color="#333",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_motif_quadrant(motifs: dict, out_path: Path) -> None:
    base_rate = motifs["__base_rate__"]["base_resolve_rate"]
    items = [(m, p) for m, p in motifs.items() if m != "__base_rate__" and p["n_trajectories_with"] >= 10 and p["n_atoms"] >= 2]
    xs = np.array([p["mean_tokens_per_use"] for _, p in items])
    ys = np.array([p["success_rate_when_used"] for _, p in items])
    sizes = np.array([np.sqrt(p["n_trajectories_with"]) * 8 for _, p in items])

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.scatter(xs, ys, s=sizes, color="#5d90e0", edgecolor="white", alpha=0.75)
    ax.axhline(base_rate, color="#bbb", lw=1, alpha=0.8, label=f"base resolve rate = {base_rate:.2f}")

    median_cost = float(np.median(xs))
    ax.axvline(median_cost, color="#bbb", lw=1, alpha=0.8, label=f"median motif cost = {median_cost:.0f} tokens")

    ax.set_xlabel("estimated mean tokens per use of this motif")
    ax.set_ylabel("fraction of trajectories using this motif that resolved the task")
    ax.set_title(
        "Motif cost vs success: which procedures actually pay off?\n"
        "Upper-left = cheap + high success (efficient). Lower-right = expensive + low success (wasteful).",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, frameon=False, loc="lower right")

    # annotate outliers: highest-success and highest-cost and wasteful
    def abbrev(m: str, maxl: int = 32) -> str:
        parts = m.split("+")
        if len(parts) <= 2:
            s = m.replace("+", "->")
        else:
            s = f"{parts[0]}->...->{parts[-1]} ({len(parts)})"
        return s if len(s) <= maxl else s[:maxl-1] + "..."

    efficient = sorted(items, key=lambda kv: (-kv[1]["success_rate_when_used"], kv[1]["mean_tokens_per_use"]))[:3]
    expensive = sorted(items, key=lambda kv: -kv[1]["mean_tokens_per_use"])[:3]
    wasteful = sorted(
        [(m, p) for m, p in items if p["mean_tokens_per_use"] >= median_cost],
        key=lambda kv: kv[1]["success_rate_when_used"],
    )[:3]
    seen = set()
    for (m, p) in efficient + expensive + wasteful:
        if m in seen:
            continue
        seen.add(m)
        ax.annotate(
            abbrev(m), xy=(p["mean_tokens_per_use"], p["success_rate_when_used"]),
            xytext=(5, 3), textcoords="offset points",
            fontsize=7, color="#333",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_wasteful(motifs: dict, out_path: Path, top_n: int = 12) -> None:
    items = [(m, p) for m, p in motifs.items() if m != "__base_rate__"
             and p["n_trajectories_with"] >= 20 and p["n_atoms"] >= 2]
    ranked = sorted(items, key=lambda kv: (kv[1]["success_rate_when_used"] - (kv[1]["mean_tokens_per_use"] / max(1, kv[1]["mean_tokens_per_use"] + 1)) * 0.0))
    # Compose a composite wasteful index: high cost, low success relative to base
    base_rate = motifs["__base_rate__"]["base_resolve_rate"]
    scored = []
    max_cost = max(p["mean_tokens_per_use"] for _, p in items) if items else 1
    for m, p in items:
        cost_norm = p["mean_tokens_per_use"] / max_cost
        success_deficit = base_rate - p["success_rate_when_used"]
        scored.append((m, p, cost_norm - success_deficit * 0))  # preserve flexible
        # Simpler: waste_score = cost_norm * (base_rate - success_rate) with floor
        waste_score = cost_norm * max(0, base_rate - p["success_rate_when_used"])
        scored[-1] = (m, p, waste_score)

    scored.sort(key=lambda t: -t[2])
    top = scored[:top_n]

    def abbrev(m: str, maxl: int = 36) -> str:
        parts = m.split("+")
        if len(parts) <= 2:
            s = m.replace("+", "->")
        else:
            s = f"{parts[0]}->...->{parts[-1]} ({len(parts)})"
        return s if len(s) <= maxl else s[:maxl-1] + "..."

    names = [abbrev(t[0]) for t in top]
    costs = [t[1]["mean_tokens_per_use"] / 1000 for t in top]
    success = [t[1]["success_rate_when_used"] for t in top]

    fig, ax1 = plt.subplots(figsize=(11, 5.2))
    y = np.arange(len(top))
    ax1.barh(y, costs, color="#c9b5dd", edgecolor="white", label="cost per use (k tokens)")
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("mean tokens per use (thousands)", color="#6a4a9f")
    ax1.tick_params(axis="x", labelcolor="#6a4a9f")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = ax1.twiny()
    ax2.plot(success, y, "o-", color="#333333", lw=2, markersize=6, label="resolve-rate when used")
    ax2.axvline(base_rate, color="#bbb", lw=1, alpha=0.8)
    ax2.set_xlim(0, max(0.8, max(success) + 0.05))
    ax2.set_xlabel(f"resolve rate when motif is used  (thin line = base rate {base_rate:.2f})", color="#333333")
    ax2.tick_params(axis="x", labelcolor="#333333")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Top {top_n} candidate 'wasteful' motifs\n"
        "High cost (bars) combined with low resolve rate (red line) = procedures that burn tokens without helping.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency_frontier(frontier: dict, out_path: Path) -> None:
    agents = sorted(frontier.keys())

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for a in agents:
        f = frontier[a]
        ax.scatter(
            f["mean_cost_per_task_usd"], f["resolve_rate"],
            s=220, color=AGENT_COLORS.get(a, "#888"),
            edgecolor="white", linewidth=1.2, zorder=3,
        )
        ax.annotate(
            f"{a}\n${f['mean_cost_per_resolved_usd']:.2f}/resolved",
            xy=(f["mean_cost_per_task_usd"], f["resolve_rate"]),
            xytext=(10, -4), textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("mean cost per task attempted (USD)")
    ax.set_ylabel("resolve rate (fraction of tasks solved)")
    ax.set_title(
        "Efficiency frontier: cost vs solve rate per agent\n"
        "Up-and-left is better. Annotation shows cost per resolved task.",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    print("Loading data...")
    seqs = load_sequences()
    stats = load_model_stats()
    resolved = load_resolved_set()
    print(f"  {len(seqs)} sequences, {len(stats)} model_stats, {len(resolved)} resolved (agent, instance) pairs")

    atoms = per_atom_profile(seqs, stats)
    motifs = per_motif_profile(seqs, stats, resolved)
    frontier = efficiency_frontier(seqs, stats, resolved)

    base_rate = motifs["__base_rate__"]["base_resolve_rate"]
    print(f"\nBase resolve rate: {base_rate:.3f}")

    print("\nTop 10 atoms by total cost attributed:")
    top_atoms = sorted(
        [(a, p) for a, p in atoms.items() if p["occurrences"] >= 5],
        key=lambda kv: -kv[1]["total_tokens_attributed"],
    )[:10]
    for a, p in top_atoms:
        print(f"  {a:<30s}  occ={p['occurrences']:>5d}  mean_tok/use={p['mean_tokens_per_use']:>8.0f}")

    print("\nMost 'efficient' motifs (high success, low cost):")
    motif_items = [(m, p) for m, p in motifs.items() if m != "__base_rate__"
                   and p["n_trajectories_with"] >= 20 and p["n_atoms"] >= 2]
    median_cost = float(np.median([p["mean_tokens_per_use"] for _, p in motif_items]))
    cheap_success = sorted(
        [(m, p) for m, p in motif_items if p["mean_tokens_per_use"] <= median_cost],
        key=lambda kv: -kv[1]["success_rate_when_used"],
    )[:5]
    for m, p in cheap_success:
        print(f"  {m[:60]:<60s}  resolve={p['success_rate_when_used']:.3f}  "
              f"cost={p['mean_tokens_per_use']:.0f}  n_traj={p['n_trajectories_with']}")

    print("\nMost 'wasteful' motifs (high cost, low success):")
    max_cost = max(p["mean_tokens_per_use"] for _, p in motif_items)
    scored = []
    for m, p in motif_items:
        cost_norm = p["mean_tokens_per_use"] / max_cost
        success_deficit = max(0, base_rate - p["success_rate_when_used"])
        scored.append((m, p, cost_norm * success_deficit))
    scored.sort(key=lambda t: -t[2])
    for m, p, score in scored[:5]:
        print(f"  {m[:60]:<60s}  resolve={p['success_rate_when_used']:.3f}  "
              f"cost={p['mean_tokens_per_use']:.0f}  waste_score={score:.3f}")

    print("\nEfficiency frontier:")
    for a, f in frontier.items():
        print(f"  {a:<12s}  resolve_rate={f['resolve_rate']:.2f}  "
              f"${f['mean_cost_per_task_usd']:.2f}/task  ${f['mean_cost_per_resolved_usd']:.2f}/resolved")

    (OUT / "step_resources.json").write_text(json.dumps({
        "atoms": atoms,
        "motifs": motifs,
        "efficiency_frontier": frontier,
    }, indent=2, default=str))
    plot_atoms(atoms, OUT / "step_resources_atoms.png")
    plot_motif_quadrant(motifs, OUT / "step_resources_motifs.png")
    plot_wasteful(motifs, OUT / "step_resources_wasteful.png")
    plot_efficiency_frontier(frontier, OUT / "step_resources_efficiency.png")

    print(f"\nSaved:")
    for n in ["step_resources.json", "step_resources_atoms.png",
              "step_resources_motifs.png", "step_resources_wasteful.png",
              "step_resources_efficiency.png"]:
        print(f"  {OUT / n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
