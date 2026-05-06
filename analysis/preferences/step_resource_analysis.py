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

import sys

import altair as alt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.theme import register, BLUE, ORANGE, GREEN, VERMILLION, SKY, GRAY, NEAR_BLACK
register()

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


def plot_atoms(atoms: dict, out_path: Path, top_n: int = 15) -> None:
    items = sorted(
        [(a, p) for a, p in atoms.items() if p["occurrences"] >= 5],
        key=lambda kv: -kv[1]["total_tokens_attributed"],
    )[:top_n]

    df = pd.DataFrame([
        {
            "atom_name": a,
            "total_tokens_m": p["total_tokens_attributed"] / 1e6,
            "occurrences": p["occurrences"],
        }
        for a, p in items
    ])

    sort_order = df["atom_name"].tolist()
    panel_h = top_n * 26

    axis_opts = alt.Axis(domain=False, ticks=False)

    panel1 = (
        alt.Chart(df)
        .mark_bar(color=BLUE)
        .encode(
            x=alt.X("total_tokens_m:Q", title="Total attributed tokens (millions)", axis=axis_opts),
            y=alt.Y("atom_name:N", sort=sort_order, title=None,
                    axis=alt.Axis(domain=False, ticks=False, labelFontSize=9)),
        )
        .properties(width=240, height=panel_h)
    )

    panel2 = (
        alt.Chart(df)
        .mark_bar(color=SKY)
        .encode(
            x=alt.X("occurrences:Q", title="Occurrences across corpus", axis=axis_opts),
            y=alt.Y("atom_name:N", sort=sort_order, title=None,
                    axis=alt.Axis(domain=False, ticks=False, labelFontSize=9)),
        )
        .properties(width=240, height=panel_h)
    )

    chart = (
        alt.hconcat(panel1, panel2)
        .resolve_scale(y="shared")
        .properties(
            title=alt.TitleParams(
                text=f"Top {top_n} action types by total estimated cost",
                fontSize=13, color="#111111", anchor="start",
            )
        )
        .configure_view(strokeWidth=0)
        .configure_axisX(grid=True, gridColor="#F0F0F0", gridWidth=0.3)
    )

    chart.save(str(out_path), scale_factor=2)


def plot_motif_quadrant(motifs: dict, out_path: Path) -> None:
    base_rate = motifs["__base_rate__"]["base_resolve_rate"]
    items = [
        (m, p) for m, p in motifs.items()
        if m != "__base_rate__" and p["n_trajectories_with"] >= 10 and p["n_atoms"] >= 2
    ]

    xs = np.array([p["mean_tokens_per_use"] for _, p in items])
    median_cost = float(np.median(xs))

    def abbrev(m: str, maxl: int = 32) -> str:
        parts = m.split("+")
        if len(parts) <= 2:
            s = m.replace("+", "->")
        else:
            s = f"{parts[0]}->...{parts[-1]} ({len(parts)})"
        return s if len(s) <= maxl else s[:maxl - 1] + "..."

    efficient = sorted(items, key=lambda kv: (-kv[1]["success_rate_when_used"], kv[1]["mean_tokens_per_use"]))[:3]
    expensive = sorted(items, key=lambda kv: -kv[1]["mean_tokens_per_use"])[:3]
    wasteful = sorted(
        [(m, p) for m, p in items if p["mean_tokens_per_use"] >= median_cost],
        key=lambda kv: kv[1]["success_rate_when_used"],
    )[:3]
    seen: set[str] = set()
    annotated: list[str] = []
    for (m, _) in efficient + expensive + wasteful:
        if m not in seen:
            seen.add(m)
            annotated.append(m)

    df = pd.DataFrame([
        {
            "motif": m,
            "mean_tokens_per_use": p["mean_tokens_per_use"],
            "success_rate_when_used": p["success_rate_when_used"],
            "n_traj": p["n_trajectories_with"],
            "label": abbrev(m),
            "annotate": m in seen,
        }
        for m, p in items
    ])

    axis_opts = alt.Axis(domain=False, ticks=False)

    scatter = (
        alt.Chart(df)
        .mark_point(filled=True, opacity=0.7, color=BLUE)
        .encode(
            x=alt.X("mean_tokens_per_use:Q",
                    title="Estimated mean tokens per use",
                    axis=axis_opts),
            y=alt.Y("success_rate_when_used:Q",
                    title="Fraction of trajectories that resolved the task",
                    axis=alt.Axis(domain=False, ticks=False, format=".0%")),
            size=alt.Size("n_traj:Q", scale=alt.Scale(range=[20, 200]), legend=None),
        )
    )

    hline = (
        alt.Chart(pd.DataFrame({"y": [base_rate]}))
        .mark_rule(color=GRAY, strokeDash=[4, 4])
        .encode(y=alt.Y("y:Q"))
    )

    vline = (
        alt.Chart(pd.DataFrame({"x": [median_cost]}))
        .mark_rule(color=GRAY, strokeDash=[4, 4])
        .encode(x=alt.X("x:Q"))
    )

    ann_df = df[df["annotate"]].copy()
    annotations = (
        alt.Chart(ann_df)
        .mark_text(xOffset=5, yOffset=-8, fontSize=8, color="#333333", align="left")
        .encode(
            x=alt.X("mean_tokens_per_use:Q"),
            y=alt.Y("success_rate_when_used:Q"),
            text=alt.Text("label:N"),
        )
    )

    chart = (
        alt.layer(hline, vline, scatter, annotations)
        .properties(
            width=500,
            height=300,
            title=alt.TitleParams(
                text="Motif cost vs. solve rate",
                fontSize=13, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axisX(grid=True, gridColor="#F0F0F0", gridWidth=0.3)
        .configure_axisY(grid=True, gridColor="#F0F0F0", gridWidth=0.3)
    )

    chart.save(str(out_path), scale_factor=2)


def plot_wasteful(motifs: dict, out_path: Path, top_n: int = 12) -> None:
    base_rate = motifs["__base_rate__"]["base_resolve_rate"]
    items = [
        (m, p) for m, p in motifs.items()
        if m != "__base_rate__" and p["n_trajectories_with"] >= 20 and p["n_atoms"] >= 2
    ]

    max_cost = max(p["mean_tokens_per_use"] for _, p in items) if items else 1
    scored = []
    for m, p in items:
        cost_norm = p["mean_tokens_per_use"] / max_cost
        waste_score = cost_norm * max(0, base_rate - p["success_rate_when_used"])
        scored.append((m, p, waste_score))
    scored.sort(key=lambda t: -t[2])
    top = scored[:top_n]

    def abbrev(m: str, maxl: int = 36) -> str:
        parts = m.split("+")
        if len(parts) <= 2:
            s = m.replace("+", "->")
        else:
            s = f"{parts[0]}->...{parts[-1]} ({len(parts)})"
        return s if len(s) <= maxl else s[:maxl - 1] + "..."

    df = pd.DataFrame([
        {
            "motif_name": abbrev(t[0]),
            "cost_k": t[1]["mean_tokens_per_use"] / 1000,
            "success_rate": t[1]["success_rate_when_used"],
        }
        for t in top
    ])

    sort_order = df["motif_name"].tolist()
    panel_h = top_n * 26
    axis_opts = alt.Axis(domain=False, ticks=False)

    panel1 = (
        alt.Chart(df)
        .mark_bar(color=SKY)
        .encode(
            x=alt.X("cost_k:Q", title="Mean tokens per use (thousands)", axis=axis_opts),
            y=alt.Y("motif_name:N", sort=sort_order, title=None,
                    axis=alt.Axis(domain=False, ticks=False, labelFontSize=9)),
        )
        .properties(width=240, height=panel_h)
    )

    base_df = pd.DataFrame({"x": [base_rate]})
    vline = (
        alt.Chart(base_df)
        .mark_rule(color=GRAY, strokeDash=[4, 4])
        .encode(x=alt.X("x:Q"))
    )

    bars2 = (
        alt.Chart(df)
        .mark_bar(color=ORANGE)
        .encode(
            x=alt.X("success_rate:Q", title="Resolve rate when used",
                    axis=alt.Axis(domain=False, ticks=False, format=".0%")),
            y=alt.Y("motif_name:N", sort=sort_order, title=None,
                    axis=alt.Axis(domain=False, ticks=False, labelFontSize=9)),
        )
    )
    panel2 = (
        alt.layer(bars2, vline)
        .properties(width=240, height=panel_h)
    )

    chart = (
        alt.hconcat(panel1, panel2)
        .resolve_scale(y="shared")
        .properties(
            title=alt.TitleParams(
                text=f"Top {top_n} candidate wasteful motifs",
                fontSize=13, color="#111111", anchor="start",
            )
        )
        .configure_view(strokeWidth=0)
        .configure_axisX(grid=True, gridColor="#F0F0F0", gridWidth=0.3)
    )

    chart.save(str(out_path), scale_factor=2)


def plot_efficiency_frontier(frontier: dict, out_path: Path) -> None:
    df = pd.DataFrame([
        {
            "agent": a,
            "mean_cost_per_task_usd": f["mean_cost_per_task_usd"],
            "resolve_rate": f["resolve_rate"],
            "label": f"${f['mean_cost_per_resolved_usd']:.2f}/resolved",
        }
        for a, f in frontier.items()
    ])

    color_domain = list(AGENT_COLORS.keys())
    color_range = [AGENT_COLORS[k] for k in color_domain]
    axis_opts = alt.Axis(domain=False, ticks=False)

    points = (
        alt.Chart(df)
        .mark_point(filled=True, size=200)
        .encode(
            x=alt.X("mean_cost_per_task_usd:Q",
                    title="Mean cost per task attempted (USD)",
                    axis=axis_opts),
            y=alt.Y("resolve_rate:Q",
                    title="Resolve rate",
                    axis=alt.Axis(domain=False, ticks=False, format=".0%")),
            color=alt.Color(
                "agent:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=None,
            ),
        )
    )

    labels = (
        alt.Chart(df)
        .mark_text(dy=-16, fontSize=9)
        .encode(
            x=alt.X("mean_cost_per_task_usd:Q"),
            y=alt.Y("resolve_rate:Q"),
            text=alt.Text("label:N"),
        )
    )

    chart = (
        alt.layer(points, labels)
        .properties(
            width=360,
            height=240,
            title=alt.TitleParams(
                text="Token efficiency by agent",
                fontSize=13, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axisX(grid=True, gridColor="#F0F0F0", gridWidth=0.3)
        .configure_axisY(grid=True, gridColor="#F0F0F0", gridWidth=0.3)
    )

    chart.save(str(out_path), scale_factor=2)


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

    print(f"\nSaved:")
    for n in ["step_resources.json", "step_resources_atoms.png",
              "step_resources_motifs.png", "step_resources_wasteful.png"]:
        print(f"  {OUT / n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
