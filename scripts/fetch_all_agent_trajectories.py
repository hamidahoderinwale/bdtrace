#!/usr/bin/env python3
"""
Fetch full trajectories (all 300 instances) for all 4 SWE-agent models,
tagging each with outcome (resolved / failed). Produces:

  1. cross_agent_patches_all.jsonl  — all patches incl. failed (tagged)
  2. cross_agent_distances_all.parquet — structural distances with outcome cols
  3. localization_metrics.parquet — per-(instance, model) localization metrics
  4. localization_metrics.png — histograms of localization metrics by outcome
  5. history_flow.png — History Flow for top-K divergent instances

The History Flow shows, for each selected instance, the sequence of files
each agent touched across trajectory steps — revealing where procedures diverge
between a model that resolved the task and one that did not.

Usage:
  uv run python scripts/fetch_all_agent_trajectories.py
  uv run python scripts/fetch_all_agent_trajectories.py --limit 30
  uv run python scripts/fetch_all_agent_trajectories.py --history-only
"""

import argparse
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence, patch_to_chunks

S3_BASE = "https://swe-bench-submissions.s3.amazonaws.com"

MODELS = {
    "20240402_sweagent_gpt4":            "GPT-4",
    "20240620_sweagent_claude3.5sonnet": "Claude 3.5",
    "20240728_sweagent_gpt4o":           "GPT-4o",
    "20240402_sweagent_claude3opus":     "Claude 3 Opus",
}

# Wong colorblind-safe
MODEL_COLORS = {
    "GPT-4":        "#0072B2",
    "Claude 3.5":   "#E69F00",
    "GPT-4o":       "#009E73",
    "Claude 3 Opus":"#CC79A7",
}

DS_OUT    = ROOT / "output" / "datasets" / "cross_agent_all"
PLOTS_OUT = ROOT / "notebooks" / "plots" / "cross_agent_all"
DS_OUT.mkdir(parents=True, exist_ok=True)
PLOTS_OUT.mkdir(parents=True, exist_ok=True)

RESOLVED_MARKER = "submitted"   # exit_status prefix for resolved


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_trajectory(instance_id: str, model_id: str, timeout: int = 20) -> dict | None:
    """Fetch full .traj and return dict with patch, outcome, and step sequence."""
    url = f"{S3_BASE}/lite/{model_id}/trajs/{instance_id}.traj"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read())
    except Exception:
        return None

    info       = data.get("info", {})
    exit_status = info.get("exit_status", "")
    patch      = info.get("submission") or ""
    traj_steps = data.get("trajectory", [])

    resolved = exit_status.startswith(RESOLVED_MARKER)

    return {
        "patch":       patch,
        "resolved":    resolved,
        "exit_status": exit_status,
        "steps":       _extract_file_steps(traj_steps),
        "n_steps":     len(traj_steps),
    }


def _extract_file_steps(traj_steps: list[dict]) -> list[dict]:
    """
    Walk trajectory steps and extract file-touch events.
    Returns list of {step, file, action_type} dicts.

    State machine: 'open <file>' sets current_file; subsequent 'edit N'
    and 'create <file>' are attributed to that file.
    """
    events = []
    current_file = None
    open_pat   = re.compile(r'^open\s+([\w./\-]+\.\w+)', re.IGNORECASE)
    create_pat = re.compile(r'^create\s+([\w./\-]+\.\w+)', re.IGNORECASE)
    edit_pat   = re.compile(r'^edit\s+\d')
    view_pat   = re.compile(r'^(?:view|cat|less)\s+([\w./\-]+\.\w+)', re.IGNORECASE)

    for i, step in enumerate(traj_steps):
        action = step.get("action", "").strip()

        m_open = open_pat.match(action)
        m_create = create_pat.match(action)
        m_view = view_pat.match(action)

        if m_create:
            current_file = m_create.group(1)
            events.append({"step": i, "file": current_file, "action_type": "create"})
        elif m_open:
            current_file = m_open.group(1)
            events.append({"step": i, "file": current_file, "action_type": "open"})
        elif m_view:
            f = m_view.group(1)
            events.append({"step": i, "file": f, "action_type": "view"})
        elif edit_pat.match(action) and current_file:
            events.append({"step": i, "file": current_file, "action_type": "edit"})

    return events


def fetch_all(instance_ids: list[str], workers: int = 16, limit: int | None = None) -> dict:
    """
    Returns {instance_id: {model_label: traj_dict}}.
    Keeps ALL instances, not just those with >=2 patches.
    """
    if limit:
        instance_ids = instance_ids[:limit]

    tasks = [
        (iid, mid, label)
        for iid in instance_ids
        for mid, label in MODELS.items()
    ]
    results: dict = {iid: {} for iid in instance_ids}

    def _fetch(iid, mid, label):
        return iid, label, fetch_trajectory(iid, mid)

    print(f"Fetching {len(tasks)} trajectories ({len(instance_ids)} instances × {len(MODELS)} models)...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch, iid, mid, lbl): (iid, lbl) for iid, mid, lbl in tasks}
        done = 0
        for fut in as_completed(futures):
            iid, label, traj = fut.result()
            if traj is not None:
                results[iid][label] = traj
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(tasks)}")

    present = sum(1 for v in results.values() if len(v) >= 1)
    print(f"Instances with ≥1 trajectory: {present}/{len(instance_ids)}")
    return results


# ---------------------------------------------------------------------------
# Structural features (reused from fetch_cross_agent_patches.py)
# ---------------------------------------------------------------------------

def _op_types(patch: str) -> set[str]:
    chunks = patch_to_chunks(patch)
    types: set[str] = set()
    for chunk in chunks:
        for tok in chunk.sequence:
            parts = tok.split("_", 1)
            if len(parts) == 2:
                types.add(parts[1])
    return types


def _modules(patch: str) -> set[str]:
    stems: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("diff --git"):
            m = re.search(r'b/([\w/._-]+\.py)$', line)
            if m:
                stems.add(Path(m.group(1)).stem)
    return stems


def _token_seq(patch: str) -> list[str]:
    return patch_to_ast_sequence(patch)


def _levenshtein(a: list[str], b: list[str]) -> float:
    sa, sb = " ".join(a), " ".join(b)
    if not sa and not sb:
        return 0.0
    try:
        import Levenshtein
        d = Levenshtein.distance(sa, sb)
        return min(d / max(len(sa), len(sb), 1), 1.0)
    except ImportError:
        sa_set, sb_set = set(a), set(b)
        if not sa_set and not sb_set:
            return 0.0
        inter = len(sa_set & sb_set)
        union = len(sa_set | sb_set)
        return 0.0 if union == 0 else 1.0 - inter / union


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 0.0 if not (a | b) else 1.0 - len(a & b) / len(a | b)


def _sym_diff(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    total = len(a) + len(b)
    return len(a ^ b) / total if total else 0.0


# ---------------------------------------------------------------------------
# Build distance table (all instances, with outcome metadata)
# ---------------------------------------------------------------------------

def compute_distances(all_trajs: dict) -> pd.DataFrame:
    rows = []
    for iid, model_trajs in all_trajs.items():
        labels = [l for l, t in model_trajs.items() if t["patch"]]
        if len(labels) < 2:
            continue

        feats = {}
        for lbl in labels:
            patch = model_trajs[lbl]["patch"]
            feats[lbl] = {
                "tokens":  _token_seq(patch),
                "op_types": _op_types(patch),
                "modules": _modules(patch),
            }

        for i, la in enumerate(labels):
            for lb in labels[i + 1:]:
                fa, fb = feats[la], feats[lb]
                ra = model_trajs[la]["resolved"]
                rb = model_trajs[lb]["resolved"]
                outcome = (
                    "both_resolved"   if ra and rb else
                    "both_failed"     if not ra and not rb else
                    "one_resolved"
                )
                rows.append({
                    "instance_id": iid,
                    "agent_a":     la,
                    "agent_b":     lb,
                    "outcome":     outcome,
                    "a_resolved":  ra,
                    "b_resolved":  rb,
                    "d_tokens":    _levenshtein(fa["tokens"], fb["tokens"]),
                    "d_edits":     _sym_diff(fa["op_types"], fb["op_types"]),
                    "d_modules":   _jaccard(fa["modules"], fb["modules"]),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Localization metrics
# ---------------------------------------------------------------------------

def _gini(counts: list[int]) -> float:
    """Gini coefficient over a list of counts (0 = uniform, 1 = all on one item)."""
    n = len(counts)
    if n == 0:
        return float("nan")
    if n == 1:
        return 1.0
    counts = sorted(counts)
    total = sum(counts)
    if total == 0:
        return float("nan")
    gini_sum = sum((2 * (i + 1) - n - 1) * c for i, c in enumerate(counts))
    return gini_sum / (n * total)


def _patch_files(patch: str) -> set[str]:
    """Extract file paths touched by a patch (from diff --git headers)."""
    files: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("diff --git"):
            m = re.search(r' b/(.+)$', line)
            if m:
                files.add(m.group(1).strip())
    return files


def compute_localization_metrics(all_trajs: dict) -> pd.DataFrame:
    """
    Per-(instance, model) localization metrics derived from step sequences.

    Columns:
      files_edited        — unique files with edit/create actions
      step_of_first_edit  — normalised step of first edit (nan if no edits)
      file_concentration  — Gini over per-file edit counts (1 = single file)
      early_patch_focus   — fraction of edits in first 25% of steps that hit
                            a patch-relevant file (nan if no patch files / no
                            early edits)
      resolved            — outcome

    patch_files are pooled from all agents on the same instance so every agent
    is evaluated against the same ground-truth file set.
    """
    from collections import Counter
    rows = []
    for iid, model_trajs in all_trajs.items():
        # Ground-truth files: union of files touched by any agent's patch
        gt_files: set[str] = set()
        for traj in model_trajs.values():
            if traj.get("patch"):
                gt_files |= _patch_files(traj["patch"])
        gt_basenames = {Path(f).name for f in gt_files}

        for label, traj in model_trajs.items():
            steps    = traj["steps"]
            n_steps  = traj["n_steps"] or 1
            resolved = traj["resolved"]

            if not steps:
                rows.append(dict(
                    instance_id=iid, model=label, resolved=resolved,
                    files_edited=0, step_of_first_edit=float("nan"),
                    file_concentration=float("nan"), early_patch_focus=float("nan"),
                ))
                continue

            edit_steps = [ev for ev in steps if ev["action_type"] in ("edit", "create")]

            files_edited = len({ev["file"] for ev in edit_steps})

            step_of_first_edit = (
                edit_steps[0]["step"] / n_steps if edit_steps else float("nan")
            )

            file_concentration = (
                _gini(list(Counter(ev["file"] for ev in edit_steps).values()))
                if edit_steps else float("nan")
            )

            if gt_files and edit_steps:
                early_cutoff = 0.25
                early = [ev for ev in edit_steps if ev["step"] / n_steps <= early_cutoff]
                if early:
                    early_patch_focus = sum(
                        1 for ev in early
                        if ev["file"] in gt_files or Path(ev["file"]).name in gt_basenames
                    ) / len(early)
                else:
                    early_patch_focus = float("nan")
            else:
                early_patch_focus = float("nan")

            rows.append(dict(
                instance_id=iid, model=label, resolved=resolved,
                files_edited=files_edited,
                step_of_first_edit=step_of_first_edit,
                file_concentration=file_concentration,
                early_patch_focus=early_patch_focus,
            ))

    return pd.DataFrame(rows)


def plot_localization(df_loc: pd.DataFrame, out_path: Path) -> None:
    """
    Four panels, one per localization metric, split by resolved/failed.
    Overlapping histograms + mean vlines, same style as plot_outcome_comparison.
    """
    metrics = [
        ("files_edited",       "Files edited (unique)"),
        ("step_of_first_edit", "Step of first edit (normalised)"),
        ("file_concentration", "File concentration (Gini)"),
        ("early_patch_focus",  "Early patch focus (first 25% of steps)"),
    ]
    colors = {True: "#0072B2", False: "#D55E00"}

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    fig.suptitle(
        "Localization metrics: resolved vs. failed trajectories",
        fontsize=10, y=1.03,
    )

    for ax, (col, label) in zip(axes, metrics):
        for resolved, color in [(True, colors[True]), (False, colors[False])]:
            vals = df_loc[df_loc["resolved"] == resolved][col].dropna()
            if len(vals) == 0:
                continue
            outcome_label = f"{'resolved' if resolved else 'failed'} (n={len(vals)})"
            ax.hist(vals, bins=20, alpha=0.55, color=color,
                    label=outcome_label, edgecolor="none", density=True)
            ax.axvline(vals.mean(), color=color, linewidth=1.2, linestyle="--")

        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("density" if ax is axes[0] else "")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# History Flow
# ---------------------------------------------------------------------------

ACTION_MARKERS = {"create": "s", "open": "^", "edit": "o", "view": "."}
ACTION_ALPHA   = {"create": 0.9, "open": 0.7, "edit": 1.0, "view": 0.3}
ACTION_SIZE    = {"create": 60,  "open": 50,  "edit": 40,  "view": 20}


def _select_history_flow_instances(
    all_trajs: dict,
    df_dist: pd.DataFrame,
    n: int = 6,
) -> list[str]:
    """
    Pick instances where:
    - At least one model resolved and at least one failed
    - At least 3 models have step data (can include failed with no patch)
    - Sorted by step-count variance (most behaviorally divergent first)
    """
    # Find mixed-outcome instances directly from trajectory data
    mixed = []
    for iid, mt in all_trajs.items():
        outcomes = [t["resolved"] for t in mt.values()]
        if not (any(outcomes) and not all(outcomes)):
            continue
        # Need at least 3 models with actual steps
        models_with_steps = [l for l, t in mt.items() if len(t["steps"]) > 0]
        if len(models_with_steps) < 3:
            continue
        # Sort by variance in step counts (most divergent behaviour)
        step_counts = [t["n_steps"] for t in mt.values()]
        variance = np.var(step_counts)
        mixed.append((iid, variance))

    mixed.sort(key=lambda x: -x[1])
    return [iid for iid, _ in mixed[:n]]


def plot_history_flow(
    all_trajs: dict,
    instance_ids: list[str],
    out_path: Path,
) -> None:
    """
    History Flow figure: one row per instance, showing step-by-step
    file-touch sequences for each agent. Inspired by Wattenberg & Viégas
    History Flow (CHI 2004).

    Layout: n_instances rows × 4 model columns.
    Each cell: x = normalized step, y = file index (shared within instance),
    color = model, marker = action type.
    A resolved agent's trajectory has a solid border; failed has dashed.
    """
    n = len(instance_ids)
    if n == 0:
        print("No instances selected for History Flow.")
        return

    fig, axes = plt.subplots(
        n, 4,
        figsize=(14, 2.8 * n),
        sharex=False, sharey="row",
    )
    if n == 1:
        axes = axes[None, :]

    for row_i, iid in enumerate(instance_ids):
        mt = all_trajs[iid]

        # Collect all files touched across all agents for this instance
        all_files: list[str] = []
        for traj in mt.values():
            for ev in traj["steps"]:
                if ev["file"] not in all_files:
                    all_files.append(ev["file"])

        # Shorten file paths for display
        def _shorten(f: str) -> str:
            parts = Path(f).parts
            return "/".join(parts[-2:]) if len(parts) > 2 else f

        file_labels = [_shorten(f) for f in all_files]
        file_idx    = {f: i for i, f in enumerate(all_files)}
        n_files     = len(all_files)

        for col_i, (label, color) in enumerate(MODEL_COLORS.items()):
            ax = axes[row_i, col_i]
            traj = mt.get(label)

            resolved = traj["resolved"] if traj else False
            spine_ls = "-" if resolved else "--"
            for spine in ax.spines.values():
                spine.set_linestyle(spine_ls)
                spine.set_linewidth(1.2)
                spine.set_edgecolor(color)

            if not traj or not traj["steps"]:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=8)
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.5, max(n_files - 0.5, 0.5))
                continue

            steps  = traj["steps"]
            n_s    = traj["n_steps"] or 1
            xs     = [ev["step"] / n_s for ev in steps]
            ys     = [file_idx.get(ev["file"], 0) for ev in steps]
            atypes = [ev["action_type"] for ev in steps]

            # Draw trajectory line (light)
            if xs:
                ax.plot(xs, ys, color=color, alpha=0.15, linewidth=0.8, zorder=1)

            for x, y, at in zip(xs, ys, atypes):
                ax.scatter(x, y,
                           marker=ACTION_MARKERS.get(at, "o"),
                           s=ACTION_SIZE.get(at, 30),
                           color=color,
                           alpha=ACTION_ALPHA.get(at, 0.7),
                           linewidths=0,
                           zorder=3)

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.5, n_files - 0.5)
            ax.set_yticks(range(n_files))
            ax.set_yticklabels(
                file_labels if col_i == 0 else [],
                fontsize=6.5,
            )
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_xticklabels(["0", "½", "1"], fontsize=7)

            status = "✓" if resolved else "✗"
            ax.set_title(
                f"{label} {status}",
                fontsize=8, color=color,
                fontweight="bold" if resolved else "normal",
            )

        # Row label: short instance id
        axes[row_i, 0].set_ylabel(
            iid.replace("__", "\n"), fontsize=7, rotation=0,
            ha="right", va="center", labelpad=55,
        )

    # Legend for action types
    legend_handles = [
        mpatches.Patch(facecolor="none", edgecolor="black",
                       linestyle="-",  label="resolved (solid border)"),
        mpatches.Patch(facecolor="none", edgecolor="black",
                       linestyle="--", label="failed (dashed border)"),
    ]
    for at, marker in ACTION_MARKERS.items():
        if at == "view":
            continue
        legend_handles.append(
            plt.scatter([], [], marker=marker, s=40, color="gray",
                        label=f"action: {at}")
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=5,
        fontsize=8, frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        "History Flow: file-touch sequences across agents on the same task\n"
        "x = normalised step position, y = file, colour = model",
        fontsize=11, y=1.01,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Outcome comparison plot
# ---------------------------------------------------------------------------

def plot_outcome_comparison(df: pd.DataFrame, out_path: Path) -> None:
    """
    Three panels (tokens / edits / modules), each showing distance distributions
    split by outcome: both_resolved, one_resolved, both_failed.
    Shows that failed trajectories have higher structural variance.
    """
    outcomes = ["both_resolved", "one_resolved", "both_failed"]
    labels   = ["Both resolved", "One resolved", "Both failed"]
    colors   = ["#0072B2", "#E69F00", "#D55E00"]
    stages   = [
        ("d_tokens",  "Tokens (Levenshtein)"),
        ("d_edits",   "Edits (AST sym diff)"),
        ("d_modules", "Modules (Jaccard)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    fig.suptitle(
        "Structural distance by outcome: resolved vs. failed trajectories",
        fontsize=10, y=1.03,
    )

    for ax, (col, stage_label) in zip(axes, stages):
        for outcome, label, color in zip(outcomes, labels, colors):
            vals = df[df["outcome"] == outcome][col].dropna()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=25, alpha=0.55, color=color,
                    label=f"{label} (n={len(vals)})", edgecolor="none",
                    density=True)
            ax.axvline(vals.mean(), color=color, linewidth=1.2, linestyle="--")

        ax.set_xlabel(stage_label, fontsize=9)
        ax.set_ylabel("density" if ax is axes[0] else "")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",        type=int, default=None)
    parser.add_argument("--workers",      type=int, default=16)
    parser.add_argument("--no-cache",     action="store_true")
    parser.add_argument("--history-only", action="store_true",
                        help="Skip fetch, just regenerate History Flow from cache")
    parser.add_argument("--history-n",    type=int, default=6,
                        help="Number of instances in History Flow")
    args = parser.parse_args()

    trajs_path = DS_OUT / "all_trajectories.jsonl"
    dist_path  = DS_OUT / "cross_agent_distances_all.parquet"

    # --- Load or fetch ---
    if trajs_path.exists() and not args.no_cache and not args.history_only:
        print(f"Loading cached trajectories from {trajs_path.name}...")
        all_trajs: dict = {}
        with open(trajs_path) as f:
            for line in f:
                rec = json.loads(line)
                all_trajs[rec["instance_id"]] = rec["models"]
        print(f"  {len(all_trajs)} instances loaded")
    elif not args.history_only:
        from datasets import load_dataset
        ds = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
        instance_ids = [str(r["instance_id"]) for r in ds]

        all_trajs = fetch_all(instance_ids, workers=args.workers, limit=args.limit)

        print(f"Caching to {trajs_path.name}...")
        with open(trajs_path, "w") as f:
            for iid, models in all_trajs.items():
                # Steps are serializable as-is; keep only what we need
                f.write(json.dumps({"instance_id": iid, "models": models}) + "\n")
    else:
        print(f"--history-only: loading from {trajs_path.name}...")
        all_trajs = {}
        with open(trajs_path) as f:
            for line in f:
                rec = json.loads(line)
                all_trajs[rec["instance_id"]] = rec["models"]

    # --- Distances ---
    if dist_path.exists() and not args.no_cache and not args.history_only:
        print(f"Loading cached distances...")
        df = pd.read_parquet(dist_path)
    else:
        print("Computing structural distances...")
        df = compute_distances(all_trajs)
        df.to_parquet(dist_path, index=False)
        print(f"Saved {len(df)} rows → {dist_path.name}")

    # Summary
    print("\nOutcome breakdown:")
    print(df.groupby("outcome")[["d_tokens", "d_edits", "d_modules"]].mean().round(4))

    # --- Outcome comparison plot ---
    plot_outcome_comparison(df, PLOTS_OUT / "outcome_distance_comparison.png")

    # --- Localization metrics ---
    df_loc = compute_localization_metrics(all_trajs)
    df_loc.to_parquet(DS_OUT / "localization_metrics.parquet", index=False)
    print("\nLocalization metrics (mean by outcome):")
    print(
        df_loc.groupby("resolved")[
            ["files_edited", "step_of_first_edit", "file_concentration", "early_patch_focus"]
        ].mean().round(3).to_string()
    )
    plot_localization(df_loc, PLOTS_OUT / "localization_metrics.png")

    # --- History Flow ---
    selected = _select_history_flow_instances(all_trajs, df, n=args.history_n)
    print(f"\nSelected {len(selected)} instances for History Flow:")
    for iid in selected:
        mt = all_trajs[iid]
        resolved = {l: mt[l]["resolved"] for l in mt}
        print(f"  {iid}: {resolved}")

    plot_history_flow(all_trajs, selected, PLOTS_OUT / "history_flow.png")


if __name__ == "__main__":
    main()
