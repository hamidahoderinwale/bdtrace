#!/usr/bin/env python3
"""
Fetch per-agent patches from S3 for all 3 SWE-agent models on SWE-bench Lite,
then compute within-instance cross-agent structural distances.

For each instance where ≥2 agents submitted a patch, computes:
  - token distance: Levenshtein on AST token sequences (from patch hunks)
  - edits distance: symmetric set diff on AST op types
  - modules distance: Jaccard on touched file stems

Output:
  output/datasets/swe_bench_lite_resolved/cross_agent_patches.jsonl
  output/datasets/swe_bench_lite_resolved/cross_agent_distances.parquet
  notebooks/plots/swe_bench_lite_resolved/cross_agent_distances.png

Usage:
  uv run python scripts/fetch_cross_agent_patches.py
  uv run python scripts/fetch_cross_agent_patches.py --limit 20
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
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence, patch_to_chunks

S3_BASE = "https://swe-bench-submissions.s3.amazonaws.com"

MODELS = {
    "20240402_sweagent_gpt4":              "GPT-4",
    "20240620_sweagent_claude3.5sonnet":   "Claude 3.5",
    "20240728_sweagent_gpt4o":             "GPT-4o",
    "20240402_sweagent_claude3opus":       "Claude 3 Opus",
}

DS_OUT = ROOT / "output" / "datasets" / "swe_bench_lite_resolved"
PLOTS_OUT = ROOT / "notebooks" / "plots" / "swe_bench_lite_resolved"
DS_OUT.mkdir(parents=True, exist_ok=True)
PLOTS_OUT.mkdir(parents=True, exist_ok=True)

# Wong colorblind-safe palette
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
GRAY   = "#999999"

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_patch(instance_id: str, model_id: str, timeout: int = 15) -> str | None:
    url = f"{S3_BASE}/lite/{model_id}/trajs/{instance_id}.traj"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read())
        return data.get("info", {}).get("submission") or None
    except Exception:
        return None


def fetch_all_patches(
    instance_ids: list[str],
    workers: int = 12,
    limit: int | None = None,
) -> dict[str, dict[str, str]]:
    """
    Returns {instance_id: {model_label: patch_str}} for all available patches.
    """
    if limit:
        instance_ids = instance_ids[:limit]

    tasks = [
        (iid, mid, label)
        for iid in instance_ids
        for mid, label in MODELS.items()
    ]
    results: dict[str, dict[str, str]] = {iid: {} for iid in instance_ids}

    def _fetch(iid, mid, label):
        patch = fetch_patch(iid, mid)
        return iid, label, patch

    print(f"Fetching {len(tasks)} patches ({len(instance_ids)} instances x {len(MODELS)} models)...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch, iid, mid, lbl): (iid, lbl) for iid, mid, lbl in tasks}
        done = 0
        for fut in as_completed(futures):
            iid, label, patch = fut.result()
            if patch:
                results[iid][label] = patch
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(tasks)}")

    coverage = {iid: patches for iid, patches in results.items() if len(patches) >= 2}
    print(f"Instances with >=2 agent patches: {len(coverage)}/{len(instance_ids)}")
    return coverage


# ---------------------------------------------------------------------------
# Feature extraction from patch
# ---------------------------------------------------------------------------

def _op_types_from_patch(patch: str) -> set[str]:
    """Extract AST operation types from a unified diff."""
    chunks = patch_to_chunks(patch)
    types: set[str] = set()
    for chunk in chunks:
        for tok in chunk.sequence:
            # e.g. DEL_If, ADD_Call -> extract node type
            parts = tok.split("_", 1)
            if len(parts) == 2:
                types.add(parts[1])
    return types


def _module_stems_from_patch(patch: str) -> set[str]:
    """Extract touched .py file stems from diff --git headers."""
    stems: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("diff --git"):
            m = re.search(r'b/([\w/._-]+\.py)$', line)
            if m:
                stems.add(Path(m.group(1)).stem)
    return stems


def _token_seq_from_patch(patch: str) -> list[str]:
    """Flat AST token sequence for Levenshtein."""
    return patch_to_ast_sequence(patch)


def _levenshtein(a: list[str], b: list[str]) -> float:
    """Normalized Levenshtein on token sequences."""
    sa = " ".join(a)
    sb = " ".join(b)
    if not sa and not sb:
        return 0.0
    try:
        import Levenshtein
        d = Levenshtein.distance(sa, sb)
        return min(d / max(len(sa), len(sb), 1), 1.0)
    except ImportError:
        # fallback: Jaccard on token sets
        sa_set, sb_set = set(a), set(b)
        if not sa_set and not sb_set:
            return 0.0
        inter = len(sa_set & sb_set)
        union = len(sa_set | sb_set)
        return 0.0 if union == 0 else 1.0 - inter / union


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 0.0 if union == 0 else 1.0 - inter / union


def _sym_diff(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    diff = len(a ^ b)
    total = len(a) + len(b)
    return diff / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Build distance table
# ---------------------------------------------------------------------------

def compute_cross_agent_distances(
    coverage: dict[str, dict[str, str]],
) -> pd.DataFrame:
    rows = []
    for iid, patches in coverage.items():
        labels = list(patches.keys())
        feats = {}
        for lbl, patch in patches.items():
            feats[lbl] = {
                "tokens":  _token_seq_from_patch(patch),
                "op_types": _op_types_from_patch(patch),
                "modules": _module_stems_from_patch(patch),
            }

        for i, la in enumerate(labels):
            for lb in labels[i + 1:]:
                fa, fb = feats[la], feats[lb]
                rows.append({
                    "instance_id":   iid,
                    "agent_a":       la,
                    "agent_b":       lb,
                    "d_tokens":      _levenshtein(fa["tokens"], fb["tokens"]),
                    "d_edits":       _sym_diff(fa["op_types"], fb["op_types"]),
                    "d_modules":     _jaccard(fa["modules"], fb["modules"]),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_cross_agent_distances(df: pd.DataFrame) -> None:
    pairs = df[["agent_a", "agent_b"]].drop_duplicates()
    pair_labels = [f"{r.agent_a} / {r.agent_b}" for _, r in pairs.iterrows()]

    stages = [
        ("d_tokens",  "Tokens\n(Levenshtein)",        BLUE),
        ("d_edits",   "Edits\n(AST sym diff)",         ORANGE),
        ("d_modules", "Modules\n(Jaccard on files)",   GREEN),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    fig.suptitle(
        "Cross-agent structural distance — same instance, different agents\n"
        "(SWE-bench Lite, 3 SWE-agent models: GPT-4, Claude 3.5, GPT-4o)",
        fontsize=10, y=1.03,
    )

    for ax, (col, label, color) in zip(axes, stages):
        data_by_pair = [
            df[df["agent_a"] == r.agent_a][col].dropna().values if True else []
            for _, r in pairs.iterrows()
        ]
        # Use all pairs together for overall distribution; break out by pair as violin
        all_vals = df[col].dropna()
        ax.hist(all_vals, bins=30, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        mean = all_vals.mean()
        ax.axvline(mean, color="black", linewidth=1.2, linestyle="--")
        ax.set_xlabel(label, fontsize=8.5)
        ax.set_ylabel("instance pairs" if ax is axes[0] else "")
        ax.text(
            0.97, 0.95,
            f"mean = {mean:.3f}\nstd = {all_vals.std():.3f}\nn = {len(all_vals)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRAY, linewidth=0.6),
        )

    fig.tight_layout()
    out_path = PLOTS_OUT / "cross_agent_distances.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path.relative_to(ROOT)}")

    # Second plot: per-agent-pair breakdown
    fig2, axes2 = plt.subplots(1, 3, figsize=(11, 3.8))
    fig2.suptitle(
        "Cross-agent distance by model pair — per structural stage",
        fontsize=10, y=1.03,
    )
    pair_colors = {p: c for p, c in zip(sorted(df[["agent_a", "agent_b"]]
                   .apply(lambda r: f"{r.agent_a} / {r.agent_b}", axis=1).unique()), [BLUE, ORANGE, GREEN])}

    for ax, (col, label, _) in zip(axes2, stages):
        df["pair"] = df["agent_a"] + " / " + df["agent_b"]
        pair_order = sorted(df["pair"].unique())
        positions = range(len(pair_order))
        bplot = ax.boxplot(
            [df[df["pair"] == p][col].dropna().values for p in pair_order],
            positions=list(positions),
            widths=0.5,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, pair in zip(bplot["boxes"], pair_order):
            patch.set_facecolor(pair_colors.get(pair, GRAY))
            patch.set_alpha(0.8)
        SHORT = {"Claude 3.5": "C3.5", "GPT-4": "G4", "GPT-4o": "G4o", "Claude 3 Opus": "C3O"}
        short_labels = [
            " / ".join(SHORT.get(p, p) for p in pair.split(" / "))
            for pair in pair_order
        ]
        ax.set_xticks(list(positions))
        ax.set_xticklabels(short_labels, fontsize=8, rotation=45, ha="right")
        ax.set_ylabel(label if ax is axes2[0] else "")
        ax.set_title(label, fontsize=8.5)

    fig2.tight_layout()
    out_path2 = PLOTS_OUT / "cross_agent_distances_by_pair.png"
    fig2.savefig(out_path2)
    plt.close(fig2)
    print(f"Saved {out_path2.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit instances (for testing)")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--no-cache", action="store_true", help="Re-fetch even if cache exists")
    args = parser.parse_args()

    patches_path = DS_OUT / "cross_agent_patches.jsonl"
    dist_path    = DS_OUT / "cross_agent_distances.parquet"

    # Load or fetch patches
    if patches_path.exists() and not args.no_cache and not args.limit:
        print(f"Loading cached patches from {patches_path.name}...")
        coverage: dict[str, dict[str, str]] = {}
        with open(patches_path) as f:
            for line in f:
                rec = json.loads(line)
                coverage[rec["instance_id"]] = rec["patches"]
        print(f"  {len(coverage)} instances loaded")
    else:
        from datasets import load_dataset
        ds = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
        instance_ids = [str(r["instance_id"]) for r in ds]

        coverage = fetch_all_patches(instance_ids, workers=args.workers, limit=args.limit)

        # Cache patches
        with open(patches_path, "w") as f:
            for iid, patches in coverage.items():
                f.write(json.dumps({"instance_id": iid, "patches": patches}) + "\n")
        print(f"Cached {len(coverage)} instances to {patches_path.name}")

    # Compute distances
    print("Computing cross-agent distances...")
    df = compute_cross_agent_distances(coverage)
    df.to_parquet(dist_path, index=False)
    print(f"Saved {len(df)} rows to {dist_path.name}")

    # Summary
    print("\nSummary by stage:")
    for col in ["d_tokens", "d_edits", "d_modules"]:
        print(f"  {col}: mean={df[col].mean():.4f}  std={df[col].std():.4f}  "
              f"min={df[col].min():.4f}  max={df[col].max():.4f}")

    print("\nBy agent pair:")
    print(df.groupby(["agent_a", "agent_b"])[["d_tokens", "d_edits", "d_modules"]].mean().round(4))

    # Plot
    plot_cross_agent_distances(df)


if __name__ == "__main__":
    main()
