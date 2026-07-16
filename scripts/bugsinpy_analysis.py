#!/usr/bin/env python3
"""
BugsInPy generalization analysis.

Downloads patches from BugsInPy (Widyasari et al. 2020), computes edit
certificates, runs FIM, and compares pattern distributions to SWE-bench Lite.

Outputs:
  bugsinpy_patches.json         -- raw patches keyed by bug ID
  bugsinpy_certs.json           -- per-bug edit certificates
  bugsinpy_fim_patterns.json    -- FIM closed frequent itemsets
  bugsinpy_comparison.json      -- comparison stats vs SWE-bench Lite
  fig_pattern_comparison.png    -- distribution comparison figure (Altair)

Usage:
  uv run python scripts/bugsinpy_analysis.py
"""

import base64
import json
import os
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence
from scripts.build_canonical_forms import _NORMALIZE_OPS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "bugsinpy"
OUT.mkdir(parents=True, exist_ok=True)

GITHUB_API = "https://api.github.com/repos/soarsmu/BugsInPy/contents"

# Wong colorblind-safe palette
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
PINK = "#CC79A7"
GRAY = "#999999"

PROJECTS = [
    "PySnooper", "ansible", "black", "cookiecutter", "fastapi",
    "httpie", "keras", "luigi", "matplotlib", "pandas",
    "sanic", "scrapy", "spacy", "thefuck", "tornado",
    "tqdm", "youtube-dl",
]


# --- Step 1: Download patches ---


def _api_get(url: str) -> dict | list:
    """GET from GitHub API with basic rate-limit handling."""
    token = os.environ.get("GITHUB_TOKEN", "")
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github.v3+json")

    for attempt in range(3):
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            remaining = resp.headers.get("X-RateLimit-Remaining", "?")
            if remaining != "?" and int(remaining) < 10:
                print(f"    Rate limit low ({remaining} remaining), pausing 30s...")
                time.sleep(30)
            return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 403:
                reset = int(e.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset - int(time.time()), 10)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif e.code == 404:
                return []
            else:
                raise
    return []


def download_patches(cache_path: Path) -> dict[str, str]:
    """Download all bug_patch.txt files from BugsInPy. Caches to disk."""
    if cache_path.exists():
        with open(cache_path) as f:
            patches = json.load(f)
        print(f"  Loaded {len(patches)} cached patches from {cache_path.name}")
        return patches

    patches = {}
    for project in PROJECTS:
        print(f"  {project}...", end="", flush=True)
        bugs_url = f"{GITHUB_API}/projects/{project}/bugs"
        bug_dirs = _api_get(bugs_url)
        if not bug_dirs:
            print(" (no bugs found)")
            continue

        count = 0
        for bug_dir in bug_dirs:
            bug_num = bug_dir["name"]
            bug_id = f"{project}__{bug_num}"
            patch_url = f"{GITHUB_API}/projects/{project}/bugs/{bug_num}/bug_patch.txt"
            data = _api_get(patch_url)

            if not data or "content" not in data:
                continue

            content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            if content.strip():
                patches[bug_id] = content
                count += 1

            # Small delay to avoid hammering the API
            time.sleep(0.3)

        print(f" {count} patches")

    with open(cache_path, "w") as f:
        json.dump(patches, f)
    print(f"  Saved {len(patches)} patches to {cache_path.name}")

    return patches


# --- Step 2: Compute edit certificates ---


def compute_certs(patches: dict[str, str]) -> dict[str, list[str]]:
    """Compute normalized edit certificate for each patch."""
    certs = {}
    skipped = 0

    for bug_id, patch in patches.items():
        try:
            ops = patch_to_ast_sequence(patch)
        except Exception:
            skipped += 1
            continue

        if not ops:
            skipped += 1
            continue

        normalized = sorted(set(_NORMALIZE_OPS.get(op, op) for op in ops))
        certs[bug_id] = normalized

    print(f"  {len(certs)} bugs with valid certificates, {skipped} skipped")
    return certs


# --- Step 3: FIM ---


def compute_fim(certs: dict[str, list[str]], min_support: float = 0.10) -> list[dict]:
    """Run fpgrowth and return closed frequent itemsets."""
    transactions = list(certs.values())
    ids = list(certs.keys())

    te = TransactionEncoder()
    binary = pd.DataFrame(
        te.fit_transform(transactions),
        columns=te.columns_,
        index=ids,
    )

    all_items = fpgrowth(binary, min_support=min_support, use_colnames=True)
    if all_items.empty:
        print(f"  No itemsets found at support={min_support}")
        return []

    # Keep only closed itemsets (no superset with same support)
    support_map = {
        frozenset(row["itemsets"]): row["support"]
        for _, row in all_items.iterrows()
    }

    closed = []
    for _, row in all_items.iterrows():
        itemset = frozenset(row["itemsets"])
        s = row["support"]
        is_closed = True
        for other, other_s in support_map.items():
            if other > itemset and abs(other_s - s) < 1e-9:
                is_closed = False
                break
        if is_closed and len(itemset) >= 2:
            closed.append({
                "pattern": sorted(itemset),
                "support": float(s),
                "size": len(itemset),
            })

    closed.sort(key=lambda x: -x["support"])
    print(f"  {len(all_items)} total itemsets, {len(closed)} closed (size >= 2)")
    return closed


# --- Step 4: Compare to SWE-bench Lite ---


def load_swebench_patterns(path: Path) -> list[dict]:
    """Load canonical forms from SWE-bench Lite analysis."""
    with open(path) as f:
        data = json.load(f)
    forms = []
    for form in data["forms"]:
        forms.append({
            "name": form["name"],
            "pattern": sorted(form["pattern"]),
            "n_instances": form["n_instances"],
        })
    return forms


def compare_distributions(
    bugsinpy_certs: dict[str, list[str]],
    bugsinpy_fim: list[dict],
    swebench_forms: list[dict],
) -> dict:
    """Compare edit type and pattern distributions."""
    # Unique edit types
    bp_types = Counter()
    for cert in bugsinpy_certs.values():
        bp_types.update(cert)

    # Load SWE-bench Lite certs for edit type comparison
    swe_types_from_forms = Counter()
    for form in swebench_forms:
        for op in form["pattern"]:
            swe_types_from_forms[op] += form["n_instances"]

    bp_type_set = set(bp_types.keys())
    swe_type_set = set(swe_types_from_forms.keys())
    shared_types = bp_type_set & swe_type_set
    bp_only = bp_type_set - swe_type_set
    swe_only = swe_type_set - bp_type_set

    # Pattern overlap: which SWE-bench patterns also appear in BugsInPy FIM?
    bp_pattern_set = {frozenset(p["pattern"]) for p in bugsinpy_fim}
    swe_pattern_set = {frozenset(f["pattern"]) for f in swebench_forms}

    shared_patterns = bp_pattern_set & swe_pattern_set
    bp_only_patterns = bp_pattern_set - swe_pattern_set
    swe_only_patterns = swe_pattern_set - bp_pattern_set

    # For each SWE-bench pattern, check if it's a subset of any BugsInPy pattern
    swe_subset_of_bp = 0
    for swe_pat in swe_pattern_set:
        for bp_pat in bp_pattern_set:
            if swe_pat.issubset(bp_pat):
                swe_subset_of_bp += 1
                break

    # Edit type frequency correlation
    shared_ops = sorted(shared_types)
    bp_freqs = [bp_types[op] for op in shared_ops]
    swe_freqs = [swe_types_from_forms[op] for op in shared_ops]
    if len(shared_ops) >= 3:
        correlation = float(np.corrcoef(bp_freqs, swe_freqs)[0, 1])
    else:
        correlation = None

    # Certificate size distribution
    bp_sizes = [len(c) for c in bugsinpy_certs.values()]

    return {
        "edit_types": {
            "bugsinpy_unique": len(bp_type_set),
            "swebench_unique": len(swe_type_set),
            "shared": len(shared_types),
            "bugsinpy_only": sorted(bp_only),
            "swebench_only": sorted(swe_only),
            "frequency_correlation": correlation,
        },
        "patterns": {
            "bugsinpy_fim_count": len(bugsinpy_fim),
            "swebench_form_count": len(swebench_forms),
            "exact_overlap": len(shared_patterns),
            "bugsinpy_only": len(bp_only_patterns),
            "swebench_only": len(swe_only_patterns),
            "swebench_subset_of_bugsinpy": swe_subset_of_bp,
            "shared_patterns": [sorted(p) for p in shared_patterns],
        },
        "certificate_sizes": {
            "bugsinpy_mean": float(np.mean(bp_sizes)),
            "bugsinpy_median": float(np.median(bp_sizes)),
            "bugsinpy_std": float(np.std(bp_sizes)),
            "bugsinpy_min": int(min(bp_sizes)),
            "bugsinpy_max": int(max(bp_sizes)),
        },
        "top_bugsinpy_ops": [
            {"op": op, "count": count}
            for op, count in bp_types.most_common(20)
        ],
    }


# --- Step 5: Figures ---


def fig_pattern_comparison(
    bugsinpy_certs: dict[str, list[str]],
    swebench_forms: list[dict],
    bugsinpy_fim: list[dict],
    out_path: Path,
):
    """Three-panel figure: edit type frequencies, pattern sizes, FIM support."""

    # Panel A: Edit type frequencies (top 15 shared types, side by side)
    bp_types = Counter()
    for cert in bugsinpy_certs.values():
        bp_types.update(cert)

    swe_types = Counter()
    for form in swebench_forms:
        for op in form["pattern"]:
            swe_types[op] += form["n_instances"]

    shared_types = sorted(set(bp_types.keys()) & set(swe_types.keys()))

    # Normalize to fractions
    bp_total = sum(bp_types.values())
    swe_total = sum(swe_types.values())

    type_rows = []
    for op in shared_types:
        type_rows.append({
            "op": op,
            "fraction": bp_types[op] / bp_total,
            "dataset": "BugsInPy",
        })
        type_rows.append({
            "op": op,
            "fraction": swe_types[op] / swe_total,
            "dataset": "SWE-bench Lite",
        })

    type_df = pd.DataFrame(type_rows)

    # Keep top 12 by combined frequency
    top_ops = sorted(
        shared_types,
        key=lambda op: bp_types[op] / bp_total + swe_types[op] / swe_total,
        reverse=True,
    )[:12]
    type_df = type_df[type_df["op"].isin(top_ops)]

    panel_a = alt.Chart(type_df).mark_bar().encode(
        x=alt.X(
            "op:N",
            sort=top_ops,
            axis=alt.Axis(title=None, labelAngle=-40, labelFontSize=8),
        ),
        y=alt.Y(
            "fraction:Q",
            axis=alt.Axis(title="Frequency (fraction)", titleFontSize=9, format=".0%"),
        ),
        color=alt.Color(
            "dataset:N",
            scale=alt.Scale(
                domain=["BugsInPy", "SWE-bench Lite"],
                range=[BLUE, ORANGE],
            ),
            legend=alt.Legend(
                title=None,
                orient="top-right",
                labelFontSize=9,
            ),
        ),
        xOffset="dataset:N",
    ).properties(
        width=350,
        height=220,
        title=alt.TitleParams(
            "A. Edit type frequency",
            fontSize=11,
            fontWeight="normal",
            anchor="start",
        ),
    )

    # Panel B: Certificate size distributions
    bp_sizes = [len(c) for c in bugsinpy_certs.values()]

    size_rows = []
    for s in bp_sizes:
        size_rows.append({"cert_size": s, "dataset": "BugsInPy"})

    # SWE-bench sizes from form patterns (weighted by instance count)
    for form in swebench_forms:
        for _ in range(form["n_instances"]):
            size_rows.append({"cert_size": len(form["pattern"]), "dataset": "SWE-bench Lite"})

    size_df = pd.DataFrame(size_rows)

    panel_b = alt.Chart(size_df).mark_bar(
        opacity=0.7,
        binSpacing=0,
    ).encode(
        x=alt.X(
            "cert_size:Q",
            bin=alt.Bin(maxbins=20),
            axis=alt.Axis(title="Certificate size (unique ops)", titleFontSize=9),
        ),
        y=alt.Y(
            "count():Q",
            stack=None,
            axis=alt.Axis(title="Count", titleFontSize=9),
        ),
        color=alt.Color(
            "dataset:N",
            scale=alt.Scale(
                domain=["BugsInPy", "SWE-bench Lite"],
                range=[BLUE, ORANGE],
            ),
            legend=alt.Legend(
                title=None,
                orient="top-right",
                labelFontSize=9,
            ),
        ),
    ).properties(
        width=350,
        height=220,
        title=alt.TitleParams(
            "B. Certificate size distribution",
            fontSize=11,
            fontWeight="normal",
            anchor="start",
        ),
    )

    # Panel C: FIM pattern support comparison (BugsInPy patterns)
    if bugsinpy_fim:
        fim_df = pd.DataFrame(bugsinpy_fim)
        fim_df["pattern_label"] = fim_df["pattern"].apply(
            lambda p: " + ".join(p[:3]) + ("..." if len(p) > 3 else "")
        )
        # Top 15 by support
        fim_top = fim_df.head(15)

        panel_c = alt.Chart(fim_top).mark_bar().encode(
            x=alt.X(
                "support:Q",
                axis=alt.Axis(title="Support (fraction of bugs)", titleFontSize=9, format=".0%"),
            ),
            y=alt.Y(
                "pattern_label:N",
                sort=alt.EncodingSortField(field="support", order="descending"),
                axis=alt.Axis(title=None, labelFontSize=8),
            ),
            color=alt.value(BLUE),
        ).properties(
            width=350,
            height=220,
            title=alt.TitleParams(
                "C. Top BugsInPy FIM patterns (support >= 10%)",
                fontSize=11,
                fontWeight="normal",
                anchor="start",
            ),
        )
    else:
        panel_c = alt.Chart(pd.DataFrame({"x": [0]})).mark_text(
            text="No FIM patterns found"
        ).encode(x="x:Q").properties(width=350, height=220)

    fig = (panel_a | panel_b).resolve_scale(
        color="shared"
    ) & panel_c

    fig = fig.configure_axis(
        grid=False,
        labelFontSize=9,
        titleFontSize=10,
    ).configure_view(strokeWidth=0)

    fig.save(str(out_path), scale_factor=2)
    print(f"  Saved {out_path.name}")


# --- Main ---


def main():
    print("=" * 60)
    print("BugsInPy generalization analysis")
    print("=" * 60)

    # Step 1: Download patches
    print("\nStep 1: Downloading BugsInPy patches...")
    patches = download_patches(OUT / "bugsinpy_patches.json")
    print(f"  Total patches: {len(patches)}")

    # Project breakdown
    project_counts = Counter()
    for bug_id in patches:
        project = bug_id.split("__")[0]
        project_counts[project] += 1
    print("  Per-project:")
    for proj, count in project_counts.most_common():
        print(f"    {proj}: {count}")

    # Step 2: Compute edit certificates
    print(f"\nStep 2: Computing edit certificates...")
    certs = compute_certs(patches)

    # Save certificates
    with open(OUT / "bugsinpy_certs.json", "w") as f:
        json.dump(certs, f, indent=2)
    print(f"  Saved bugsinpy_certs.json")

    # Unique edit types
    all_ops = Counter()
    for cert in certs.values():
        all_ops.update(cert)
    print(f"  Unique edit types: {len(all_ops)}")
    print(f"  Top 10:")
    for op, count in all_ops.most_common(10):
        pct = 100 * count / len(certs)
        print(f"    {op}: {count} ({pct:.1f}% of bugs)")

    # Step 3: Run FIM
    print(f"\nStep 3: Running FIM (fpgrowth, support=0.10)...")
    fim_patterns = compute_fim(certs, min_support=0.10)

    with open(OUT / "bugsinpy_fim_patterns.json", "w") as f:
        json.dump(fim_patterns, f, indent=2)
    print(f"  Saved bugsinpy_fim_patterns.json")

    if fim_patterns:
        print(f"  Top 10 patterns:")
        for p in fim_patterns[:10]:
            print(f"    support={p['support']:.2f}, size={p['size']}: {p['pattern']}")

    # Step 4: Compare to SWE-bench Lite
    print(f"\nStep 4: Comparing to SWE-bench Lite...")
    swe_forms_path = ROOT / "output" / "canonical_forms" / "canonical_forms.json"
    if not swe_forms_path.exists():
        print(f"  WARNING: {swe_forms_path} not found. Skipping comparison.")
        comparison = {"error": "SWE-bench Lite canonical forms not found"}
    else:
        swe_forms = load_swebench_patterns(swe_forms_path)
        print(f"  SWE-bench Lite: {len(swe_forms)} canonical forms")
        comparison = compare_distributions(certs, fim_patterns, swe_forms)

        print(f"\n  Edit types:")
        print(f"    BugsInPy unique: {comparison['edit_types']['bugsinpy_unique']}")
        print(f"    SWE-bench Lite unique: {comparison['edit_types']['swebench_unique']}")
        print(f"    Shared: {comparison['edit_types']['shared']}")
        if comparison["edit_types"]["frequency_correlation"] is not None:
            print(f"    Frequency correlation: {comparison['edit_types']['frequency_correlation']:.3f}")
        if comparison["edit_types"]["bugsinpy_only"]:
            print(f"    BugsInPy-only types: {comparison['edit_types']['bugsinpy_only']}")
        if comparison["edit_types"]["swebench_only"]:
            print(f"    SWE-bench-only types: {comparison['edit_types']['swebench_only']}")

        print(f"\n  Patterns:")
        print(f"    BugsInPy FIM patterns: {comparison['patterns']['bugsinpy_fim_count']}")
        print(f"    SWE-bench Lite forms: {comparison['patterns']['swebench_form_count']}")
        print(f"    Exact overlap: {comparison['patterns']['exact_overlap']}")
        print(f"    SWE-bench patterns that are subsets of BugsInPy: "
              f"{comparison['patterns']['swebench_subset_of_bugsinpy']}")

        print(f"\n  Certificate sizes (BugsInPy):")
        cs = comparison["certificate_sizes"]
        print(f"    mean={cs['bugsinpy_mean']:.1f}, median={cs['bugsinpy_median']:.0f}, "
              f"std={cs['bugsinpy_std']:.1f}, range=[{cs['bugsinpy_min']}, {cs['bugsinpy_max']}]")

    with open(OUT / "bugsinpy_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Saved bugsinpy_comparison.json")

    # Step 5: Figure
    print(f"\nStep 5: Generating comparison figure...")
    if swe_forms_path.exists():
        fig_pattern_comparison(
            certs, swe_forms, fim_patterns,
            OUT / "fig_pattern_comparison.png",
        )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  BugsInPy bugs processed: {len(patches)}")
    print(f"  Valid edit certificates: {len(certs)}")
    coverage = 100 * len(certs) / len(patches) if patches else 0
    print(f"  Coverage: {coverage:.1f}%")
    print(f"  Unique edit types: {len(all_ops)}")
    print(f"  FIM patterns (closed, size >= 2): {len(fim_patterns)}")
    if "edit_types" in comparison:
        et = comparison["edit_types"]
        print(f"  Edit type overlap with SWE-bench Lite: "
              f"{et['shared']}/{et['bugsinpy_unique']} BugsInPy types shared")
    print(f"\n  Outputs in {OUT}")


if __name__ == "__main__":
    main()
