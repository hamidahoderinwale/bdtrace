"""Cross-benchmark vocabulary transfer (Week 2 kickoff).

Builds a reference vocabulary from ALL of Lite, then measures the fraction
of Verified / SWE-Smith edit-ops and modules covered by it.

This is the compositional-compactness claim scaled up from the 50-trace
smoke test to full benchmarks: if ~95% of Verified edit-ops already appear
in Lite, edit-op vocabulary saturates well below the benchmark scale.

Runs two coverage passes (min_count=1 includes singletons; min_count=2
excludes them, which is the more conservative reference), plus pairwise
intersection sizes on raw edit-op sets.

Writes output/pdiff_smoke_test/cross_benchmark_transfer.json.

Usage:
    python -m analysis.pdiff.run_cross_benchmark_transfer
"""

from __future__ import annotations

import json
from pathlib import Path

from analysis.pdiff import (
    build_reference_vocabulary,
    ood_score,
    view_from_trace,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "output" / "pdiff_smoke_test"
LITE = PROJECT_ROOT / "output" / "resolved_traces_lite_full.jsonl"
VERIFIED = PROJECT_ROOT / "output" / "resolved_traces_verified_full.jsonl"
SWE_SMITH = PROJECT_ROOT / "output" / "resolved_traces_swe_smith.jsonl"

SWE_SMITH_CAP = 5000


def _load_views(path: Path, cap: int | None = None) -> list:
    views = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                trace = json.loads(line)
            except json.JSONDecodeError:
                continue
            views.append(view_from_trace(trace))
            if cap is not None and len(views) >= cap:
                break
    return views


def _bucket_of(score: float) -> str:
    if score == 0.0:
        return "zero"
    if score == 1.0:
        return "one"
    if score < 0.25:
        return "0_to_25"
    if score < 0.50:
        return "25_to_50"
    if score < 0.75:
        return "50_to_75"
    return "75_to_100"


def _histogram(scores: list[float]) -> dict[str, int]:
    buckets = ["zero", "0_to_25", "25_to_50", "50_to_75", "75_to_100", "one"]
    counts = dict.fromkeys(buckets, 0)
    for s in scores:
        counts[_bucket_of(s)] += 1
    return counts


def _stats(scores: list[float]) -> dict:
    if not scores:
        return {"n": 0}
    n = len(scores)
    scores_sorted = sorted(scores)
    mean = sum(scores) / n
    if n % 2:
        median = scores_sorted[n // 2]
    else:
        median = 0.5 * (scores_sorted[n // 2 - 1] + scores_sorted[n // 2])
    frac_zero = sum(1 for s in scores if s == 0.0) / n
    frac_one = sum(1 for s in scores if s == 1.0) / n
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "frac_fully_covered": frac_zero,
        "frac_fully_novel": frac_one,
        "histogram": _histogram(scores),
    }


def _vocab_sets(views: list) -> tuple[set[str], set[str]]:
    edits: set[str] = set()
    modules: set[str] = set()
    for v in views:
        edits.update(v.edits)
        modules.update(v.modules)
    return edits, modules


def _coverage_pass(views: list, reference: dict, label: str) -> dict:
    edit_scores = [ood_score(v, reference, level="edits") for v in views if v.has_edits]
    module_scores = [ood_score(v, reference, level="modules") for v in views if v.has_modules]
    return {
        "label": label,
        "n_views": len(views),
        "n_with_edits": len(edit_scores),
        "n_with_modules": len(module_scores),
        "edits": _stats(edit_scores),
        "modules": _stats(module_scores),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Lite views...")
    lite_views = _load_views(LITE)
    print(f"  lite views: {len(lite_views)}")

    print("Loading Verified views...")
    verified_views = _load_views(VERIFIED)
    print(f"  verified views: {len(verified_views)}")

    print(f"Loading SWE-Smith views (cap {SWE_SMITH_CAP})...")
    swe_smith_views = _load_views(SWE_SMITH, cap=SWE_SMITH_CAP)
    print(f"  swe-smith views: {len(swe_smith_views)}")

    print()
    print("Building reference vocabularies from ALL Lite...")
    ref_mc1 = build_reference_vocabulary(lite_views, min_count=1)
    ref_mc2 = build_reference_vocabulary(lite_views, min_count=2)
    print(f"  min_count=1 edit-ops: {len(ref_mc1['edits'])}")
    print(f"  min_count=2 edit-ops: {len(ref_mc2['edits'])}")
    print(f"  min_count=1 modules:  {len(ref_mc1['modules'])}")
    print(f"  min_count=2 modules:  {len(ref_mc2['modules'])}")
    print()

    lite_edits, lite_modules = _vocab_sets(lite_views)
    verified_edits, verified_modules = _vocab_sets(verified_views)
    swe_edits, swe_modules = _vocab_sets(swe_smith_views)

    vocab_summary = {
        "lite": {"edit_ops": len(lite_edits), "modules": len(lite_modules)},
        "verified": {
            "edit_ops": len(verified_edits),
            "modules": len(verified_modules),
            "edit_ops_intersect_lite": len(verified_edits & lite_edits),
            "modules_intersect_lite": len(verified_modules & lite_modules),
            "edit_ops_minus_lite": len(verified_edits - lite_edits),
            "modules_minus_lite": len(verified_modules - lite_modules),
        },
        "swe_smith": {
            "edit_ops": len(swe_edits),
            "modules": len(swe_modules),
            "edit_ops_intersect_lite": len(swe_edits & lite_edits),
            "modules_intersect_lite": len(swe_modules & lite_modules),
            "edit_ops_minus_lite": len(swe_edits - lite_edits),
            "modules_minus_lite": len(swe_modules - lite_modules),
        },
    }

    print("Vocabulary set sizes:")
    print(f"  Lite:      edit_ops={len(lite_edits):4d} modules={len(lite_modules):5d}")
    print(
        f"  Verified:  edit_ops={len(verified_edits):4d} "
        f"modules={len(verified_modules):5d} "
        f"(intersect={len(verified_edits & lite_edits)}, "
        f"minus_lite={len(verified_edits - lite_edits)})"
    )
    print(
        f"  SWE-Smith: edit_ops={len(swe_edits):4d} "
        f"modules={len(swe_modules):5d} "
        f"(intersect={len(swe_edits & lite_edits)}, "
        f"minus_lite={len(swe_edits - lite_edits)})"
    )
    print()

    results = {
        "caps": {"swe_smith": SWE_SMITH_CAP},
        "reference_vocab_sizes": {
            "min_count_1": {k: len(v) for k, v in ref_mc1.items()},
            "min_count_2": {k: len(v) for k, v in ref_mc2.items()},
        },
        "vocab_summary": vocab_summary,
        "coverage": {
            "verified_vs_lite_mc1": _coverage_pass(verified_views, ref_mc1, "verified_mc1"),
            "verified_vs_lite_mc2": _coverage_pass(verified_views, ref_mc2, "verified_mc2"),
            "swe_smith_vs_lite_mc1": _coverage_pass(swe_smith_views, ref_mc1, "swe_smith_mc1"),
            "swe_smith_vs_lite_mc2": _coverage_pass(swe_smith_views, ref_mc2, "swe_smith_mc2"),
        },
    }

    print("Coverage table (OOD score = fraction of trace items NOT in reference):")
    print(
        f"{'slice':<25} {'level':<8} {'n':>5} {'mean':>8} "
        f"{'median':>8} {'frac_0':>8} {'frac_1':>8}"
    )
    for key, data in results["coverage"].items():
        for lvl in ("edits", "modules"):
            s = data[lvl]
            print(
                f"{key:<25} {lvl:<8} {s['n']:>5} "
                f"{s.get('mean', float('nan')):>8.4f} "
                f"{s.get('median', float('nan')):>8.4f} "
                f"{s.get('frac_fully_covered', float('nan')):>8.4f} "
                f"{s.get('frac_fully_novel', float('nan')):>8.4f}"
            )
    print()

    print("Edit OOD histograms (counts per bucket):")
    print(
        f"{'slice':<25} {'zero':>6} {'0-25':>6} {'25-50':>6} "
        f"{'50-75':>6} {'75-100':>6} {'one':>6}"
    )
    for key, data in results["coverage"].items():
        h = data["edits"].get("histogram", {})
        print(
            f"{key:<25} {h.get('zero', 0):>6} {h.get('0_to_25', 0):>6} "
            f"{h.get('25_to_50', 0):>6} {h.get('50_to_75', 0):>6} "
            f"{h.get('75_to_100', 0):>6} {h.get('one', 0):>6}"
        )

    out_path = OUT_DIR / "cross_benchmark_transfer.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
