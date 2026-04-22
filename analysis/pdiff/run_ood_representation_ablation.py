"""OOD representation ablation: edits vs modules vs scopes.

Robustness check for the committed cross-benchmark transfer finding
(Verified 95.6 percent / SWE-Smith 91.4 percent fully covered by Lite reference
at level=edits). This script extends it to three representation levels
(edits, modules, scopes) and two min_count thresholds (1 and 2).

Interpretation (written into the docstring for future readers)
--------------------------------------------------------------
The compositional-compactness finding is sharpest at the edit-op level,
where 91-96 percent of held-out trajectories are fully covered by Lite's
reference vocabulary. Module-level coverage is benchmark-dependent --
synthetic benchmarks like SWE-Smith use disjoint file-stem vocabularies
(swesmith/* repos), so cross-benchmark module-OOD is trivially high. That
is a data-generation artifact, not a finding. Scope-level falls between
the two.

Caveats (also flagged in the output table)
------------------------------------------
* SWE-Smith modules will show approx 100 percent OOD. Expected because
  SWE-Smith uses synthetic repos whose file stems do not overlap with
  Django/Sphinx/Astropy.
* Scopes coverage may drop on SWE-Smith because trace_to_scoped_cert
  fails to extract scopes for some traces (missing or unparseable
  before_content). We report actual n_scored so the denominator is visible.
* Edit-level should mirror the committed cross-benchmark transfer numbers
  (Verified mc1 approx 0.956 fully covered, SWE-Smith mc1 approx 0.914).
  Divergence beyond 0.5 percentage points indicates a pipeline regression.

Usage:
    python -m analysis.pdiff.run_ood_representation_ablation
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
LEVELS = ("edits", "modules", "scopes")
MIN_COUNTS = (1, 2)


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


def _has_level(view, level: str) -> bool:
    attr = getattr(view, level, None)
    return bool(attr)


def _median(sorted_values: list[float]) -> float:
    n = len(sorted_values)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


def _cell_stats(
    views: list,
    reference: dict,
    level: str,
    lite_vocab_at_level: frozenset,
    benchmark_vocab_at_level: set,
) -> dict:
    scored = [
        ood_score(v, reference, level=level)
        for v in views
        if _has_level(v, level)
    ]
    n = len(scored)
    novel_count = len(benchmark_vocab_at_level - lite_vocab_at_level)
    vocab_size = len(reference.get(level, ()))
    if n == 0:
        return _empty_cell(vocab_size, novel_count)
    scored_sorted = sorted(scored)
    mean_ood = sum(scored) / n
    median_ood = _median(scored_sorted)
    pct_fully_covered = sum(1 for s in scored if s == 0.0) / n
    pct_fully_novel = sum(1 for s in scored if s == 1.0) / n
    return _make_cell(n, mean_ood, median_ood, pct_fully_covered, pct_fully_novel, vocab_size, novel_count)


def _empty_cell(vocab_size, novel_count):
    nan = float("nan")
    return {**{"n_scored": 0, "mean_ood": nan, "median_ood": nan}, **{"pct_fully_covered": nan, "pct_fully_novel": nan, "vocab_size": vocab_size, "novel_item_count": novel_count}}


def _make_cell(n, mean_ood, median_ood, pct_cov, pct_nov, vocab_size, novel_count):
    return {**{"n_scored": n, "mean_ood": mean_ood, "median_ood": median_ood}, **{"pct_fully_covered": pct_cov, "pct_fully_novel": pct_nov, "vocab_size": vocab_size, "novel_item_count": novel_count}}


def _vocab_union(views: list, level: str) -> set:
    out: set = set()
    for v in views:
        out.update(getattr(v, level, ()))
    return out


def _build_references(lite_views: list) -> dict:
    return {mc: build_reference_vocabulary(lite_views, min_count=mc) for mc in MIN_COUNTS}


def _benchmark_cells(views: list, references: dict, lite_vocab_by_level: dict) -> dict:
    bench_vocab_by_level = {lv: _vocab_union(views, lv) for lv in LEVELS}
    cells: dict = {}
    for mc, ref in references.items():
        for level in LEVELS:
            cells[f"mc{mc}_{level}"] = _cell_stats(
                views,
                ref,
                level,
                lite_vocab_at_level=frozenset(lite_vocab_by_level[level]),
                benchmark_vocab_at_level=bench_vocab_by_level[level],
            )
    return cells


def _fmt_pct(x: float) -> str:
    if x != x:
        return "   nan"
    return f"{100 * x:5.1f}%"


def _fmt_int(x: int) -> str:
    return f"{x:>5d}"


def _print_header(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _row_pct(label: str, cells: dict, mc: int) -> str:
    parts = [f"{label:<18}"]
    for level in LEVELS:
        cell = cells[f"mc{mc}_{level}"]
        parts.append(f"{_fmt_pct(cell['pct_fully_covered']):>10}")
    return " ".join(parts)


def _row_n(label: str, cells: dict, mc: int) -> str:
    parts = [f"{label:<18}"]
    for level in LEVELS:
        cell = cells[f"mc{mc}_{level}"]
        parts.append(f"{_fmt_int(cell['n_scored']):>10}")
    return " ".join(parts)


def _row_mean(label: str, cells: dict, mc: int) -> str:
    parts = [f"{label:<18}"]
    for level in LEVELS:
        cell = cells[f"mc{mc}_{level}"]
        m = cell["mean_ood"]
        if m == m:
            parts.append(f"{m:>10.4f}")
        else:
            parts.append(f"{'   nan':>10}")
    return " ".join(parts)


def _print_compact_table(verified_cells: dict, swe_cells: dict) -> None:
    _print_header("pct_fully_covered (headline): OOD=0 trajectories / n_scored")
    header = f"{'slice':<18} {'edits':>10} {'modules':>10} {'scopes':>10}"
    print(header)
    print("-" * len(header))
    print("# Verified (n=500 views; shares Django/Sphinx/Astropy family with Lite)")
    print(_row_pct("Verified mc1", verified_cells, 1))
    print(_row_pct("Verified mc2", verified_cells, 2))
    print()
    print("# SWE-Smith (first 5000 views; modules ~100% OOD by construction)")
    print(_row_pct("SWE-Smith mc1", swe_cells, 1))
    print(_row_pct("SWE-Smith mc2", swe_cells, 2))

    _print_header("n_scored (denominator for each cell)")
    print(header)
    print("-" * len(header))
    print(_row_n("Verified mc1", verified_cells, 1))
    print(_row_n("Verified mc2", verified_cells, 2))
    print(_row_n("SWE-Smith mc1", swe_cells, 1))
    print(_row_n("SWE-Smith mc2", swe_cells, 2))

    _print_header("mean_ood")
    print(header)
    print("-" * len(header))
    print(_row_mean("Verified mc1", verified_cells, 1))
    print(_row_mean("Verified mc2", verified_cells, 2))
    print(_row_mean("SWE-Smith mc1", swe_cells, 1))
    print(_row_mean("SWE-Smith mc2", swe_cells, 2))


def _sanity_check_edit_level(verified_cells: dict, swe_cells: dict) -> list:
    """Compare edit-level mc1 headline against the committed finding."""
    warnings: list = []
    v_edits = verified_cells["mc1_edits"]["pct_fully_covered"]
    s_edits = swe_cells["mc1_edits"]["pct_fully_covered"]
    if abs(v_edits - 0.956) > 0.005:
        warnings.append(f"Verified mc1 edits pct_fully_covered={v_edits:.4f} diverges from committed 0.9560 by >0.5pp.")
    if abs(s_edits - 0.914) > 0.005:
        warnings.append(f"SWE-Smith mc1 edits pct_fully_covered={s_edits:.4f} diverges from committed 0.9140 by >0.5pp.")
    return warnings


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
    print("Building Lite reference vocabularies at min_count=1 and min_count=2...")
    references = _build_references(lite_views)
    for mc, ref in references.items():
        sizes = {lv: len(ref[lv]) for lv in LEVELS}
        print(f"  min_count={mc}: {sizes}")

    lite_vocab_by_level = {lv: _vocab_union(lite_views, lv) for lv in LEVELS}

    print()
    print("Scoring Verified against Lite reference...")
    verified_cells = _benchmark_cells(verified_views, references, lite_vocab_by_level)
    print("Scoring SWE-Smith against Lite reference...")
    swe_cells = _benchmark_cells(swe_smith_views, references, lite_vocab_by_level)

    _print_compact_table(verified_cells, swe_cells)

    warnings = _sanity_check_edit_level(verified_cells, swe_cells)
    if warnings:
        _print_header("WARNINGS: edit-level headline divergence")
        for w in warnings:
            print(f"  {w}")
    else:
        print()
        print("Edit-level mc1 matches committed cross-benchmark numbers within 0.5pp.")

    _print_header("Caveats")
    print("- SWE-Smith modules ~100% OOD is a known data-generation artifact:")
    print("  swesmith/* repos have file stems disjoint from Django/Sphinx/Astropy.")
    print("  Module-level is NOT a portable representation across synthetic benchmarks.")
    print("- Scope n_scored may be lower than n_views when trace_to_scoped_cert")
    print("  fails to parse before_content. See n_scored column above.")
    print("- Edit-level is the representation the compositional-compactness claim")
    print("  is made at; module/scope ablations document where it does NOT hold.")

    results = _build_results_dict(lite_views, verified_views, swe_smith_views, references, verified_cells, swe_cells, warnings)
    out_path = OUT_DIR / "ood_representation_ablation.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print()
    print(f"Wrote {out_path}")
    return 0


def _build_results_dict(lite_views, verified_views, swe_smith_views, references, verified_cells, swe_cells, warnings):
    text = "Edit-op level is portable; module-level is not; scope-level in between."
    vocab_sizes = {}
    for mc, ref in references.items():
        vocab_sizes["min_count_" + str(mc)] = {lv: len(ref[lv]) for lv in LEVELS}
    n_views = {}
    n_views["lite"] = len(lite_views)
    n_views["verified"] = len(verified_views)
    n_views["swe_smith"] = len(swe_smith_views)
    cells = {}
    cells["verified"] = verified_cells
    cells["swe_smith"] = swe_cells
    caps = {}
    caps["swe_smith"] = SWE_SMITH_CAP
    out = {}
    out["caps"] = caps
    out["n_views"] = n_views
    out["reference_vocab_sizes"] = vocab_sizes
    out["cells"] = cells
    out["edit_level_sanity_warnings"] = warnings
    out["interpretation"] = text
    return out


if __name__ == "__main__":
    raise SystemExit(main())
