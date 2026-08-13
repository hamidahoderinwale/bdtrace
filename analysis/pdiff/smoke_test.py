"""End-to-end smoke test for analysis.pdiff.

Loads resolved trajectories from Lite / Verified / SWE-Smith, computes:
  - pairwise diffs for ~10 instances
  - signature aggregates per benchmark
  - reference vocabulary from Lite
  - OOD scores for held-out Verified + SWE-Smith samples against Lite reference

Prints a summary table and saves full results to output/pdiff_smoke_test/.

Usage:
    python -m analysis.pdiff.smoke_test
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from analysis.pdiff import (
    build_reference_vocabulary,
    diff,
    ood_items,
    ood_score,
    signature,
    view_from_trace,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "output" / "pdiff_smoke_test"

RESOLVED = {
    "lite": PROJECT_ROOT / "output" / "resolved_traces_lite_full.jsonl",
    "verified": PROJECT_ROOT / "output" / "resolved_traces_verified_full.jsonl",
    "swe_smith": PROJECT_ROOT / "output" / "resolved_traces_swe_smith.jsonl",
}

SAMPLE_N = 50  # trajectories to load per benchmark
PAIR_N = 10  # pairwise diffs to compute per benchmark
OOD_N = 25  # OOD scores to compute per held-out benchmark


def load_traces(path: Path, limit: int) -> list[dict]:
    if not path.exists():
        return []
    traces: list[dict] = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            traces.append(json.loads(line))
            if len(traces) >= limit:
                break
    return traces


def describe_view_counts(views: list) -> dict[str, float]:
    return {
        "n": len(views),
        "with_edits": sum(1 for v in views if v.has_edits),
        "with_tokens": sum(1 for v in views if v.has_tokens),
        "with_scopes": sum(1 for v in views if v.has_scopes),
        "with_modules": sum(1 for v in views if v.has_modules),
    }


def pairwise_diff_stats(views: list, limit: int) -> dict[str, Any]:
    pairs: list[dict[str, float | None]] = []
    for i in range(min(limit, len(views) - 1)):
        d = diff(views[i], views[i + 1])
        pairs.append({
            "tokens": d.tokens,
            "edits": d.edits,
            "scopes": d.scopes,
            "modules": d.modules,
            "mean": d.mean(),
        })

    def _mean(key: str) -> float | None:
        vals = [p[key] for p in pairs if p[key] is not None]
        return statistics.mean(vals) if vals else None

    return {
        "n_pairs": len(pairs),
        "mean_tokens": _mean("tokens"),
        "mean_edits": _mean("edits"),
        "mean_scopes": _mean("scopes"),
        "mean_modules": _mean("modules"),
        "mean_overall": _mean("mean"),
        "pairs": pairs,
    }


def ood_stats(views: list, reference: dict[str, frozenset[str]], limit: int) -> dict[str, Any]:
    scores_edits: list[float] = []
    scores_modules: list[float] = []
    novel_edits_per_trace: list[int] = []
    for v in views[:limit]:
        if v.has_edits:
            scores_edits.append(ood_score(v, reference, level="edits"))
            novel_edits_per_trace.append(len(ood_items(v, reference, level="edits")))
        if v.has_modules:
            scores_modules.append(ood_score(v, reference, level="modules"))

    return {
        "n_scored_edits": len(scores_edits),
        "mean_ood_edits": statistics.mean(scores_edits) if scores_edits else None,
        "median_ood_edits": statistics.median(scores_edits) if scores_edits else None,
        "n_fully_covered_edits": sum(1 for s in scores_edits if s == 0.0),
        "n_fully_novel_edits": sum(1 for s in scores_edits if s == 1.0),
        "mean_novel_ops_per_trace": statistics.mean(novel_edits_per_trace) if novel_edits_per_trace else None,
        "n_scored_modules": len(scores_modules),
        "mean_ood_modules": statistics.mean(scores_modules) if scores_modules else None,
    }


def serialize_signature(sig) -> dict[str, Any]:
    return {
        "n": sig.n,
        "edit_vocab_size": len(sig.edit_vocab),
        "module_vocab_size": len(sig.module_vocab),
        "scope_vocab_size": len(sig.scope_vocab),
        "mean_tokens": sig.mean_tokens,
        "top_10_edits": sig.edit_freq.most_common(10),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    print(f"=== pdiff smoke test ===\nProject root: {PROJECT_ROOT}\nOutput: {OUT_DIR}\n")

    traces: dict[str, list[dict]] = {}
    views: dict[str, list] = {}
    for name, path in RESOLVED.items():
        loaded = load_traces(path, SAMPLE_N)
        traces[name] = loaded
        views[name] = [view_from_trace(t) for t in loaded]
        counts = describe_view_counts(views[name])
        results[f"views_{name}"] = counts
        print(f"[{name}] loaded {counts['n']} traces | "
              f"edits={counts['with_edits']} tokens={counts['with_tokens']} "
              f"scopes={counts['with_scopes']} modules={counts['with_modules']}")

    print()
    for name in RESOLVED:
        if not views[name]:
            print(f"[{name}] no data — skipping signature/diff")
            continue
        sig = signature(views[name])
        serialized_sig = serialize_signature(sig)
        results[f"signature_{name}"] = serialized_sig
        print(f"[{name}] signature: n={sig.n} "
              f"edit_vocab={len(sig.edit_vocab)} "
              f"module_vocab={len(sig.module_vocab)} "
              f"mean_tokens={sig.mean_tokens:.1f}")

    print()
    for name in RESOLVED:
        if len(views[name]) < 2:
            continue
        pstats = pairwise_diff_stats(views[name], PAIR_N)
        results[f"pairwise_{name}"] = pstats
        print(f"[{name}] {pstats['n_pairs']} pairwise diffs: "
              f"tokens={pstats['mean_tokens']} "
              f"edits={pstats['mean_edits']} "
              f"scopes={pstats['mean_scopes']} "
              f"modules={pstats['mean_modules']}")

    print()
    if not views.get("lite"):
        print("No Lite traces — cannot build reference vocabulary. Exiting.")
        _write_results(results)
        return 1

    reference = build_reference_vocabulary(views["lite"])
    results["reference_vocab_from_lite"] = {
        "n_traces": len(views["lite"]),
        "edits_vocab_size": len(reference["edits"]),
        "modules_vocab_size": len(reference["modules"]),
        "scopes_vocab_size": len(reference["scopes"]),
    }
    print(f"[reference] built from {len(views['lite'])} Lite traces: "
          f"edits={len(reference['edits'])} "
          f"modules={len(reference['modules'])} "
          f"scopes={len(reference['scopes'])}")

    print()
    for name in ("verified", "swe_smith"):
        if not views.get(name):
            continue
        stats = ood_stats(views[name], reference, OOD_N)
        results[f"ood_{name}_vs_lite"] = stats
        print(f"[ood {name} vs lite] n={stats['n_scored_edits']} "
              f"mean_ood_edits={stats['mean_ood_edits']} "
              f"fully_covered={stats['n_fully_covered_edits']} "
              f"fully_novel={stats['n_fully_novel_edits']}")

    _write_results(results)
    print(f"\nFull results: {OUT_DIR / 'results.json'}")
    return 0


def _write_results(results: dict[str, Any]) -> None:
    out = OUT_DIR / "results.json"
    with open(out, "w") as fh:
        json.dump(_json_safe(results), fh, indent=2, default=str)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(str(x) for x in obj)
    return obj


if __name__ == "__main__":
    raise SystemExit(main())
