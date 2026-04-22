"""Smoke test for the V-measure battery.

Loads resolved Lite trajectories, k-means clusters them on edit-op vocab,
computes V-measure / ARI / NMI against reference partitions built from the
trace metadata (repo name, patch-size bucket, module). Bootstrap stability
on one reference. Writes to output/pdiff_smoke_test/vmeasure.json.

Usage:
    python -m analysis.pdiff.run_vmeasure_smoke
"""

from __future__ import annotations

import json
from pathlib import Path

from analysis.pdiff import view_from_trace
from analysis.pdiff.vmeasure import (
    bootstrap_stability,
    cluster_edits_by_vocab,
    run_vmeasure_battery,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "output" / "pdiff_smoke_test"
RESOLVED_LITE = PROJECT_ROOT / "output" / "resolved_traces_lite_full.jsonl"

SAMPLE_N = 200
K = 10


def _repo_of(trace: dict) -> str | None:
    return trace.get("repo") if isinstance(trace, dict) else None


def _size_bucket(view) -> str:
    n = len(view.edits)
    if n == 0:
        return "empty"
    if n <= 3:
        return "small"
    if n <= 8:
        return "medium"
    return "large"


def _module_of(view) -> str | None:
    return next(iter(view.modules), None)


def _first_edit_op(view) -> str | None:
    edits = sorted(view.edits)
    return edits[0] if edits else None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RESOLVED_LITE.exists():
        print(f"No resolved traces at {RESOLVED_LITE}")
        return 1

    traces: list[dict] = []
    with open(RESOLVED_LITE) as fh:
        for line in fh:
            if not line.strip():
                continue
            traces.append(json.loads(line))
            if len(traces) >= SAMPLE_N:
                break

    views = [view_from_trace(t) for t in traces]
    usable = [(t, v) for t, v in zip(traces, views, strict=True) if v.has_edits]
    if len(usable) < 10:
        print(f"Too few usable traces ({len(usable)})")
        return 1

    t_list = [t for t, _ in usable]
    v_list = [v for _, v in usable]
    predicted = cluster_edits_by_vocab(v_list, k=K)

    references = {
        "repo": [_repo_of(t) for t in t_list],
        "patch_size_bucket": [_size_bucket(v) for v in v_list],
        "module": [_module_of(v) for v in v_list],
        "first_edit_op": [_first_edit_op(v) for v in v_list],
    }

    print(f"=== V-measure battery ===\nn usable traces: {len(usable)}\n")

    table = run_vmeasure_battery(predicted, references)
    print(table.to_string(index=False))
    print()

    boot = bootstrap_stability(predicted, references["repo"], n_bootstrap=50)
    print(f"Bootstrap ARI vs repo: mean={boot['mean_ari']:.4f} "
          f"std={boot['std_ari']:.4f} (n={boot['n_bootstrap']})")

    results = {
        "n_traces": len(usable),
        "k": K,
        "table": table.to_dict(orient="records"),
        "bootstrap_repo": boot,
    }
    (OUT_DIR / "vmeasure.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {OUT_DIR / 'vmeasure.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
