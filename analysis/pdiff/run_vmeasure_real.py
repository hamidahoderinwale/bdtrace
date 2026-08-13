"""V-measure on real edit-certificate clusters (canonical forms).

Unlike `run_vmeasure_smoke` which uses a k-means-on-edit-vocab baseline,
this loads the existing canonical-form assignments from
``output/canonical_forms/instance_assignments.parquet`` and tests whether
those clusters align with repo / patch-size / module references.

Canonical forms were chosen over fix_forms / intent_forms / hunk_clusters
because they have the richest partition (69 distinct labels on 289 Lite
instances) and the cleanest human-readable names. The other three
sources are documented in the handoff report for future comparison.

Writes output/pdiff_smoke_test/vmeasure_real.json.

Usage:
    python -m analysis.pdiff.run_vmeasure_real
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.pdiff import view_from_trace
from analysis.pdiff.vmeasure import (
    bootstrap_stability,
    headline_table,
    run_vmeasure_battery,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "output" / "pdiff_smoke_test"
RESOLVED_LITE = PROJECT_ROOT / "output" / "resolved_traces_lite_full.jsonl"
ASSIGNMENTS = PROJECT_ROOT / "output" / "canonical_forms" / "instance_assignments.parquet"


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


def _load_lite_traces() -> dict[str, dict]:
    traces: dict[str, dict] = {}
    with open(RESOLVED_LITE) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            iid = d.get("instance_id")
            if iid:
                traces[iid] = d
    return traces


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RESOLVED_LITE.exists():
        print(f"missing {RESOLVED_LITE}")
        return 1
    if not ASSIGNMENTS.exists():
        print(f"missing {ASSIGNMENTS}")
        return 1

    assigns = pd.read_parquet(ASSIGNMENTS)
    assigns = assigns[assigns["assigned"].astype(bool)].copy()
    print(f"canonical-form assignments: {len(assigns)} rows, "
          f"{assigns['form_name'].nunique()} distinct forms")

    traces = _load_lite_traces()
    print(f"Lite traces loaded: {len(traces)}")

    rows = []
    for _, row in assigns.iterrows():
        iid = row["instance_id"]
        trace = traces.get(iid)
        if trace is None:
            continue
        view = view_from_trace(trace)
        if not view.has_edits:
            continue
        rows.append({
            "instance_id": iid,
            "cluster": row["form_name"],
            "repo": trace.get("repo"),
            "patch_size_bucket": _size_bucket(view),
            "module": _module_of(view),
        })

    if len(rows) < 10:
        print(f"too few joined rows ({len(rows)}); aborting")
        return 1

    df = pd.DataFrame(rows)
    print(f"joined rows with edits: {len(df)}")
    print(f"distinct clusters in join: {df['cluster'].nunique()}")

    predicted = df["cluster"].tolist()
    references = {
        "repo": df["repo"].tolist(),
        "patch_size_bucket": df["patch_size_bucket"].tolist(),
        "module": df["module"].tolist(),
    }

    print()
    print("=== pdiff V-measure on canonical-form clusters ===")
    print("ARI near 0 means predicted clustering is no more aligned with reference")
    print("than chance; ARI near 1 means identical partitions.")
    print()
    print(f"n: {len(df)}  n_clusters: {df['cluster'].nunique()}")
    print()

    headline = headline_table(predicted, references)
    print("Headline (ARI + V-measure):")
    print(headline.to_string(index=False))
    print()

    full = run_vmeasure_battery(predicted, references)

    boot = bootstrap_stability(predicted, references["repo"], n_bootstrap=100)
    print(f"Bootstrap ARI vs repo (n={boot['n_bootstrap']}): "
          f"mean={boot['mean_ari']:.4f} std={boot['std_ari']:.4f} "
          f"min={boot.get('min_ari', float('nan')):.4f} "
          f"max={boot.get('max_ari', float('nan')):.4f}")

    results = {
        "cluster_source": str(ASSIGNMENTS.relative_to(PROJECT_ROOT)),
        "n_joined": len(df),
        "n_clusters": int(df["cluster"].nunique()),
        "headline": headline.to_dict(orient="records"),
        "table": full.to_dict(orient="records"),
        "bootstrap_repo": boot,
    }
    out_path = OUT_DIR / "vmeasure_real.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
