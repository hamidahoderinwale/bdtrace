#!/usr/bin/env python3
"""Figure 3: V-measure headline dot plot on canonical-form clusters.

Horizontal dot plot: one row per reference partition (repo, module,
patch_size_bucket). X-axis is ARI with a dashed rule at zero. Dots
are placed at the point ARI estimate; horizontal bars show 95%
bootstrap CI (mean +/- 1.96 * std, n_bootstrap=100). V-measure is
annotated as small text at the right of each dot.

Rebuilds the joined predicted/reference columns the same way
run_vmeasure_real.py does, then calls bootstrap_stability for
each reference (not only repo, so the CI lines are consistent).

Reads:
  output/resolved_traces_lite_full.jsonl
  output/canonical_forms/instance_assignments.parquet
  output/pdiff_smoke_test/vmeasure_real.json  (for point estimates)

Writes:
  figures/procedural-diff/fig3_vmeasure_headline.{png,svg}

Usage:
    python -m scripts.figures.build_vmeasure_headline
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd

from analysis.pdiff import view_from_trace
from analysis.pdiff.vmeasure import bootstrap_stability, compute_metrics

ROOT = Path(__file__).resolve().parents[2]
LITE = ROOT / "output" / "resolved_traces_lite_full.jsonl"
ASSIGNMENTS = ROOT / "output" / "canonical_forms" / "instance_assignments.parquet"
OUT_DIR = ROOT / "figures" / "procedural-diff"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 100

REFERENCE_LABELS = {
    "repo": "Repo",
    "module": "Module",
    "patch_size_bucket": "Patch size bucket",
}


def _size_bucket(view) -> str:
    n = len(view.edits)
    if n == 0:
        return "empty"
    if n <= 3:
        return "small"
    if n <= 8:
        return "medium"
    return "large"


def _module_of(view):
    return next(iter(view.modules), None)


def _load_lite_traces() -> dict[str, dict]:
    traces: dict[str, dict] = {}
    with open(LITE) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            iid = d.get("instance_id")
            if iid:
                traces[iid] = d
    return traces


def _build_join() -> pd.DataFrame:
    assigns = pd.read_parquet(ASSIGNMENTS)
    assigns = assigns[assigns["assigned"].astype(bool)].copy()
    traces = _load_lite_traces()
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
    return pd.DataFrame(rows)


def build_chart(df: pd.DataFrame) -> alt.Chart:
    # Sort rows by ARI desc.
    df = df.sort_values("ari", ascending=False).reset_index(drop=True)
    order = df["reference"].tolist()

    base = alt.Chart(df).encode(
        y=alt.Y("reference:N", title=None, sort=order,
                axis=alt.Axis(labelFontSize=10)),
    )

    ci = base.mark_rule(strokeWidth=2, color="#666666").encode(
        x=alt.X("ci_low:Q", title="ARI (chance-corrected)",
                scale=alt.Scale(domain=[-0.2, 0.4])),
        x2="ci_high:Q",
    )

    dots = base.mark_point(size=120, filled=True, color="#0072B2").encode(
        x=alt.X("ari:Q"),
    )

    rule0 = alt.Chart(pd.DataFrame([{"x": 0.0}])).mark_rule(
        strokeDash=[4, 4], color="gray", opacity=0.6, strokeWidth=0.8,
    ).encode(x="x:Q")

    vlabels = base.mark_text(
        align="left", dx=8, fontSize=9, color="#333333",
    ).encode(
        x=alt.X("ci_high:Q"),
        text=alt.Text("v_label:N"),
    )

    chart = (rule0 + ci + dots + vlabels).properties(
        width=380,
        height=150,
        title=alt.TitleParams(
            text="Canonical-form clusters vs reference partitions",
            subtitle="Dots: ARI point estimate; bars: 95% bootstrap CI (n=100); text: V-measure",
            fontSize=11,
            subtitleFontSize=9,
            anchor="start",
        ),
    ).configure_axis(
        grid=False,
        labelFontSize=9,
        titleFontSize=10,
    ).configure_view(
        strokeWidth=0,
    )
    return chart


def main() -> int:
    print("Building join from canonical_forms + Lite traces...")
    df = _build_join()
    print(f"joined rows: {len(df)}  n_clusters: {df['cluster'].nunique()}")

    predicted = df["cluster"].tolist()
    rows = []
    for ref_key, ref_label in REFERENCE_LABELS.items():
        ref = df[ref_key].tolist()
        m = compute_metrics(predicted, ref, reference_name=ref_key)
        boot = bootstrap_stability(predicted, ref, n_bootstrap=N_BOOTSTRAP)
        mean = boot.get("mean_ari", float("nan"))
        std = boot.get("std_ari", float("nan"))
        ci_half = 1.96 * std if std == std else 0.0
        rows.append({
            "reference": ref_label,
            "ari": m.ari,
            "v_measure": m.v_measure,
            "v_label": f"V = {m.v_measure:.2f}",
            "ci_low": mean - ci_half,
            "ci_high": mean + ci_half,
            "boot_mean": mean,
            "boot_std": std,
        })

    out_df = pd.DataFrame(rows)
    print(out_df.to_string(index=False))

    chart = build_chart(out_df)
    png = OUT_DIR / "fig3_vmeasure_headline.png"
    svg = OUT_DIR / "fig3_vmeasure_headline.svg"
    chart.save(str(png), scale_factor=2)
    chart.save(str(svg))
    print(f"Wrote {png}")
    print(f"Wrote {svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
