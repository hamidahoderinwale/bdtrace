"""Difficulty vs fix type as predictors of pass rate (9-agent extended corpus).

Two-panel comparison: same x-axis, same CI method (Wilson 95%).
Left: pass rate by difficulty bucket (n_resolved / 9 agents).
Right: pass rate by fix type taxonomy.

The visual contrast is the finding: difficulty spans 0--100% with
non-overlapping CIs; fix type clusters with heavily overlapping CIs.

Reads:  output/datasets/swe_bench_lite_resolved/fix_types.json
        output/paper2_pilot/extended_pass_fail.json (via _extended_pass_fail_df)
        output/paper2_pilot/bpe_sequences_extended.jsonl
Writes: output/figures/fig_difficulty_vs_fixtype.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, COPPER, GRAY
register()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _extended_pass_fail_df import load_extended_traj_pass_fail

OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MIN_N = 10

FIX_TYPE_LABELS = {
    "logic_fix":          "Logic fix",
    "exception_handling": "Exception handling",
    "api_change":         "API change",
    "config_fix":         "Config fix",
    "guard_clause":       "Guard clause",
    "refactor":           "Refactor",
    "string_fix":         "String fix",
    "test_fix":           "Test fix",
    "type_coercion":      "Type coercion",
    "import_fix":         "Import fix",
    "other":              "Other",
}

DIFFICULTY_LABELS = {n: f"{n} / 9" for n in range(10)}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    mean = k / n
    denom = 1 + z**2 / n
    centre = (mean + z**2 / (2 * n)) / denom
    half = z * np.sqrt(mean * (1 - mean) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mutual_info(df: pd.DataFrame, group_col: str) -> tuple[float, float]:
    """Returns (MI in bits, H(Y) in bits)."""
    n = len(df)
    p_pass = df["passed"].mean()
    def _h(p):
        if p <= 0 or p >= 1:
            return 0.0
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    h_y = _h(p_pass)
    h_y_given_x = 0.0
    for _, sub in df.groupby(group_col):
        p_g = len(sub) / n
        h_y_given_x += p_g * _h(sub["passed"].mean())
    return h_y - h_y_given_x, h_y


def build_panel(
    rows: list[dict],
    order: list[str],
    title: str,
    color: str,
) -> alt.Chart:
    plot_df = pd.DataFrame(rows)

    x_enc = alt.X(
        "mean:Q",
        title="Pass rate",
        scale=alt.Scale(domain=[-0.03, 1.0]),
        axis=alt.Axis(format=".0%", values=[0, 0.25, 0.5, 0.75, 1.0]),
    )

    # Alternating row background bands
    stripe_df = pd.DataFrame({
        "label": order,
        "fill": ["#F5F5F5" if i % 2 == 0 else "white" for i in range(len(order))],
    })
    stripes = (
        alt.Chart(stripe_df)
        .mark_rect()
        .encode(
            y=alt.Y("label:N", sort=order, axis=alt.Axis(title=None)),
            color=alt.Color("fill:N", scale=None, legend=None),
        )
    )

    base = alt.Chart(plot_df).encode(
        y=alt.Y("label:N", sort=order, axis=alt.Axis(title=None)),
        color=alt.value(color),
    )

    points = base.mark_point(filled=True, size=80, strokeWidth=0).encode(x=x_enc)

    errors = base.mark_errorbar().encode(
        x=alt.X("lo:Q", title="Pass rate", scale=alt.Scale(domain=[-0.03, 1.0])),
        x2=alt.X2("hi:Q"),
    )

    # Phantom points force x-axis to span full range.
    phantom = (
        alt.Chart(pd.DataFrame({"x": [-0.03, 1.0]}))
        .mark_point(opacity=0)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-0.03, 1.0]),
                    axis=alt.Axis(format=".0%", values=[0, 0.25, 0.5, 0.75, 1.0])),
            y=alt.value(0),
        )
    )

    return (
        (stripes + points + errors + phantom)
        .resolve_scale(x="shared")
        .properties(
            width=280,
            height=200,
            title=alt.TitleParams(title, fontSize=12, color="#111111", anchor="start"),
        )
    )


def main() -> None:
    # Per-instance fix type (canonical fix_types.json; 300 of 300 covered).
    ft_data = json.loads(
        (ROOT / "output/datasets/swe_bench_lite_resolved/fix_types.json").read_text()
    )
    fix_df = pd.DataFrame([
        {"instance_id": r["instance_id"], "fix_type_hand": r["fix_type"]}
        for r in ft_data["results"]
    ]).drop_duplicates()

    # 9-agent pass/fail and n_resolved already computed by the helper.
    traj_df = load_extended_traj_pass_fail()[["instance_id", "agent", "passed", "n_resolved"]]
    df = traj_df.merge(fix_df, on="instance_id", how="inner")
    df["passed"] = df["passed"].astype(float)

    # ── mutual information values ─────────────────────────────────────────────
    mi_d, h_y = mutual_info(df, "n_resolved")
    valid_ft = [
        k for k in df["fix_type_hand"].unique()
        if df["fix_type_hand"].eq(k).sum() >= MIN_N
    ]
    df_ft = df[df["fix_type_hand"].isin(valid_ft)]
    mi_f, _ = mutual_info(df_ft, "fix_type_hand")

    # ── Panel A: difficulty ───────────────────────────────────────────────────
    diff_rows = []
    diff_order = []
    for nr in sorted(df["n_resolved"].unique()):
        sub = df[df["n_resolved"] == nr]
        k = int(sub["passed"].sum())
        n = len(sub)
        lo, hi = wilson_ci(k, n)
        label = DIFFICULTY_LABELS[nr]
        diff_rows.append({"label": label, "mean": k / n, "lo": lo, "hi": hi, "n": n})
        diff_order.append(label)

    panel_a = build_panel(
        diff_rows,
        diff_order,
        "Difficulty bucket",
        BLUE,
    )

    # ── Panel B: fix type ─────────────────────────────────────────────────────
    ft_rows = []
    for ft in valid_ft:
        sub = df_ft[df_ft["fix_type_hand"] == ft]
        k = int(sub["passed"].sum())
        n = len(sub)
        lo, hi = wilson_ci(k, n)
        ft_rows.append({
            "label": FIX_TYPE_LABELS.get(ft, ft),
            "mean": k / n,
            "lo": lo,
            "hi": hi,
            "n": n,
        })

    ft_order = (
        pd.DataFrame(ft_rows)
        .sort_values("mean", ascending=True)["label"]
        .tolist()
    )

    panel_b = build_panel(
        ft_rows,
        ft_order,
        "Pass rate by fix type",
        COPPER,
    )

    for panel, name in [(panel_a, "fig_difficulty.png"), (panel_b, "fig_fixtype.png")]:
        out = OUT / name
        (
            panel
            .configure_view(strokeWidth=0)
            .configure_axis(grid=False)
        ).save(str(out), scale_factor=2)
        print(f"Saved {out}")
    print(f"  H(Y) = {h_y:.4f} bits (pass rate = {df['passed'].mean():.3f})")
    print(f"  MI(difficulty; outcome) = {mi_d:.4f} bits = {mi_d/h_y:.1%} of H(Y)")
    print(f"  MI(fix_type;   outcome) = {mi_f:.4f} bits = {mi_f/h_y:.1%} of H(Y)")
    print(f"  Ratio: {mi_d/mi_f:.1f}x")


if __name__ == "__main__":
    main()
