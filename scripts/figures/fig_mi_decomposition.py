"""Mutual information decomposition: 3 predictors x 3 outcomes.

Shows how much of each outcome's uncertainty is removed by knowing
the predictor. Agent identity predicts behavior; difficulty predicts pass/fail.

Predictors:
    agent      — 4 categories (model identity)
    difficulty — n_resolved: how many of the 4 agents passed (0-4)
    fix_type   — fix_type_hand (categories with n >= 5)

Outcomes:
    passed          — binary pass/fail
    length_bin      — canonical_length quartile
    compression_bin — compression quartile

Reads:
    output/paper2_pilot/bpe_sequences.jsonl
    output/trajectories/lite_all_models.parquet
    output/fix_forms/form_assignments.parquet
Writes:
    output/figures/fig_mi_decomposition.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, GREEN, BLUE, COPPER, AGENT_SHORT

register()

BPE_FILE   = ROOT / "output" / "paper2_pilot" / "bpe_sequences.jsonl"
LITE_FILE  = ROOT / "output" / "trajectories" / "lite_all_models.parquet"
FORMS_FILE = ROOT / "output" / "fix_forms" / "form_assignments.parquet"
FIG_OUT    = ROOT / "output" / "figures"

# Predictor display colors
PRED_COLORS = {
    "Agent":      GREEN,    # #20A380
    "Difficulty": BLUE,     # #5692E5
    "Fix type":   COPPER,   # #CB4D20
}
PRED_ORDER = ["Agent", "Difficulty", "Fix type"]


# ── MI utility ────────────────────────────────────────────────────────────────

def mi_bits(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[float, float]:
    """Return (MI in bits, H(Y) in bits) for categorical X and categorical Y."""
    n = len(df)

    def h(series: pd.Series) -> float:
        vc = series.value_counts(normalize=True)
        return float(-(vc * np.log2(vc + 1e-15)).sum())

    h_y = h(df[y_col])
    h_y_given_x = 0.0
    for _, sub in df.groupby(x_col, observed=True):
        h_y_given_x += (len(sub) / n) * h(sub[y_col])
    mi = h_y - h_y_given_x
    return max(0.0, mi), h_y


# ── Data loading and joining ───────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    # BPE sequences: agent, instance_id, canonical_length, compression
    bpe_rows = []
    with BPE_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            bpe_rows.append({
                "agent":            rec["agent"],
                "instance_id":      rec["instance_id"],
                "canonical_length": rec.get("canonical_length"),
                "compression":      rec.get("compression"),
            })
    bpe_df = pd.DataFrame(bpe_rows)

    # Lite parquet: map model_id → agent short name, keep agent/instance_id/passed
    lite_df = pd.read_parquet(LITE_FILE)
    lite_df["agent"] = lite_df["model_id"].map(AGENT_SHORT)
    lite_df = lite_df[lite_df["agent"].notna()][["agent", "instance_id", "passed"]].copy()

    # Fix forms: instance_id, fix_type_hand (one row per instance)
    forms_df = pd.read_parquet(FORMS_FILE)[["instance_id", "fix_type_hand"]].drop_duplicates("instance_id")

    # Join BPE + lite on (agent, instance_id)
    df = bpe_df.merge(lite_df, on=["agent", "instance_id"], how="inner")

    # Join with fix forms on instance_id
    df = df.merge(forms_df, on="instance_id", how="inner")

    # n_resolved = number of agents that passed each instance (across all 4 agents)
    n_resolved = (
        lite_df.groupby("instance_id")["passed"]
        .sum()
        .astype(int)
        .rename("n_resolved")
    )
    df = df.merge(n_resolved, on="instance_id", how="left")

    # Drop rows with missing fix_type_hand or canonical_length/compression
    df = df.dropna(subset=["fix_type_hand", "canonical_length", "compression"])

    # Outcome bins
    df["length_bin"] = pd.qcut(
        df["canonical_length"], 4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    )
    df["compression_bin"] = pd.qcut(
        df["compression"], 4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    )

    # Predictor columns: ensure correct types
    df["agent"]      = df["agent"].astype(str)
    df["difficulty"] = df["n_resolved"].astype(str)   # categorical (0–4)
    df["fix_type"]   = df["fix_type_hand"].astype(str)
    df["passed_str"] = df["passed"].astype(str)        # "True"/"False" for MI

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    df = load_data()

    # Filter fix_type to categories with >= 5 occurrences
    ft_counts = df["fix_type"].value_counts()
    valid_ft  = ft_counts[ft_counts >= 5].index
    df_ft     = df[df["fix_type"].isin(valid_ft)].copy()

    # Outcome configurations: (column_name, display_name)
    outcomes = [
        ("passed_str",       "Pass/fail"),
        ("length_bin",       "Trajectory length"),
        ("compression_bin",  "Compression ratio"),
    ]
    # Outcome display order (top to bottom in chart)
    outcome_order = ["Pass/fail", "Trajectory length", "Compression ratio"]

    # Compute MI for all 9 combinations
    rows = []
    for col, label in outcomes:
        for pred_col, pred_label, subset_df in [
            ("agent",      "Agent",      df),
            ("difficulty", "Difficulty", df),
            ("fix_type",   "Fix type",   df_ft),
        ]:
            # Drop rows where outcome bin is NaN (qcut edge cases)
            sub = subset_df[[pred_col, col]].dropna()
            mi, hy = mi_bits(sub, pred_col, col)
            pct = (mi / hy * 100) if hy > 0 else 0.0
            rows.append({
                "outcome":   label,
                "predictor": pred_label,
                "mi_bits":   mi,
                "h_y":       hy,
                "pct":       round(pct, 1),
            })

    mi_df = pd.DataFrame(rows)

    # Y-offset values to separate 3 dots per outcome row
    offset_map = {"Agent": -12, "Difficulty": 0, "Fix type": 12}
    mi_df["yOffset"] = mi_df["predictor"].map(offset_map)
    mi_df["label"]   = mi_df["pct"].apply(lambda v: f"{v:.0f}%")

    color_scale = alt.Scale(
        domain=PRED_ORDER,
        range=[PRED_COLORS[p] for p in PRED_ORDER],
    )

    base = alt.Chart(mi_df).encode(
        y=alt.Y(
            "outcome:N",
            sort=outcome_order,
            axis=alt.Axis(title=None),
        ),
        yOffset=alt.YOffset("yOffset:Q"),
        color=alt.Color(
            "predictor:N",
            scale=color_scale,
            legend=alt.Legend(
                title=None,
                orient="bottom",
                columns=3,
            ),
        ),
    )

    dots = base.mark_point(filled=True, size=80).encode(
        x=alt.X(
            "pct:Q",
            title="Outcome uncertainty removed (%)",
            scale=alt.Scale(domain=[0, 70]),
        ),
    )

    labels = base.mark_text(
        align="left",
        dx=8,
        fontSize=10,
        color="#444444",
    ).encode(
        x=alt.X("pct:Q"),
        text="label:N",
        color=alt.value("#444444"),
    )

    chart = (
        alt.layer(dots, labels)
        .properties(
            width=320,
            height=160,
            title=alt.TitleParams(
                "Outcome uncertainty removed by predictor",
                fontSize=12,
                color="#111111",
                anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    out_fig = FIG_OUT / "fig_mi_decomposition.png"
    chart.save(str(out_fig), scale_factor=2)
    print(f"Saved {out_fig}")


if __name__ == "__main__":
    main()
