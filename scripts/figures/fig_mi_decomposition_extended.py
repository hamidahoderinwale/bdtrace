"""Phase 7: MI decomposition on the 8-submission extended corpus.

Computes MI(predictor; outcome) on the same 3x3 grid as the original
fig_mi_decomposition.py, but with agent now having 8 levels.

Predictors:
    agent      — 8 categories (5 SWE-agent + dars + agentless + moatless)
    difficulty — n_resolved across the 8 agents (0-8)
    fix_type   — fix_type_hand from output/fix_forms/form_assignments.parquet (n>=5)

Outcomes:
    passed          — binary pass/fail (resolved list per submission)
    length_bin      — canonical_length quartile (Q1-Q4)
    compression_bin — compression quartile (Q1-Q4)

Reads:
    output/paper2_pilot/bpe_sequences_extended.jsonl
    output/paper2_pilot/extended_pass_fail.json
    output/fix_forms/form_assignments.parquet
Writes:
    output/figures/fig_mi_decomposition_extended.png
    output/paper2_pilot/mi_decomposition_extended.json

Usage:
    python -m scripts.figures.fig_mi_decomposition_extended
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, GREEN, BLUE, COPPER
register()

OUT_FIG = ROOT / "output" / "figures"
OUT_DAT = ROOT / "output" / "paper2_pilot"

BPE_FILE = OUT_DAT / "bpe_sequences_extended.jsonl"
PASS_FILE = OUT_DAT / "extended_pass_fail.json"
FORMS_FILE = ROOT / "output" / "fix_forms" / "form_assignments.parquet"

# Submission ID -> short label  (match build_extended_bpe.SUBMISSION_LABEL)
SUBMISSION_LABEL = {
    "20240402_sweagent_claude3opus":                                   "Claude-3",
    "20240402_sweagent_gpt4":                                          "GPT-4",
    "20240620_sweagent_claude3.5sonnet":                               "Claude-3.5",
    "20240728_sweagent_gpt4o":                                         "GPT-4o",
    "20250226_sweagent_claude-3-7-sonnet-20250219":                    "Claude-3.7-thinking",
    "20250205_dars_agent_claude_3.5_sonnet_deepseek_r1":               "DARS+R1",
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022":               "Agentless+Claude-3.5",
    "20250111_moatless_deepseek_v3":                                   "Moatless+V3",
    "20250526_sweagent_claude-4-sonnet-20250514":                      "Claude-4",
}

PRED_COLORS = {"Agent": GREEN, "Difficulty": BLUE, "Fix type": COPPER}
PRED_ORDER = ["Agent", "Difficulty", "Fix type"]


def mi_bits(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[float, float]:
    n = len(df)

    def h(series: pd.Series) -> float:
        vc = series.value_counts(normalize=True)
        return float(-(vc * np.log2(vc + 1e-15)).sum())

    h_y = h(df[y_col])
    h_y_given_x = 0.0
    for _, sub in df.groupby(x_col, observed=True):
        h_y_given_x += (len(sub) / n) * h(sub[y_col])
    return max(0.0, h_y - h_y_given_x), h_y


def load_data() -> pd.DataFrame:
    # BPE sequences
    bpe_rows = []
    with BPE_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            bpe_rows.append({
                "submission":      rec["submission"],
                "agent":           rec["agent"],
                "instance_id":     rec["instance_id"],
                "canonical_length": rec.get("canonical_length"),
                "compression":     rec.get("compression"),
            })
    bpe_df = pd.DataFrame(bpe_rows)

    # Pass/fail per submission
    pass_data = json.loads(PASS_FILE.read_text())

    def is_resolved(row) -> bool:
        sub = row["submission"]
        info = pass_data.get(sub, {})
        return row["instance_id"] in set(info.get("resolved", []))

    bpe_df["passed"] = bpe_df.apply(is_resolved, axis=1)

    # Difficulty: how many of the 8 agents resolved each instance
    n_resolved = (
        bpe_df.groupby("instance_id")["passed"]
        .sum()
        .astype(int)
        .rename("n_resolved")
    )
    df = bpe_df.merge(n_resolved, on="instance_id", how="left")

    # Fix type
    forms_df = pd.read_parquet(FORMS_FILE)[["instance_id", "fix_type_hand"]].drop_duplicates("instance_id")
    df = df.merge(forms_df, on="instance_id", how="left")

    df = df.dropna(subset=["canonical_length", "compression"])

    df["length_bin"] = pd.qcut(df["canonical_length"], 4,
                               labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    df["compression_bin"] = pd.qcut(df["compression"], 4,
                                    labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")

    df["agent"] = df["agent"].astype(str)
    df["difficulty"] = df["n_resolved"].astype(str)
    df["fix_type"] = df["fix_type_hand"].astype(str)
    df["passed_str"] = df["passed"].astype(str)
    return df


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_DAT.mkdir(parents=True, exist_ok=True)

    df = load_data()

    print(f"loaded n={len(df)} rows, agents={df['agent'].nunique()}, "
          f"instances={df['instance_id'].nunique()}")
    print("agents:", sorted(df['agent'].unique()))
    print("difficulty distribution (n_resolved across 8 agents):")
    print(df["difficulty"].value_counts().sort_index())

    ft_counts = df["fix_type"].value_counts()
    valid_ft = ft_counts[ft_counts >= 5].index
    df_ft = df[df["fix_type"].isin(valid_ft)].copy()

    outcomes = [
        ("passed_str", "Pass/fail"),
        ("length_bin", "Trajectory length"),
        ("compression_bin", "Compression ratio"),
    ]
    outcome_order = ["Pass/fail", "Trajectory length", "Compression ratio"]

    rows = []
    for col, label in outcomes:
        for pred_col, pred_label, sub_df in [
            ("agent", "Agent", df),
            ("difficulty", "Difficulty", df),
            ("fix_type", "Fix type", df_ft),
        ]:
            sub = sub_df[[pred_col, col]].dropna()
            mi, hy = mi_bits(sub, pred_col, col)
            pct = (mi / hy * 100) if hy > 0 else 0.0
            rows.append({
                "outcome": label,
                "predictor": pred_label,
                "mi_bits": mi,
                "h_y": hy,
                "pct": round(pct, 1),
                "n": len(sub),
            })

    mi_df = pd.DataFrame(rows)
    print("\nMI decomposition:")
    print(mi_df.to_string(index=False))

    offset_map = {"Agent": -12, "Difficulty": 0, "Fix type": 12}
    mi_df["yOffset"] = mi_df["predictor"].map(offset_map)
    mi_df["label"] = mi_df["pct"].apply(lambda v: f"{v:.0f}%")

    color_scale = alt.Scale(
        domain=PRED_ORDER,
        range=[PRED_COLORS[p] for p in PRED_ORDER],
    )

    base = alt.Chart(mi_df).encode(
        y=alt.Y("outcome:N", sort=outcome_order, axis=alt.Axis(title=None)),
        yOffset=alt.YOffset("yOffset:Q"),
        color=alt.Color("predictor:N", scale=color_scale,
                        legend=alt.Legend(title=None, orient="bottom", columns=3)),
    )

    dots = base.mark_point(filled=True, size=80).encode(
        x=alt.X("pct:Q",
                title="Outcome uncertainty removed (%)",
                scale=alt.Scale(domain=[0, max(70, mi_df["pct"].max() * 1.15)])),
    )

    labels = base.mark_text(align="left", dx=8, fontSize=10, color="#444444").encode(
        x=alt.X("pct:Q"),
        text="label:N",
        color=alt.value("#444444"),
    )

    chart = (
        alt.layer(dots, labels)
        .properties(
            width=320, height=180,
            title=alt.TitleParams(
                "Outcome uncertainty removed by predictor",
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    out_fig = OUT_FIG / "fig_mi_decomposition_extended.png"
    chart.save(str(out_fig), scale_factor=2)
    print(f"\nSaved {out_fig}")

    payload = {
        "n_rows": int(len(df)),
        "n_agents": int(df["agent"].nunique()),
        "n_instances": int(df["instance_id"].nunique()),
        "results": rows,
    }
    out_json = OUT_DAT / "mi_decomposition_extended.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
