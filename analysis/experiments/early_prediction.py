"""Early trajectory prediction: can you predict resolution from first N steps?

Uses action type counts from the first N steps as features for logistic
regression. Reports AUC vs N steps with cross-validation confidence bands.

Outputs:
    output/experiments/early_prediction.json
    output/experiments/early_prediction_auc.png
"""
from __future__ import annotations
import json, sys
import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, ORANGE, GREEN
register()

OUT = ROOT / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)

AGENT_COLORS = {"Claude-3.5": GREEN, "GPT-4": BLUE, "GPT-4o": ORANGE}
AGENT_SHORT = {
    "20240402_sweagent_gpt4": "GPT-4",
    "20240620_sweagent_claude3.5sonnet": "Claude-3.5",
    "20240728_sweagent_gpt4o": "GPT-4o",
}
N_STEPS_LIST = [3, 5, 10, 15, 20, 30, 50]


def build_features(sequences: list[str], n_steps: int, vocab: list[str]) -> np.ndarray:
    rows = []
    for seq in sequences:
        acts = seq.split()[:n_steps]
        c = Counter(acts)
        rows.append([c.get(v, 0) / max(len(acts), 1) for v in vocab])
    return np.array(rows)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    df = pd.read_parquet(ROOT / "output/trajectories/lite_all_models.parquet")
    df["agent_short"] = df["model_id"].map(AGENT_SHORT)
    df = df.dropna(subset=["agent_short"])

    # Build vocabulary from all action types
    all_acts: list[str] = []
    for seq in df["action_sequence"]:
        all_acts.extend(str(seq).split())
    vocab = sorted(set(all_acts) - {"SUBMIT", "OTHER"})
    print(f"Action vocabulary: {vocab}")

    results = []
    for agent_short in ["Claude-3.5", "GPT-4", "GPT-4o"]:
        sub = df[df["agent_short"] == agent_short].copy()
        y = sub["passed"].astype(int).values
        seqs = sub["action_sequence"].tolist()
        print(f"\n{agent_short}: {len(sub)} trajectories, {y.mean():.1%} resolved")

        # Full trajectory baseline
        X_full = build_features(seqs, 999, vocab)
        clf = LogisticRegression(max_iter=500, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_full = cross_val_score(clf, X_full, y, cv=cv, scoring="roc_auc")

        agent_rows = []
        for n in N_STEPS_LIST:
            X = build_features(seqs, n, vocab)
            aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
            agent_rows.append({
                "agent": agent_short,
                "n_steps": n,
                "auc_mean": float(aucs.mean()),
                "auc_std": float(aucs.std()),
                "auc_lo": float(aucs.mean() - 1.96 * aucs.std()),
                "auc_hi": float(aucs.mean() + 1.96 * aucs.std()),
            })
            print(f"  N={n:3d}: AUC={aucs.mean():.3f} +/- {aucs.std():.3f}")

        # Add full trajectory point
        max_steps = int(sub["n_steps"].max())
        agent_rows.append({
            "agent": agent_short,
            "n_steps": max_steps,
            "auc_mean": float(auc_full.mean()),
            "auc_std": float(auc_full.std()),
            "auc_lo": float(auc_full.mean() - 1.96 * auc_full.std()),
            "auc_hi": float(auc_full.mean() + 1.96 * auc_full.std()),
        })
        results.extend(agent_rows)

    (OUT / "early_prediction.json").write_text(json.dumps(results, indent=2))

    # --- Plot: AUC vs N steps, one line per agent ---
    plot_df = pd.DataFrame(results)
    agent_order = ["Claude-3.5", "GPT-4", "GPT-4o"]
    cscale = alt.Scale(
        domain=agent_order,
        range=[AGENT_COLORS[a] for a in agent_order],
    )

    band = (
        alt.Chart(plot_df)
        .mark_area(opacity=0.12)
        .encode(
            x=alt.X("n_steps:Q",
                    title="Steps used for prediction",
                    scale=alt.Scale(domain=[0, plot_df["n_steps"].max() + 5]),
                    axis=alt.Axis(domain=False, ticks=False,
                                  values=[0, 10, 20, 30, 40, 50])),
            y=alt.Y("auc_lo:Q", scale=alt.Scale(domain=[0.4, 1.0])),
            y2="auc_hi:Q",
            color=alt.Color("agent:N", scale=cscale, legend=None),
        )
    )
    line = (
        alt.Chart(plot_df)
        .mark_line(strokeWidth=2)
        .encode(
            x="n_steps:Q",
            y=alt.Y("auc_mean:Q",
                    title="AUC (5-fold CV)",
                    scale=alt.Scale(domain=[0.4, 1.0]),
                    axis=alt.Axis(domain=False, ticks=False,
                                  values=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])),
            color=alt.Color("agent:N", scale=cscale,
                            legend=alt.Legend(title=None, orient="bottom")),
        )
    )
    pts = (
        alt.Chart(plot_df)
        .mark_point(size=50, filled=True, strokeWidth=0)
        .encode(
            x="n_steps:Q",
            y="auc_mean:Q",
            color=alt.Color("agent:N", scale=cscale, legend=None),
        )
    )
    chart = (
        (band + line + pts)
        .properties(
            title=alt.TitleParams(
                "Resolution prediction from first N steps",
                subtitle="Shaded band: 95% CI; rightmost point is full trajectory",
                subtitleFontSize=10, subtitleColor="#888888",
                fontSize=13, color="#111111", anchor="start",
            ),
            width=420, height=240,
        )
        .configure_view(strokeWidth=0)
    )
    chart.save(str(OUT / "early_prediction_auc.png"), scale_factor=2)
    print("\nSaved early_prediction_auc.png")


if __name__ == "__main__":
    main()
