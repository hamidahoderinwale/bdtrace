"""Repo ease figure: horizontal dot plot with fix-signal annotations.

Produces both interactive HTML and static PNG.
Usage:
    python -m scripts.paper_repo_ease
"""

from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.theme import register, GREEN, ORANGE, GRAY, NEAR_BLACK

register()

OUT = PROJECT_ROOT / "output" / "paper2_pilot"

FIX_SIGNAL = {
    "requests":     "run and verify",
    "scikit-learn": "run and verify",
    "django":       "run and verify",
    "pytest":       "run and verify",
    "seaborn":      "run and verify",   # exception-throwing even for visual bugs
    "astropy":      "run and verify",
    "sympy":        "inspect to verify",
    "xarray":       "run and verify",
    "matplotlib":   "inspect to verify",
    "pylint":       "inspect to verify",
    "sphinx":       "inspect to verify",
    "flask":        "run and verify",
}

# Short callout text for outlier repos shown in the static figure
CALLOUTS = {
    "sphinx":     "docs rendering",
    "matplotlib": "visual output",
    "sympy":      "symbolic math",
    "pylint":     "static analysis output",
}


def build_df() -> pd.DataFrame:
    ids   = json.load(open(PROJECT_ROOT / "output/issue_embeddings/instance_ids.json"))
    ease  = json.load(open(PROJECT_ROOT / "output/issue_embeddings/ease_scores.json"))
    repos = [r.split("/")[-1]
             for r in json.load(open(PROJECT_ROOT / "output/issue_embeddings/repos.json"))]

    repo_data: dict[str, list] = defaultdict(list)
    repo_ids:  dict[str, list] = defaultdict(list)
    for iid, repo in zip(ids, repos):
        repo_data[repo].append(ease[iid])
        repo_ids[repo].append(iid)

    rows = []
    for repo, scores in repo_data.items():
        rows.append({
            "repo":        repo,
            "repo_label":  f"{repo}  (n={len(scores)})",
            "ease":        float(np.mean(scores)),
            "n":           len(scores),
            "n_resolved":  int(sum(1 for s in scores if s > 0)),
            "fix_signal":  FIX_SIGNAL.get(repo, "run and verify"),
            "callout":     CALLOUTS.get(repo, ""),
            "examples":    ", ".join(repo_ids[repo][:3]),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("ease", ascending=False).reset_index(drop=True)
    return df


def make_chart(df: pd.DataFrame, overall_mean: float) -> alt.LayerChart:
    label_order = df["repo_label"].tolist()

    color_scale = alt.Scale(
        domain=["run and verify", "inspect to verify"],
        range=[GREEN, ORANGE],
    )

    # Main dots — fixed size, no size encoding
    dots = (
        alt.Chart(df)
        .mark_circle(size=130, opacity=0.92)
        .encode(
            y=alt.Y("repo_label:N", sort=label_order,
                    axis=alt.Axis(title=None, labelFontSize=11)),
            x=alt.X("ease:Q",
                    scale=alt.Scale(domain=[-0.015, 0.38]),
                    axis=alt.Axis(
                        title="Mean ease (fraction of agents that resolved)",
                        titleFontSize=11,
                        format=".0%",
                        values=[0, 0.1, 0.2, 0.3],
                        grid=False,
                    )),
            color=alt.Color("fix_signal:N",
                            scale=color_scale,
                            legend=alt.Legend(
                                title=None,
                                orient="bottom",
                                direction="horizontal",
                                labelFontSize=11,
                                symbolSize=80,
                            )),
            tooltip=[
                alt.Tooltip("repo:N",        title="Repo"),
                alt.Tooltip("ease:Q",         title="Mean ease", format=".1%"),
                alt.Tooltip("n:Q",            title="Instances"),
                alt.Tooltip("n_resolved:Q",   title="Resolved by any agent"),
                alt.Tooltip("fix_signal:N",   title="Fix signal"),
                alt.Tooltip("examples:N",     title="Example instances"),
            ],
        )
    )

    # Percentage labels to the right of each dot
    pct_labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=10, fontSize=10, color="#666666")
        .encode(
            y=alt.Y("repo_label:N", sort=label_order),
            x=alt.X("ease:Q"),
            text=alt.Text("ease:Q", format=".0%"),
        )
    )

    # Callout annotations for inspect-to-verify outliers
    callout_df = df[df["callout"] != ""].copy()
    callouts = (
        alt.Chart(callout_df)
        .mark_text(align="left", dx=42, fontSize=9,
                   color="#999999", fontStyle="italic")
        .encode(
            y=alt.Y("repo_label:N", sort=label_order),
            x=alt.X("ease:Q"),
            text="callout:N",
        )
    )

    return (
        alt.layer(dots, pct_labels, callouts)
        .properties(
            title=alt.TitleParams(
                "Task resolution rate by repository",
                fontSize=13, fontWeight="normal",
                anchor="start", color=NEAR_BLACK,
            ),
            width=500,
            height=340,
        )
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_df()
    overall_mean = df["ease"].mul(df["n"]).sum() / df["n"].sum()

    chart = make_chart(df, overall_mean)

    html_path = OUT / "repo_ease.html"
    chart.save(str(html_path))
    print(f"Saved interactive: {html_path}")

    png_path = OUT / "repo_ease.png"
    try:
        import vl_convert as vlc
        png = vlc.vegalite_to_png(chart.to_json(), scale=2)
        png_path.write_bytes(png)
        print(f"Saved static:      {png_path}")
    except ImportError:
        print("vl_convert not available — HTML only")


if __name__ == "__main__":
    main()
