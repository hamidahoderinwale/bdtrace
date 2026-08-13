"""Confound analysis: has_reproducer and PyPI download proxy.

Tests two competing explanations for repo ease variation beyond fix-signal:
  1. Whether the issue includes a reproducing test case (has_reproducer)
  2. Library popularity as a proxy for training data presence (PyPI downloads)

Outputs:
    output/experiments/confounds.json
    output/experiments/reproducer_ease.png
    output/experiments/pypi_ease.png
"""
from __future__ import annotations
import json, sys, urllib.request
import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, GREEN, ORANGE, BLUE, GRAY
register()

OUT = ROOT / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)

FIX_SIGNAL = {
    "requests": "run and verify", "scikit-learn": "run and verify",
    "django": "run and verify", "pytest": "run and verify",
    "seaborn": "run and verify", "astropy": "run and verify",
    "xarray": "run and verify", "flask": "run and verify",
    "sympy": "inspect to verify", "matplotlib": "inspect to verify",
    "pylint": "inspect to verify", "sphinx": "inspect to verify",
}

PYPI_PACKAGES = {
    "requests": "requests", "scikit-learn": "scikit-learn",
    "django": "Django", "pytest": "pytest", "seaborn": "seaborn",
    "astropy": "astropy", "xarray": "xarray", "flask": "Flask",
    "sympy": "sympy", "matplotlib": "matplotlib", "pylint": "pylint",
    "sphinx": "Sphinx",
}


def fetch_pypi_downloads(package: str) -> int | None:
    try:
        url = f"https://pypistats.org/api/packages/{package.lower()}/recent"
        req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
        with urllib.request.urlopen(req, timeout=8) as r:
            d = json.loads(r.read())
        return d["data"]["last_month"]
    except Exception:
        return None


def main():
    # Load ease scores and repo labels
    ease = json.loads((ROOT / "output/issue_embeddings/ease_scores.json").read_text())
    ids  = json.loads((ROOT / "output/issue_embeddings/instance_ids.json").read_text())
    repos_raw = json.loads((ROOT / "output/issue_embeddings/repos.json").read_text())
    repo_map = {iid: r.split("/")[-1] for iid, r in zip(ids, repos_raw)}

    # Load SWE-bench Lite for has_reproducer
    print("Loading SWE-bench Lite for has_reproducer...")
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    reproducer_map = {}
    for r in ds:
        iid = r["instance_id"]
        # hint_text or problem_statement contains a test case if has_reproducer
        has_rep = bool(r.get("hints_text", "").strip())
        reproducer_map[iid] = has_rep

    # Build per-instance rows
    rows = []
    for iid in ids:
        repo = repo_map.get(iid, "")
        rows.append({
            "instance_id": iid,
            "repo": repo,
            "ease": ease.get(iid, 0.0),
            "has_reproducer": reproducer_map.get(iid, False),
            "fix_signal": FIX_SIGNAL.get(repo, "run and verify"),
        })
    df = pd.DataFrame(rows)

    # Per-repo summary
    repo_df = (
        df.groupby("repo")
        .agg(
            ease=("ease", "mean"),
            has_reproducer_frac=("has_reproducer", "mean"),
            n=("ease", "count"),
            fix_signal=("fix_signal", "first"),
        )
        .reset_index()
    )

    # PyPI downloads
    print("Fetching PyPI download counts...")
    downloads = {}
    for repo, pkg in PYPI_PACKAGES.items():
        d = fetch_pypi_downloads(pkg)
        downloads[repo] = d
        print(f"  {repo}: {d}")

    repo_df["pypi_downloads"] = repo_df["repo"].map(downloads)
    repo_df["pypi_downloads_m"] = repo_df["pypi_downloads"].apply(
        lambda x: x / 1e6 if x else None
    )

    (OUT / "confounds.json").write_text(
        json.dumps(repo_df.to_dict(orient="records"), indent=2, default=str)
    )

    # --- Plot 1: ease by has_reproducer status ---
    rep_rows = []
    for has_rep in [True, False]:
        sub = df[df["has_reproducer"] == has_rep]
        rep_rows.append({
            "has_reproducer": "Has reproducer" if has_rep else "No reproducer",
            "ease": sub["ease"].mean(),
            "zero": 0.0,
            "n": len(sub),
        })
    rep_df = pd.DataFrame(rep_rows)
    rep_order = ["Has reproducer", "No reproducer"]
    cscale = alt.Scale(domain=rep_order, range=[BLUE, GRAY])

    rule = (
        alt.Chart(rep_df)
        .mark_rule(strokeWidth=2.5, opacity=0.55)
        .encode(
            y=alt.Y("has_reproducer:N", sort=rep_order,
                    axis=alt.Axis(title=None, domain=False, ticks=False, labelFontSize=12)),
            x=alt.X("zero:Q", scale=alt.Scale(domain=[0, 0.25]),
                    axis=alt.Axis(title="Mean ease (fraction of agents resolving)",
                                  domain=False, ticks=False, format=".0%",
                                  values=[0, 0.05, 0.10, 0.15, 0.20])),
            x2="ease:Q",
            color=alt.Color("has_reproducer:N", scale=cscale, legend=None),
        )
    )
    pts = (
        alt.Chart(rep_df)
        .mark_point(size=120, filled=True, strokeWidth=1.5, stroke="white")
        .encode(
            y=alt.Y("has_reproducer:N", sort=rep_order),
            x=alt.X("ease:Q", scale=alt.Scale(domain=[0, 0.25])),
            color=alt.Color("has_reproducer:N", scale=cscale, legend=None),
        )
    )
    labels = (
        alt.Chart(rep_df)
        .mark_text(align="left", dx=10, fontSize=11, color="#444444")
        .encode(
            y=alt.Y("has_reproducer:N", sort=rep_order),
            x=alt.X("ease:Q", scale=alt.Scale(domain=[0, 0.25])),
            text=alt.Text("ease:Q", format=".1%"),
        )
    )
    rep_chart = (
        (rule + pts + labels)
        .properties(
            title=alt.TitleParams("Ease by presence of reproducing test case",
                                  fontSize=13, color="#111111", anchor="start"),
            width=320, height=100,
        )
        .configure_view(strokeWidth=0)
    )
    rep_chart.save(str(OUT / "reproducer_ease.png"), scale_factor=2)
    print("Saved reproducer_ease.png")

    # --- Plot 2: PyPI downloads vs ease (repo-level) ---
    pypi_df = repo_df.dropna(subset=["pypi_downloads_m"]).copy()
    cscale2 = alt.Scale(
        domain=["run and verify", "inspect to verify"],
        range=[GREEN, ORANGE],
    )
    scatter = (
        alt.Chart(pypi_df)
        .mark_circle(size=100, opacity=0.85)
        .encode(
            x=alt.X("pypi_downloads_m:Q",
                    title="PyPI downloads last month (millions)",
                    axis=alt.Axis(domain=False, ticks=False)),
            y=alt.Y("ease:Q",
                    title="Mean ease",
                    scale=alt.Scale(domain=[-0.01, 0.38]),
                    axis=alt.Axis(domain=False, ticks=False, format=".0%",
                                  values=[0, 0.1, 0.2, 0.3])),
            color=alt.Color("fix_signal:N", scale=cscale2,
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["repo:N", alt.Tooltip("ease:Q", format=".1%"),
                     alt.Tooltip("pypi_downloads_m:Q", format=".1f")],
        )
    )
    repo_labels = (
        alt.Chart(pypi_df)
        .mark_text(align="left", dx=7, fontSize=9, color="#666666")
        .encode(
            x="pypi_downloads_m:Q",
            y="ease:Q",
            text="repo:N",
        )
    )
    pypi_chart = (
        (scatter + repo_labels)
        .properties(
            title=alt.TitleParams("Library popularity vs task ease",
                                  fontSize=13, color="#111111", anchor="start"),
            width=400, height=260,
        )
        .configure_view(strokeWidth=0)
    )
    pypi_chart.save(str(OUT / "pypi_ease.png"), scale_factor=2)
    print("Saved pypi_ease.png")

    # Correlation stats
    valid = repo_df.dropna(subset=["pypi_downloads"])
    r = np.corrcoef(valid["pypi_downloads"], valid["ease"])[0, 1]
    print(f"\nPyPI downloads vs ease Pearson r = {r:.3f} (n={len(valid)} repos)")
    print(f"\nHas reproducer: mean ease = {df[df['has_reproducer']]['ease'].mean():.3f}")
    print(f"No reproducer:  mean ease = {df[~df['has_reproducer']]['ease'].mean():.3f}")


if __name__ == "__main__":
    main()
