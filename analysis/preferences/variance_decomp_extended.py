"""9-agent variance decomposition. Same idiom as variance_decomp.py.

For each per-trajectory behavioral feature we compute one number per grouping:
the standard deviation of the group means. The ratios

    sd(task_means)     / sd(agent_means)   -> task structure vs agent identity
    sd(scaffold_means) / sd(agent_means)   -> agent variation captured by scaffold
    sd(paradigm_means) / sd(agent_means)   -> agent variation captured by paradigm

answer the question directly. No ANOVA, no F-tests, no eta-squared.

Two readings live in the same chart:
  - task / fix_type / repo groupings: BETWEEN-task vs BETWEEN-agent variation
    (task ratio > 1 means tasks vary more than agents)
  - scaffold / paradigm groupings: WITHIN-agent-set substructure
    (scaffold ratio close to 1 means agent spread aligns with scaffold class)

Per-trajectory features come from the canonical atom sequence in
bpe_sequences_extended.jsonl, so all 9 agents share the same feature definition.

Output:
  output/paper2_pilot/variance_decomposition_extended.json
  output/paper2_pilot/variance_decomposition_extended.csv
  output/paper2_pilot/variance_decomposition_extended.png

Usage:
    python -m analysis.preferences.variance_decomp_extended
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.theme import register, BLUE, COPPER, GREEN, MAGENTA, OLIVE
register()

OUT_DIR = PROJECT_ROOT / "output" / "paper2_pilot"
SEQ_PATH = OUT_DIR / "bpe_sequences_extended.jsonl"
FIX_TYPES_PATH = PROJECT_ROOT / "output" / "datasets" / "swe_bench_lite_resolved" / "fix_types.json"

AGENT_SCAFFOLD = {
    "Claude-3":              "SWE-agent",
    "Claude-3.5":            "SWE-agent",
    "Claude-3.7-thinking":   "SWE-agent",
    "Claude-4":              "SWE-agent",
    "GPT-4":                 "SWE-agent",
    "GPT-4o":                "SWE-agent",
    "DARS+R1":               "DARS",
    "Agentless+Claude-3.5":  "Agentless",
    "Moatless+V3":           "Moatless",
}
AGENT_PARADIGM = {
    "Claude-3":              "RLHF dense",
    "Claude-3.5":            "RLHF dense",
    "Claude-3.7-thinking":   "Extended-thinking",
    "Claude-4":              "Extended-thinking",
    "GPT-4":                 "RLHF dense",
    "GPT-4o":                "RLHF dense",
    "DARS+R1":               "RL reasoning",
    "Agentless+Claude-3.5":  "RLHF dense",
    "Moatless+V3":           "MoE pretrain",
}

# Atom-prefix buckets for derived features (matches the original SWE-agent ones).
ATOM_BUCKETS = {
    "n_edits":    ("EDIT_", "CREATE_"),
    "n_searches": ("SEARCH", "FIND_FILE", "GREP"),
    "n_opens":    ("OPEN_",),
    "n_runs":     ("RUN_",),
    "n_nav":      ("NAV_",),
    "n_shell":    ("SHELL_",),
}

FEATURES = [
    "n_steps",
    "n_motifs",
    "compression",
    "n_edits",
    "n_searches",
    "n_opens",
    "n_runs",
    "n_nav",
    "n_shell",
    "edit_share",
]
FEATURE_LABELS = {
    "n_steps":     "Total atoms",
    "n_motifs":    "Motif count",
    "compression": "Compression",
    "n_edits":     "Edits",
    "n_searches":  "Searches",
    "n_opens":     "File opens",
    "n_runs":      "Script runs",
    "n_nav":       "Navigation",
    "n_shell":     "Shell ops",
    "edit_share":  "Edit share",
}

GROUPINGS = ["instance_id", "fix_type", "repo", "scaffold", "paradigm"]
GROUPING_LABEL = {
    "instance_id": "Task identity",
    "fix_type":    "Fix type",
    "repo":        "Repository",
    "scaffold":    "Scaffold class",
    "paradigm":    "Training paradigm",
}
GROUPING_KIND = {
    "instance_id": "task",
    "fix_type":    "task",
    "repo":        "task",
    "scaffold":    "agent-substructure",
    "paradigm":    "agent-substructure",
}
GROUPING_COLORS = {
    "instance_id": BLUE,
    "fix_type":    COPPER,
    "repo":        GREEN,
    "scaffold":    MAGENTA,
    "paradigm":    OLIVE,
}


def derive_features(canonical: list[str], n_motifs: int) -> dict:
    n_steps = len(canonical)
    counts = {k: 0 for k in ATOM_BUCKETS}
    for atom in canonical:
        for key, prefixes in ATOM_BUCKETS.items():
            if any(atom.startswith(p) for p in prefixes):
                counts[key] += 1
                break
    compression = (n_motifs / n_steps) if n_steps > 0 else 0.0
    edit_share = (counts["n_edits"] / n_steps) if n_steps > 0 else 0.0
    return {
        "n_steps":     n_steps,
        "n_motifs":    n_motifs,
        "compression": compression,
        "edit_share":  edit_share,
        **counts,
    }


def load_corpus() -> pd.DataFrame:
    labels = json.loads(FIX_TYPES_PATH.read_text())["results"]
    lut = {r["instance_id"]: r for r in labels}

    rows = []
    with SEQ_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            agent = rec["agent"]
            iid = rec["instance_id"]
            feat = derive_features(rec["canonical"], rec["bpe_length"])
            meta = lut.get(iid, {})
            rows.append({
                "agent":       agent,
                "scaffold":    AGENT_SCAFFOLD[agent],
                "paradigm":    AGENT_PARADIGM[agent],
                "instance_id": iid,
                "fix_type":    meta.get("fix_type", "unknown"),
                "repo":        meta.get("repo", "unknown"),
                **feat,
            })
    return pd.DataFrame(rows)


def decompose(df: pd.DataFrame, feature: str) -> dict:
    out = {
        "feature":         feature,
        "n_trajectories":  int(len(df)),
        "grand_mean":      float(df[feature].mean()),
    }
    agent_sd = df.groupby("agent")[feature].mean().std(ddof=1)
    out["sd_agent_means"] = float(agent_sd) if not np.isnan(agent_sd) else 0.0
    out["n_agent_groups"] = int(df["agent"].nunique())

    for g in GROUPINGS:
        grp_means = df.groupby(g)[feature].mean()
        sd = grp_means.std(ddof=1)
        out[f"sd_{g}_means"] = float(sd) if len(grp_means) > 1 else 0.0
        out[f"n_{g}_groups"] = int(df[g].nunique())
        out[f"ratio_{g}_over_agent"] = (
            float(sd / agent_sd)
            if agent_sd > 0 and len(grp_means) > 1 else float("nan")
        )
    return out


def build_report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([decompose(df, f) for f in FEATURES])


def plot(report: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for _, row in report.iterrows():
        for g in GROUPINGS:
            rows.append({
                "feature":        FEATURE_LABELS.get(row["feature"], row["feature"]),
                "feature_raw":    row["feature"],
                "grouping_label": GROUPING_LABEL[g],
                "grouping_kind":  GROUPING_KIND[g],
                "ratio":          row[f"ratio_{g}_over_agent"],
            })
    df_plot = pd.DataFrame(rows)

    feat_order = (
        df_plot[df_plot["grouping_label"] == GROUPING_LABEL["instance_id"]]
        .sort_values("ratio", ascending=True)["feature"]
        .tolist()
    )

    color_scale = alt.Scale(
        domain=[GROUPING_LABEL[g] for g in GROUPINGS],
        range=[GROUPING_COLORS[g] for g in GROUPINGS],
    )

    bars = (
        alt.Chart(df_plot)
        .mark_bar(height=8)
        .encode(
            y=alt.Y(
                "feature:N",
                sort=feat_order,
                axis=alt.Axis(title=None, ticks=False, domain=False, labelFontSize=10),
            ),
            x=alt.X(
                "ratio:Q",
                title="SD of group means / SD of agent means",
                axis=alt.Axis(ticks=False, domain=False, labelFontSize=10),
            ),
            yOffset=alt.YOffset(
                "grouping_label:N",
                sort=[GROUPING_LABEL[g] for g in GROUPINGS],
                scale=alt.Scale(range=[-12, 12]),
            ),
            color=alt.Color(
                "grouping_label:N",
                scale=color_scale,
                legend=alt.Legend(orient="bottom", title=None, labelFontSize=10),
            ),
        )
    )

    chart = (
        bars
        .properties(
            width=440,
            height=max(280, len(report) * 38),
            title=alt.TitleParams(
                text="Behavioral feature variance by grouping (9-agent corpus)",
                fontSize=13,
                color="#111111",
                anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    chart.save(str(out_path), scale_factor=2)


def show(report: pd.DataFrame) -> None:
    head = ["feature"] + [GROUPING_LABEL[g][:18] for g in GROUPINGS]
    print("\n  " + "  ".join(f"{h:<18s}" for h in head))
    for _, row in report.iterrows():
        cells = [f"{FEATURE_LABELS.get(row['feature'], row['feature'])[:18]:<18s}"]
        for g in GROUPINGS:
            r = row[f"ratio_{g}_over_agent"]
            n = row[f"n_{g}_groups"]
            cells.append(f"{r:>5.2f}x n={n:<3d}      "[:18])
        print("  " + "  ".join(cells))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_corpus()

    print(f"=== 9-agent corpus ===")
    print(f"  {len(df)} trajectories, "
          f"{df['instance_id'].nunique()} tasks, "
          f"{df['fix_type'].nunique()} fix types, "
          f"{df['repo'].nunique()} repos, "
          f"{df['agent'].nunique()} agents, "
          f"{df['scaffold'].nunique()} scaffolds, "
          f"{df['paradigm'].nunique()} paradigms")

    report = build_report(df)
    show(report)

    base = "variance_decomposition_extended"
    (OUT_DIR / f"{base}.csv").write_text(report.to_csv(index=False))
    (OUT_DIR / f"{base}.json").write_text(json.dumps({
        "n_trajectories": int(len(df)),
        "n_agents":       int(df["agent"].nunique()),
        "groupings":      GROUPINGS,
        "grouping_kind":  GROUPING_KIND,
        "features":       report.to_dict(orient="records"),
    }, indent=2))
    plot(report, OUT_DIR / f"{base}.png")

    print(f"\nSaved:")
    for n in [f"{base}.json", f"{base}.csv", f"{base}.png"]:
        print(f"  {OUT_DIR / n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
