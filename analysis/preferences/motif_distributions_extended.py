"""9-agent per-agent motif distributions + JSD matrix.

Companion to motif_distributions.py: reads the 9-agent corpus
(bpe_sequences_extended.jsonl) and produces the corresponding 9-agent
motif-distribution and JSD-heatmap figures.

Outputs:
  output/paper2_pilot/agent_motif_distributions_extended.png
  output/paper2_pilot/agent_jsd_matrix_extended.png
  output/paper2_pilot/agent_motif_distributions_extended.json

Usage:
    python -m analysis.preferences.motif_distributions_extended
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.theme import (
    register, BLUE, COPPER, GREEN, MAGENTA, OLIVE, GREEN_D, BLUE_D, MAGENTA_D,
)
register()

OUT = PROJECT_ROOT / "output" / "paper2_pilot"
SEQ_PATH = OUT / "bpe_sequences_extended.jsonl"

AGENT_COLORS = {
    "Claude-3":              COPPER,
    "Claude-3.5":            GREEN,
    "Claude-3.7-thinking":   GREEN_D,
    "Claude-4":              "#187860",
    "GPT-4":                 BLUE,
    "GPT-4o":                MAGENTA,
    "DARS+R1":               MAGENTA_D,
    "Agentless+Claude-3.5":  BLUE_D,
    "Moatless+V3":           OLIVE,
}
AGENT_ORDER = list(AGENT_COLORS)


def load_sequences() -> list[dict]:
    records = []
    with open(SEQ_PATH) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def per_agent_token_counts(records: list[dict]) -> dict[str, Counter]:
    counts: dict[str, Counter] = {}
    for r in records:
        counts.setdefault(r["agent"], Counter()).update(r["bpe"])
    return counts


def per_agent_canonical_counts(records: list[dict]) -> dict[str, Counter]:
    counts: dict[str, Counter] = {}
    for r in records:
        counts.setdefault(r["agent"], Counter()).update(r["canonical"])
    return counts


def per_agent_n_trajectories(records: list[dict]) -> dict[str, int]:
    n: dict[str, int] = {}
    for r in records:
        n[r["agent"]] = n.get(r["agent"], 0) + 1
    return n


def normalize(counter: Counter, vocab: list[str]) -> np.ndarray:
    total = sum(counter.values())
    if total == 0:
        return np.zeros(len(vocab))
    return np.array([counter.get(v, 0) / total for v in vocab])


def compute_jsd_matrix(distributions: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    agents = list(distributions.keys())
    for a, b in combinations(agents, 2):
        d = float(jensenshannon(distributions[a], distributions[b], base=2)) ** 2
        out[(a, b)] = d
    return out


_STAGE_ORDER = ["Explore", "Browse", "Edit", "Test", "Finish"]
_ATOM_STAGE: dict[str, str] = {
    "SEARCH":              "Explore",
    "FIND_FILE":           "Explore",
    "SHELL_LS":            "Explore",
    "SHELL_CD":            "Explore",
    "SHELL_CAT":           "Explore",
    "SHELL_GREP":          "Explore",
    "SHELL_MKDIR":         "Explore",
    "OPEN_SRC_PY":         "Browse",
    "NAV_SRC_PY":          "Browse",
    "OPEN_TEST_PY":        "Browse",
    "EDIT_SRC_PY":         "Edit",
    "EDIT_TEST_PY":        "Edit",
    "EDIT_REPRO_PY":       "Edit",
    "EDIT_CONFIG_PY":      "Edit",
    "CREATE_TEST_PY":      "Edit",
    "EDIT_OTHER":          "Edit",
    "RUN_PYTHON_SRC_PY":   "Test",
    "RUN_PYTHON_ALL":      "Test",
    "RUN_PYTHON_TEST_PY":  "Test",
    "RUN_PYTHON_REPRO_PY": "Test",
    "SUBMIT":              "Finish",
    "EXIT_ERROR":          "Finish",
}
_ATOM_LABEL: dict[str, str] = {
    "SEARCH":              "Search codebase",
    "FIND_FILE":           "Find file",
    "SHELL_LS":            "List directory",
    "SHELL_CD":            "Change directory",
    "SHELL_CAT":           "Read file contents",
    "SHELL_GREP":          "Search with grep",
    "SHELL_MKDIR":         "Create directory",
    "OPEN_SRC_PY":         "Open source file",
    "NAV_SRC_PY":          "Browse source",
    "OPEN_TEST_PY":        "Open test file",
    "EDIT_SRC_PY":         "Edit source",
    "EDIT_TEST_PY":        "Edit test",
    "EDIT_REPRO_PY":       "Edit repro script",
    "EDIT_CONFIG_PY":      "Edit config",
    "CREATE_TEST_PY":      "Write new test",
    "EDIT_OTHER":          "Edit other file",
    "RUN_PYTHON_SRC_PY":   "Run source file",
    "RUN_PYTHON_ALL":      "Run full test suite",
    "RUN_PYTHON_TEST_PY":  "Run test file",
    "RUN_PYTHON_REPRO_PY": "Run repro script",
    "SUBMIT":              "Submit fix",
    "EXIT_ERROR":          "Exit on error",
}


def plot_atom_breakdown(
    canonical_counts: dict[str, Counter],
    n_traj: dict[str, int],
    out_path: Path,
) -> None:
    agents = [a for a in AGENT_ORDER if a in canonical_counts]

    all_atoms: Counter = Counter()
    for c in canonical_counts.values():
        for tok, n in c.items():
            if "+" not in tok:
                all_atoms[tok] += n

    stage_idx = {s: i for i, s in enumerate(_STAGE_ORDER)}
    focus_atoms = [tok for tok in _ATOM_STAGE if tok in all_atoms]

    def mean_rate(atom: str) -> float:
        return sum(canonical_counts[a].get(atom, 0) for a in agents) / sum(
            n_traj[a] for a in agents
        )

    focus_atoms.sort(key=lambda a: (stage_idx[_ATOM_STAGE[a]], -mean_rate(a)))

    rows = []
    for atom in focus_atoms:
        lbl = _ATOM_LABEL[atom]
        stage = _ATOM_STAGE[atom]
        for agent in agents:
            n = n_traj[agent]
            rate = canonical_counts[agent].get(atom, 0) / n if n > 0 else 0.0
            rows.append({"atom": atom, "label": lbl, "stage": stage, "agent": agent, "rate": rate})
    df = pd.DataFrame(rows)

    label_order = [_ATOM_LABEL[a] for a in focus_atoms]

    x_clip = 6.0
    x_max = x_clip * 1.30

    color_scale = alt.Scale(
        domain=agents,
        range=[AGENT_COLORS[a] for a in agents],
    )

    df["rate_plot"] = df["rate"].clip(upper=x_clip)
    df["is_outlier"] = df["rate"] > x_clip

    rng2 = df.groupby("label")["rate_plot"].agg(["min", "max"]).reset_index()
    rng2.columns = ["label", "rp_min", "rp_max"]
    df = df.merge(rng2, on="label")

    band_rows = [
        {"label": lbl, "band_color": "#F5F5F5" if i % 2 == 0 else "#FFFFFF",
         "xmin": 0.0, "xmax": x_max}
        for i, lbl in enumerate(label_order)
    ]
    bands = (
        alt.Chart(pd.DataFrame(band_rows))
        .mark_rect()
        .encode(
            y=alt.Y("label:N", sort=label_order),
            x=alt.X("xmin:Q", scale=alt.Scale(domain=[0, x_max])),
            x2=alt.X2("xmax:Q"),
            color=alt.Color("band_color:N", scale=None),
        )
    )

    range_bars = (
        alt.Chart(df.drop_duplicates(subset=["label", "rp_min", "rp_max"]))
        .mark_rule(color="#CCCCCC", strokeWidth=1.5)
        .encode(
            y=alt.Y("label:N", sort=label_order),
            x=alt.X("rp_min:Q", scale=alt.Scale(domain=[0, x_max])),
            x2=alt.X2("rp_max:Q"),
        )
    )

    dots = (
        alt.Chart(df)
        .mark_point(size=70, filled=True, strokeWidth=0)
        .encode(
            y=alt.Y(
                "label:N",
                sort=label_order,
                axis=alt.Axis(title=None, domain=False, ticks=False,
                              labelFontSize=10, labelLimit=180),
            ),
            x=alt.X(
                "rate_plot:Q",
                scale=alt.Scale(domain=[0, x_max]),
                axis=alt.Axis(labels=False, title=None, domain=False, ticks=False),
            ),
            color=alt.Color(
                "agent:N", scale=color_scale,
                legend=alt.Legend(orient="bottom", title=None, columns=5,
                                  symbolSize=70, labelFontSize=10),
            ),
            tooltip=["agent:N", "label:N", alt.Tooltip("rate:Q", format=".2f")],
        )
    )

    label_df = df[df["rate"] >= 0.5].copy()
    label_df["label_text"] = label_df["rate"].apply(lambda r: f"{r:.1f}")
    label_df["label_x"] = label_df["rate_plot"]
    label_df.loc[label_df["is_outlier"], "label_text"] = (
        label_df.loc[label_df["is_outlier"], "rate"].apply(lambda r: f"→{r:.1f}")
    )
    label_df.loc[label_df["is_outlier"], "label_x"] = x_clip - 0.05

    value_labels = (
        alt.Chart(label_df)
        .mark_text(dy=-9, fontSize=8, fontWeight="normal")
        .encode(
            y=alt.Y("label:N", sort=label_order),
            x=alt.X("label_x:Q", scale=alt.Scale(domain=[0, x_max])),
            text=alt.Text("label_text:N"),
            color=alt.Color("agent:N", scale=color_scale, legend=None),
        )
    )

    chart = (
        alt.layer(bands, range_bars, dots, value_labels)
        .properties(
            width=480,
            height=len(focus_atoms) * 26,
            title=alt.TitleParams(
                text="How often agents use each action (9-agent corpus)",
                fontSize=13,
                fontWeight="normal",
                color="#111111",
                anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    chart.save(str(out_path), scale_factor=2)


def plot_jsd_matrix(
    jsd_full: dict[tuple[str, str], float],
    jsd_motifs: dict[tuple[str, str], float],
    out_path: Path,
) -> None:
    agents = [a for a in AGENT_ORDER if a in {ag for pair in jsd_full for ag in pair}]
    n = len(agents)

    def make_matrix(d: dict[tuple[str, str], float]) -> np.ndarray:
        M = np.zeros((n, n))
        for (a, b), v in d.items():
            i, j = agents.index(a), agents.index(b)
            M[i, j] = v
            M[j, i] = v
        return M

    M_full = make_matrix(jsd_full)
    M_motifs = make_matrix(jsd_motifs)
    max_val = max(M_full.max(), M_motifs.max())
    color_scale = alt.Scale(scheme="blues", domain=[0, max_val * 1.05])

    def make_panel(M: np.ndarray, title: str):
        rows_off, rows_diag = [], []
        for i, agent_a in enumerate(agents):
            for j, agent_b in enumerate(agents):
                is_diag = i == j
                value = 0.0 if is_diag else float(M[i, j])
                entry = {"agent_a": agent_a, "agent_b": agent_b, "value": value}
                if is_diag:
                    rows_diag.append(entry)
                else:
                    rows_off.append({**entry, "label": f"{value:.2f}"})
        df_all = pd.DataFrame(rows_off + rows_diag)
        df_off = pd.DataFrame(rows_off) if rows_off else pd.DataFrame(
            columns=["agent_a", "agent_b", "value", "label"])
        df_diag = pd.DataFrame(rows_diag)

        def base_enc(df):
            return alt.Chart(df).encode(
                x=alt.X("agent_b:N", sort=agents,
                        axis=alt.Axis(title=None, domain=False, ticks=False,
                                      labelFontSize=9, labelAngle=-35)),
                y=alt.Y("agent_a:N", sort=agents,
                        axis=alt.Axis(title=None, domain=False, ticks=False,
                                      labelFontSize=9)),
            )

        heatmap = base_enc(df_all).mark_rect().encode(
            color=alt.Color("value:Q", scale=color_scale, legend=None),
        )
        text_off = base_enc(df_off).mark_text(fontSize=8, color="white").encode(
            text=alt.Text("label:N"),
        )
        text_diag = base_enc(df_diag).mark_text(fontSize=10, color="#B0C4DE").encode(
            text=alt.value("-"),
        )
        return (
            alt.layer(heatmap, text_diag, text_off)
            .properties(
                width=320,
                height=320,
                title=alt.TitleParams(text=title, fontSize=11,
                                      fontWeight="normal", color="#555555",
                                      anchor="middle"),
            )
        )

    panel_a = make_panel(M_full, "All action types")
    panel_b = make_panel(M_motifs, "Multi-step sequences")

    chart = (
        alt.hconcat(panel_a, panel_b, spacing=24)
        .properties(
            title=alt.TitleParams(
                text="Pairwise procedural divergence (Jensen-Shannon, base 2)",
                fontSize=13, fontWeight="normal",
                color="#111111", anchor="start",
            )
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    chart.save(str(out_path), scale_factor=2)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading 9-agent BPE-expressed sequences...")
    records = load_sequences()
    print(f"  {len(records)} records across {len({r['agent'] for r in records})} agents")

    per_agent_counts = per_agent_token_counts(records)
    canonical_counts = per_agent_canonical_counts(records)
    n_traj = per_agent_n_trajectories(records)

    full_vocab = sorted({t for c in per_agent_counts.values() for t in c})
    motif_vocab = [t for t in full_vocab if "+" in t]
    print(f"  full vocabulary: {len(full_vocab)} items ({len(motif_vocab)} motifs)")

    dist_full = {a: normalize(c, full_vocab) for a, c in per_agent_counts.items()}
    dist_motifs = {
        a: normalize(Counter({t: c[t] for t in motif_vocab if t in c}), motif_vocab)
        for a, c in per_agent_counts.items()
    }

    jsd_full = compute_jsd_matrix(dist_full)
    jsd_motifs = compute_jsd_matrix(dist_motifs)

    plot_atom_breakdown(canonical_counts, n_traj,
                        OUT / "agent_motif_distributions_extended.png")
    plot_jsd_matrix(jsd_full, jsd_motifs,
                    OUT / "agent_jsd_matrix_extended.png")

    summary = {
        "n_records": len(records),
        "n_agents": len(per_agent_counts),
        "full_vocab_size": len(full_vocab),
        "n_motifs": len(motif_vocab),
        "per_agent_total_tokens": {a: sum(c.values()) for a, c in per_agent_counts.items()},
        "per_agent_n_trajectories": n_traj,
        "jsd_full_vocab": {f"{a}__{b}": v for (a, b), v in jsd_full.items()},
        "jsd_motifs_only": {f"{a}__{b}": v for (a, b), v in jsd_motifs.items()},
        "interpretation": {
            "jsd_range": "[0, 1] with log2; 0 = identical distributions, 1 = disjoint",
        },
    }
    (OUT / "agent_motif_distributions_extended.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    print("\nSaved:")
    for n in [
        "agent_motif_distributions_extended.png",
        "agent_jsd_matrix_extended.png",
        "agent_motif_distributions_extended.json",
    ]:
        print(f"  {OUT / n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
