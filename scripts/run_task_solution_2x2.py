#!/usr/bin/env python3
"""
2x2 pass rate table: structural tractability × procedural alignment.

  coverage  = fraction of task's K nearest structural neighbors solved by a
              *different* model (cross-model, to avoid circularity)
  aligned   = agent's action sequence is closer to the local structural template
              than to the global template  (local_dist < global_dist)

If the decomposition P(solved) ≈ P(tractable) × P(right procedure | type) holds,
the top-left cell (high coverage, aligned) should have the highest pass rate and
the bottom-right the lowest.

Usage:
  uv run python scripts/run_task_solution_2x2.py
"""

import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path("output/datasets/swe_bench_lite_resolved")
TRAJ_PATH = Path("output/trajectories/lite_all_models.parquet")
PLOTS_DIR = Path("notebooks/plots/behavioral")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

K = 10
_MODELS = {
    "20240402_sweagent_gpt4": "GPT-4",
    "20240620_sweagent_claude3.5sonnet": "Claude 3.5",
    "20240728_sweagent_gpt4o": "GPT-4o",
}
_WONG = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]


## ── helpers ─────────────────────────────────────────────────────────────────

def levenshtein(a: list, b: list) -> int:
    """Token-level edit distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def medoid_sequence(seqs: list[list]) -> list:
    """Return the sequence with the minimum total Levenshtein to all others."""
    if len(seqs) == 1:
        return seqs[0]
    total = [sum(levenshtein(s, other) for other in seqs) for s in seqs]
    return seqs[int(np.argmin(total))]


def global_template(seqs: list[list]) -> list:
    """Most representative sequence across all passed runs."""
    return medoid_sequence(seqs)


def local_template(task_idx: int, dist_row: np.ndarray, solved_idx: list[int],
                   idx_to_seq: dict[int, list]) -> list | None:
    """Medoid of K nearest solved neighbors (excluding task_idx itself)."""
    row = dist_row.copy()
    row[task_idx] = np.inf
    candidates = [i for i in solved_idx if i != task_idx]
    if not candidates:
        return None
    nn = sorted(candidates, key=lambda i: row[i])[:K]
    return medoid_sequence([idx_to_seq[i] for i in nn if i in idx_to_seq])


## ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    labels = pd.read_parquet(DATA_DIR / "labels.parquet")
    instance_ids = labels["instance_id"].tolist()
    id_to_idx = {iid: i for i, iid in enumerate(instance_ids)}

    mats = np.load(DATA_DIR / "matrices.npz")
    D_struct = mats["edits_set_diff"]   # coverage: structural tractability
    D_modules = mats["modules"]          # local template: file-level neighbor retrieval

    trajs = pd.read_parquet(TRAJ_PATH)

    model_ids = list(_MODELS.keys())
    all_rows = []

    for mid, label in _MODELS.items():
        sub = trajs[trajs["model_id"] == mid].copy()
        sub["seq"] = sub["action_sequence"].apply(
            lambda s: s.split() if isinstance(s, str) else []
        )
        iid_to_row = {r["instance_id"]: r for _, r in sub.iterrows()}

        ## Tasks this model attempted and their pass/fail
        attempted_idx = [id_to_idx[iid] for iid in sub["instance_id"] if iid in id_to_idx]
        passed_idx = [id_to_idx[r["instance_id"]] for _, r in sub.iterrows()
                      if r["passed"] and r["instance_id"] in id_to_idx]
        idx_to_seq = {id_to_idx[r["instance_id"]]: r["seq"]
                      for _, r in sub.iterrows() if r["instance_id"] in id_to_idx}

        ## Global template from all passed runs for this model
        passed_seqs = [idx_to_seq[i] for i in passed_idx if i in idx_to_seq]
        if not passed_seqs:
            continue
        g_template = global_template(passed_seqs)

        ## Cross-model coverage: average fraction solved by OTHER models
        other_mids = [m for m in model_ids if m != mid]
        other_passed: list[set] = []
        for om in other_mids:
            os = trajs[trajs["model_id"] == om]
            other_passed.append(
                {id_to_idx[r["instance_id"]] for _, r in os.iterrows()
                 if r["passed"] and r["instance_id"] in id_to_idx}
            )

        for task_iid, row in iid_to_row.items():
            tidx = id_to_idx.get(task_iid)
            if tidx is None:
                continue

            ## Coverage: fraction of K nearest neighbors solved by other models
            dist_row = D_struct[tidx].copy()
            dist_row[tidx] = np.inf
            nn_idx = np.argsort(dist_row)[:K]
            cov_scores = [
                sum(1 for ni in nn_idx if ni in op) / K
                for op in other_passed
            ]
            coverage = float(np.mean(cov_scores))

            ## Local template alignment
            lt = local_template(tidx, D_modules[tidx], passed_idx, idx_to_seq)
            agent_seq = row["seq"]
            if lt is None or not agent_seq:
                continue
            local_d = levenshtein(agent_seq, lt)
            global_d = levenshtein(agent_seq, g_template)

            all_rows.append({
                "model": label,
                "instance_id": task_iid,
                "passed": bool(row["passed"]),
                "coverage": coverage,
                "local_dist": local_d,
                "global_dist": global_d,
                "locally_aligned": local_d < global_d,
            })

    df = pd.DataFrame(all_rows)
    df.to_parquet(DATA_DIR / "task_solution_2x2.parquet", index=False)
    print(f"Built {len(df)} rows across {df['model'].nunique()} models")

    ## ── 2x2 table ────────────────────────────────────────────────────────────
    df["high_coverage"] = df.groupby("model")["coverage"].transform(
        lambda x: x >= x.median()
    )

    table_rows = []
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        for hc, hc_label in [(True, "high coverage"), (False, "low coverage")]:
            for la, la_label in [(True, "locally aligned"), (False, "global pattern")]:
                cell = sub[(sub["high_coverage"] == hc) & (sub["locally_aligned"] == la)]
                pass_rate = cell["passed"].mean() if len(cell) > 0 else float("nan")
                n = len(cell)
                table_rows.append({
                    "model": model,
                    "coverage_bin": hc_label,
                    "alignment_bin": la_label,
                    "pass_rate": pass_rate,
                    "n": n,
                })

    tdf = pd.DataFrame(table_rows)

    ## Print raw table
    print("\n── 2×2 pass rates ──")
    for model in tdf["model"].unique():
        print(f"\n{model}")
        pivot = tdf[tdf["model"] == model].pivot(
            index="coverage_bin", columns="alignment_bin", values="pass_rate"
        ).round(3)
        print(pivot.to_string())

    ## ── Plot ─────────────────────────────────────────────────────────────────
    tdf["pass_pct"] = (tdf["pass_rate"] * 100).round(1)
    tdf["label"] = tdf["pass_pct"].apply(lambda v: f"{v:.0f}%" if not np.isnan(v) else "–")
    tdf["label_n"] = tdf.apply(lambda r: f"{r['pass_pct']:.0f}%\n(n={r['n']})", axis=1)

    x_enc = alt.X(
        "alignment_bin:N",
        sort=["locally aligned", "global pattern"],
        title="procedural alignment",
        axis=alt.Axis(labelAngle=-20),
    )
    y_enc = alt.Y(
        "coverage_bin:N",
        sort=["high coverage", "low coverage"],
        title="structural coverage",
    )

    base = alt.Chart(tdf).encode(x_enc, y_enc)

    heatmap = base.mark_rect().encode(
        alt.Color(
            "pass_rate:Q",
            scale=alt.Scale(scheme="blues", domain=[0, 0.45]),
            title="pass rate",
        )
    )

    text = base.mark_text(fontSize=13, fontWeight="bold").encode(
        alt.Text("label:N"),
        color=alt.condition(
            "datum.pass_rate > 0.25",
            alt.value("white"),
            alt.value("#333333"),
        ),
    )

    (
        (heatmap + text)
        .facet("model:N", columns=3)
        .properties(
            title=alt.TitleParams(
                text="Pass rate by structural coverage × procedural alignment",
                subtitle=(
                    "Coverage = fraction of K=10 structural neighbors solved by other models. "
                    "Aligned = agent sequence closer to local structural template than global."
                ),
                anchor="start",
            )
        )
        .resolve_scale(color="shared")
        .save(PLOTS_DIR / "task_solution_2x2.png")
    )
    print("\nSaved task_solution_2x2.png")


if __name__ == "__main__":
    main()
