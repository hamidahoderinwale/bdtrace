"""Observation uptake: does the agent incorporate feedback from its environment?

For each trajectory step t, we measure semantic similarity between
observation[t] (what the environment returned) and thought[t+1] (what the
agent said next). A high score means the agent's next thought references what
it just saw. A low score means it ploughs ahead regardless.

Per-trajectory uptake = mean similarity across all valid (obs[t], thought[t+1])
pairs. We compare uptake scores between passing and failing trajectories and
across agents.

Reads:
    output/trajectories/.cache/{agent}/*.json   (raw trajectories)
    output/trajectories/lite_all_models.parquet  (ground-truth pass/fail)
Writes:
    output/paper2_pilot/observation_uptake.json
    output/figures/fig_observation_uptake.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, ORANGE, GRAY, GREEN, AGENT_COLORS, AGENT_ORDER
register()

CACHE = ROOT / "output" / "trajectories" / ".cache"
OUT = ROOT / "output" / "paper2_pilot"
FIG_OUT = ROOT / "output" / "figures"

AGENT_MAP = {
    "20240402_sweagent_gpt4": "GPT-4",
    "20240620_sweagent_claude3.5sonnet": "Claude-3.5",
    "20240728_sweagent_gpt4o": "GPT-4o",
}

OBS_TRUNCATE = 300
MIN_OBS_LEN = 30
MIN_PAIRS = 3


def load_pass_fail() -> dict[tuple[str, str], bool]:
    df = pd.read_parquet(ROOT / "output/trajectories/lite_all_models.parquet")
    return {
        (row["model_id"], row["instance_id"]): bool(row["passed"])
        for _, row in df.iterrows()
    }


def extract_pairs(traj: list[dict]) -> list[tuple[str, str]]:
    pairs = []
    for i in range(len(traj) - 1):
        obs = traj[i].get("observation", "").strip()[:OBS_TRUNCATE]
        thought = traj[i + 1].get("thought", "").strip()
        if len(obs) < MIN_OBS_LEN or not thought:
            continue
        pairs.append((obs, thought))
    return pairs


def compute_uptake_scores(
    all_pairs: list[tuple[str, str]],
    pair_indices: list[tuple[int, int]],
) -> list[float | None]:
    if not all_pairs:
        return []

    obs_texts = [p[0] for p in all_pairs]
    thought_texts = [p[1] for p in all_pairs]

    vectorizer = TfidfVectorizer(
        min_df=1, max_df=0.95, ngram_range=(1, 2), sublinear_tf=True
    )
    vectorizer.fit(obs_texts + thought_texts)
    obs_vecs = vectorizer.transform(obs_texts)
    thought_vecs = vectorizer.transform(thought_texts)

    sims = np.array(
        cosine_similarity(obs_vecs, thought_vecs).diagonal()
    )

    scores: list[float | None] = []
    for start, end in pair_indices:
        if end <= start:
            scores.append(None)
        else:
            scores.append(float(np.mean(sims[start:end])))
    return scores


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    pass_fail = load_pass_fail()

    records = []
    all_pairs: list[tuple[str, str]] = []
    pair_boundaries: list[tuple[int, int]] = []

    for agent_dir in sorted(CACHE.iterdir()):
        if not agent_dir.is_dir():
            continue
        agent_short = AGENT_MAP.get(agent_dir.name)
        if agent_short is None:
            continue

        for traj_file in sorted(agent_dir.glob("*.json")):
            instance_id = traj_file.stem
            passed = pass_fail.get((agent_dir.name, instance_id))
            if passed is None:
                continue

            raw = json.loads(traj_file.read_text())
            traj = raw.get("trajectory", [])
            pairs = extract_pairs(traj)

            if len(pairs) < MIN_PAIRS:
                continue

            start = len(all_pairs)
            all_pairs.extend(pairs)
            end = len(all_pairs)
            pair_boundaries.append((start, end))

            records.append({
                "agent": agent_short,
                "instance_id": instance_id,
                "passed": passed,
                "n_pairs": len(pairs),
                "n_steps": len(traj),
            })

    print(f"Loaded {len(records)} trajectories, {len(all_pairs)} obs-thought pairs")

    scores = compute_uptake_scores(all_pairs, pair_boundaries)
    for rec, score in zip(records, scores):
        rec["uptake"] = score

    records = [r for r in records if r["uptake"] is not None]
    print(f"Scored {len(records)} trajectories")

    df = pd.DataFrame(records)

    print("\nUptake by agent x outcome:")
    for agent in AGENT_ORDER:
        sub = df[df["agent"] == agent]
        passed = sub[sub["passed"]]["uptake"]
        failed = sub[~sub["passed"]]["uptake"]
        print(f"  {agent}  pass={passed.mean():.3f} (n={len(passed)})  "
              f"fail={failed.mean():.3f} (n={len(failed)})")
        if len(passed) >= 5 and len(failed) >= 5:
            stat, p = mannwhitneyu(passed, failed, alternative="greater")
            print(f"    Mann-Whitney p (pass > fail): {p:.4f}")

    print("\nOverall:")
    passed_all = df[df["passed"]]["uptake"]
    failed_all = df[~df["passed"]]["uptake"]
    print(f"  pass mean={passed_all.mean():.3f}  fail mean={failed_all.mean():.3f}")
    stat, p = mannwhitneyu(passed_all, failed_all, alternative="greater")
    print(f"  Mann-Whitney p (pass > fail): {p:.4f}")

    summary = {
        "n_trajectories": len(df),
        "n_pairs_total": len(all_pairs),
        "overall": {
            "pass_mean": float(passed_all.mean()),
            "fail_mean": float(failed_all.mean()),
            "mw_p_pass_gt_fail": float(p),
        },
        "by_agent": {},
    }
    for agent in AGENT_ORDER:
        sub = df[df["agent"] == agent]
        pa = sub[sub["passed"]]["uptake"]
        fa = sub[~sub["passed"]]["uptake"]
        entry: dict = {
            "pass_mean": float(pa.mean()) if len(pa) else None,
            "fail_mean": float(fa.mean()) if len(fa) else None,
            "n_pass": int(len(pa)),
            "n_fail": int(len(fa)),
        }
        if len(pa) >= 5 and len(fa) >= 5:
            _, ap = mannwhitneyu(pa, fa, alternative="greater")
            entry["mw_p"] = float(ap)
        summary["by_agent"][agent] = entry

    (OUT / "observation_uptake.json").write_text(json.dumps(summary, indent=2))

    # Figure
    color_scale = alt.Scale(
        domain=[True, False],
        range=[BLUE, GRAY],
    )

    chart = (
        alt.Chart(df)
        .mark_point(size=30, opacity=0.5, filled=True, strokeWidth=0)
        .encode(
            x=alt.X("uptake:Q",
                    title="Observation uptake score (TF-IDF cosine similarity)",
                    scale=alt.Scale(domain=[0, df["uptake"].max() * 1.05]),
                    axis=alt.Axis(format=".2f")),
            y=alt.Y("agent:N",
                    sort=AGENT_ORDER,
                    axis=alt.Axis(title=None)),
            color=alt.Color("passed:N", scale=color_scale,
                            legend=alt.Legend(title="Passed", orient="bottom",
                                              labelExpr="datum.value ? 'pass' : 'fail'")),
            yOffset=alt.YOffset("passed:N", sort=[True, False],
                                scale=alt.Scale(range=[-8, 8])),
        )
        .properties(
            width=360, height=160,
            title=alt.TitleParams(
                "Observation uptake by agent and outcome",
                fontSize=13, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
    )

    out_path = FIG_OUT / "fig_observation_uptake.png"
    chart.save(str(out_path), scale_factor=2)
    print(f"\nSaved {out_path}")
    print(f"Saved {OUT / 'observation_uptake.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
