"""CI drift check across two SWE-agent model upgrades.

Validates the "agent CI / regression testing" integration claim on the
procgrep landing page (Integrations #1). Treats two real model-bump
pairs (Claude-3 -> Claude-3.5, GPT-4 -> GPT-4o) inside the SWE-agent
scaffold as if they were sequential builds of the same agent stack
and runs the Mode-2 controlled-eval recipe on each pair.

For each pair:
  * Across-arm JSD: jensenshannon(arm_A_mean, arm_B_mean)^2, base 2.
  * Within-arm noise floor (arm-mean scale): random-split each arm in
    half K times, compute JSD between the two half-means, take the
    median. This puts numerator and denominator on the same units
    (both are arm-mean to arm-mean distances).
  * SNR = across / within_median.
  * Binary-classifier probe: logistic regression on raw motif counts
    with 5-fold CV; report mean accuracy. (This is the per-trajectory
    discriminability question, which is independent of arm-mean SNR.)

The data is *not* a literal CI lineage (these are different vendors'
releases captured in the public SWE-bench corpus). The framing is:
"if a team's CI had captured both builds, here is what procgrep would
have surfaced."

Reads:
    output/paper2_pilot/bpe_sequences_extended.jsonl

Writes:
    output/paper2_pilot/ci_drift_check.json
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SEQ_PATH = ROOT / "output" / "paper2_pilot" / "bpe_sequences_extended.jsonl"
OUT_JSON = ROOT / "output" / "paper2_pilot" / "ci_drift_check.json"

PAIRS: list[tuple[str, str]] = [
    ("Claude-3", "Claude-3.5"),
    ("GPT-4",    "GPT-4o"),
]

SEED = 0
N_SPLIT_HALVES = 200   # split-halves resamples per arm for within-arm noise
N_CV_FOLDS = 5


def jsd_sq(p: np.ndarray, q: np.ndarray) -> float:
    """Squared Jensen-Shannon distance in bits; standard procgrep convention."""
    d = float(jensenshannon(p, q, base=2))
    return d * d


def to_distribution(motif_list: list[str], vocab_idx: dict[str, int]) -> np.ndarray:
    """L1-normalized motif distribution over a fixed vocabulary."""
    v = np.zeros(len(vocab_idx), dtype=np.float64)
    for m in motif_list:
        if m in vocab_idx:
            v[vocab_idx[m]] += 1
    s = v.sum()
    return v / s if s > 0 else v


def arm_mean_distribution(arm_records: list[dict], vocab_idx: dict[str, int]) -> np.ndarray:
    """Pool all arm motif counts, then normalize."""
    pooled = Counter()
    for r in arm_records:
        pooled.update(r["bpe"])
    v = np.zeros(len(vocab_idx), dtype=np.float64)
    for m, n in pooled.items():
        if m in vocab_idx:
            v[vocab_idx[m]] = n
    s = v.sum()
    return v / s if s > 0 else v


def within_arm_noise(
    arm_records: list[dict],
    vocab_idx: dict[str, int],
    n_splits: int,
    rng: random.Random,
) -> tuple[float, float]:
    """Median + IQR of JSD between two random half-means of the same arm.

    The half-mean is the analog of the arm-mean: pool every motif count
    in one half of the arm's trajectories, normalize. Comparing two such
    halves gives the same-units sampling-noise estimate of an arm-mean
    fingerprint, which is the right denominator when the numerator is
    `jsd(arm_A_mean, arm_B_mean)`.
    """
    n = len(arm_records)
    if n < 4:
        return float("nan"), float("nan")
    samples: list[float] = []
    for _ in range(n_splits):
        shuffled = arm_records[:]
        rng.shuffle(shuffled)
        h = n // 2
        half_a = shuffled[:h]
        half_b = shuffled[h:2 * h]
        m_a = arm_mean_distribution(half_a, vocab_idx)
        m_b = arm_mean_distribution(half_b, vocab_idx)
        samples.append(jsd_sq(m_a, m_b))
    arr = np.asarray(samples)
    return float(np.median(arr)), float(np.subtract(*np.percentile(arr, [75, 25])))


def binary_probe_accuracy(
    arm_a_records: list[dict],
    arm_b_records: list[dict],
    vocab_idx: dict[str, int],
    n_folds: int,
    seed: int,
) -> tuple[float, float]:
    """Mean + std of logistic-regression 5-fold CV accuracy on raw motif counts.
    Predicts arm label (A vs B) from each trajectory's motif-count vector."""
    X = []
    y = []
    for r in arm_a_records:
        v = np.zeros(len(vocab_idx), dtype=np.float64)
        for m in r["bpe"]:
            if m in vocab_idx:
                v[vocab_idx[m]] += 1
        X.append(v)
        y.append(0)
    for r in arm_b_records:
        v = np.zeros(len(vocab_idx), dtype=np.float64)
        for m in r["bpe"]:
            if m in vocab_idx:
                v[vocab_idx[m]] += 1
        X.append(v)
        y.append(1)
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    accs: list[float] = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X_arr, y_arr):
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        clf.fit(X_arr[train_idx], y_arr[train_idx])
        accs.append(float(clf.score(X_arr[test_idx], y_arr[test_idx])))
    return float(np.mean(accs)), float(np.std(accs))


def main() -> int:
    print(f"Loading {SEQ_PATH} ...")
    records = [json.loads(line) for line in SEQ_PATH.open() if line.strip()]
    print(f"  {len(records)} trajectories")

    # Build the shared BPE vocabulary across the full corpus (matches the
    # controlled-eval recipe's "fit one shared vocab across all arms" rule).
    vocab_set: set[str] = set()
    for r in records:
        vocab_set.update(r["bpe"])
    vocab = sorted(vocab_set)
    vocab_idx = {m: i for i, m in enumerate(vocab)}
    print(f"  vocab size: {len(vocab)}")

    by_agent: dict[str, list[dict]] = {}
    for r in records:
        by_agent.setdefault(r["agent"], []).append(r)

    rng = random.Random(SEED)
    results: list[dict] = []
    print()
    for agent_a, agent_b in PAIRS:
        if agent_a not in by_agent or agent_b not in by_agent:
            print(f"SKIP {agent_a} -> {agent_b}: missing arm in corpus")
            continue
        arm_a = by_agent[agent_a]
        arm_b = by_agent[agent_b]

        mean_a = arm_mean_distribution(arm_a, vocab_idx)
        mean_b = arm_mean_distribution(arm_b, vocab_idx)
        across = jsd_sq(mean_a, mean_b)

        within_a_med, within_a_iqr = within_arm_noise(arm_a, vocab_idx, N_SPLIT_HALVES, rng)
        within_b_med, within_b_iqr = within_arm_noise(arm_b, vocab_idx, N_SPLIT_HALVES, rng)
        within_median = float(np.median([within_a_med, within_b_med]))

        snr = across / within_median if within_median > 0 else float("inf")

        acc_mean, acc_std = binary_probe_accuracy(
            arm_a, arm_b, vocab_idx, N_CV_FOLDS, SEED
        )

        result = {
            "arm_a": agent_a,
            "arm_b": agent_b,
            "n_a": len(arm_a),
            "n_b": len(arm_b),
            "across_arm_jsd": round(across, 4),
            "within_arm_jsd_median": round(within_median, 4),
            "within_arm_jsd_per_arm": {
                agent_a: {"median": round(within_a_med, 4), "iqr": round(within_a_iqr, 4)},
                agent_b: {"median": round(within_b_med, 4), "iqr": round(within_b_iqr, 4)},
            },
            "snr": round(snr, 2),
            "snr_threshold_for_meaningful_drift": 2.0,
            "drift_verdict": "moved procedure" if snr >= 2.0 else "sampled noise",
            "binary_probe_accuracy_mean": round(acc_mean, 3),
            "binary_probe_accuracy_std": round(acc_std, 3),
            "binary_probe_chance": 0.5,
            "probe_verdict": (
                "arms distinguishable above chance"
                if acc_mean > 0.55 else "arms not reliably distinguishable"
            ),
        }
        results.append(result)

        print(f"== {agent_a} -> {agent_b} ==")
        print(f"  N: {len(arm_a)} / {len(arm_b)}")
        print(f"  across-arm JSD:     {across:.4f}")
        print(f"  within-arm median:  {within_median:.4f}  ({agent_a}: {within_a_med:.4f}; {agent_b}: {within_b_med:.4f})")
        print(f"  SNR:                {snr:.2f}x   ({result['drift_verdict']})")
        print(f"  probe accuracy:     {acc_mean:.3f} ± {acc_std:.3f}  ({result['probe_verdict']})")
        print()

    payload = {
        "framing": (
            "Two model bumps inside the SWE-agent scaffold (Claude-3 -> Claude-3.5; "
            "GPT-4 -> GPT-4o) treated as sequential CI builds. The data is the "
            "public SWE-bench corpus, not a literal CI lineage."
        ),
        "config": {
            "seed": SEED,
            "n_split_halves_per_arm": N_SPLIT_HALVES,
            "n_cv_folds": N_CV_FOLDS,
            "vocab_size": len(vocab),
            "within_arm_noise_method": (
                "split-halves: randomly partition each arm into halves K times, "
                "compute JSD between the two half-means; report median + IQR."
            ),
        },
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
