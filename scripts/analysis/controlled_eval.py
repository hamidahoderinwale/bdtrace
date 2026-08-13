"""Generic controlled-eval harness for procgrep Mode 2.

Consumes a JSONL of BPE-canonicalized trajectory records (one per line)
tagged with an arm label, and emits the controlled-eval comparison the
procgrep page describes: per-arm summaries plus per-pair across-arm JSD,
within-arm split-halves noise floor, SNR, and a binary-classifier probe
accuracy.

This script does NOT orchestrate agent runs. It is the *analysis* half
of the workflow. The capture half lives in whichever harness drives the
agent; that harness writes JSONL records of the shape:

    {"agent": "swe-agent+claude3.5", "instance_id": "django/foo",
     "canonical": [...], "bpe": [...], "arm": "0.3", "seed": 0}

where the arm field is whatever knob you swept. The script supports
both pre-fit BPE motifs (the `bpe` field is present and motif-tokenized)
and a shared-vocab fit at load time.

Two comparison modes:
  - `sweep`        consecutive pairs in `--arms` order (the right mode
                   for a continuous-knob sweep over temperature, top_p,
                   max_tokens, etc.)
  - `pairwise-all` every (n choose 2) pair (the right mode for
                   discrete-arm comparisons such as
                   {prompt_A, prompt_B, prompt_C})

Usage:
    python controlled_eval.py \\
        --input sweep_traces.jsonl \\
        --group-field temperature \\
        --arms 0.0,0.3,0.7,1.0 \\
        --mode sweep \\
        --output controlled_eval_temperature.json

If --arms is omitted, arms are taken from the data and sorted in
alphanumeric order, which is wrong for continuous sweeps; always pass
--arms explicitly when running a sweep so the consecutive-pair chain
matches the knob's natural ordering.

Reads:
    input JSONL with fields: canonical, bpe, <group-field>, instance_id

Writes:
    output JSON with per-arm summary and per-pair stats.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# Defaults
DEFAULT_GROUP_FIELD = "arm"
DEFAULT_MODE = "sweep"
DEFAULT_SEED = 0
DEFAULT_SPLIT_HALVES = 200
DEFAULT_CV_FOLDS = 5
SNR_THRESHOLD = 2.0
PROBE_THRESHOLD = 0.55


def jsd_sq(p: np.ndarray, q: np.ndarray) -> float:
    """Squared Jensen-Shannon distance in bits."""
    d = float(jensenshannon(p, q, base=2))
    return d * d


def build_vocab(records: list[dict]) -> dict[str, int]:
    vocab: set[str] = set()
    for r in records:
        vocab.update(r["bpe"])
    return {m: i for i, m in enumerate(sorted(vocab))}


def trajectory_dist(motifs: list[str], vocab_idx: dict[str, int]) -> np.ndarray:
    v = np.zeros(len(vocab_idx), dtype=np.float64)
    for m in motifs:
        i = vocab_idx.get(m)
        if i is not None:
            v[i] += 1
    s = v.sum()
    return v / s if s > 0 else v


def trajectory_counts(motifs: list[str], vocab_idx: dict[str, int]) -> np.ndarray:
    v = np.zeros(len(vocab_idx), dtype=np.float64)
    for m in motifs:
        i = vocab_idx.get(m)
        if i is not None:
            v[i] += 1
    return v


def arm_mean(arm_records: list[dict], vocab_idx: dict[str, int]) -> np.ndarray:
    pooled: Counter = Counter()
    for r in arm_records:
        pooled.update(r["bpe"])
    v = np.zeros(len(vocab_idx), dtype=np.float64)
    for m, n in pooled.items():
        i = vocab_idx.get(m)
        if i is not None:
            v[i] = n
    s = v.sum()
    return v / s if s > 0 else v


def within_arm_noise(
    arm_records: list[dict],
    vocab_idx: dict[str, int],
    n_splits: int,
    rng: random.Random,
) -> tuple[float, float]:
    """Median + IQR of JSD between two random half-means of the same arm."""
    n = len(arm_records)
    if n < 4:
        return float("nan"), float("nan")
    samples: list[float] = []
    for _ in range(n_splits):
        shuffled = arm_records[:]
        rng.shuffle(shuffled)
        h = n // 2
        m_a = arm_mean(shuffled[:h], vocab_idx)
        m_b = arm_mean(shuffled[h : 2 * h], vocab_idx)
        samples.append(jsd_sq(m_a, m_b))
    arr = np.asarray(samples)
    iqr = float(np.subtract(*np.percentile(arr, [75, 25])))
    return float(np.median(arr)), iqr


def _coerce_bool(v) -> int | None:
    """Coerce a resolved-field value to 0/1 or return None if unparseable."""
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(bool(v))
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "y", "resolved", "pass", "passed"):
            return 1
        if s in ("false", "0", "no", "n", "unresolved", "fail", "failed"):
            return 0
    return None


def binary_probe(
    arm_a_records: list[dict],
    arm_b_records: list[dict],
    vocab_idx: dict[str, int],
    n_folds: int,
    seed: int,
) -> tuple[float, float]:
    """Classifier accuracy on arm-label prediction (which arm did this come from?)."""
    X = []
    y = []
    for r in arm_a_records:
        X.append(trajectory_counts(r["bpe"], vocab_idx))
        y.append(0)
    for r in arm_b_records:
        X.append(trajectory_counts(r["bpe"], vocab_idx))
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


def success_probe(
    arm_a_records: list[dict],
    arm_b_records: list[dict],
    vocab_idx: dict[str, int],
    resolved_field: str,
    n_folds: int,
    seed: int,
) -> dict:
    """Classifier accuracy on outcome prediction (does fingerprint predict success?).

    Trained on the pooled (arm_a + arm_b) records using motif counts as features
    and the resolved field as the label. Reports cross-validated accuracy plus
    the resolve-pct baseline (predict-the-majority-class), per-arm resolve-pct,
    and a lift number = (acc - majority_baseline) / (1 - majority_baseline)
    that says "how much of the room above majority-class did we actually capture."
    """
    X: list[np.ndarray] = []
    y: list[int] = []
    skipped = 0
    resolved_per_arm = {"a": [0, 0], "b": [0, 0]}  # [resolved_count, total]
    for r in arm_a_records:
        label = _coerce_bool(r.get(resolved_field))
        if label is None:
            skipped += 1
            continue
        X.append(trajectory_counts(r["bpe"], vocab_idx))
        y.append(label)
        resolved_per_arm["a"][0] += label
        resolved_per_arm["a"][1] += 1
    for r in arm_b_records:
        label = _coerce_bool(r.get(resolved_field))
        if label is None:
            skipped += 1
            continue
        X.append(trajectory_counts(r["bpe"], vocab_idx))
        y.append(label)
        resolved_per_arm["b"][0] += label
        resolved_per_arm["b"][1] += 1

    n_total = len(y)
    n_pos = sum(y)
    n_neg = n_total - n_pos
    out: dict = {
        "resolved_field": resolved_field,
        "n_with_label": n_total,
        "n_skipped_missing_label": skipped,
        "n_resolved": n_pos,
        "n_unresolved": n_neg,
        "resolve_pct_per_arm": {
            "a": (round(100 * resolved_per_arm["a"][0] / resolved_per_arm["a"][1], 1)
                  if resolved_per_arm["a"][1] else None),
            "b": (round(100 * resolved_per_arm["b"][0] / resolved_per_arm["b"][1], 1)
                  if resolved_per_arm["b"][1] else None),
        },
    }
    if n_pos == 0 or n_neg == 0:
        out["skipped_reason"] = (
            "all-one-class outcome (n_pos={}, n_neg={}); classifier undefined."
            .format(n_pos, n_neg)
        )
        out["accuracy_mean"] = None
        out["accuracy_std"] = None
        return out
    if n_total < n_folds * 2:
        out["skipped_reason"] = (
            f"too few labeled records ({n_total}) for {n_folds}-fold CV"
        )
        out["accuracy_mean"] = None
        out["accuracy_std"] = None
        return out

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    accs: list[float] = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X_arr, y_arr):
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        clf.fit(X_arr[train_idx], y_arr[train_idx])
        accs.append(float(clf.score(X_arr[test_idx], y_arr[test_idx])))
    acc_mean = float(np.mean(accs))
    acc_std = float(np.std(accs))
    majority_baseline = max(n_pos, n_neg) / n_total
    lift_above_majority = (
        (acc_mean - majority_baseline) / (1 - majority_baseline)
        if majority_baseline < 1 else 0.0
    )
    out.update({
        "accuracy_mean": round(acc_mean, 3),
        "accuracy_std": round(acc_std, 3),
        "majority_class_baseline": round(majority_baseline, 3),
        "lift_above_majority": round(lift_above_majority, 3),
        "verdict": (
            "fingerprint predicts success above majority baseline"
            if acc_mean > majority_baseline + 0.05
            else "fingerprint does not meaningfully predict success"
        ),
    })
    return out


def per_arm_summary(
    arm_records: list[dict], vocab_idx: dict[str, int]
) -> dict:
    atom_counts: Counter = Counter()
    motif_counts: Counter = Counter()
    for r in arm_records:
        atom_counts.update(r["canonical"])
        motif_counts.update(r["bpe"])
    total_atoms = sum(atom_counts.values())
    top3 = atom_counts.most_common(3)
    return {
        "n": len(arm_records),
        "n_atoms": total_atoms,
        "n_motifs_total": sum(motif_counts.values()),
        "top3_atoms": [
            {"atom": a, "share_pct": round(100 * n / total_atoms, 1) if total_atoms else 0.0}
            for a, n in top3
        ],
    }


def compare_pair(
    arm_a_name: str,
    arm_a_records: list[dict],
    arm_b_name: str,
    arm_b_records: list[dict],
    vocab_idx: dict[str, int],
    n_splits: int,
    n_folds: int,
    seed: int,
    rng: random.Random,
    resolved_field: str | None = None,
) -> dict:
    mean_a = arm_mean(arm_a_records, vocab_idx)
    mean_b = arm_mean(arm_b_records, vocab_idx)
    across = jsd_sq(mean_a, mean_b)

    within_a_med, within_a_iqr = within_arm_noise(arm_a_records, vocab_idx, n_splits, rng)
    within_b_med, within_b_iqr = within_arm_noise(arm_b_records, vocab_idx, n_splits, rng)
    within_median = float(np.median([within_a_med, within_b_med]))
    snr = across / within_median if within_median > 0 else float("inf")

    acc_mean, acc_std = binary_probe(
        arm_a_records, arm_b_records, vocab_idx, n_folds, seed
    )

    success_result: dict | None = None
    if resolved_field is not None:
        success_result = success_probe(
            arm_a_records, arm_b_records, vocab_idx,
            resolved_field, n_folds, seed,
        )

    result: dict = {
        "arm_a": arm_a_name,
        "arm_b": arm_b_name,
        "n_a": len(arm_a_records),
        "n_b": len(arm_b_records),
        "across_arm_jsd": round(across, 4),
        "within_arm_jsd_median": round(within_median, 4),
        "within_arm_jsd_per_arm": {
            arm_a_name: {"median": round(within_a_med, 4), "iqr": round(within_a_iqr, 4)},
            arm_b_name: {"median": round(within_b_med, 4), "iqr": round(within_b_iqr, 4)},
        },
        "snr": round(snr, 2),
        "snr_threshold": SNR_THRESHOLD,
        "drift_verdict": "moved procedure" if snr >= SNR_THRESHOLD else "sampled noise",
        "arm_probe_accuracy_mean": round(acc_mean, 3),
        "arm_probe_accuracy_std": round(acc_std, 3),
        "arm_probe_chance": 0.5,
        "arm_probe_verdict": (
            "arms distinguishable above chance"
            if acc_mean > PROBE_THRESHOLD
            else "arms not reliably distinguishable"
        ),
    }
    if success_result is not None:
        result["success_probe"] = success_result
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", required=True, type=Path,
                   help="JSONL of trajectory records.")
    p.add_argument("--output", required=True, type=Path,
                   help="Output JSON path.")
    p.add_argument("--group-field", default=DEFAULT_GROUP_FIELD,
                   help=f"Record field whose value names the arm "
                        f"(default: {DEFAULT_GROUP_FIELD}).")
    p.add_argument("--arms", default=None,
                   help="Comma-separated arm labels in sweep order. "
                        "If omitted, observed arms sorted alphanumerically.")
    p.add_argument("--mode", choices=["sweep", "pairwise-all"], default=DEFAULT_MODE,
                   help=f"sweep: consecutive pairs in --arms order. "
                        f"pairwise-all: every C(n,2) pair "
                        f"(default: {DEFAULT_MODE}).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Random seed (default: {DEFAULT_SEED}).")
    p.add_argument("--n-split-halves", type=int, default=DEFAULT_SPLIT_HALVES,
                   help=f"Split-halves resamples per arm for the within-arm "
                        f"noise floor (default: {DEFAULT_SPLIT_HALVES}).")
    p.add_argument("--n-cv-folds", type=int, default=DEFAULT_CV_FOLDS,
                   help=f"Stratified CV folds for the binary probes "
                        f"(default: {DEFAULT_CV_FOLDS}).")
    p.add_argument("--resolved-field", default=None,
                   help="Optional field name carrying a per-record outcome "
                        "(True/False, 1/0, 'resolved'/'unresolved', etc.). "
                        "When set, runs a second classifier per pair on "
                        "fingerprint -> resolved, alongside the arm probe.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"Loading {args.input} ...")
    records = [json.loads(line) for line in args.input.open() if line.strip()]
    print(f"  {len(records)} records")

    # Group by arm
    by_arm: dict[str, list[dict]] = {}
    skipped = 0
    for r in records:
        if args.group_field not in r:
            skipped += 1
            continue
        arm = str(r[args.group_field])
        by_arm.setdefault(arm, []).append(r)
    if skipped:
        print(f"  WARN: {skipped} records missing field '{args.group_field}'")
    print(f"  arms observed: {sorted(by_arm.keys())}")

    # Resolve arm order
    if args.arms:
        arm_order = [a.strip() for a in args.arms.split(",")]
        missing = [a for a in arm_order if a not in by_arm]
        if missing:
            raise SystemExit(f"--arms references arms not in data: {missing}")
        # Filter records to only the named arms (drop unnamed)
        by_arm = {a: by_arm[a] for a in arm_order}
    else:
        arm_order = sorted(by_arm.keys())
        if args.mode == "sweep":
            print("  WARN: --arms not given; sweep mode will use "
                  "alphanumeric order, which is wrong for continuous knobs.")

    if len(arm_order) < 2:
        raise SystemExit("Need at least 2 arms to compare.")

    # Build shared BPE vocab across all included arms (Mode-2 recipe rule).
    pooled = [r for a in arm_order for r in by_arm[a]]
    vocab_idx = build_vocab(pooled)
    print(f"  shared vocab size: {len(vocab_idx)}")

    # Per-arm summary
    per_arm = {a: per_arm_summary(by_arm[a], vocab_idx) for a in arm_order}

    # Pair list
    pairs: list[tuple[str, str]] = []
    if args.mode == "sweep":
        pairs = list(zip(arm_order[:-1], arm_order[1:]))
    else:  # pairwise-all
        for i in range(len(arm_order)):
            for j in range(i + 1, len(arm_order)):
                pairs.append((arm_order[i], arm_order[j]))

    print(f"  comparing {len(pairs)} pair(s) in {args.mode} mode")

    pair_results: list[dict] = []
    print()
    for a, b in pairs:
        result = compare_pair(
            a, by_arm[a], b, by_arm[b],
            vocab_idx, args.n_split_halves, args.n_cv_folds, args.seed, rng,
            resolved_field=args.resolved_field,
        )
        pair_results.append(result)
        print(f"== {a} -> {b} ==")
        print(f"  N: {result['n_a']} / {result['n_b']}")
        print(f"  across-arm JSD:    {result['across_arm_jsd']:.4f}")
        print(f"  within-arm median: {result['within_arm_jsd_median']:.4f}")
        print(f"  SNR:               {result['snr']:.2f}x   ({result['drift_verdict']})")
        print(f"  arm probe acc:     {result['arm_probe_accuracy_mean']:.3f} "
              f"± {result['arm_probe_accuracy_std']:.3f}  "
              f"({result['arm_probe_verdict']})")
        if "success_probe" in result:
            sp = result["success_probe"]
            if sp.get("accuracy_mean") is None:
                print(f"  success probe:     SKIPPED ({sp.get('skipped_reason', 'unknown')})")
            else:
                ra, rb = sp["resolve_pct_per_arm"]["a"], sp["resolve_pct_per_arm"]["b"]
                print(f"  resolve%:          {a}={ra}%  {b}={rb}%")
                print(f"  success probe acc: {sp['accuracy_mean']:.3f} ± {sp['accuracy_std']:.3f}  "
                      f"(majority baseline {sp['majority_class_baseline']:.3f}, "
                      f"lift {sp['lift_above_majority']:+.3f})")
                print(f"                     {sp['verdict']}")
        print()

    payload = {
        "input": str(args.input),
        "group_field": args.group_field,
        "arm_order": arm_order,
        "mode": args.mode,
        "config": {
            "seed": args.seed,
            "n_split_halves": args.n_split_halves,
            "n_cv_folds": args.n_cv_folds,
            "snr_threshold": SNR_THRESHOLD,
            "probe_threshold": PROBE_THRESHOLD,
            "vocab_size": len(vocab_idx),
            "within_arm_noise_method": (
                "split-halves: randomly partition each arm in half K times, "
                "compute JSD between the two half-means; report median + IQR."
            ),
        },
        "per_arm": per_arm,
        "pairs": pair_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
