"""OOD-predicts-difficulty correlation (Paper 1 load-bearing experiment).

For each SWE-bench Lite instance with a resolved trace, compute a leave-one-out
OOD score against the edit-op vocabulary of all OTHER Lite instances. Correlate
with per-instance pass rate across 84 leaderboard agents. Baseline: does OOD
predict difficulty better than raw patch size?

Positive outcome: higher OOD → lower pass rate (negative correlation),
|OOD corr| > |patch_size corr|. Would validate OOD as a meaningful difficulty
signal beyond trivial complexity proxies.

Usage:
    python -m analysis.pdiff.run_ood_difficulty_correlation
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from analysis.pdiff import view_from_trace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "output" / "pdiff_smoke_test"

RESOLVED_LITE = PROJECT_ROOT / "output" / "resolved_traces_lite_full.jsonl"
LEADERBOARD = PROJECT_ROOT / "output" / "leaderboard" / "lite_results.json"

N_BOOTSTRAP = 1000
RNG_SEED = 0


def load_agent_pass_rates(path: Path, all_instance_ids: set[str]) -> dict[str, float]:
    """Compute per-instance pass rate across leaderboard agents.

    lite_results.json stores only successes per agent: each agent's map contains
    the subset of instances that agent resolved. Absent entries mean "this agent
    failed or did not attempt". We use the total number of agents in the file
    as the denominator, so every instance in `all_instance_ids` gets a rate in
    [0, 1], including 0.0 for instances no agent resolved.
    """
    with open(path) as fh:
        data = json.load(fh)

    n_agents = len(data)
    passes: Counter = Counter()
    for result_map in data.values():
        for inst, passed in result_map.items():
            if passed:
                passes[inst] += 1

    return {iid: passes.get(iid, 0) / n_agents for iid in all_instance_ids}


def load_traces_with_views(path: Path) -> dict[str, Any]:
    """Load resolved traces and build views keyed by instance_id.

    Only instances where view_from_trace yields has_edits are kept.
    """
    views: dict[str, Any] = {}
    traces: dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            t = json.loads(line)
            iid = t.get("instance_id")
            if not iid:
                continue
            v = view_from_trace(t)
            if v.has_edits:
                views[iid] = v
                traces[iid] = t
    return {"views": views, "traces": traces}


def per_instance_patch_size(traces: dict[str, dict]) -> dict[str, int]:
    """Sum of lines_added + lines_removed across code_change events."""
    out: dict[str, int] = {}
    for iid, trace in traces.items():
        total = 0
        for ev in trace.get("events", []) or []:
            if not isinstance(ev, dict) or ev.get("type") != "code_change":
                continue
            details = ev.get("details") or {}
            total += int(details.get("lines_added", 0) or 0)
            total += int(details.get("lines_removed", 0) or 0)
        out[iid] = total
    return out


def build_global_edit_counter(views: dict[str, Any]) -> Counter:
    """Total count of each edit-op across all instances (used for LOO)."""
    c: Counter = Counter()
    for v in views.values():
        c.update(v.edits)
    return c


def loo_ood_score(view: Any, global_counter: Counter) -> float:
    """Leave-one-out OOD score at the edit level.

    For instance i, an op appears in the LOO reference iff at least one
    OTHER instance uses it. Since a view's edits is a frozenset, each op
    appears once per instance, so global_counter[op] > 1 means the op is
    present in at least one other instance.
    """
    items = view.edits
    if not items:
        return 0.0
    novel = sum(1 for op in items if global_counter[op] <= 1)
    return novel / len(items)


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap 95% CI for Spearman correlation."""
    n = len(x)
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        rho, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(rho):
            samples.append(rho)
    if not samples:
        return (float("nan"), float("nan"))
    arr = np.array(samples)
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


def correlate(x: np.ndarray, y: np.ndarray, *, n_bootstrap: int = 0) -> dict[str, Any]:
    """Compute Spearman and Pearson. Bootstrap CI only if n_bootstrap > 0."""
    rho, _ = stats.spearmanr(x, y)
    r, _ = stats.pearsonr(x, y)
    out: dict[str, Any] = {
        "spearman": float(rho),
        "pearson": float(r),
        "n": int(len(x)),
    }
    if n_bootstrap > 0:
        rng = np.random.default_rng(RNG_SEED)
        lo, hi = bootstrap_spearman(x, y, n_bootstrap, rng)
        out["spearman_ci_95"] = [lo, hi]
        out["n_bootstrap"] = n_bootstrap
    return out


def describe(arr: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RESOLVED_LITE.exists():
        print(f"Missing: {RESOLVED_LITE}")
        return 1
    if not LEADERBOARD.exists():
        print(f"Missing: {LEADERBOARD}")
        return 1

    print("=== OOD-predicts-difficulty correlation ===\n")

    loaded = load_traces_with_views(RESOLVED_LITE)
    views = loaded["views"]
    traces = loaded["traces"]
    print(f"Loaded {len(views)} instances with extractable edits "
          f"(dropped {300 - len(views)} of 300 due to no extractable edits)")

    pass_rates_all = load_agent_pass_rates(LEADERBOARD, set(views))
    n_unsolved = sum(1 for v in pass_rates_all.values() if v == 0.0)
    print(f"Leaderboard pass rates computed for all {len(pass_rates_all)} views "
          f"(denominator = number of agents in file)")
    print(f"Instances no agent resolved: {n_unsolved} (expect ~35 per prior findings)")

    shared = sorted(set(views) & set(pass_rates_all))
    dropped_no_leaderboard = len(views) - len(shared)
    print(f"Intersection: {len(shared)} instances "
          f"(dropped {dropped_no_leaderboard} views)\n")

    if len(shared) < 30:
        print("Too few shared instances for correlation; aborting.")
        return 1

    global_counter = build_global_edit_counter({k: views[k] for k in shared})
    patch_sizes = per_instance_patch_size(traces)

    ood_list: list[float] = []
    ps_list: list[float] = []
    pr_list: list[float] = []
    per_instance: list[dict[str, Any]] = []
    for iid in shared:
        ood = loo_ood_score(views[iid], global_counter)
        ps = patch_sizes.get(iid, 0)
        pr = pass_rates_all[iid]
        ood_list.append(ood)
        ps_list.append(ps)
        pr_list.append(pr)
        per_instance.append({
            "instance_id": iid,
            "ood": ood,
            "pass_rate": pr,
            "patch_size": ps,
        })

    ood_arr = np.array(ood_list)
    ps_arr = np.array(ps_list, dtype=float)
    pr_arr = np.array(pr_list)

    ood_stats = describe(ood_arr)
    pr_stats = describe(pr_arr)
    ps_stats = describe(ps_arr)

    ood_vs_pr = correlate(ood_arr, pr_arr, n_bootstrap=N_BOOTSTRAP)
    ps_vs_pr = correlate(ps_arr, pr_arr, n_bootstrap=0)

    print(f"Predictor -> pass_rate correlations (n = {len(shared)})")
    print(f"{'predictor':<14} {'spearman':>10} {'95% CI':>22} {'pearson':>10}")
    ci = ood_vs_pr.get("spearman_ci_95", [float("nan"), float("nan")])
    print(f"{'OOD (LOO)':<14} {ood_vs_pr['spearman']:>10.4f} "
          f"[{ci[0]:>+.4f}, {ci[1]:>+.4f}] {ood_vs_pr['pearson']:>10.4f}")
    print(f"{'patch_size':<14} {ps_vs_pr['spearman']:>10.4f} "
          f"{'':>22} {ps_vs_pr['pearson']:>10.4f}")

    ood_beats_ps = abs(ood_vs_pr["spearman"]) > abs(ps_vs_pr["spearman"])
    delta = abs(ood_vs_pr["spearman"]) - abs(ps_vs_pr["spearman"])
    print(f"\nOOD |spearman| - patch_size |spearman| = {delta:+.4f}")
    print(f"OOD beats patch_size as difficulty predictor: {ood_beats_ps}")

    print(f"\nOOD distribution:        {ood_stats}")
    print(f"pass_rate distribution:  {pr_stats}")
    print(f"patch_size distribution: {ps_stats}")

    result = {
        "n_instances": len(shared),
        "n_dropped_no_edits": 300 - len(views),
        "n_dropped_no_leaderboard": dropped_no_leaderboard,
        "ood_stats": ood_stats,
        "pass_rate_stats": pr_stats,
        "patch_size_stats": ps_stats,
        "ood_vs_pass_rate": ood_vs_pr,
        "patch_size_vs_pass_rate": ps_vs_pr,
        "comparison": {
            "ood_beats_patch_size": ood_beats_ps,
            "delta_abs_spearman": float(delta),
        },
        "per_instance": per_instance,
    }

    out_path = OUT_DIR / "ood_difficulty_correlation.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
