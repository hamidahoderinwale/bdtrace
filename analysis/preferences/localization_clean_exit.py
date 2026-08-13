"""Localization analysis stratified by exit status.

Addresses the budget-cut confounder: a type-A failure (never reached gold file)
is ambiguous — did the agent fail to navigate, or did it run out of budget
mid-search? This script:

1. Classifies each trajectory as clean-exit vs budget-cut.
2. Recomputes type-A / type-B failure mode rates for clean-exit failures only.
3. Bootstraps 95% CIs on all behavioral metrics (type-A rate, localization rate).

Reads:
    output/trajectories/.cache/{agent}/*.json
    output/trajectories/lite_all_models.parquet
    output/resolved_traces_lite_full.jsonl
Writes:
    output/paper2_pilot/localization_clean_exit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.preferences.localization import (
    load_gold_files, load_pass_fail, first_localization_step, AGENT_MAP
)
from scripts.theme import AGENT_ORDER

CACHE = ROOT / "output" / "trajectories" / ".cache"
OUT   = ROOT / "output" / "paper2_pilot"

BUDGET_STATUSES = {"submitted (exit_cost)", "exit_cost"}


def bootstrap_ci(
    arr: np.ndarray,
    stat_fn,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(rng_seed)
    boot = np.array([stat_fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)])
    lo = float(np.percentile(boot, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boot, 100 * (1 - (1 - ci) / 2)))
    return lo, hi


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    gold_files = load_gold_files()
    pass_fail  = load_pass_fail()

    records = []
    for agent_dir in sorted(CACHE.iterdir()):
        if not agent_dir.is_dir():
            continue
        agent_short = AGENT_MAP.get(agent_dir.name)
        if agent_short is None:
            continue
        for traj_file in sorted(agent_dir.glob("*.json")):
            iid    = traj_file.stem
            gold   = gold_files.get(iid)
            passed = pass_fail.get((agent_dir.name, iid))
            if gold is None or passed is None:
                continue
            raw  = json.loads(traj_file.read_text())
            traj = raw.get("trajectory", [])
            n    = len(traj)
            if n == 0:
                continue
            exit_status = raw.get("info", {}).get("exit_status", "")
            is_budget_cut = exit_status in BUDGET_STATUSES

            loc_step  = first_localization_step(traj, gold)
            localized = loc_step is not None
            records.append({
                "agent":        agent_short,
                "instance_id":  iid,
                "passed":       passed,
                "n_steps":      n,
                "localized":    localized,
                "loc_step":     loc_step,
                "steps_after":  (n - loc_step) if localized else None,
                "exit_status":  exit_status,
                "budget_cut":   is_budget_cut,
            })

    df = pd.DataFrame(records)

    print("\n=== Exit status breakdown ===")
    for agent in AGENT_ORDER:
        sub = df[df["agent"] == agent]
        bc  = sub["budget_cut"].sum()
        print(f"  {agent:12s}  total={len(sub)}  budget_cut={bc} ({bc/len(sub):.0%})  "
              f"clean={len(sub)-bc} ({(len(sub)-bc)/len(sub):.0%})")

    results = {}
    for agent in AGENT_ORDER:
        sub  = df[df["agent"] == agent]
        n    = len(sub)
        fails_all   = sub[~sub["passed"]]
        fails_clean = sub[~sub["passed"] & ~sub["budget_cut"]]

        # --- type-A rate among ALL failures ---
        if len(fails_all):
            typeA_all = (~fails_all["localized"]).mean()
            arr_all   = (~fails_all["localized"]).astype(float).values
            lo_all, hi_all = bootstrap_ci(arr_all, np.mean)
        else:
            typeA_all = lo_all = hi_all = float("nan")

        # --- type-A rate among CLEAN-EXIT failures only ---
        if len(fails_clean):
            typeA_clean = (~fails_clean["localized"]).mean()
            arr_clean   = (~fails_clean["localized"]).astype(float).values
            lo_clean, hi_clean = bootstrap_ci(arr_clean, np.mean)
        else:
            typeA_clean = lo_clean = hi_clean = float("nan")

        # --- overall localization rate (all trajectories) ---
        loc_arr = sub["localized"].astype(float).values
        loc_rate = loc_arr.mean()
        lo_loc, hi_loc = bootstrap_ci(loc_arr, np.mean)

        # --- localization rate for failing trajectories ---
        fail_loc_arr = fails_all["localized"].astype(float).values
        fail_loc     = fail_loc_arr.mean() if len(fail_loc_arr) else float("nan")
        lo_fl, hi_fl = bootstrap_ci(fail_loc_arr, np.mean) if len(fail_loc_arr) > 1 else (float("nan"), float("nan"))

        budget_cut_n    = int(sub["budget_cut"].sum())
        budget_cut_rate = budget_cut_n / n

        results[agent] = {
            "n_total":            n,
            "n_fail_all":         len(fails_all),
            "n_fail_clean":       len(fails_clean),
            "budget_cut_n":       budget_cut_n,
            "budget_cut_rate":    float(budget_cut_rate),
            "typeA_rate_all_failures": {
                "mean": float(typeA_all),
                "ci95_lo": float(lo_all),
                "ci95_hi": float(hi_all),
            },
            "typeA_rate_clean_exit_failures": {
                "mean": float(typeA_clean),
                "ci95_lo": float(lo_clean),
                "ci95_hi": float(hi_clean),
            },
            "localization_rate_all": {
                "mean": float(loc_rate),
                "ci95_lo": float(lo_loc),
                "ci95_hi": float(hi_loc),
            },
            "localization_rate_failing": {
                "mean": float(fail_loc),
                "ci95_lo": float(lo_fl),
                "ci95_hi": float(hi_fl),
            },
        }

        print(f"\n{agent}:")
        print(f"  Type-A rate (all failures):        {typeA_all:.0%}  [{lo_all:.0%}, {hi_all:.0%}]")
        print(f"  Type-A rate (clean-exit failures): {typeA_clean:.0%}  [{lo_clean:.0%}, {hi_clean:.0%}]")
        print(f"  Localization rate (all):           {loc_rate:.0%}  [{lo_loc:.0%}, {hi_loc:.0%}]")
        print(f"  Localization rate (failing):        {fail_loc:.0%}  [{lo_fl:.0%}, {hi_fl:.0%}]")

    out_path = OUT / "localization_clean_exit.json"
    out_path.write_text(json.dumps({"by_agent": results}, indent=2, default=float))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
