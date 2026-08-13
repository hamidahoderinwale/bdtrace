"""File localization analysis.

For each trajectory, measures whether the agent ever opened or edited the
gold-patch file, and at what step. Compares localization between passing
and failing trajectories and across agents.

The hypothesis: agents that find the right file early pass more often.
Failing trajectories either never reach the gold file or reach it too late.

Reads:
    output/trajectories/.cache/{agent}/*.json     (raw trajectories)
    output/trajectories/lite_all_models.parquet   (ground-truth pass/fail)
    output/resolved_traces_lite_full.jsonl        (gold patch files)
Writes:
    output/paper2_pilot/localization.json
    output/figures/fig_localization.png
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import altair as alt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, GRAY, AGENT_COLORS, AGENT_ORDER
register()

CACHE   = ROOT / "output" / "trajectories" / ".cache"
OUT     = ROOT / "output" / "paper2_pilot"
FIG_OUT = ROOT / "output" / "figures"

AGENT_MAP = {
    "20240402_sweagent_claude3opus":    "Claude-3",
    "20240402_sweagent_gpt4":           "GPT-4",
    "20240620_sweagent_claude3.5sonnet": "Claude-3.5",
    "20240728_sweagent_gpt4o":           "GPT-4o",
}

# Patterns that indicate the agent accessed a file
_FILE_RE = re.compile(r"(?:^|\s)([a-zA-Z0-9_./\-]+\.py)", re.MULTILINE)


def load_gold_files() -> dict[str, set[str]]:
    gold: dict[str, set[str]] = {}
    with open(ROOT / "output/resolved_traces_lite_full.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            files = {
                ev["details"]["file_path"]
                for ev in r["events"]
                if ev["type"] == "code_change" and ev["details"].get("file_path")
            }
            if files:
                gold[r["instance_id"]] = files
    return gold


def load_pass_fail() -> dict[tuple[str, str], bool]:
    df = pd.read_parquet(ROOT / "output/trajectories/lite_all_models.parquet")
    return {
        (row["model_id"], row["instance_id"]): bool(row["passed"])
        for _, row in df.iterrows()
    }


def files_accessed_at_step(step: dict) -> set[str]:
    """Extract .py file paths mentioned in action or state at this step."""
    files: set[str] = set()
    state = step.get("state", "{}")
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except Exception:
            state = {}
    open_file = state.get("open_file", "")
    if open_file and open_file != "n/a" and open_file.endswith(".py"):
        # Strip leading repo path (e.g. /astropy__astropy/)
        parts = Path(open_file).parts
        # Find first non-root component that looks like a package path
        for i, p in enumerate(parts):
            if "__" in p:  # repo dir like astropy__astropy-12907
                rel = "/".join(parts[i + 1:])
                if rel:
                    files.add(rel)
                break
        else:
            files.add(open_file.lstrip("/"))

    action = step.get("action", "")
    for m in _FILE_RE.finditer(action):
        fp = m.group(1)
        if not fp.startswith(".") and "/" in fp:
            files.add(fp)

    return files


def first_localization_step(
    traj: list[dict],
    gold: set[str],
) -> int | None:
    """Return 0-indexed step at which agent first accesses the gold file, or None."""
    for i, step in enumerate(traj):
        accessed = files_accessed_at_step(step)
        if accessed & gold:
            return i
    return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)

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

            loc_step = first_localization_step(traj, gold)
            localized = loc_step is not None

            records.append({
                "agent":        agent_short,
                "instance_id":  iid,
                "passed":       passed,
                "n_steps":      n,
                "localized":    localized,
                "loc_step":     loc_step if localized else n,   # n = never
                "loc_frac":     (loc_step / n) if localized else 1.0,
            })

    df = pd.DataFrame(records)

    # ── aggregate stats ────────────────────────────────────────────────────────
    print(f"\n{len(df)} trajectories\n")

    print("Localization rate (% that ever reach gold file):")
    for agent in AGENT_ORDER:
        sub = df[df["agent"] == agent]
        p = sub[sub["passed"]]
        f = sub[~sub["passed"]]
        print(f"  {agent:12s}  pass={p['localized'].mean():.0%} (n={len(p)})  "
              f"fail={f['localized'].mean():.0%} (n={len(f)})")

    print("\nAmong trajectories that localized — median step:")
    for agent in AGENT_ORDER:
        sub = df[(df["agent"] == agent) & df["localized"]]
        p = sub[sub["passed"]]["loc_step"]
        f = sub[~sub["passed"]]["loc_step"]
        if len(p) and len(f):
            stat, pval = mannwhitneyu(p, f, alternative="less")
            print(f"  {agent:12s}  pass median={p.median():.0f}  "
                  f"fail median={f.median():.0f}  MW p={pval:.4f}")

    print("\nOverall localization rate:")
    p_all = df[df["passed"]]
    f_all = df[~df["passed"]]
    print(f"  pass: {p_all['localized'].mean():.0%}  fail: {f_all['localized'].mean():.0%}")

    stat, pval = mannwhitneyu(
        p_all["loc_frac"], f_all["loc_frac"], alternative="less"
    )
    print(f"  loc_frac MW p (pass < fail): {pval:.4f}")

    # ── summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "n_trajectories": len(df),
        "overall": {
            "pass_localization_rate": float(p_all["localized"].mean()),
            "fail_localization_rate": float(f_all["localized"].mean()),
            "mw_p_loc_frac": float(pval),
        },
        "by_agent": {},
    }
    for agent in AGENT_ORDER:
        sub = df[df["agent"] == agent]
        p = sub[sub["passed"]]
        f = sub[~sub["passed"]]
        entry: dict = {
            "pass_localization_rate": float(p["localized"].mean()),
            "fail_localization_rate": float(f["localized"].mean()),
            "pass_median_loc_step":   float(p[p["localized"]]["loc_step"].median()) if p["localized"].any() else None,
            "fail_median_loc_step":   float(f[f["localized"]]["loc_step"].median()) if f["localized"].any() else None,
        }
        summary["by_agent"][agent] = entry

    (OUT / "localization.json").write_text(json.dumps(summary, indent=2))

    # ── figure: mean localization rate ± CI, by agent × outcome ───────────────
    rows = []
    for agent in AGENT_ORDER:
        for passed, label in [(True, "pass"), (False, "fail")]:
            sub = df[(df["agent"] == agent) & (df["passed"] == passed)]
            n   = len(sub)
            if n == 0:
                continue
            mean = sub["localized"].mean()
            se   = np.sqrt(mean * (1 - mean) / n)
            rows.append({
                "agent":  agent,
                "outcome": label,
                "mean":   mean,
                "lo":     max(0.0, mean - 1.96 * se),
                "hi":     min(1.0, mean + 1.96 * se),
            })
    plot_df = pd.DataFrame(rows)

    color_scale = alt.Scale(
        domain=AGENT_ORDER,
        range=[AGENT_COLORS[a] for a in AGENT_ORDER],
    )

    base = alt.Chart(plot_df).encode(
        x=alt.X("mean:Q",
                title="Proportion reaching gold file",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", values=[0, 0.25, 0.5, 0.75, 1.0])),
        y=alt.Y("agent:N",
                sort=AGENT_ORDER,
                axis=alt.Axis(title=None)),
        color=alt.Color("agent:N", scale=color_scale, legend=None),
        yOffset=alt.YOffset("outcome:N", sort=["pass", "fail"],
                            scale=alt.Scale(range=[-10, 10])),
    )

    points = base.mark_point(filled=True, size=80, strokeWidth=0)
    errors = base.mark_errorbar().encode(
        x=alt.X("lo:Q", title="Proportion reaching gold file"),
        x2=alt.X2("hi:Q"),
    )
    value_labels = base.mark_text(align="left", dx=7, fontSize=10, color="#555555").encode(
        text=alt.Text("mean:Q", format=".0%"),
        color=alt.value("#555555"),
    )
    outcome_labels = base.mark_text(align="right", dx=-7, fontSize=9).encode(
        text="outcome:N",
        color=alt.value("#999999"),
        x=alt.value(0),
    )

    chart = (
        (points + errors + value_labels + outcome_labels)
        .resolve_scale(x="shared")
        .properties(
            width=320, height=160,
            title=alt.TitleParams(
                "Gold file localization rate by agent and outcome",
                fontSize=13, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
    )

    out_path = FIG_OUT / "fig_localization.png"
    chart.save(str(out_path), scale_factor=2)
    print(f"\nSaved {out_path}")
    print(f"Saved {OUT / 'localization.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
