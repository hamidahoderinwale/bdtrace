"""
Controlled evaluation analysis for the procedural fingerprinting paper.

Defines 5 procedural constraints as predicates on canonical action sequences,
then measures pass rates among satisfying vs non-satisfying trajectories for
each constraint × agent combination.

Framing: instead of re-running agents with enforced constraints, we retroactively
partition the trace corpus by whether each trajectory satisfies a procedural spec,
then compare outcomes — controlling for task difficulty via the matched-task design.
"""

import json
import os
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "output", "paper2_pilot")
FIG_DIR = os.path.join(REPO_ROOT, "output", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_dataset():
    """Return list of dicts with keys: agent, instance_id, resolved, canonical."""
    # BPE sequences carry canonical atom sequences
    seq_path = os.path.join(OUT_DIR, "bpe_sequences_extended.jsonl")
    seqs = {}
    with open(seq_path) as f:
        for line in f:
            r = json.loads(line)
            seqs[(r["submission"], r["instance_id"])] = r["canonical"]

    # extended_pass_fail carries per-submission resolved lists
    pf_path = os.path.join(OUT_DIR, "extended_pass_fail.json")
    pf = json.load(open(pf_path))

    # submission → agent label (from megatable)
    mega = json.load(open(os.path.join(OUT_DIR, "per_agent_megatable.json")))
    # build from bpe sequences: submission → agent
    sub_to_agent = {}
    with open(seq_path) as f:
        for line in f:
            r = json.loads(line)
            sub_to_agent[r["submission"]] = r["agent"]

    rows = []
    for submission, outcome_dict in pf.items():
        agent = sub_to_agent.get(submission)
        if agent is None:
            continue
        resolved_set = set(outcome_dict.get("resolved", []))
        no_gen = set(outcome_dict.get("no_generation", []))
        no_logs = set(outcome_dict.get("no_logs", []))
        skip = no_gen | no_logs

        for (sub, iid), canonical in seqs.items():
            if sub != submission:
                continue
            if iid in skip:
                continue
            rows.append({
                "agent": agent,
                "submission": submission,
                "instance_id": iid,
                "resolved": iid in resolved_set,
                "canonical": canonical,
            })

    return rows


# ---------------------------------------------------------------------------
# Procedural constraints
# Predicate: canonical (list of atom strings) → bool
# ---------------------------------------------------------------------------

TEST_ATOMS = {
    "RUN_PYTHON_TEST_PY", "RUN_PYTEST_TEST_PY", "RUN_PYTEST_ALL",
    "RUN_PYTHON_ALL", "RUN_TEST_SCRIPT",
}
EXPLORE_ATOMS = {"SEARCH", "FIND_FILE", "SHELL_GREP", "SHELL_LS"}
EDIT_ATOMS = {"EDIT_SRC_PY", "EDIT", "EDIT_CONFIG_PY", "EDIT_OTHER"}
RUN_ATOMS = {"RUN_PYTHON_REPRO_PY", "RUN_PYTHON_SRC_PY"} | TEST_ATOMS


def c_test_before_submit(canonical):
    """At least one test-run action appears before SUBMIT."""
    last_submit = None
    for i, a in enumerate(canonical):
        if a == "SUBMIT":
            last_submit = i
            break
    if last_submit is None:
        last_submit = len(canonical)
    return any(a in TEST_ATOMS for a in canonical[:last_submit])


def c_repro_step(canonical):
    """Trajectory includes a reproduction script step."""
    return "CREATE_REPRO_PY" in canonical


def c_search_before_edit(canonical):
    """At least one search/browse action precedes the first source edit."""
    first_edit = None
    for i, a in enumerate(canonical):
        if a in EDIT_ATOMS:
            first_edit = i
            break
    if first_edit is None:
        return True  # no edit at all — neutral; exclude from analysis
    return any(a in EXPLORE_ATOMS for a in canonical[:first_edit])


def c_no_exit_error(canonical):
    """Trajectory exits cleanly (no EXIT_ERROR atom).
    NOTE: EXIT_ERROR trajectories never reach SUBMIT (0/421 in corpus),
    so this constraint is near-trivially predictive. Excluded from main analysis.
    """
    return "EXIT_ERROR" not in canonical


def c_low_edit_retry(canonical):
    """Edit retry rate below 50%: fewer than half of edits are consecutive retries."""
    edit_seq = [a for a in canonical if a in EDIT_ATOMS | {"EDIT_SRC_PY"}]
    if len(edit_seq) < 2:
        return True
    retries = sum(1 for i in range(1, len(edit_seq)) if edit_seq[i] == edit_seq[i - 1])
    return retries / len(edit_seq) < 0.50


CONSTRAINTS = {
    "Test before submit": c_test_before_submit,
    "Repro step": c_repro_step,
    "Search before edit": c_search_before_edit,
    "Low edit retry rate": c_low_edit_retry,
}

# EXIT_ERROR excluded: trajectories with EXIT_ERROR never reach SUBMIT (0/421),
# making it trivially predictive of failure rather than a meaningful procedural signal.


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return (max(0, centre - margin), min(1, centre + margin))


def run_analysis(rows):
    """
    For each constraint × agent, compute:
      - n_satisfies, pass_rate_satisfies, ci_satisfies
      - n_not, pass_rate_not, ci_not
      - delta = pass_rate_satisfies - pass_rate_not
      - p_value (Fisher exact test)
    """
    # group by agent
    by_agent = defaultdict(list)
    for r in rows:
        by_agent[r["agent"]].append(r)

    results = {}  # constraint_name → {agent → stats_dict}

    for cname, cfunc in CONSTRAINTS.items():
        results[cname] = {}
        for agent, agent_rows in sorted(by_agent.items()):
            satisfies, not_satisfies = [], []
            for r in agent_rows:
                try:
                    sat = cfunc(r["canonical"])
                except Exception:
                    continue
                if sat:
                    satisfies.append(r["resolved"])
                else:
                    not_satisfies.append(r["resolved"])

            n_s = len(satisfies)
            n_n = len(not_satisfies)
            k_s = sum(satisfies)
            k_n = sum(not_satisfies)

            pr_s = k_s / n_s if n_s else float("nan")
            pr_n = k_n / n_n if n_n else float("nan")
            ci_s = wilson_ci(k_s, n_s)
            ci_n = wilson_ci(k_n, n_n)
            delta = pr_s - pr_n if (n_s and n_n) else float("nan")

            # Fisher exact test
            if n_s >= 5 and n_n >= 5:
                table = [[k_s, n_s - k_s], [k_n, n_n - k_n]]
                _, p = stats.fisher_exact(table)
            else:
                p = float("nan")

            results[cname][agent] = {
                "n_satisfies": n_s,
                "n_not": n_n,
                "k_satisfies": k_s,
                "k_not": k_n,
                "pr_satisfies": pr_s,
                "pr_not": pr_n,
                "ci_satisfies": ci_s,
                "ci_not": ci_n,
                "delta": delta,
                "p_value": p,
            }

    return results


def print_table(results, agents):
    print("\n=== Pass rate: satisfies constraint vs does not ===")
    for cname, agent_stats in results.items():
        print(f"\n{cname}")
        print(f"  {'Agent':<28} {'n_sat':>6} {'n_not':>6} {'pr_sat':>7} {'pr_not':>7} {'delta':>7} {'p':>8}")
        for agent in agents:
            if agent not in agent_stats:
                continue
            s = agent_stats[agent]
            p_str = f"{s['p_value']:.3f}" if not math.isnan(s['p_value']) else "  n/a"
            d_str = f"{s['delta']:+.3f}" if not math.isnan(s['delta']) else "  n/a"
            print(f"  {agent:<28} {s['n_satisfies']:>6} {s['n_not']:>6} "
                  f"{s['pr_satisfies']:>7.3f} {s['pr_not']:>7.3f} {d_str:>7} {p_str:>8}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# Agent display order and colors (matching paper palette)
AGENT_ORDER = [
    "Claude-3", "Claude-3.5", "Claude-3.7-thinking", "Claude-4",
    "GPT-4", "GPT-4o",
    "DARS+R1", "Agentless+Claude-3.5", "Moatless+V3",
]

AGENT_COLORS = {
    "Claude-3":             "#c9693a",
    "Claude-3.5":           "#c9693a",
    "Claude-3.7-thinking":  "#c9693a",
    "Claude-4":             "#c9693a",
    "GPT-4":                "#3a6fc9",
    "GPT-4o":               "#3a6fc9",
    "DARS+R1":              "#5a8a50",
    "Agentless+Claude-3.5": "#5a8a50",
    "Moatless+V3":          "#5a8a50",
}

AGENT_ALPHA = {
    "Claude-3": 0.40, "Claude-3.5": 0.60, "Claude-3.7-thinking": 0.80, "Claude-4": 1.0,
    "GPT-4": 0.55, "GPT-4o": 1.0,
    "DARS+R1": 0.65, "Agentless+Claude-3.5": 0.80, "Moatless+V3": 1.0,
}

AGENT_MARKERS = {
    "Claude-3": "o", "Claude-3.5": "o", "Claude-3.7-thinking": "o", "Claude-4": "o",
    "GPT-4": "s", "GPT-4o": "s",
    "DARS+R1": "^", "Agentless+Claude-3.5": "^", "Moatless+V3": "^",
}


def fig_delta_heatmap(results, agents, out_path):
    """
    Heatmap: rows = constraints, cols = agents.
    Cell value = Δ pass rate (satisfies − not).
    """
    constraints = list(results.keys())
    data = np.full((len(constraints), len(agents)), np.nan)
    pvals = np.full((len(constraints), len(agents)), np.nan)

    for i, cname in enumerate(constraints):
        for j, agent in enumerate(agents):
            if agent in results[cname]:
                s = results[cname][agent]
                data[i, j] = s["delta"]
                pvals[i, j] = s["p_value"]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    vmax = 0.25
    im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(constraints)))
    ax.set_yticklabels(constraints, fontsize=9)

    # annotate cells with Δ value; mark near-significant cells
    for i in range(len(constraints)):
        for j in range(len(agents)):
            v = data[i, j]
            p = pvals[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="#888")
                continue
            sign = "*" if (not np.isnan(p) and p < 0.10) else ""
            ax.text(j, i, f"{v:+.2f}{sign}", ha="center", va="center",
                    fontsize=7.5, color="black" if abs(v) < 0.15 else "white",
                    fontweight="bold" if sign else "normal")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Δ pass rate (sat − not)", fontsize=8)
    ax.set_title("Pass rate lift when procedural constraint is satisfied\n"
                 "(* = p < 0.10, Fisher exact)", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def fig_dot_chart(results, agents, out_path):
    """
    Dot chart: for each constraint, show pass rate among satisfiers vs non-satisfiers
    per agent, with Wilson CIs. Constraints as rows, agents as colored dots.
    """
    constraints = list(results.keys())
    n_c = len(constraints)

    fig, axes = plt.subplots(1, n_c, figsize=(3.2 * n_c, 4.0), sharey=True)
    if n_c == 1:
        axes = [axes]

    for ax, cname in zip(axes, constraints):
        # plot each agent: satisfies (filled) vs not (open)
        agents_present = [a for a in agents if a in results[cname]]
        y_positions = range(len(agents_present))

        for y, agent in zip(y_positions, agents_present):
            s = results[cname][agent]
            color = AGENT_COLORS.get(agent, "#777")
            alpha = AGENT_ALPHA.get(agent, 0.7)
            mk = AGENT_MARKERS.get(agent, "o")

            # satisfies
            pr_s = s["pr_satisfies"]
            ci_s = s["ci_satisfies"]
            if not math.isnan(pr_s) and s["n_satisfies"] >= 3:
                ax.plot([ci_s[0], ci_s[1]], [y, y], color=color, alpha=0.4, lw=1.5)
                ax.scatter([pr_s], [y], color=color, alpha=alpha, s=50,
                           marker=mk, zorder=3, label=agent if cname == constraints[0] else "")

            # not satisfies
            pr_n = s["pr_not"]
            ci_n = s["ci_not"]
            if not math.isnan(pr_n) and s["n_not"] >= 3:
                ax.plot([ci_n[0], ci_n[1]], [y + 0.25, y + 0.25], color=color, alpha=0.3, lw=1.5)
                ax.scatter([pr_n], [y + 0.25], color=color, alpha=alpha * 0.5, s=40,
                           marker=mk, zorder=3, facecolors="none", edgecolors=color)

        ax.set_title(cname, fontsize=8, pad=4)
        ax.set_xlim(-0.05, 0.80)
        ax.axvline(x=0, color="#ccc", lw=0.7, ls="--")
        ax.set_xlabel("Pass rate", fontsize=7)
        ax.tick_params(axis="x", labelsize=7)

        if ax is axes[0]:
            ax.set_yticks(list(y_positions))
            ax.set_yticklabels(agents_present, fontsize=7.5)
        ax.invert_yaxis()

    # legend: filled = satisfies, open = doesn't satisfy
    sat_patch = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#666",
                            markersize=7, label="Satisfies constraint")
    not_patch = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                            markeredgecolor="#666", markersize=7, label="Does not satisfy")
    fig.legend(handles=[sat_patch, not_patch], loc="lower center", ncol=2,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Pass rates under procedural constraints (filled = satisfies, open = does not)",
                 fontsize=8.5, y=1.01)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def fig_satisfaction_rates(results, agents, out_path):
    """
    Bar chart: what fraction of each agent's trajectories satisfy each constraint?
    Tells you which constraints are already baked into an agent's default behavior.
    """
    constraints = list(results.keys())
    n_c = len(constraints)
    n_a = len(agents)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    x = np.arange(n_a)
    width = 0.8 / n_c
    offsets = np.linspace(-(n_c - 1) * width / 2, (n_c - 1) * width / 2, n_c)

    cmap = plt.cm.get_cmap("tab10", n_c)
    for ci, (cname, offset) in enumerate(zip(constraints, offsets)):
        rates = []
        for agent in agents:
            if agent not in results[cname]:
                rates.append(np.nan)
                continue
            s = results[cname][agent]
            total = s["n_satisfies"] + s["n_not"]
            rates.append(s["n_satisfies"] / total if total else np.nan)
        ax.bar(x + offset, rates, width=width, color=cmap(ci), alpha=0.8,
               label=cname, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of trajectories satisfying constraint", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_title("Natural satisfaction rates: how often do agents already follow each constraint?",
                 fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def matched_task_analysis(rows):
    """
    For each constraint, look at task instances where some agents satisfy it
    and others don't. Compare pass rates in the two groups within the same task.
    This controls for task difficulty by construction.

    Returns: {constraint_name: {'sat_pass': k/n, 'not_pass': k/n, 'n_tasks': int, 'delta': float}}
    """
    by_instance = defaultdict(list)
    for r in rows:
        by_instance[r["instance_id"]].append(r)

    results = {}
    for cname, cfunc in CONSTRAINTS.items():
        sat_outcomes, not_outcomes = [], []
        n_tasks_used = 0

        for iid, irows in by_instance.items():
            sat = [r for r in irows if cfunc(r["canonical"])]
            not_ = [r for r in irows if not cfunc(r["canonical"])]
            if not sat or not not_:
                continue  # task not informative for this constraint
            n_tasks_used += 1
            sat_outcomes.extend(r["resolved"] for r in sat)
            not_outcomes.extend(r["resolved"] for r in not_)

        k_s, n_s = sum(sat_outcomes), len(sat_outcomes)
        k_n, n_n = sum(not_outcomes), len(not_outcomes)
        pr_s = k_s / n_s if n_s else float("nan")
        pr_n = k_n / n_n if n_n else float("nan")
        delta = pr_s - pr_n if (n_s and n_n) else float("nan")

        if n_s >= 5 and n_n >= 5:
            _, p = stats.fisher_exact([[k_s, n_s - k_s], [k_n, n_n - k_n]])
        else:
            p = float("nan")

        results[cname] = {
            "sat_pass_rate": round(pr_s, 4),
            "not_pass_rate": round(pr_n, 4),
            "delta": round(delta, 4),
            "n_sat": n_s,
            "n_not": n_n,
            "n_tasks": n_tasks_used,
            "p_value": round(p, 4) if not math.isnan(p) else None,
        }

    return results


def print_matched_table(matched):
    print("\n=== Matched-task analysis (difficulty-controlled) ===")
    print(f"{'Constraint':<25} {'n_tasks':>8} {'n_sat':>6} {'pr_sat':>7} {'n_not':>6} {'pr_not':>7} {'delta':>7} {'p':>8}")
    for cname, s in matched.items():
        p_str = f"{s['p_value']:.3f}" if s["p_value"] is not None else "  n/a"
        d_str = f"{s['delta']:+.3f}" if not math.isnan(s["delta"]) else "  n/a"
        print(f"{cname:<25} {s['n_tasks']:>8} {s['n_sat']:>6} {s['sat_pass_rate']:>7.3f} "
              f"{s['n_not']:>6} {s['not_pass_rate']:>7.3f} {d_str:>7} {p_str:>8}")


def fig_matched_bars(matched, out_path):
    """Bar chart of matched-task pass rates: satisfies vs does not, per constraint."""
    constraints = list(matched.keys())
    x = np.arange(len(constraints))
    width = 0.35

    sat_rates = [matched[c]["sat_pass_rate"] for c in constraints]
    not_rates = [matched[c]["not_pass_rate"] for c in constraints]

    # Wilson CIs
    sat_cis = [wilson_ci(int(matched[c]["sat_pass_rate"] * matched[c]["n_sat"]),
                         matched[c]["n_sat"]) for c in constraints]
    not_cis = [wilson_ci(int(matched[c]["not_pass_rate"] * matched[c]["n_not"]),
                         matched[c]["n_not"]) for c in constraints]

    sat_err = [(r - l, u - r) for (l, u), r in zip(sat_cis, sat_rates)]
    not_err = [(r - l, u - r) for (l, u), r in zip(not_cis, not_rates)]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars_s = ax.bar(x - width / 2, sat_rates, width, color="#3a7d44", alpha=0.8,
                    label="Satisfies constraint", capsize=4,
                    yerr=np.array([[e[0] for e in sat_err], [e[1] for e in sat_err]]))
    bars_n = ax.bar(x + width / 2, not_rates, width, color="#c0392b", alpha=0.7,
                    label="Does not satisfy", capsize=4,
                    yerr=np.array([[e[0] for e in not_err], [e[1] for e in not_err]]))

    # annotate delta and p-value
    for xi, cname in zip(x, constraints):
        s = matched[cname]
        d = s["delta"]
        p = s["p_value"]
        sign = "†" if (p is not None and p < 0.10) else ""
        sign = "*" if (p is not None and p < 0.05) else sign
        d_label = f"Δ{d:+.2f}{sign}"
        ypos = max(s["sat_pass_rate"], s["not_pass_rate"]) + 0.04
        ax.text(xi, ypos, d_label, ha="center", va="bottom", fontsize=7.5,
                color="#222", fontweight="bold" if sign else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(constraints, fontsize=8.5)
    ax.set_ylabel("Pass rate", fontsize=9)
    ax.set_ylim(0, 0.70)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Matched-task pass rates under procedural constraints\n"
                 "(same tasks, agents differ in whether they satisfy the constraint)\n"
                 "* p < 0.05   † p < 0.10   error bars = 95% Wilson CI", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_results_json(results, agents, out_path):
    """Save structured results for downstream use."""
    out = {}
    for cname, agent_stats in results.items():
        out[cname] = {}
        for agent in agents:
            if agent in agent_stats:
                s = agent_stats[agent]
                out[cname][agent] = {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in s.items()
                    if not isinstance(v, tuple)
                }
                out[cname][agent]["ci_satisfies"] = [round(x, 4) for x in s["ci_satisfies"]]
                out[cname][agent]["ci_not"] = [round(x, 4) for x in s["ci_not"]]
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading dataset...")
    rows = load_dataset()
    print(f"  Total rows: {len(rows)}")

    agents_present = sorted(set(r["agent"] for r in rows))
    agents = [a for a in AGENT_ORDER if a in agents_present]
    print(f"  Agents: {agents}")

    # per-agent baseline pass rates
    by_agent = defaultdict(list)
    for r in rows:
        by_agent[r["agent"]].append(r["resolved"])
    print("\nBaseline pass rates:")
    for agent in agents:
        vals = by_agent[agent]
        print(f"  {agent:<30} {sum(vals):>4}/{len(vals):>4} = {sum(vals)/len(vals):.3f}")

    print("\nRunning constraint analysis...")
    results = run_analysis(rows)

    print_table(results, agents)

    # figures
    fig_delta_heatmap(
        results, agents,
        os.path.join(FIG_DIR, "controlled_eval_delta_heatmap.png"),
    )
    fig_dot_chart(
        results, agents,
        os.path.join(FIG_DIR, "controlled_eval_dot_chart.png"),
    )
    fig_satisfaction_rates(
        results, agents,
        os.path.join(FIG_DIR, "controlled_eval_satisfaction_rates.png"),
    )
    save_results_json(
        results, agents,
        os.path.join(OUT_DIR, "controlled_eval_results.json"),
    )

    print("\nRunning matched-task analysis (difficulty-controlled)...")
    matched = matched_task_analysis(rows)
    print_matched_table(matched)

    fig_matched_bars(
        matched,
        os.path.join(FIG_DIR, "controlled_eval_matched_bars.png"),
    )

    # add matched results to saved JSON
    json_path = os.path.join(OUT_DIR, "controlled_eval_results.json")
    combined = json.load(open(json_path))
    combined["matched_task"] = matched
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Updated: {json_path}")

    print("\nDone.")
