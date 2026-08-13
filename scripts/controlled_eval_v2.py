"""
Controlled evaluations — three experiments.

Experiment A: Vocabulary controls
  For each canonical action type, measure the pass rate among trajectories
  that include it vs those that exclude it. Reports an "outcome lift" per atom type,
  showing which operations are load-bearing for success.

Experiment B: Phase controls
  Segment each trajectory into four phases: Explore, Diagnose, Repair, Verify.
  Enumerate which phase combinations are present and compare pass rates.
  Shows whether structural ordering matters beyond vocabulary presence.

Experiment C: Temporal controls (early prediction)
  Using BPE motif frequency vectors from the first k steps, predict outcome
  with logistic regression. Reports AUC vs step k across all available agents.
  Identifies the intervention timing — at what point is the outcome already predictable.
"""

import json
import math
import os
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import normalize

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "output", "paper2_pilot")
FIG_DIR = os.path.join(REPO_ROOT, "output", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

AGENT_ORDER = [
    "Claude-3", "Claude-3.5", "Claude-3.7-thinking", "Claude-4",
    "GPT-4", "GPT-4o",
    "DARS+R1", "Agentless+Claude-3.5", "Moatless+V3",
]
PALETTE = {
    "Claude-3": "#c9693a", "Claude-3.5": "#c9693a",
    "Claude-3.7-thinking": "#c9693a", "Claude-4": "#c9693a",
    "GPT-4": "#3a6fc9", "GPT-4o": "#3a6fc9",
    "DARS+R1": "#5a8a50", "Agentless+Claude-3.5": "#5a8a50", "Moatless+V3": "#5a8a50",
}
ALPHA = {
    "Claude-3": 0.35, "Claude-3.5": 0.55, "Claude-3.7-thinking": 0.75, "Claude-4": 1.0,
    "GPT-4": 0.55, "GPT-4o": 1.0,
    "DARS+R1": 0.65, "Agentless+Claude-3.5": 0.80, "Moatless+V3": 1.0,
}
MARKERS = {
    "Claude-3": "o", "Claude-3.5": "o", "Claude-3.7-thinking": "o", "Claude-4": "o",
    "GPT-4": "s", "GPT-4o": "s",
    "DARS+R1": "^", "Agentless+Claude-3.5": "^", "Moatless+V3": "^",
}

# ─── Phase definitions ────────────────────────────────────────────────────────
PHASE_ATOMS = {
    "Explore": {"SEARCH", "FIND_FILE", "SHELL_GREP", "SHELL_LS",
                "OPEN_SRC_PY", "OPEN_TEST_PY", "OPEN_OTHER", "OPEN_CONFIG_PY",
                "NAV_SRC_PY", "NAV", "SHELL_CD", "SHELL_CAT"},
    "Diagnose": {"CREATE_REPRO_PY", "EDIT_REPRO_PY", "RUN_PYTHON_REPRO_PY",
                 "OPEN_REPRO_PY"},
    "Repair":   {"EDIT_SRC_PY", "EDIT", "EDIT_CONFIG_PY", "EDIT_OTHER",
                 "EDIT_TEST_PY", "CREATE_SRC_PY", "CREATE_TEST_PY",
                 "SHELL_SED", "SHELL_MV"},
    "Verify":   {"RUN_PYTHON_TEST_PY", "RUN_PYTEST_TEST_PY", "RUN_PYTEST_ALL",
                 "RUN_PYTHON_ALL", "RUN_TEST_SCRIPT", "RUN_PYTHON_SRC_PY"},
}


def atom_phase(atom):
    for phase, atoms in PHASE_ATOMS.items():
        if atom in atoms:
            return phase
    return None


def trajectory_phases(canonical):
    """Return frozenset of phases present in this trajectory."""
    return frozenset(p for a in canonical for p in [atom_phase(a)] if p)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_dataset():
    seq_path = os.path.join(OUT_DIR, "bpe_sequences_extended.jsonl")
    pf_path = os.path.join(OUT_DIR, "extended_pass_fail.json")
    pf = json.load(open(pf_path))

    sub_to_agent = {}
    with open(seq_path) as f:
        for line in f:
            r = json.loads(line)
            sub_to_agent[r["submission"]] = r["agent"]

    rows = []
    with open(seq_path) as f:
        for line in f:
            r = json.loads(line)
            sub = r["submission"]
            if sub not in pf:
                continue
            skip = set(pf[sub].get("no_generation", [])) | set(pf[sub].get("no_logs", []))
            if r["instance_id"] in skip:
                continue
            resolved_set = set(pf[sub].get("resolved", []))
            rows.append({
                "agent": sub_to_agent[sub],
                "submission": sub,
                "instance_id": r["instance_id"],
                "resolved": r["instance_id"] in resolved_set,
                "canonical": r["canonical"],
                "bpe": r["bpe"],
            })
    return rows


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    m = (z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / d
    return (max(0.0, c - m), min(1.0, c + m))


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A — Vocabulary controls
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_a(rows):
    """
    For each canonical action type, compute pass rate in trajectories
    that use it (present=True) vs those that don't (present=False).
    Pool across all agents; control for agent by computing within-agent Δ
    then averaging.

    Returns a list of dicts sorted by pooled Δ (descending).
    """
    # Gather all atom types
    all_atoms = Counter()
    for r in rows:
        for a in r["canonical"]:
            all_atoms[a] += 1

    # Minimum: atom must appear in at least 30 trajectories to be reportable
    MIN_N = 30
    atoms = [a for a, cnt in all_atoms.items() if cnt >= MIN_N]

    by_agent = defaultdict(list)
    for r in rows:
        by_agent[r["agent"]].append(r)

    atom_results = []
    for atom in atoms:
        # Pooled (all agents)
        present = [r["resolved"] for r in rows if atom in r["canonical"]]
        absent = [r["resolved"] for r in rows if atom not in r["canonical"]]
        k_p, n_p = sum(present), len(present)
        k_a, n_a = sum(absent), len(absent)
        pr_p = k_p / n_p if n_p else float("nan")
        pr_a = k_a / n_a if n_a else float("nan")
        delta_pooled = pr_p - pr_a if (n_p and n_a) else float("nan")
        if n_p >= 5 and n_a >= 5:
            _, p_val = stats.fisher_exact([[k_p, n_p - k_p], [k_a, n_a - k_a]])
        else:
            p_val = float("nan")

        # Within-agent Δ (average over agents with both groups)
        agent_deltas = []
        for agent, agent_rows in by_agent.items():
            pres_a = [r["resolved"] for r in agent_rows if atom in r["canonical"]]
            abs_a = [r["resolved"] for r in agent_rows if atom not in r["canonical"]]
            if len(pres_a) < 3 or len(abs_a) < 3:
                continue
            d = sum(pres_a) / len(pres_a) - sum(abs_a) / len(abs_a)
            agent_deltas.append(d)
        within_agent_delta = float(np.mean(agent_deltas)) if agent_deltas else float("nan")

        atom_results.append({
            "atom": atom,
            "n_present": n_p,
            "n_absent": n_a,
            "pr_present": round(pr_p, 4),
            "pr_absent": round(pr_a, 4),
            "delta_pooled": round(delta_pooled, 4),
            "within_agent_delta": round(within_agent_delta, 4),
            "p_value": round(p_val, 4) if not math.isnan(p_val) else None,
            "n_agents_contributing": len(agent_deltas),
        })

    atom_results.sort(key=lambda x: x["within_agent_delta"] if not math.isnan(x["within_agent_delta"]) else 0,
                      reverse=True)
    return atom_results


def fig_vocabulary_lifts(atom_results, out_path, top_n=20):
    """
    Horizontal bar chart of within-agent pass rate lift per action type.
    Top N by absolute delta. Color by direction.
    """
    # Take top N by absolute within-agent delta
    valid = [r for r in atom_results if not math.isnan(r["within_agent_delta"])]
    # Top positive + top negative
    pos = sorted(valid, key=lambda x: x["within_agent_delta"], reverse=True)[:top_n // 2]
    neg = sorted(valid, key=lambda x: x["within_agent_delta"])[:top_n // 2]
    combined = pos + neg
    combined.sort(key=lambda x: x["within_agent_delta"])

    labels = [r["atom"] for r in combined]
    deltas = [r["within_agent_delta"] for r in combined]
    pvals = [r["p_value"] for r in combined]
    colors = ["#2a7a2a" if d > 0 else "#b02020" for d in deltas]

    fig, ax = plt.subplots(figsize=(7, 0.35 * len(labels) + 1.2))
    bars = ax.barh(range(len(labels)), deltas, color=colors, alpha=0.78, edgecolor="white")

    # significance markers
    for i, (d, p) in enumerate(zip(deltas, pvals)):
        if p is not None and p < 0.05:
            ax.text(d + (0.003 if d >= 0 else -0.003), i, "*",
                    ha="left" if d >= 0 else "right", va="center",
                    fontsize=10, color="#333", fontweight="bold")
        elif p is not None and p < 0.10:
            ax.text(d + (0.003 if d >= 0 else -0.003), i, "†",
                    ha="left" if d >= 0 else "right", va="center",
                    fontsize=9, color="#555")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5, fontfamily="monospace")
    ax.axvline(0, color="#888", lw=0.8, ls="--")
    ax.set_xlabel("Within-agent Δ pass rate (uses type − doesn't)", fontsize=8)
    ax.set_title("Outcome lift by action type\n"
                 "(* p<0.05  † p<0.10  averaged across agents with ≥3 trajectories in each group)",
                 fontsize=8.5)
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — Phase controls
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_b(rows):
    """
    For each trajectory, identify which phases are present.
    Compute pass rates for all observed phase combinations (min 10 trajectories).
    Also compute per-phase marginal pass rates (is this phase necessary?).
    """
    phase_combos = defaultdict(list)
    for r in rows:
        phases = trajectory_phases(r["canonical"])
        phase_combos[phases].append(r["resolved"])

    combo_results = []
    for phases, outcomes in phase_combos.items():
        n = len(outcomes)
        k = sum(outcomes)
        pr = k / n
        ci = wilson_ci(k, n)
        label = "+".join(sorted(phases)) if phases else "(none)"
        combo_results.append({
            "phases": label,
            "phase_set": phases,
            "n": n,
            "k": k,
            "pass_rate": round(pr, 4),
            "ci_lo": round(ci[0], 4),
            "ci_hi": round(ci[1], 4),
        })
    combo_results.sort(key=lambda x: x["pass_rate"], reverse=True)

    # Marginal effect of each phase: with vs without, controlling for other phases
    marginal = {}
    for phase in PHASE_ATOMS:
        with_phase = [r["resolved"] for r in rows if phase in trajectory_phases(r["canonical"])]
        without_phase = [r["resolved"] for r in rows if phase not in trajectory_phases(r["canonical"])]
        k_w, n_w = sum(with_phase), len(with_phase)
        k_wo, n_wo = sum(without_phase), len(without_phase)
        pr_w = k_w / n_w if n_w else float("nan")
        pr_wo = k_wo / n_wo if n_wo else float("nan")
        _, p = stats.fisher_exact([[k_w, n_w - k_w], [k_wo, n_wo - k_wo]]) \
            if (n_w >= 5 and n_wo >= 5) else (None, float("nan"))
        marginal[phase] = {
            "pr_with": round(pr_w, 4),
            "pr_without": round(pr_wo, 4),
            "delta": round(pr_w - pr_wo, 4),
            "n_with": n_w,
            "n_without": n_wo,
            "p_value": round(p, 4) if not math.isnan(p) else None,
        }

    return combo_results, marginal


def fig_phase_controls(combo_results, marginal, out_path):
    fig = plt.figure(figsize=(11, 5.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], wspace=0.4)

    # Left: phase combination pass rates (top 12 by n, min 10)
    ax_left = fig.add_subplot(gs[0])
    top = [r for r in combo_results if r["n"] >= 10][:14]
    top.sort(key=lambda x: x["pass_rate"])
    labels = [r["phases"] for r in top]
    prs = [r["pass_rate"] for r in top]
    ci_lo = [r["ci_lo"] for r in top]
    ci_hi = [r["ci_hi"] for r in top]
    ns = [r["n"] for r in top]

    colors = plt.cm.RdYlGn([p for p in prs])
    bars = ax_left.barh(range(len(top)), prs, color=colors, alpha=0.85, edgecolor="white")
    ax_left.errorbar(prs, range(len(top)),
                     xerr=[np.array(prs) - np.array(ci_lo),
                           np.array(ci_hi) - np.array(prs)],
                     fmt="none", color="#555", capsize=3, lw=1)
    for i, (pr, n) in enumerate(zip(prs, ns)):
        ax_left.text(pr + 0.012, i, f"n={n}", va="center", fontsize=6.5, color="#555")

    ax_left.set_yticks(range(len(top)))
    ax_left.set_yticklabels(labels, fontsize=7.5)
    ax_left.set_xlabel("Pass rate (95% Wilson CI)", fontsize=8)
    ax_left.set_title("Pass rate by phase combination\n(min 10 trajectories per combo)",
                       fontsize=8.5)
    ax_left.set_xlim(0, 0.72)
    ax_left.grid(axis="x", alpha=0.25)

    # Right: marginal effect per phase
    ax_right = fig.add_subplot(gs[1])
    phases_order = ["Explore", "Diagnose", "Repair", "Verify"]
    m_deltas = [marginal[p]["delta"] for p in phases_order]
    m_with = [marginal[p]["pr_with"] for p in phases_order]
    m_without = [marginal[p]["pr_without"] for p in phases_order]
    m_p = [marginal[p]["p_value"] for p in phases_order]

    x = np.arange(len(phases_order))
    w = 0.32
    phase_colors = {"Explore": "#4a90c4", "Diagnose": "#c48a2a",
                    "Repair": "#7a4a94", "Verify": "#2a9a5a"}
    b1 = ax_right.bar(x - w / 2, m_with, w, label="Phase present",
                       color=[phase_colors[p] for p in phases_order], alpha=0.85)
    b2 = ax_right.bar(x + w / 2, m_without, w, label="Phase absent",
                       color=[phase_colors[p] for p in phases_order], alpha=0.35)

    for xi, (d, p) in enumerate(zip(m_deltas, m_p)):
        sign = "*" if (p is not None and p < 0.05) else ("†" if (p is not None and p < 0.10) else "")
        ypos = max(m_with[xi], m_without[xi]) + 0.025
        ax_right.text(xi, ypos, f"Δ{d:+.2f}{sign}", ha="center", va="bottom",
                      fontsize=7.5, fontweight="bold" if sign else "normal")

    ax_right.set_xticks(x)
    ax_right.set_xticklabels(phases_order, fontsize=8.5)
    ax_right.set_ylabel("Pass rate", fontsize=8)
    ax_right.set_ylim(0, 0.72)
    ax_right.legend(fontsize=7.5, loc="upper right")
    ax_right.set_title("Marginal phase effect\n(* p<0.05  † p<0.10)", fontsize=8.5)
    ax_right.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT C — Temporal controls (early prediction with BPE motif features)
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_c(rows):
    """
    For each agent, train a logistic regression on BPE motif frequency vectors
    from the first k canonical steps. Report AUC vs k.

    BPE motif features = frequency of each top-M motif within the first k atoms.
    """
    K_STEPS = [1, 2, 3, 5, 8, 12, 18, 25, 35, 50]
    MIN_TRAJ = 30

    # Build global motif vocabulary from BPE sequences
    motif_counts = Counter()
    for r in rows:
        for m in r["bpe"]:
            motif_counts[m] += 1
    top_motifs = [m for m, _ in motif_counts.most_common(150)]
    motif_idx = {m: i for i, m in enumerate(top_motifs)}

    # For each agent, compute BPE motifs from first k canonical steps
    # by mapping canonical prefixes back to BPE tokens greedily
    by_agent = defaultdict(list)
    for r in rows:
        by_agent[r["agent"]].append(r)

    results = []
    for agent in AGENT_ORDER:
        agent_rows = by_agent.get(agent, [])
        if len(agent_rows) < MIN_TRAJ:
            continue

        y = np.array([int(r["resolved"]) for r in agent_rows])
        if y.mean() < 0.01 or y.mean() > 0.99:
            continue

        # For each k, build feature matrix using canonical atom counts
        # (atom type frequency within first k steps — simpler but consistent)
        atom_vocab = sorted(set(a for r in agent_rows for a in r["canonical"]))
        atom_idx = {a: i for i, a in enumerate(atom_vocab)}

        prev_auc = None
        agent_rows_k = []
        for k in K_STEPS:
            X = np.zeros((len(agent_rows), len(atom_idx) + len(top_motifs)))
            for i, r in enumerate(agent_rows):
                prefix = r["canonical"][:k]
                for a in prefix:
                    if a in atom_idx:
                        X[i, atom_idx[a]] += 1
                # Also add motif features from bpe (take motifs that fit in k steps)
                step_count = 0
                for m in r["bpe"]:
                    atoms_in_motif = len(m.split("+"))
                    if step_count + atoms_in_motif > k:
                        break
                    if m in motif_idx:
                        X[i, len(atom_vocab) + motif_idx[m]] += 1
                    step_count += atoms_in_motif

            # L1 normalize per row
            row_norms = np.linalg.norm(X, axis=1, keepdims=True)
            row_norms[row_norms == 0] = 1.0
            X = X / row_norms

            clf = LogisticRegression(max_iter=500, C=0.5, random_state=42, solver="lbfgs")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")

            results.append({
                "agent": agent,
                "k": k,
                "auc_mean": float(aucs.mean()),
                "auc_std": float(aucs.std()),
                "auc_lo": float(aucs.mean() - 1.96 * aucs.std()),
                "auc_hi": float(aucs.mean() + 1.96 * aucs.std()),
            })
            print(f"  {agent} @ k={k:3d}: AUC={aucs.mean():.3f} ± {aucs.std():.3f}")

    return results


def fig_temporal_controls(results, out_path):
    """AUC vs step k, one line per agent family."""
    agents_present = sorted(set(r["agent"] for r in results))
    agents_plot = [a for a in AGENT_ORDER if a in agents_present]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for agent in agents_plot:
        agent_data = sorted([r for r in results if r["agent"] == agent], key=lambda x: x["k"])
        if not agent_data:
            continue
        ks = [r["k"] for r in agent_data]
        auc = [r["auc_mean"] for r in agent_data]
        lo = [r["auc_lo"] for r in agent_data]
        hi = [r["auc_hi"] for r in agent_data]
        color = PALETTE.get(agent, "#777")
        alpha = ALPHA.get(agent, 0.7)
        mk = MARKERS.get(agent, "o")

        ax.fill_between(ks, lo, hi, alpha=0.10, color=color)
        ax.plot(ks, auc, color=color, alpha=alpha, lw=1.8,
                marker=mk, markersize=5, label=agent)

    ax.axhline(0.5, color="#aaa", lw=0.8, ls="--", label="Chance (AUC=0.5)")
    ax.set_xlabel("Steps used for prediction (prefix length k)", fontsize=9)
    ax.set_ylabel("AUC (5-fold CV)", fontsize=9)
    ax.set_title("Temporal control: how early is outcome predictable?\n"
                 "(atom + BPE motif features; shaded = 95% CI)",
                 fontsize=9)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.set_ylim(0.40, 0.95)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading dataset...")
    rows = load_dataset()
    print(f"  {len(rows)} trajectories, {len(set(r['agent'] for r in rows))} agents\n")

    # ── Experiment A ──────────────────────────────────────────────────────────
    print("=== Experiment A: Vocabulary controls ===")
    atom_results = experiment_a(rows)
    print(f"\nTop 10 positive lifts (within-agent Δ):")
    for r in atom_results[:10]:
        p = f"p={r['p_value']:.3f}" if r["p_value"] is not None else "n/a"
        print(f"  {r['atom']:<30} Δ={r['within_agent_delta']:+.3f}  {p}  n_agents={r['n_agents_contributing']}")
    print(f"\nTop 10 negative lifts:")
    for r in atom_results[-10:]:
        p = f"p={r['p_value']:.3f}" if r["p_value"] is not None else "n/a"
        print(f"  {r['atom']:<30} Δ={r['within_agent_delta']:+.3f}  {p}  n_agents={r['n_agents_contributing']}")

    fig_vocabulary_lifts(atom_results, os.path.join(FIG_DIR, "controlled_eval_vocab_lifts.png"))
    with open(os.path.join(OUT_DIR, "controlled_eval_vocab_lifts.json"), "w") as f:
        json.dump(atom_results, f, indent=2)

    # ── Experiment B ──────────────────────────────────────────────────────────
    print("\n=== Experiment B: Phase controls ===")
    combo_results, marginal = experiment_b(rows)
    print("\nTop phase combinations (n ≥ 10):")
    for r in [x for x in combo_results if x["n"] >= 10][:8]:
        print(f"  {r['phases']:<40} pass={r['pass_rate']:.3f}  n={r['n']}")
    print("\nMarginal phase effects:")
    for phase, m in marginal.items():
        p = f"p={m['p_value']:.3f}" if m["p_value"] is not None else "n/a"
        print(f"  {phase:<12} with={m['pr_with']:.3f}  without={m['pr_without']:.3f}  Δ={m['delta']:+.3f}  {p}")

    fig_phase_controls(combo_results, marginal, os.path.join(FIG_DIR, "controlled_eval_phases.png"))
    with open(os.path.join(OUT_DIR, "controlled_eval_phases.json"), "w") as f:
        json.dump({"combos": combo_results, "marginal": marginal}, f, indent=2, default=list)

    # ── Experiment C ──────────────────────────────────────────────────────────
    print("\n=== Experiment C: Temporal controls ===")
    temporal_results = experiment_c(rows)
    fig_temporal_controls(temporal_results, os.path.join(FIG_DIR, "controlled_eval_temporal.png"))
    with open(os.path.join(OUT_DIR, "controlled_eval_temporal.json"), "w") as f:
        json.dump(temporal_results, f, indent=2)

    print("\nAll done.")
