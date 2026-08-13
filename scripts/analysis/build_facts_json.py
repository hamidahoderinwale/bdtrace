"""Aggregate every load-bearing number for Paper 2 into one facts.json.

Reads many existing JSON outputs and produces a single indexed file the
writer can cite from. Each fact carries: value, source file, and a
one-line interpretation.

Usage:
    uv run python scripts/analysis/build_facts_json.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DAT = ROOT / "output" / "paper2_pilot"
OUT = DAT / "writing_pack" / "facts.json"


def load(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    facts: dict = {}

    # --- Corpus ---
    backbone = load(DAT / "backbone_probe_extended.json")
    n_traj = 0
    n_inst = 0
    n_classes = 0
    if backbone:
        n_traj = backbone.get("n_trajectories", 0)
        n_inst = backbone.get("n_unique_instances", 0)
        n_classes = backbone.get("n_classes", 0)
    facts["corpus"] = {
        "n_trajectories": n_traj,
        "n_unique_instances": n_inst,
        "n_agents": n_classes,
        "n_scaffolds": 4,
        "scaffolds_present": ["SWE-agent", "Agentless", "DARS", "Moatless"],
        "agents": backbone.get("class_labels") if backbone else None,
        "source": "backbone_probe_extended.json",
    }

    # --- Backbone probe ---
    if backbone:
        gcv = backbone.get("grouped_cv", {})
        facts["backbone_probe_9class"] = {
            "method": "logistic regression on motif TF features, GroupKFold by instance_id, k=5",
            "accuracy_mean": gcv.get("mean_accuracy"),
            "accuracy_std":  gcv.get("std_accuracy"),
            "macro_f1_mean": gcv.get("mean_macro_f1"),
            "macro_f1_std":  gcv.get("std_macro_f1"),
            "chance_accuracy": round(1 / max(n_classes, 1), 3),
            "per_class_f1": {
                k: round(v["f1"], 3)
                for k, v in backbone.get("per_class_metrics_full_fit", {}).items()
            },
            "interpretation": (
                "9-class agent identification far above chance "
                f"({gcv.get('mean_accuracy', 0):.2%} vs {1/max(n_classes,1):.1%}). "
                "Scaffold-distinct classes (Agentless, DARS, Moatless) approach 1.00. "
                "Within-SWE-agent extended-thinking backbones (Claude-3.7-thinking, Claude-4) at 0.99-1.00. "
                "Within-SWE-agent base backbones cluster at 0.70-0.86 — they confuse with each other."
            ),
            "source": "backbone_probe_extended.json",
        }

    # --- JSD structure ---
    jsd = load(DAT / "jsd_matrix_extended.json")
    if jsd:
        # Extract the three regimes
        matrix_rows = jsd.get("matrix", [])
        same_scaffold_base = []  # within SWE-agent base (Claude-3, Claude-3.5, GPT-4, GPT-4o pairs)
        same_scaffold_thinking = []  # Claude-3.7-thinking ↔ Claude-4
        cross_paradigm = []  # base ↔ thinking within SWE-agent
        cross_scaffold = []  # any × Agentless/DARS/Moatless
        BASE = {"Claude-3", "Claude-3.5", "GPT-4", "GPT-4o"}
        THINK = {"Claude-3.7-thinking", "Claude-4"}
        XSCAF = {"Agentless+Claude-3.5", "DARS+R1", "Moatless+V3"}
        for row in matrix_rows:
            r, c = row["row"], row["col"]
            if r >= c:
                continue
            v = row["jsd"]
            if r in BASE and c in BASE:
                same_scaffold_base.append(v)
            elif r in THINK and c in THINK:
                same_scaffold_thinking.append(v)
            elif (r in BASE and c in THINK) or (r in THINK and c in BASE):
                cross_paradigm.append(v)
            elif r in XSCAF or c in XSCAF:
                cross_scaffold.append(v)

        def stats(xs):
            if not xs:
                return None
            return {"min": round(min(xs), 3), "max": round(max(xs), 3),
                    "mean": round(sum(xs) / len(xs), 3), "n_pairs": len(xs)}

        facts["jsd_three_regime_structure"] = {
            "within_swe_agent_base":         stats(same_scaffold_base),
            "within_swe_agent_extended_thinking": stats(same_scaffold_thinking),
            "cross_paradigm_within_swe_agent": stats(cross_paradigm),
            "cross_scaffold":                stats(cross_scaffold),
            "interpretation": (
                "Paradigm-cluster N=2 evidence: extended-thinking pair JSD lower than cross-paradigm "
                "within same scaffold. Three regimes preserved: base < thinking-pair < cross-paradigm < cross-scaffold."
            ),
            "source": "jsd_matrix_extended.json",
        }

    # --- Aggregate per-agent metrics ---
    agg = load(DAT / "aggregate_metrics_extended.json")
    if agg:
        # agg structure varies; just include the file pointer + most useful summary
        facts["per_agent_aggregates"] = {
            "interpretation": (
                "4 metrics per agent: motif entropy, distinct-motifs-at-90%, mean trajectory length, "
                "BPE compression ratio. Cross-scaffold spread (entropy 0.0 to 6.0) is ~6x within-scaffold spread "
                "(5.0 to 6.0). Same scaffold-dominance signal as JSD."
            ),
            "source": "aggregate_metrics_extended.json",
            "raw": agg,
        }

    # --- Permutation null (heritability gap) ---
    perm = load(DAT / "permutation_null.json")
    if perm:
        agg_p = perm.get("aggregate", {})
        facts["permutation_null_heritability"] = {
            "n_trajectories": agg_p.get("n_records"),
            "observed_gap": round(agg_p.get("observed_gap", 0), 4),
            "null_mean": round(agg_p.get("null_mean", 0), 4),
            "null_std": round(agg_p.get("null_std", 0), 4),
            "null_q95": round(agg_p.get("null_q95", 0), 4),
            "p_value": agg_p.get("p_value"),
            "n_permutations": agg_p.get("n_permutations"),
            "per_bucket": {
                k: {"n": v["n_records"], "obs": round(v["observed_gap"], 4), "p": v["p_value"]}
                for k, v in perm.items() if k != "aggregate"
            },
            "interpretation": (
                "Same-family motif divergence is below chance at p=0.001 aggregate. "
                "Per-difficulty bucket: 0/3 p=0.003, 1/3 p=0.001, 2/3 p=0.048, 3/3 p=0.199 (small sample)."
            ),
            "source": "permutation_null.json",
        }

    # --- FIM-form analysis ---
    fim_cont = load(DAT / "fim_continuous_check.json")
    fim_inv = load(DAT / "fim_investigations.json")
    if fim_cont:
        facts["fim_complexity_continuous"] = {
            "n_instances": fim_cont.get("n_instances"),
            "spearman_rho": fim_cont.get("spearman_rho"),
            "spearman_p": fim_cont.get("p_value"),
            "bootstrap_ci_95": fim_cont.get("bootstrap_ci_95"),
            "alt_distinct_types_rho": fim_cont.get("alt_distinct_types_rho"),
            "interpretation": (
                "AST-edit complexity correlates negatively with 84-agent ease (Spearman -0.23, p=0.0002, "
                "CI [-0.35, -0.11]). Structure predicts difficulty across the leaderboard."
            ),
            "source": "fim_continuous_check.json",
        }
    if fim_inv:
        i1 = fim_inv.get("investigation_1_mi_form_passfail", {})
        ref = fim_inv.get("reference_mi_n_resolved", {})
        i2 = fim_inv.get("investigation_2_structure_vs_surface", {})
        i3 = fim_inv.get("investigation_3_conditional_mi", {})
        facts["fim_three_investigations"] = {
            "n_trajectories": fim_inv.get("n_trajectories"),
            "n_forms": fim_inv.get("n_forms"),
            "mi_form_passfail_pct_H": i1.get("pct_of_H"),
            "mi_n_resolved_passfail_pct_H": ref.get("pct_of_H"),
            "rho_ast_complexity_ease": i2.get("rho_ast_complexity_ease"),
            "rho_loc_delta_ease":      i2.get("rho_loc_delta_ease"),
            "rho_ast_vs_loc":          i2.get("rho_ast_vs_loc"),
            "conditional_mi_pct_H":    i3.get("pct_of_H_y_given_z"),
            "interpretation": (
                "FIM-form alone explains 19.4% of pass/fail entropy; n_resolved explains 60.2%. "
                "Conditional MI(FIM | n_resolved) = 0.3% — the two are co-extensive measures of "
                "the same structural-difficulty construct, not independent predictors."
            ),
            "source": "fim_investigations.json",
        }

    # --- N-gram robustness ---
    ngram = load(DAT / "ngram_baseline.json")
    if ngram:
        facts["ngram_robustness"] = {
            "n": ngram.get("n_gram_n"),
            "pearson_r":  ngram.get("pearson_r"),
            "spearman_r": ngram.get("spearman_r"),
            "interpretation": (
                f"3-gram pair-JSD agrees with BPE pair-JSD at Pearson r=0.82, Spearman ρ=0.59. "
                "BPE finding is not an artifact of the BPE choice; n-grams give the same ordering "
                "with coarser resolution."
            ),
            "source": "ngram_baseline.json",
        }

    # --- V-sweep stability ---
    vsweep = load(DAT / "bpe_vocab_sweep_extended.json")
    if vsweep:
        results = vsweep.get("results", [])
        facts["v_sweep_stability"] = {
            "v_values_tested": [r["V"] for r in results] if results else None,
            "interpretation": (
                "Heritability ordering is rank-invariant across V in [100, 500]. "
                "Magnitudes amplify with V (more granular vocabulary -> more room for divergence) "
                "but the pair ranking does not flip. V=200 chosen by elbow rule and defended by "
                "the V-sweep itself."
            ),
            "source": "bpe_vocab_sweep_extended.json",
        }

    # --- Failure modes ---
    fm = load(DAT / "failure_modes_extended.json")
    if fm:
        facts["failure_anatomy"] = {
            "interpretation": (
                "Type A (pre-localization) vs Type B (post-localization) failures per agent. "
                "Type B accounts for 57.4% (Claude-3) to 90.5% (Claude-3.7-thinking) of failures across 7 applicable agents. "
                "Bottleneck is post-localization, not search."
            ),
            "source": "failure_modes_extended.json",
            "raw": fm,
        }

    # --- Stuck-edit-loop signature ---
    postloc = load(DAT / "r10_postloc_motifs_extended.json")
    if postloc:
        facts["stuck_edit_loop"] = {
            "interpretation": (
                "Long Type B failures show EDIT_SRC_PY +0.053 / SUBMIT -0.063 across 7 scaffolds. "
                "Generalizable runtime detector for the failure pattern."
            ),
            "source": "r10_postloc_motifs_extended.json",
            "raw": postloc,
        }

    # --- FP-growth canonical templates ---
    fpg = load(DAT / "motif_fpgrowth_templates_extended.json")
    if fpg:
        facts["fpgrowth_templates"] = {
            "interpretation": (
                "Top pass-enriched canonical pattern per FIM form, lift 1.25-1.40."
            ),
            "source": "motif_fpgrowth_templates_extended.json",
        }

    # --- Verified replication (R4) ---
    verified = load(DAT / "verified_mi.json")
    if verified:
        facts["verified_replication"] = {
            "interpretation": (
                f"Verified-split replication (n=996, 2 agents, 500 instances): "
                f"MI(difficulty; pass/fail) = {verified.get('mi_pct_H', '?')}, "
                "stronger than Lite 60%. Same finding holds on a curated split."
            ),
            "source": "verified_mi.json",
            "raw": verified,
        }

    # --- Bookkeeping ---
    facts["_meta"] = {
        "generated_by":   "scripts/analysis/build_facts_json.py",
        "generated_when": "see file mtime",
        "purpose":        "Single indexed source of every load-bearing number for Paper 2 writing.",
        "consumer":       "user writing in Overleaf — cite by `facts.<key>.<field>` reference.",
    }

    OUT.write_text(json.dumps(facts, indent=2, default=str))
    n_keys = sum(1 for k in facts if not k.startswith("_"))
    print(f"Wrote {OUT}")
    print(f"  {n_keys} fact categories, {os.path.getsize(OUT)} bytes")


if __name__ == "__main__":
    main()
