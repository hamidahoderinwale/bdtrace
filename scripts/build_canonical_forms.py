#!/usr/bin/env python3
"""
Build canonical fix strategy forms.

Pipeline:
  1. Closed frequent itemsets from edit certificates (structural, deterministic)
  2. Assign each instance to its largest matching closed pattern
  3. Filter descriptions to convergent ones (cross-model agreement >= threshold)
  4. Name each form using convergent descriptions via LLM
  5. Validate: distinct pass rates, coverage, outlier characterization

Outputs:
  canonical_forms.json            -- forms with pattern, instances, pass rate, name
  instance_assignments.parquet    -- per-instance form assignment
  fig1_form_pass_rates.png        -- pass rate per canonical form
  fig2_agent_coverage.png         -- agent x form coverage heatmap
  fig3_coverage_stats.png         -- coverage, outlier rate, form sizes

Usage:
  uv run python scripts/build_canonical_forms.py
  uv run python scripts/build_canonical_forms.py --support 0.10 --convergence 0.65
"""

import argparse
import difflib
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from openai import OpenAI
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _env in [
    Path(__file__).resolve().parent.parent / ".venv" / ".env",
    Path(__file__).resolve().parent.parent / ".env",
]:
    if _env.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_env)
        except ImportError:
            pass

from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "canonical_forms"

PANEL_BG = "#f5f5f5"
PANEL_EDGE = "#dddddd"
TEAL = "#0C6583"
ORANGE = "#EE7733"
NAVY = "#2B2D42"
GRAY = "#AAAAAA"

_NORMALIZE_OPS = {
    "ADD_if": "ADD_If", "DEL_if": "DEL_If",
    "ADD_for": "ADD_For", "DEL_for": "DEL_For",
    "ADD_return": "ADD_Return", "DEL_return": "DEL_Return",
    "ADD_raise": "ADD_Raise", "DEL_raise": "DEL_Raise",
    "ADD_try": "ADD_Try", "DEL_try": "DEL_Try",
    "ADD_while": "ADD_While", "DEL_while": "DEL_While",
    "ADD_with": "ADD_With", "DEL_with": "DEL_With",
    "ADD_def": "ADD_FunctionDef", "DEL_def": "DEL_FunctionDef",
    "ADD_class": "ADD_ClassDef", "DEL_class": "DEL_ClassDef",
    "ADD_elif": "ADD_If", "DEL_elif": "DEL_If",
    "ADD_else": "ADD_If", "DEL_else": "DEL_If",
    "ADD_except": "ADD_ExceptHandler", "DEL_except": "DEL_ExceptHandler",
    "ADD_assert": "ADD_Assert",
}

AGENT_LABELS = {
    "lite_20240402_sweagent_gpt4": "SWE-agent GPT-4",
    "lite_20240620_sweagent_claude3.5sonnet": "SWE-agent Claude 3.5",
    "lite_20240728_sweagent_gpt4o": "SWE-agent GPT-4o",
    "lite_20241128_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor_20241128": "SWE-Fixer Qwen2.5",
}

MODEL_DIRS = {
    "gpt_4o": "GPT-4o",
    "gpt_4o_mini": "GPT-4o mini",
    "qwen_2.5_72b_instruct": "Qwen 2.5 72B",
    "llama_3.3_70b_instruct": "Llama 3.3 70B",
}


# --- Step 1: Edit certificates ---

def load_certs(traces_path: Path) -> dict[str, frozenset[str]]:
    certs = {}
    with open(traces_path) as f:
        for line in f:
            trace = json.loads(line)
            ops = []
            for ev in trace["events"]:
                if ev["type"] != "code_change":
                    continue
                d = ev["details"]
                if not d["file_path"].endswith(".py"):
                    continue
                before = d["before_content"].splitlines(keepends=True)
                after = d["after_content"].splitlines(keepends=True)
                raw = "".join(difflib.unified_diff(
                    before, after, fromfile=d["file_path"], tofile=d["file_path"]
                ))
                if not raw:
                    continue
                diff = f"diff --git a/{d['file_path']} b/{d['file_path']}\n" + raw
                ops.extend(patch_to_ast_sequence(diff))
            if ops:
                norm = frozenset(_NORMALIZE_OPS.get(op, op) for op in ops)
                certs[trace["instance_id"]] = norm
    return certs


# --- Step 2: Closed frequent itemsets ---

def compute_closed_itemsets(certs: dict[str, frozenset],
                             min_support: float) -> pd.DataFrame:
    instances = list(certs.keys())
    transactions = [list(cert) for cert in certs.values()]
    te = TransactionEncoder()
    binary = pd.DataFrame(
        te.fit_transform(transactions),
        columns=te.columns_,
        index=instances,
    )
    all_items = fpgrowth(binary, min_support=min_support, use_colnames=True)

    # Keep only closed: no proper superset has the same support
    support_map = {frozenset(row["itemsets"]): row["support"]
                   for _, row in all_items.iterrows()}
    closed = []
    for _, row in all_items.iterrows():
        itemset = frozenset(row["itemsets"])
        s = row["support"]
        is_closed = True
        for other, other_s in support_map.items():
            if other > itemset and abs(other_s - s) < 1e-9:
                is_closed = False
                break
        if is_closed and len(itemset) >= 2:
            closed.append({"itemsets": itemset, "support": s})

    return pd.DataFrame(closed).sort_values("support", ascending=False)


def assign_instances(certs: dict[str, frozenset],
                     closed: pd.DataFrame) -> dict[str, frozenset | None]:
    assignments = {}
    for iid, cert in certs.items():
        matching = [(row["itemsets"], len(row["itemsets"]), row["support"])
                    for _, row in closed.iterrows()
                    if row["itemsets"].issubset(cert)]
        if not matching:
            assignments[iid] = None
        else:
            # Largest pattern wins; break ties by support
            best = max(matching, key=lambda x: (x[1], x[2]))
            assignments[iid] = best[0]
    return assignments


# --- Step 3: Convergence filter ---

def compute_convergence(study_dir: Path,
                        condition: str = "no_context") -> dict[str, float]:
    """Mean pairwise cosine similarity of responses across models per instance."""
    model_responses: dict[str, dict[str, str]] = {}
    for model_key in MODEL_DIRS:
        rpath = study_dir / model_key / "records.json"
        if not rpath.exists():
            continue
        with open(rpath) as f:
            records = json.load(f)
        for r in records:
            iid = r["instance_id"]
            resp = r["conditions"].get(condition, {}).get("response", "")
            if resp:
                model_responses.setdefault(iid, {})[model_key] = resp

    # Only instances with >= 2 models
    multi = {iid: resps for iid, resps in model_responses.items()
             if len(resps) >= 2}
    if not multi:
        return {}

    print(f"  Computing convergence for {len(multi)} instances with >=2 models...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    convergence = {}

    all_texts = []
    all_keys = []
    for iid, resps in multi.items():
        for model_key, text in resps.items():
            all_texts.append(text[:1000])
            all_keys.append((iid, model_key))

    embeddings = encoder.encode(all_texts, show_progress_bar=False, batch_size=32)

    # Group by instance
    from collections import defaultdict
    iid_embs: dict[str, list[np.ndarray]] = defaultdict(list)
    for (iid, _), emb in zip(all_keys, embeddings):
        iid_embs[iid].append(emb)

    for iid, embs in iid_embs.items():
        emb_mat = np.array(embs)
        norms = np.linalg.norm(emb_mat, axis=1, keepdims=True)
        emb_mat = emb_mat / np.maximum(norms, 1e-9)
        sim_mat = emb_mat @ emb_mat.T
        n = len(embs)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        mean_sim = np.mean([sim_mat[i, j] for i, j in pairs]) if pairs else 0.0
        convergence[iid] = float(mean_sim)

    return convergence


# --- Step 4: Name forms via LLM ---

def name_form(pattern: frozenset, descriptions: list[str],
              client: OpenAI, model: str) -> str:
    ops_str = ", ".join(sorted(pattern))
    # Summarize what the ops mean structurally
    added = sorted(op.replace("ADD_", "") for op in pattern if op.startswith("ADD_"))
    removed = sorted(op.replace("DEL_", "") for op in pattern if op.startswith("DEL_"))
    structural_summary = ""
    if added and removed:
        structural_summary = f"Adds: {', '.join(added)}. Removes: {', '.join(removed)}."
    elif added:
        structural_summary = f"Adds: {', '.join(added)}."
    else:
        structural_summary = f"Removes: {', '.join(removed)}."

    desc_sample = "\n".join(f"- {d[:150]}" for d in descriptions[:4])
    prompt = (
        f"Structural transformation: {structural_summary}\n"
        f"Raw AST ops: {ops_str}\n\n"
        f"Context from matching instances:\n{desc_sample}\n\n"
        "Name this fix strategy in 2-5 words. Be specific about what structurally changes "
        "(e.g. 'conditional branch replacement', 'return value addition', 'call chain refactor'). "
        "Do not use words like 'structural', 'behavioral', 'strategy', 'enhancement', 'maintenance'. "
        "Return only the name."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
        )
        return resp.choices[0].message.content.strip().strip('"')
    except Exception:
        return "unnamed"


# --- Plotting ---

def style_panel(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(PANEL_EDGE)
    ax.tick_params(labelsize=9)


def fig1_form_pass_rates(forms: list[dict], output_dir: Path):
    forms_with_data = [f for f in forms if f["n_instances"] >= 3]
    forms_sorted = sorted(forms_with_data, key=lambda f: -f["pass_rate"])

    names = [f["name"] for f in forms_sorted]
    pass_rates = [f["pass_rate"] for f in forms_sorted]
    sizes = [f["n_instances"] for f in forms_sorted]

    xs = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.7), 4))
    fig.subplots_adjust(bottom=0.35)
    style_panel(ax)

    bars = ax.bar(xs, pass_rates, color=TEAL, alpha=0.85)
    for bar, n in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"n={n}", ha="center", fontsize=7, color=NAVY)

    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8, rotation=40, ha="right")
    ax.set_ylabel("Pass rate", fontsize=9)
    ax.set_ylim(0, 0.7)
    ax.set_title("Pass rate by canonical fix strategy form", fontsize=11,
                 pad=6, fontweight="normal")

    fig.savefig(output_dir / "fig1_form_pass_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_form_pass_rates.png")


def fig2_agent_coverage(forms: list[dict], agents_dir: Path,
                         assignments: dict, output_dir: Path):
    form_names = [f["name"] for f in forms if f["n_instances"] >= 3]
    form_patterns = {f["name"]: f["pattern_set"] for f in forms if f["n_instances"] >= 3}

    agents = list(AGENT_LABELS.values())
    matrix = np.full((len(agents), len(form_names)), np.nan)

    for ai, (agent_key, agent_label) in enumerate(AGENT_LABELS.items()):
        fpath = agents_dir / f"{agent_key}.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)
        resolved = {d["instance_id"] for d in data if d["resolved"]}

        for fi, fname in enumerate(form_names):
            pattern = form_patterns[fname]
            form_instances = [iid for iid, pat in assignments.items()
                              if pat == pattern]
            if not form_instances:
                continue
            n_resolved = sum(1 for iid in form_instances if iid in resolved)
            matrix[ai, fi] = n_resolved / len(form_instances)

    fig, ax = plt.subplots(figsize=(max(10, len(form_names) * 0.8), 4))
    fig.subplots_adjust(bottom=0.35, left=0.2)

    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=0.6)
    ax.set_xticks(range(len(form_names)))
    ax.set_xticklabels(form_names, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents, fontsize=9)

    for i in range(len(agents)):
        for j in range(len(form_names)):
            if not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                color = "white" if val > 0.35 else NAVY
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=7, color=color)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Pass rate")
    ax.set_title("Agent pass rate by canonical fix strategy form", fontsize=11,
                 pad=6, fontweight="normal")

    fig.savefig(output_dir / "fig2_agent_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_agent_coverage.png")


def fig3_coverage_stats(forms: list[dict], n_total: int,
                         n_outliers: int, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.3, bottom=0.15)

    # Left: coverage pie
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    covered = n_total - n_outliers
    ax.pie([covered, n_outliers],
           labels=[f"Assigned\n({covered})", f"Outliers\n({n_outliers})"],
           colors=[TEAL, GRAY], autopct="%1.0f%%", startangle=90,
           textprops={"fontsize": 9})
    ax.set_title("Instance coverage by canonical forms", fontsize=10,
                 pad=6, fontweight="normal")

    # Right: form size distribution
    ax = axes[1]
    style_panel(ax)
    sizes = sorted([f["n_instances"] for f in forms], reverse=True)
    xs = np.arange(len(sizes))
    ax.bar(xs, sizes, color=ORANGE, alpha=0.85)
    ax.set_xlabel("Form rank", fontsize=9)
    ax.set_ylabel("Instances assigned", fontsize=9)
    ax.set_title("Instances per canonical form", fontsize=10,
                 pad=6, fontweight="normal")

    fig.savefig(output_dir / "fig3_coverage_stats.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_coverage_stats.png")


# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--support", type=float, default=0.10)
    parser.add_argument("--convergence", type=float, default=0.65)
    parser.add_argument("--namer-model", default="gpt-4o-mini")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    use_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = "https://openrouter.ai/api/v1" if use_openrouter else None
    namer_model = (f"openai/{args.namer_model}" if use_openrouter else args.namer_model)
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Load data
    print("Loading edit certificates...")
    certs = load_certs(ROOT / "output" / "resolved_traces_lite_full.jsonl")
    print(f"  {len(certs)} instances")

    fix_df = pd.read_parquet(
        ROOT / "notebooks" / "plots" / "fix_type_analysis" / "merged_analysis.parquet"
    )[["instance_id", "fix_type", "passed"]]
    pass_map = fix_df.set_index("instance_id")["passed"].to_dict()

    # Step 1: Closed frequent itemsets
    print(f"\nStep 1: Computing closed frequent itemsets (support={args.support})...")
    closed = compute_closed_itemsets(certs, args.support)
    print(f"  {len(closed)} closed itemsets (size >= 2)")

    # Step 2: Assign instances
    print("\nStep 2: Assigning instances to largest matching form...")
    assignments = assign_instances(certs, closed)
    n_assigned = sum(1 for v in assignments.values() if v is not None)
    n_outliers = len(assignments) - n_assigned
    print(f"  Assigned: {n_assigned}/{len(assignments)} ({100*n_assigned/len(assignments):.0f}%)")
    print(f"  Structural outliers: {n_outliers} ({100*n_outliers/len(assignments):.0f}%)")

    # Outlier pass rate
    outlier_ids = [iid for iid, pat in assignments.items() if pat is None]
    outlier_pass = np.mean([pass_map[iid] for iid in outlier_ids if iid in pass_map])
    overall_pass = np.mean([pass_map[iid] for iid in assignments if iid in pass_map])
    print(f"  Outlier pass rate: {outlier_pass:.3f} vs overall {overall_pass:.3f}")

    # Group instances by assigned pattern
    from collections import defaultdict
    pattern_to_instances: dict[frozenset, list[str]] = defaultdict(list)
    for iid, pat in assignments.items():
        if pat is not None:
            pattern_to_instances[pat].append(iid)

    print(f"  {len(pattern_to_instances)} distinct canonical forms")

    # Step 3: Convergence filter
    print(f"\nStep 3: Computing cross-model description convergence...")
    study_dir = ROOT / "output" / "prompting_study"
    convergence = compute_convergence(study_dir)
    if convergence:
        conv_values = list(convergence.values())
        print(f"  Convergence range: {min(conv_values):.3f} - {max(conv_values):.3f}, "
              f"mean={np.mean(conv_values):.3f}")
        convergent_ids = {iid for iid, s in convergence.items()
                         if s >= args.convergence}
        print(f"  Convergent instances (>={args.convergence}): {len(convergent_ids)}")
    else:
        convergent_ids = set()
        print("  No multi-model data available")

    # Load staged narratives for naming
    with open(ROOT / "output" / "staged_descriptions.json") as f:
        sd = json.load(f)
    narrative_map = {r["instance_id"]: r["staged_narrative"] for r in sd["results"]}

    # Step 4: Name each form
    print("\nStep 4: Naming canonical forms...")
    forms = []
    for pat, instances in sorted(pattern_to_instances.items(),
                                  key=lambda x: -len(x[1])):
        pass_rates = [pass_map[iid] for iid in instances if iid in pass_map]
        pr = np.mean(pass_rates) if pass_rates else 0.0

        # Prefer convergent descriptions; fall back to any staged narrative
        if convergent_ids:
            desc_instances = [iid for iid in instances if iid in convergent_ids
                              and iid in narrative_map]
        else:
            desc_instances = [iid for iid in instances if iid in narrative_map]

        descriptions = [narrative_map[iid] for iid in desc_instances[:8]]
        source = "convergent" if (convergent_ids and desc_instances) else "all_staged"

        if descriptions:
            name = name_form(pat, descriptions, client, namer_model)
        else:
            name = "unnamed_" + "_".join(sorted(pat))[:40]

        forms.append({
            "name": name,
            "pattern": sorted(pat),
            "pattern_set": pat,
            "n_instances": len(instances),
            "pass_rate": pr,
            "source": source,
            "n_convergent_descriptions": len(desc_instances),
            "instances": instances,
        })
        print(f"  [{len(instances):3d} inst, pr={pr:.2f}, {source}] {name}")
        print(f"    ops: {', '.join(sorted(pat))}")

    # Save canonical forms
    forms_out = [
        {k: v for k, v in f.items() if k != "pattern_set"}
        for f in forms
    ]
    with open(args.output_dir / "canonical_forms.json", "w") as f:
        json.dump({
            "support": args.support,
            "convergence_threshold": args.convergence,
            "n_forms": len(forms),
            "n_assigned": n_assigned,
            "n_outliers": n_outliers,
            "outlier_pass_rate": float(outlier_pass),
            "overall_pass_rate": float(overall_pass),
            "forms": forms_out,
        }, f, indent=2)
    print(f"\nSaved canonical_forms.json ({len(forms)} forms)")

    # Instance assignment table
    assign_df = pd.DataFrame([
        {
            "instance_id": iid,
            "form_name": next((f["name"] for f in forms
                               if assignments[iid] == f["pattern_set"]), None)
                         if assignments[iid] is not None else "outlier",
            "assigned": assignments[iid] is not None,
            "convergence_score": convergence.get(iid),
            "passed": pass_map.get(iid),
        }
        for iid in assignments
    ])
    assign_df.to_parquet(args.output_dir / "instance_assignments.parquet", index=False)

    # Figures
    print("\nGenerating figures...")
    fig1_form_pass_rates(forms, args.output_dir)
    fig2_agent_coverage(forms, ROOT / "output" / "swebench_results_lite_agents",
                        assignments, args.output_dir)
    fig3_coverage_stats(forms, len(assignments), n_outliers, args.output_dir)

    print(f"\nDone. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
