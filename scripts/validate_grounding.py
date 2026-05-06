#!/usr/bin/env python3
"""
Grounding validation: do model responses accurately reflect the actual edit certificate?

For each instance x condition, extracts:
  - Ground truth: AST edit operation types from the actual diff (patch_to_ast_sequence)
  - Claimed: edit operation types implied by the model's response (LLM extraction)

Computes precision, recall, F1 between claimed and actual at the unique-op-type level.
Breaks down by condition, fix type, and pass/fail.

Usage:
  uv run python scripts/validate_grounding.py --limit 50
  uv run python scripts/validate_grounding.py --model gpt_4o --conditions no_context procedural
"""

import argparse
import difflib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from openai import OpenAI

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
OUTPUT_DIR = ROOT / "output" / "grounding_validation"

CONDITIONS = ["no_context", "procedural", "raw_logs"]
CONDITION_LABELS = {"no_context": "No context", "procedural": "Procedural", "raw_logs": "Raw logs"}

# All observed op types in the corpus — used as constrained vocabulary
KNOWN_OPS = [
    "ADD_Assign", "ADD_Attribute", "ADD_BinOp", "ADD_Call", "ADD_Compare",
    "ADD_Constant", "ADD_Expr", "ADD_If", "ADD_Name", "ADD_Return",
    "ADD_Subscript", "ADD_Tuple", "ADD_UnaryOp", "ADD_keyword",
    "DEL_Assign", "DEL_Attribute", "DEL_Call", "DEL_Constant",
    "DEL_Expr", "DEL_Name", "DEL_Return", "DEL_Subscript", "DEL_Tuple",
    "DEL_UnaryOp", "DEL_if",
]

PANEL_BG = "#f5f5f5"
PANEL_EDGE = "#dddddd"
COLORS = {"no_context": "#AAAAAA", "procedural": "#0C6583", "raw_logs": "#EE7733"}


# --- Ground truth extraction ---

def trace_to_edit_cert(trace: dict) -> set[str]:
    ops = []
    for ev in trace["events"]:
        if ev["type"] != "code_change":
            continue
        d = ev["details"]
        fp = d["file_path"]
        if not fp.endswith(".py"):
            continue
        before = d["before_content"].splitlines(keepends=True)
        after = d["after_content"].splitlines(keepends=True)
        raw_diff = "".join(difflib.unified_diff(before, after, fromfile=fp, tofile=fp))
        if not raw_diff:
            continue
        diff = f"diff --git a/{fp} b/{fp}\n" + raw_diff
        ops.extend(patch_to_ast_sequence(diff))
    return set(ops)


def load_all_certs(traces_path: Path) -> dict[str, set[str]]:
    certs = {}
    with open(traces_path) as f:
        for line in f:
            trace = json.loads(line)
            cert = trace_to_edit_cert(trace)
            if cert:
                certs[trace["instance_id"]] = cert
    return certs


# --- LLM-based claimed ops extraction ---

EXTRACTION_SYSTEM = """You extract AST-level edit operations implied by a bug fix plan.

Given a model's plan for fixing a software bug, identify which Python AST node types
the plan implies will be added or removed. Return ONLY a JSON array of strings from
the provided vocabulary. Do not include operations not clearly implied by the plan.

Vocabulary meanings:
- ADD_If / DEL_if: adding or removing a conditional branch
- ADD_Return / DEL_Return: adding or removing a return statement
- ADD_Call / DEL_Call: adding or removing a function/method call
- ADD_Compare: adding a comparison expression (==, !=, <, >, is, in)
- ADD_Assign / DEL_Assign: adding or removing a variable assignment
- ADD_Attribute / DEL_Attribute: adding or removing attribute access (obj.attr)
- ADD_Constant / DEL_Constant: adding or removing a literal value (string, number, None)
- ADD_Name / DEL_Name: adding or removing a variable reference
- ADD_Raise: adding a raise/exception statement
- ADD_BinOp: adding a binary operation (+, -, *, /, etc.)
- ADD_Expr / DEL_Expr: adding or removing a standalone expression statement
- ADD_Subscript / DEL_Subscript: adding or removing subscript access (obj[key])

Return format: ["ADD_If", "ADD_Compare", ...] — only ops clearly implied, nothing else."""


def extract_claimed_ops(response_text: str, client: OpenAI, model: str) -> set[str]:
    if not response_text or not response_text.strip():
        return set()
    prompt = (
        f"Bug fix plan:\n{response_text[:2000]}\n\n"
        f"Vocabulary: {json.dumps(KNOWN_OPS)}\n\n"
        "Return a JSON array of implied edit operations."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        # Extract JSON array from response
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return set()
        ops = json.loads(raw[start:end])
        return set(op for op in ops if op in KNOWN_OPS)
    except Exception:
        return set()


def f1_score(claimed: set, actual: set) -> tuple[float, float, float]:
    if not actual:
        return (1.0, 1.0, 1.0) if not claimed else (0.0, 1.0, 0.0)
    if not claimed:
        return (0.0, 0.0, 0.0)
    claimed_lower = {c.lower() for c in claimed}
    actual_lower = {a.lower() for a in actual}
    tp = len(claimed_lower & actual_lower)
    precision = tp / len(claimed)
    recall = tp / len(actual)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# --- Plotting ---

def style_panel(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(PANEL_EDGE)
    ax.tick_params(labelsize=9)


def fig_grounding_by_condition(df: pd.DataFrame, output_dir: Path):
    metrics = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1"]
    cond_order = [c for c in CONDITIONS if c in df["condition"].unique()]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    fig.subplots_adjust(wspace=0.12, bottom=0.2)

    xs = np.arange(len(cond_order))
    width = 0.5

    for ax, metric, label in zip(axes, metrics, metric_labels):
        style_panel(ax)
        means = [df[df["condition"] == c][metric].mean() for c in cond_order]
        colors = [COLORS[c] for c in cond_order]
        ax.bar(xs, means, width, color=colors, alpha=0.85)
        ax.set_xticks(xs)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in cond_order], fontsize=9)
        ax.set_title(label, fontsize=10, pad=6, fontweight="normal")
        ax.set_ylim(0, 1)
        if metric == "precision":
            ax.set_ylabel("Score", fontsize=9)

    fig.suptitle("Grounding accuracy by condition (claimed ops vs actual edit certificate)",
                 fontsize=11, y=1.01, fontweight="normal")
    fig.savefig(output_dir / "fig1_grounding_by_condition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_grounding_by_condition.png")


def fig_grounding_by_fix_type(df: pd.DataFrame, output_dir: Path):
    cond_order = [c for c in CONDITIONS if c in df["condition"].unique()]
    type_order = (df.groupby("fix_type")["f1"].mean()
                  .sort_values(ascending=False).index.tolist())
    type_order = [t for t in type_order
                  if df["fix_type"].value_counts().get(t, 0) >= 3]

    if not type_order:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.subplots_adjust(bottom=0.3)
    style_panel(ax)

    n_conds = len(cond_order)
    width = 0.8 / n_conds
    xs = np.arange(len(type_order))

    for i, cond in enumerate(cond_order):
        means = [df[(df["condition"] == cond) & (df["fix_type"] == ft)]["f1"].mean()
                 for ft in type_order]
        offset = (i - n_conds / 2 + 0.5) * width
        ax.bar(xs + offset, means, width, color=COLORS[cond],
               alpha=0.85, label=CONDITION_LABELS[cond])

    ax.set_xticks(xs)
    ax.set_xticklabels([t.replace("_", " ") for t in type_order],
                       fontsize=8, rotation=35, ha="right")
    ax.set_ylabel("F1 (claimed vs actual ops)", fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_title("Grounding F1 by fix type and condition", fontsize=11, pad=6, fontweight="normal")

    fig.savefig(output_dir / "fig2_grounding_by_fix_type.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_grounding_by_fix_type.png")


def fig_grounding_vs_score(df: pd.DataFrame, output_dir: Path):
    cond_order = [c for c in CONDITIONS if c in df["condition"].unique()]

    fig, axes = plt.subplots(1, len(cond_order), figsize=(11, 4), sharey=True)
    fig.subplots_adjust(wspace=0.1, bottom=0.2)

    for ax, cond in zip(axes, cond_order):
        style_panel(ax)
        subset = df[df["condition"] == cond].dropna(subset=["f1", "plan_quality"])
        ax.scatter(subset["f1"], subset["plan_quality"],
                   color=COLORS[cond], alpha=0.5, s=20)
        if len(subset) > 3:
            z = np.polyfit(subset["f1"], subset["plan_quality"], 1)
            xline = np.linspace(0, 1, 50)
            ax.plot(xline, np.polyval(z, xline), color="#2B2D42", linewidth=1)
        ax.set_xlabel("Grounding F1", fontsize=9)
        ax.set_title(CONDITION_LABELS[cond], fontsize=10, pad=6, fontweight="normal")
        if cond == cond_order[0]:
            ax.set_ylabel("Plan quality score", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.8, 3.2)

    fig.suptitle("Grounding accuracy vs plan quality score", fontsize=11, y=1.01, fontweight="normal")
    fig.savefig(output_dir / "fig3_grounding_vs_score.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_grounding_vs_score.png")


# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt_4o", help="Which model's records to use")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--limit", type=int, default=None, help="Max instances (for testing)")
    parser.add_argument("--extractor-model", default="gpt-4o-mini")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = args.output_dir / f"claimed_ops_{args.model}.json"

    use_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = "https://openrouter.ai/api/v1" if use_openrouter else None
    extractor_model = (f"openai/{args.extractor_model}" if use_openrouter
                       else args.extractor_model)

    client = OpenAI(api_key=api_key, base_url=base_url)

    print("Loading edit certificates from traces...")
    certs = load_all_certs(ROOT / "output" / "resolved_traces_lite_full.jsonl")
    print(f"  {len(certs)} instances with non-empty edit certificates")

    print("Loading model records...")
    with open(ROOT / "output" / "prompting_study" / args.model / "records.json") as f:
        records = json.load(f)

    # Load fix types
    fix_df = pd.read_parquet(
        ROOT / "notebooks" / "plots" / "fix_type_analysis" / "merged_analysis.parquet"
    )[["instance_id", "fix_type", "passed"]]
    fix_map = fix_df.set_index("instance_id")[["fix_type", "passed"]].to_dict("index")

    # Build work items: (instance_id, condition, response_text, plan_quality_score)
    work = []
    for r in records:
        iid = r["instance_id"]
        if iid not in certs:
            continue
        for cond in args.conditions:
            cond_data = r["conditions"].get(cond, {})
            response = cond_data.get("response", "")
            scores = cond_data.get("scores") or {}
            plan_quality = scores.get("plan_quality")
            if not response:
                continue
            work.append((iid, cond, response, plan_quality))

    if args.limit:
        work = work[:args.limit]

    print(f"  {len(work)} response-condition pairs to process")

    # Load cache if exists
    cache: dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached extractions")

    # Extract claimed ops (skip cached)
    to_process = [(iid, cond, resp, pq) for iid, cond, resp, pq in work
                  if f"{iid}_{cond}" not in cache]
    print(f"  {len(to_process)} need LLM extraction...")

    def process_one(item):
        iid, cond, response, _ = item
        key = f"{iid}_{cond}"
        claimed = extract_claimed_ops(response, client, extractor_model)
        return key, list(claimed)

    if to_process:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one, item): item for item in to_process}
            done = 0
            for fut in as_completed(futures):
                key, claimed = fut.result()
                cache[key] = claimed
                done += 1
                if done % 20 == 0:
                    print(f"  {done}/{len(to_process)} extracted...")

        with open(cache_path, "w") as f:
            json.dump(cache, f)
        print(f"  Saved cache to {cache_path.name}")

    # Build results dataframe
    rows = []
    for iid, cond, _, plan_quality in work:
        key = f"{iid}_{cond}"
        claimed = set(cache.get(key, []))
        actual = certs[iid]
        p, r, f1 = f1_score(claimed, actual)
        meta = fix_map.get(iid, {})
        rows.append({
            "instance_id": iid,
            "condition": cond,
            "precision": p,
            "recall": r,
            "f1": f1,
            "n_claimed": len(claimed),
            "n_actual": len(actual),
            "plan_quality": plan_quality,
            "fix_type": meta.get("fix_type"),
            "passed": meta.get("passed"),
            "claimed_ops": list(claimed),
            "actual_ops": list(actual),
        })

    df = pd.DataFrame(rows)
    df.to_parquet(args.output_dir / f"grounding_{args.model}.parquet", index=False)

    # Summary
    print("\nGrounding scores by condition:")
    summary = df.groupby("condition")[["precision", "recall", "f1"]].mean().round(3)
    print(summary.to_string())

    print("\nGrounding F1 by condition x fix type:")
    ct = df.groupby(["condition", "fix_type"])["f1"].mean().round(3).unstack(fill_value=np.nan)
    print(ct.to_string())

    print("\nCorrelation between grounding F1 and plan quality:")
    for cond in df["condition"].unique():
        sub = df[df["condition"] == cond].dropna(subset=["f1", "plan_quality"])
        if len(sub) > 5:
            corr = sub[["f1", "plan_quality"]].corr().iloc[0, 1]
            print(f"  {cond}: r={corr:.3f} (n={len(sub)})")

    print("\nGenerating figures...")
    fig_grounding_by_condition(df, args.output_dir)
    fig_grounding_by_fix_type(df, args.output_dir)
    fig_grounding_vs_score(df, args.output_dir)

    print(f"\nDone. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
