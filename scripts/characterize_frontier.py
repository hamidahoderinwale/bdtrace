#!/usr/bin/env python3
"""
Characterize frontier form instances: what do the oracle patches actually do?

Reads oracle traces for the 20 frontier instances (+For++If and -Name+-Call)
and shows the actual diffs, extracted ops, and a brief LLM summary of each fix.

This validates that frontier form labels reflect genuine structural requirements,
not decision tree artifacts.

Usage:
  uv run python scripts/characterize_frontier.py
"""

import difflib
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence

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

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "frontier_characterization"

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

FRONTIER_FORMS = ["+For++If", "-Name+-Call"]


def extract_diff(trace: dict) -> tuple[str, list[str]]:
    """Return (unified_diff_text, normalized_ops) for a trace."""
    all_diff_parts = []
    all_ops = []
    for ev in trace["events"]:
        if ev["type"] != "code_change":
            continue
        d = ev["details"]
        if not d["file_path"].endswith(".py"):
            continue
        before = d["before_content"].splitlines(keepends=True)
        after = d["after_content"].splitlines(keepends=True)
        raw = "".join(difflib.unified_diff(
            before, after,
            fromfile=f"a/{d['file_path']}",
            tofile=f"b/{d['file_path']}",
            lineterm="",
        ))
        if not raw:
            continue
        all_diff_parts.append(raw)
        diff_with_header = f"diff --git a/{d['file_path']} b/{d['file_path']}\n" + raw
        ops = patch_to_ast_sequence(diff_with_header)
        all_ops.extend(ops)

    diff_text = "\n".join(all_diff_parts)
    norm_ops = [_NORMALIZE_OPS.get(op, op) for op in all_ops]
    return diff_text, norm_ops


def summarize_with_llm(instance_id: str, form: str, diff_text: str,
                       ops: list[str], problem_statement: str) -> str:
    """Use gpt-4o-mini to write a 1-sentence characterization of the fix."""
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        )
        diff_snippet = diff_text[:1500]
        prompt = (
            f"Instance: {instance_id}\n"
            f"Structural form: {form}\n"
            f"Edit operations: {', '.join(sorted(set(ops)))}\n"
            f"Problem statement (truncated): {problem_statement[:300]}\n\n"
            f"Diff:\n{diff_snippet}\n\n"
            "In one sentence, describe what this fix actually does at a programmer level "
            "(e.g. 'Adds a for-loop to iterate over items before checking a condition' or "
            "'Removes a redundant call to X and replaces with direct attribute access'). "
            "Be specific and concrete. Do not use the words 'structural', 'behavioral', or 'form'."
        )
        resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM unavailable: {e}]"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load frontier instance IDs
    form_df = pd.read_parquet(ROOT / "output" / "fix_forms" / "form_assignments.parquet")
    frontier_df = form_df[form_df["form_label"].isin(FRONTIER_FORMS)].copy()
    frontier_ids = set(frontier_df["instance_id"])
    print(f"Frontier instances: {len(frontier_ids)}")
    for form in FRONTIER_FORMS:
        n = (frontier_df["form_label"] == form).sum()
        print(f"  {form}: {n} instances")

    # Load oracle traces
    print("\nLoading traces...")
    traces = {}
    with open(ROOT / "output" / "resolved_traces_lite_full.jsonl") as f:
        for line in f:
            t = json.loads(line)
            if t["instance_id"] in frontier_ids:
                traces[t["instance_id"]] = t

    # Load problem statements from HuggingFace dataset cache or from traces
    problem_statements = {}
    for iid, trace in traces.items():
        ps = trace.get("problem_statement", "") or ""
        if not ps:
            # try metadata field
            ps = trace.get("metadata", {}).get("problem_statement", "")
        problem_statements[iid] = ps

    # Extract diffs and ops
    results = []
    for _, row in frontier_df.iterrows():
        iid = row["instance_id"]
        form = row["form_label"]
        if iid not in traces:
            print(f"  WARNING: no trace for {iid}")
            continue
        trace = traces[iid]
        diff_text, ops = extract_diff(trace)
        results.append({
            "instance_id": iid,
            "form": form,
            "ops": ops,
            "unique_ops": sorted(set(ops)),
            "diff_text": diff_text,
            "problem_statement": problem_statements.get(iid, ""),
        })

    print(f"\nExtracted diffs for {len(results)} instances\n")

    # Print summary per form
    for form in FRONTIER_FORMS:
        form_results = [r for r in results if r["form"] == form]
        print(f"{'='*60}")
        print(f"Form: {form}  ({len(form_results)} instances)")
        print(f"{'='*60}")

        # Aggregate op frequencies
        from collections import Counter
        all_ops_flat = [op for r in form_results for op in r["ops"]]
        op_counts = Counter(all_ops_flat)
        print(f"Most common ops across all instances:")
        for op, cnt in op_counts.most_common(10):
            print(f"  {op:30s}: {cnt}")

        print()
        for r in form_results:
            print(f"--- {r['instance_id']} ---")
            print(f"Unique ops: {r['unique_ops']}")
            # Show a compact diff snippet (first 30 changed lines)
            diff_lines = [l for l in r["diff_text"].splitlines()
                          if l.startswith("+") or l.startswith("-")]
            diff_lines = diff_lines[:25]
            for dl in diff_lines:
                print(f"  {dl[:120]}")
            print()

    # LLM characterization
    print("\nGenerating LLM characterizations...")
    characterizations = []
    for r in results:
        summary = summarize_with_llm(
            r["instance_id"], r["form"], r["diff_text"],
            r["ops"], r["problem_statement"]
        )
        characterizations.append({
            "instance_id": r["instance_id"],
            "form": r["form"],
            "unique_ops": r["unique_ops"],
            "summary": summary,
        })
        print(f"  [{r['form']:15s}] {r['instance_id']}: {summary}")

    # Save
    output = {
        "frontier_forms": FRONTIER_FORMS,
        "n_instances": len(results),
        "characterizations": characterizations,
    }
    with open(OUTPUT_DIR / "frontier_characterizations.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved frontier_characterizations.json")

    # Print grouped by form for easy reading
    print("\nFinal summary by form:")
    for form in FRONTIER_FORMS:
        print(f"\n{form}:")
        for c in characterizations:
            if c["form"] == form:
                print(f"  {c['instance_id']}: {c['summary']}")


if __name__ == "__main__":
    main()
