#!/usr/bin/env python3
"""
Build scoped certificates for agent patches and compute oracle alignment.

Reads cross_agent_patches.jsonl for raw agent diffs, pairs with oracle
before_content from resolved_traces_lite_full.jsonl, computes scoped
certs, and measures alignment with the oracle fix.

Outputs:
  output/pairwise_agent_comparison/agent_scoped_certs.json
  output/pairwise_agent_comparison/oracle_alignment.json

Usage:
    uv run python scripts/build_agent_scoped_certs.py
"""

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load modules manually to bypass the heavy __init__.py chain in
# analysis.procedures (which pulls numpy etc via procedure_divergence).
def _load_mod(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_m1 = _load_mod("ast_edit_sequences", "analysis/procedures/ast_edit_sequences.py")
_m2 = _load_mod("contextual_edit_ops", "analysis/procedures/contextual_edit_ops.py")
if "analysis.procedures" not in sys.modules or not hasattr(sys.modules["analysis.procedures"], "__path__"):
    _ap = type(sys)("analysis.procedures")
    _ap.__path__ = [str(ROOT / "analysis" / "procedures")]
    sys.modules["analysis.procedures"] = _ap
if "analysis" not in sys.modules or not hasattr(sys.modules["analysis"], "__path__"):
    _a = type(sys)("analysis")
    _a.__path__ = [str(ROOT / "analysis")]
    sys.modules["analysis"] = _a
sys.modules["analysis.procedures.ast_edit_sequences"] = _m1
sys.modules["analysis.procedures.contextual_edit_ops"] = _m2

from analysis.procedures.scoped_edit_ops import (
    compute_layered_similarity,
    patch_to_scoped_cert,
    trace_to_scoped_cert,
)

TRACES_PATH = ROOT / "output" / "resolved_traces_lite_full.jsonl"
PATCHES_PATH = ROOT / "output" / "datasets" / "swe_bench_lite_resolved" / "cross_agent_patches.jsonl"
OUTPUT_DIR = ROOT / "output" / "pairwise_agent_comparison"

# Map short agent labels in cross_agent_patches.jsonl to display names
AGENT_MAP = {
    "GPT-4": "SWE-agent GPT-4",
    "Claude 3.5": "SWE-agent Claude 3.5",
    "GPT-4o": "SWE-agent GPT-4o",
    "Claude 3 Opus": "SWE-agent Claude 3 Opus",
}

# Map agent_patches.json keys to leaderboard keys for pass/fail
AGENT_TO_LB = {
    "SWE-agent GPT-4": "20240402_sweagent_gpt4",
    "SWE-agent Claude 3.5": "20240620_sweagent_claude3.5sonnet",
    "SWE-agent GPT-4o": "20240728_sweagent_gpt4o",
    "SWE-agent Claude 3 Opus": "20240402_sweagent_claude3opus",
}


def _load_pass_fail() -> dict[str, dict[str, bool]]:
    """Load pass/fail from leaderboard JSON."""
    lb_path = ROOT / "output" / "leaderboard" / "lite_results.json"
    if not lb_path.exists():
        return {}
    with open(lb_path) as f:
        return json.load(f)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load oracle traces and compute oracle scoped certs
    print("Step 1: Loading oracle traces and computing scoped certs...")
    oracle_data: dict[str, dict] = {}  # instance_id -> {trace, scoped_cert}
    with open(TRACES_PATH) as f:
        for line in f:
            trace = json.loads(line)
            iid = trace["instance_id"]
            cert = trace_to_scoped_cert(trace)
            if cert is None:
                continue

            # Collect before_content for the primary file
            before_content = ""
            file_path = ""
            for ev in trace["events"]:
                if ev["type"] != "code_change":
                    continue
                d = ev["details"]
                if d["file_path"].endswith(".py"):
                    before_content = d["before_content"] or ""
                    file_path = d["file_path"]
                    break

            oracle_data[iid] = {
                "scoped_cert": cert,
                "before_content": before_content,
                "file_path": file_path,
            }
    print(f"  {len(oracle_data)} oracle certs loaded")

    # Step 2: Load agent patches
    print("\nStep 2: Loading agent patches...")
    agent_patches: dict[str, dict[str, str]] = defaultdict(dict)
    with open(PATCHES_PATH) as f:
        for line in f:
            rec = json.loads(line)
            iid = rec["instance_id"]
            for agent_short, patch in rec["patches"].items():
                agent_name = AGENT_MAP.get(agent_short, agent_short)
                if patch and patch.strip():
                    agent_patches[agent_name][iid] = patch

    for agent, patches in agent_patches.items():
        print(f"  {agent}: {len(patches)} patches")

    # Step 3: Load pass/fail data
    print("\nStep 3: Loading pass/fail results...")
    pass_fail = _load_pass_fail()
    resolved_sets: dict[str, set[str]] = {}
    for agent_name, lb_key in AGENT_TO_LB.items():
        if lb_key in pass_fail:
            resolved = {iid for iid, passed in pass_fail[lb_key].items() if passed}
            resolved_sets[agent_name] = resolved
            print(f"  {agent_name}: {len(resolved)} resolved")
        else:
            resolved_sets[agent_name] = set()
            print(f"  {agent_name}: no pass/fail data found")

    # Step 4: Compute agent scoped certs
    print("\nStep 4: Computing agent scoped certificates...")
    agent_scoped_certs: dict[str, dict[str, dict]] = {}
    for agent_name, patches in agent_patches.items():
        agent_certs = {}
        n_ok = 0
        n_fail = 0
        for iid, patch in patches.items():
            if iid not in oracle_data:
                continue
            oracle = oracle_data[iid]
            cert = patch_to_scoped_cert(
                patch, oracle["before_content"], oracle["file_path"],
            )
            if cert is None:
                n_fail += 1
                continue
            cert["instance_id"] = iid
            agent_certs[iid] = cert
            n_ok += 1
        agent_scoped_certs[agent_name] = agent_certs
        print(f"  {agent_name}: {n_ok} certs, {n_fail} failed")

    # Save agent scoped certs
    out_certs = {}
    for agent_name, certs in agent_scoped_certs.items():
        out_certs[agent_name] = {iid: cert for iid, cert in certs.items()}
    with open(OUTPUT_DIR / "agent_scoped_certs.json", "w") as f:
        json.dump(out_certs, f, indent=2)
    print(f"\nSaved agent_scoped_certs.json")

    # Step 5: Compute oracle alignment
    print("\nStep 5: Computing oracle alignment...")
    alignment_records = []
    for agent_name, certs in agent_scoped_certs.items():
        resolved = resolved_sets.get(agent_name, set())
        for iid, agent_cert in certs.items():
            if iid not in oracle_data:
                continue
            oracle_cert = oracle_data[iid]["scoped_cert"]
            sim = compute_layered_similarity(oracle_cert, agent_cert)

            # Patch minimality: agent size / oracle size
            oracle_size = oracle_cert["patch_size"]
            agent_size = agent_cert["patch_size"]
            minimality = agent_size / oracle_size if oracle_size > 0 else float("inf")

            # File navigation breakdown
            oracle_files = set(oracle_cert.get("file_paths", [oracle_cert.get("file_path", "")]))
            agent_files = set(agent_cert.get("file_paths", []))
            correct_file = bool(oracle_files & agent_files)
            extra_files = len(agent_files - oracle_files)

            if correct_file and extra_files == 0:
                file_category = "correct_only"
            elif correct_file and extra_files > 0:
                file_category = "correct_plus_extra"
            else:
                file_category = "wrong_file"

            alignment_records.append({
                "agent": agent_name,
                "instance_id": iid,
                "resolved": iid in resolved,
                "file_match": sim["file_match"],
                "file_category": file_category,
                "scope_jaccard": sim["scope_jaccard"],
                "scope_overlap_count": sim["scope_overlap_count"],
                "edit_jaccard": sim["edit_jaccard"],
                "patch_minimality": minimality,
                "agent_patch_size": agent_size,
                "oracle_patch_size": oracle_size,
                "agent_hunk_count": agent_cert["hunk_count"],
                "oracle_hunk_count": oracle_cert["hunk_count"],
                "agent_scopes": agent_cert["scopes_touched"],
                "oracle_scopes": oracle_cert["scopes_touched"],
            })

    with open(OUTPUT_DIR / "oracle_alignment.json", "w") as f:
        json.dump(alignment_records, f, indent=2)
    print(f"Saved oracle_alignment.json ({len(alignment_records)} records)")

    # Summary stats
    print("\n--- Oracle alignment summary ---")
    for agent_name in sorted(agent_scoped_certs.keys()):
        recs = [r for r in alignment_records if r["agent"] == agent_name]
        if not recs:
            continue
        n = len(recs)
        n_resolved = sum(1 for r in recs if r["resolved"])
        file_match_rate = sum(1 for r in recs if r["file_match"]) / n
        mean_scope_j = np.mean([r["scope_jaccard"] for r in recs])
        mean_edit_j = np.mean([r["edit_jaccard"] for r in recs])
        mean_min = np.median([r["patch_minimality"] for r in recs if r["patch_minimality"] < float("inf")])

        cats = Counter(r["file_category"] for r in recs)
        print(f"\n{agent_name} (n={n}, resolved={n_resolved}):")
        print(f"  File match rate: {file_match_rate:.1%}")
        print(f"  File categories: {dict(cats)}")
        print(f"  Mean scope Jaccard: {mean_scope_j:.3f}")
        print(f"  Mean edit Jaccard: {mean_edit_j:.3f}")
        print(f"  Median patch minimality: {mean_min:.2f}x oracle")


if __name__ == "__main__":
    main()
