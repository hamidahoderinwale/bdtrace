#!/usr/bin/env python3
"""
Scale the pairwise agent comparison to architecturally diverse agents.

Fetches prediction files from the swe-bench-submissions S3 bucket (public,
unsigned), computes edit certificates, and merges with existing data.

Steps:
1. Survey available agents in swe-bench/experiments GitHub repo
2. Fetch prediction files from S3 for prioritized agents
3. Compute edit certificates using patch_to_ast_sequence + _NORMALIZE_OPS
4. Merge with existing agent_patches.json (preserving existing 5 agents)
5. Recompute pairwise statistics
6. Print summary
"""

import importlib.util
import json
import math
import ssl
import statistics
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# Fix SSL certificate verification
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()
    _SSL_CTX.check_hostname = False
    _SSL_CTX.verify_mode = ssl.CERT_NONE

# Direct import of just ast_edit_sequences to avoid heavy package init chain
_ast_mod_path = _ROOT / "analysis" / "procedures" / "ast_edit_sequences.py"
_spec = importlib.util.spec_from_file_location("ast_edit_sequences", _ast_mod_path)
_ast_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ast_mod)
patch_to_ast_sequence = _ast_mod.patch_to_ast_sequence

ROOT = _ROOT
OUTPUT_DIR = ROOT / "output" / "pairwise_agent_comparison"

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


# --- Pure-python stats helpers (no numpy needed) ---

def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs):
    if not xs:
        return 0.0
    return statistics.median(xs)


def _stdev(xs):
    if len(xs) < 2:
        return 0.0
    return statistics.stdev(xs)


def _frac(xs, pred):
    """Fraction of elements in xs satisfying pred."""
    if not xs:
        return 0.0
    return sum(1 for x in xs if pred(x)) / len(xs)


# --- Agent configuration ---

AGENT_DISPLAY_NAMES = {
    "20240530_autocoderover-v20240408": "AutoCodeRover v1",
    "20240621_autocoderover-v20240620": "AutoCodeRover v2",
    "20240523_aider": "Aider",
    "20240630_agentless_gpt4o": "Agentless",
    "20240509_amazon-q-developer-agent-20240430-dev": "Amazon Q Developer",
    "20240725_opendevin_codeact_v1.8_claude35sonnet": "OpenHands CodeAct",
    "20240617_factory_code_droid": "Factory Code Droid",
    "20240620_sweagent_claude3.5sonnet": "SWE-agent Claude 3.5 Sonnet (v2)",
    "20240728_sweagent_gpt4o": "SWE-agent GPT-4o (v2)",
    "20240820_honeycomb": "Honeycomb",
    "20241025_OpenHands-CodeAct-2.1-sonnet-20241022": "OpenHands CodeAct 2.1",
    "20240811_gru": "Gru",
    "20240524_opencsg_starship_gpt4": "OpenCSG Starship",
    "20240615_appmap-navie_gpt4o": "AppMap Navie",
    "20240702_codestory_aide_mixed": "CodeStory Aide",
    "20240706_sima_gpt4o": "SIMA GPT-4o",
    "20240804_EnIGMA_GPT-4o": "EnIGMA GPT-4o",
    "20240910_autocoderover-v20240620-claude3.5sonnet": "AutoCodeRover Claude 3.5",
    "20240828_autose_mixed": "AutoSE",
    "20240721_amazon-q-developer-agent-20240719-dev": "Amazon Q v2",
    "20240925_SWE-Fixer_GPT-4o-mini": "SWE-Fixer GPT-4o-mini",
    "20241128_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor_20241128": "SWE-Fixer Qwen2.5",
    "20240604_CodeR": "CodeR",
    "20240612_MASAI_gpt4o": "MASAI",
    "20240617_moatless_gpt4o": "Moatless GPT-4o",
    "20240623_moatless_claude35sonnet": "Moatless Claude 3.5",
    "20241028_agentless-1.5_gpt4o": "Agentless 1.5",
    "20240627_abanteai_mentatbot_gpt4o": "MentatBot",
    "20240806_SuperCoder2.0": "SuperCoder 2.0",
}

PRIORITY_AGENTS = [
    "20240630_agentless_gpt4o",
    "20240530_autocoderover-v20240408",
    "20240621_autocoderover-v20240620",
    "20240910_autocoderover-v20240620-claude3.5sonnet",
    "20240523_aider",
    "20240725_opendevin_codeact_v1.8_claude35sonnet",
    "20241025_OpenHands-CodeAct-2.1-sonnet-20241022",
    "20240509_amazon-q-developer-agent-20240430-dev",
    "20240721_amazon-q-developer-agent-20240719-dev",
    "20240604_CodeR",
    "20240617_factory_code_droid",
    "20240820_honeycomb",
    "20240615_appmap-navie_gpt4o",
    "20240702_codestory_aide_mixed",
    "20240706_sima_gpt4o",
    "20240804_EnIGMA_GPT-4o",
    "20240811_gru",
    "20240925_SWE-Fixer_GPT-4o-mini",
    "20240828_autose_mixed",
    "20240524_opencsg_starship_gpt4",
    "20240612_MASAI_gpt4o",
    "20240617_moatless_gpt4o",
    "20240623_moatless_claude35sonnet",
    "20241028_agentless-1.5_gpt4o",
    "20240627_abanteai_mentatbot_gpt4o",
    "20240806_SuperCoder2.0",
]


# --- Data fetching ---

def fetch_github_api(url, timeout=30):
    """Fetch JSON from GitHub API."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX)
    return json.loads(resp.read().decode("utf-8"))


def list_available_agents():
    """List all agent directories in the experiments repo."""
    print("Step 1: Surveying available agents in swe-bench/experiments...")
    url = "https://api.github.com/repos/swe-bench/experiments/contents/evaluation/lite"
    try:
        items = fetch_github_api(url)
        dirs = [item["name"] for item in items if item["type"] == "dir"]
        print(f"  Found {len(dirs)} agent directories")
        for d in sorted(dirs):
            tag = " [PRIORITY]" if d in PRIORITY_AGENTS else ""
            print(f"    {d}{tag}")
        return dirs
    except Exception as e:
        print(f"  Failed to list agents: {e}")
        print("  Using known agent list instead")
        return list(PRIORITY_AGENTS)


_S3_BUCKET = "swe-bench-submissions"
_S3_BASE_URL = f"https://{_S3_BUCKET}.s3.amazonaws.com"


def _parse_predictions(content):
    """Parse JSONL content into {instance_id: model_patch} dict."""
    patches = {}
    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        iid = rec.get("instance_id", "")
        patch = rec.get("model_patch", rec.get("prediction", ""))
        if iid and patch and patch.strip():
            patches[iid] = patch
    return patches


def _fetch_s3_object(key):
    """Fetch an object from S3 via public HTTP URL."""
    url = f"{_S3_BASE_URL}/{key}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=120, context=_SSL_CTX)
    return resp.read().decode("utf-8")


def fetch_predictions_for_agent(agent_dir):
    """Fetch prediction file for an agent from S3 bucket via HTTP."""
    for fname in ["all_preds.jsonl", "predictions.jsonl"]:
        key = f"lite/{agent_dir}/{fname}"
        try:
            content = _fetch_s3_object(key)
            patches = _parse_predictions(content)
            if patches:
                return patches, f"s3://{_S3_BUCKET}/{key}"
        except urllib.error.HTTPError:
            continue
        except Exception as e:
            print(f"    Error fetching {key}: {e}")
            continue

    return None, None


# --- Edit certificates ---

def compute_certificate(patch_text):
    """Compute normalized edit certificate from a patch string."""
    ops = patch_to_ast_sequence(patch_text)
    if not ops:
        return frozenset()
    return frozenset(_NORMALIZE_OPS.get(op, op) for op in ops)


def jaccard(a, b):
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_pass_fail_and_lite_ids():
    """Load pass/fail results and the canonical set of Lite instance IDs."""
    # Try msgpack first
    path = ROOT / "output" / "leaderboard" / "lite_results.msgpack"
    if path.exists():
        try:
            import msgpack
            with open(path, "rb") as f:
                data = msgpack.unpack(f, raw=False)
            # Extract Lite IDs from all agents in the leaderboard
            lite_ids = set()
            for agent_data in data.values():
                lite_ids.update(agent_data.keys())
            return data, lite_ids
        except ImportError:
            print("  msgpack not available for local data")
        except Exception as e:
            print(f"  Warning: Could not load local leaderboard data: {e}")

    # Fallback: fetch results.json from GitHub for each agent
    print("  Falling back to GitHub results.json per agent...")
    base = "https://raw.githubusercontent.com/swe-bench/experiments/refs/heads/main"
    pass_fail = {}
    lite_ids = set()
    for agent_dir in PRIORITY_AGENTS:
        url = f"{base}/evaluation/lite/{agent_dir}/results/results.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=15, context=_SSL_CTX)
            data = json.loads(resp.read().decode())
            resolved = set(data.get("resolved", []))
            generated = set(data.get("generated", []))
            applied = set(data.get("applied", []))
            # Collect all known Lite IDs
            all_ids = generated | applied | resolved
            all_ids.update(data.get("no_generation", []))
            all_ids.update(data.get("no_apply", []))
            lite_ids.update(all_ids)
            # Build pass/fail dict
            pf = {}
            for iid in generated:
                pf[iid] = iid in resolved
            if pf:
                pass_fail[agent_dir] = pf
        except Exception:
            continue
    return pass_fail, lite_ids


# --- Main pipeline ---

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Survey available agents
    all_dirs = list_available_agents()

    # Step 2: Load existing agent_patches.json
    print("\nStep 2: Loading existing agent patches...")
    existing_path = OUTPUT_DIR / "agent_patches.json"
    if existing_path.exists():
        with open(existing_path) as f:
            existing_patches = json.load(f)
        print(f"  Loaded {len(existing_patches)} existing agents:")
        for agent, certs in existing_patches.items():
            print(f"    {agent}: {len(certs)} instances")
    else:
        existing_patches = {}

    # Step 3: Load pass/fail data and Lite instance IDs
    print("\nStep 3: Loading pass/fail data...")
    pass_fail, lite_ids = load_pass_fail_and_lite_ids()
    print(f"  Loaded pass/fail for {len(pass_fail)} agents")
    print(f"  SWE-bench Lite instances: {len(lite_ids)}")

    # Step 4: Fetch patches for priority agents from S3
    print("\nStep 4: Fetching patches for diverse agents from S3...")
    new_agent_raw_patches = {}
    fetch_results = {}

    for agent_dir in PRIORITY_AGENTS:
        display = AGENT_DISPLAY_NAMES.get(agent_dir, agent_dir)

        # Skip if we already have this agent under its display name
        if display in existing_patches:
            print(f"  {display}: already in existing data, skipping")
            fetch_results[display] = "exists"
            continue

        # Also skip if it matches an existing agent key exactly
        if agent_dir in existing_patches:
            print(f"  {agent_dir}: already in existing data, skipping")
            fetch_results[display] = "exists"
            continue

        patches, source = fetch_predictions_for_agent(agent_dir)
        if patches:
            print(f"  {display}: {len(patches)} patches from {source}")
            new_agent_raw_patches[agent_dir] = patches
            fetch_results[display] = f"{len(patches)} patches"
        else:
            print(f"  {display}: no predictions found in S3")
            fetch_results[display] = "not found"

    # Step 5: Compute edit certificates for new agents
    print(f"\nStep 5: Computing edit certificates for {len(new_agent_raw_patches)} new agents...")
    new_agent_certs = {}

    for agent_dir, raw_patches in new_agent_raw_patches.items():
        display = AGENT_DISPLAY_NAMES.get(agent_dir, agent_dir)

        # Check pass/fail if available
        pf = pass_fail.get(agent_dir, {})
        has_pf = bool(pf)

        certs = {}
        n_skipped_fail = 0
        n_skipped_not_lite = 0
        n_empty_cert = 0

        for iid, patch in raw_patches.items():
            # Filter to SWE-bench Lite instances only
            if lite_ids and iid not in lite_ids:
                n_skipped_not_lite += 1
                continue

            # If we have pass/fail data, only include passed instances
            if has_pf:
                if not pf.get(iid, False):
                    n_skipped_fail += 1
                    continue

            cert = compute_certificate(patch)
            if cert:
                certs[iid] = sorted(cert)
            else:
                n_empty_cert += 1

        new_agent_certs[display] = certs
        parts = [f"{len(certs)} certs"]
        if has_pf:
            parts.append(f"{n_skipped_fail} failed")
        if n_skipped_not_lite:
            parts.append(f"{n_skipped_not_lite} not-lite")
        if n_empty_cert:
            parts.append(f"{n_empty_cert} empty")
        print(f"  {display}: {', '.join(parts)}")

    # Step 6: Merge with existing data
    print(f"\nStep 6: Merging {len(new_agent_certs)} new agents with {len(existing_patches)} existing...")
    merged = dict(existing_patches)
    for agent, certs in new_agent_certs.items():
        if certs:
            merged[agent] = certs

    # Post-merge cleanup: for agents that had no pass/fail filtering,
    # fetch resolved lists from GitHub and apply retroactively
    display_to_dir = {v: k for k, v in AGENT_DISPLAY_NAMES.items()}
    base_gh = "https://raw.githubusercontent.com/swe-bench/experiments/refs/heads/main"
    agents_needing_filter = []
    for agent in merged:
        agent_dir = display_to_dir.get(agent)
        if agent_dir and agent_dir not in pass_fail:
            agents_needing_filter.append((agent, agent_dir))

    if agents_needing_filter:
        print(f"  Fetching resolved lists for {len(agents_needing_filter)} unfiltered agents...")
        for agent, agent_dir in agents_needing_filter:
            url = f"{base_gh}/evaluation/lite/{agent_dir}/results/results.json"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                resp = urllib.request.urlopen(req, timeout=15, context=_SSL_CTX)
                data = json.loads(resp.read().decode())
                resolved = set(data.get("resolved", []))
                before = len(merged[agent])
                merged[agent] = {
                    iid: cert for iid, cert in merged[agent].items()
                    if iid in resolved
                }
                after = len(merged[agent])
                if before != after:
                    print(f"    {agent}: {before} -> {after} (kept only resolved)")
            except Exception:
                pass  # Keep as-is if we can't fetch results

    # Filter all agents to Lite instances only
    if lite_ids:
        print(f"  Filtering all agents to {len(lite_ids)} Lite instances...")
        for agent in list(merged.keys()):
            before = len(merged[agent])
            merged[agent] = {
                iid: cert for iid, cert in merged[agent].items()
                if iid in lite_ids
            }
            after = len(merged[agent])
            if before != after:
                print(f"    {agent}: {before} -> {after} (removed {before - after} non-Lite)")

    # Remove agents with 0 certificates
    empty_agents = [a for a, c in merged.items() if not c]
    for a in empty_agents:
        del merged[a]
        print(f"    Removed {a}: 0 certificates")

    print(f"  Total agents after merge: {len(merged)}")
    for agent in sorted(merged.keys()):
        print(f"    {agent}: {len(merged[agent])} instances")

    with open(OUTPUT_DIR / "agent_patches.json", "w") as f:
        json.dump(merged, f, indent=2)
    print(f"  Saved agent_patches.json")

    # Step 7: Recompute pairwise statistics
    print(f"\nStep 7: Computing pairwise Jaccard similarities for {len(merged)} agents...")
    agent_names = sorted(merged.keys())
    pairwise_results = []
    all_jaccards = []

    for a1, a2 in combinations(agent_names, 2):
        certs1 = merged[a1]
        certs2 = merged[a2]
        co_solved = set(certs1.keys()) & set(certs2.keys())

        if not co_solved:
            pairwise_results.append({
                "agent_1": a1, "agent_2": a2,
                "n_co_solved": 0, "mean_jaccard": None, "identical_frac": None,
            })
            continue

        jaccards = []
        identical = 0
        for iid in co_solved:
            s1 = set(certs1[iid])
            s2 = set(certs2[iid])
            j = jaccard(s1, s2)
            jaccards.append(j)
            all_jaccards.append({
                "agent_1": a1, "agent_2": a2,
                "instance_id": iid, "jaccard": j,
                "cert_1_size": len(s1), "cert_2_size": len(s2),
            })
            if s1 == s2:
                identical += 1

        pairwise_results.append({
            "agent_1": a1, "agent_2": a2,
            "n_co_solved": len(co_solved),
            "mean_jaccard": _mean(jaccards),
            "median_jaccard": _median(jaccards),
            "std_jaccard": _stdev(jaccards),
            "identical_frac": identical / len(co_solved),
            "jaccard_gt_0.5": _frac(jaccards, lambda x: x > 0.5),
            "jaccard_gt_0.8": _frac(jaccards, lambda x: x > 0.8),
        })

    # Step 8: Agent vocabulary diversity
    print("\nStep 8: Computing agent vocabulary diversity...")
    agent_vocab = {}
    for agent, certs in merged.items():
        all_ops = set()
        for cert_ops in certs.values():
            all_ops.update(cert_ops)
        agent_vocab[agent] = {
            "n_instances": len(certs),
            "vocab_size": len(all_ops),
            "vocabulary": sorted(all_ops),
        }

    agent_avg_jaccard = defaultdict(list)
    for pr in pairwise_results:
        if pr["mean_jaccard"] is not None:
            agent_avg_jaccard[pr["agent_1"]].append(pr["mean_jaccard"])
            agent_avg_jaccard[pr["agent_2"]].append(pr["mean_jaccard"])

    for agent in agent_vocab:
        vals = agent_avg_jaccard.get(agent, [])
        agent_vocab[agent]["avg_pairwise_jaccard"] = _mean(vals) if vals else None

    # Compile and save summary
    all_j_vals = [j["jaccard"] for j in all_jaccards]
    summary = {
        "n_agents": len(merged),
        "agents": {agent: len(certs) for agent, certs in merged.items()},
        "pairwise_comparisons": pairwise_results,
        "agent_vocabulary": agent_vocab,
        "overall_stats": {},
    }

    if all_j_vals:
        summary["overall_stats"] = {
            "total_pairwise_comparisons": len(all_j_vals),
            "mean_jaccard": _mean(all_j_vals),
            "median_jaccard": _median(all_j_vals),
            "identical_frac": _frac(all_j_vals, lambda x: x == 1.0),
            "jaccard_gt_0.5_frac": _frac(all_j_vals, lambda x: x > 0.5),
            "jaccard_gt_0.8_frac": _frac(all_j_vals, lambda x: x > 0.8),
            "jaccard_zero_frac": _frac(all_j_vals, lambda x: x == 0.0),
        }

    with open(OUTPUT_DIR / "pairwise_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved pairwise_results.json")

    # Print final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total agents: {len(merged)}")
    print(f"  Existing (preserved): {len(existing_patches)}")
    print(f"  New agents added: {len(merged) - len(existing_patches)}")

    print(f"\nAgent coverage:")
    for agent in sorted(merged.keys()):
        print(f"  {agent}: {len(merged[agent])} instances")

    if all_j_vals:
        n_pairs = sum(1 for pr in pairwise_results if pr["n_co_solved"] > 0)
        print(f"\nPairwise statistics ({n_pairs} active pairs, {len(all_j_vals)} instance comparisons):")
        print(f"  Mean Jaccard: {_mean(all_j_vals):.3f}")
        print(f"  Median Jaccard: {_median(all_j_vals):.3f}")
        print(f"  Identical certificates (J=1.0): {_frac(all_j_vals, lambda x: x == 1.0):.1%}")
        print(f"  High similarity (J>0.5): {_frac(all_j_vals, lambda x: x > 0.5):.1%}")
        print(f"  Very high similarity (J>0.8): {_frac(all_j_vals, lambda x: x > 0.8):.1%}")
        print(f"  Disjoint (J=0): {_frac(all_j_vals, lambda x: x == 0.0):.1%}")

    print(f"\nFetch results:")
    for agent, result in sorted(fetch_results.items()):
        print(f"  {agent}: {result}")

    print(f"\nOutputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
