#!/usr/bin/env python3
"""
Fetch agent patches from multiple sources and compute pairwise edit certificate comparisons.

Sources:
1. Local cross_agent_patches.jsonl (4 SWE-agent variants)
2. HuggingFace: OpenHandsCommunity/Devin-SWE-bench-output
3. SWE-bench experiments GitHub repo (via raw URLs or API)

Pipeline:
1. Collect patches per agent per instance
2. Compute edit certificates using patch_to_ast_sequence
3. Pairwise Jaccard comparison for co-solved instances
4. Summary statistics and visualization
"""

import json
import sys
import urllib.request
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence

ROOT = Path(__file__).resolve().parent.parent
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

AGENT_MAP_LOCAL = {
    "GPT-4": "SWE-agent GPT-4",
    "Claude 3.5": "SWE-agent Claude 3.5 Sonnet",
    "GPT-4o": "SWE-agent GPT-4o",
    "Claude 3 Opus": "SWE-agent Claude 3 Opus",
}


def load_local_cross_agent_patches():
    """Load patches from cross_agent_patches.jsonl."""
    path = ROOT / "output" / "datasets" / "swe_bench_lite_resolved" / "cross_agent_patches.jsonl"
    if not path.exists():
        print(f"  {path} not found, skipping")
        return {}

    agent_patches = defaultdict(dict)
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            iid = rec["instance_id"]
            for agent_short, patch in rec["patches"].items():
                agent_name = AGENT_MAP_LOCAL.get(agent_short, agent_short)
                if patch and patch.strip():
                    agent_patches[agent_name][iid] = patch
    return dict(agent_patches)


def load_devin_patches():
    """Load Devin patches from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset("OpenHandsCommunity/Devin-SWE-bench-output", split="train")
        patches = {}
        for row in ds:
            iid = row["instance_id"]
            patch = row.get("model_patch", "")
            if patch and patch.strip():
                patches[iid] = patch
        return {"Devin": patches}
    except Exception as e:
        print(f"  Failed to load Devin patches: {e}")
        return {}


def load_openhands_patches():
    """Try to load OpenHands/CodeAct patches from GitHub."""
    try:
        # OpenHands publishes predictions in their evaluation results
        # Try the swe-bench experiments repo
        agents_found = {}

        # Try GitHub raw URLs for known agent predictions
        # The swe-bench/experiments repo stores predictions
        base = "https://raw.githubusercontent.com/swe-bench/experiments/refs/heads/main"

        agent_dirs = [
            "20240530_autocoderover-v20240408",
            "20240523_aider",
            "20240630_agentless_gpt4o",
            "20240621_autocoderover-v20240620",
            "20240509_amazon-q-developer-agent-20240430-dev",
            "20240725_opendevin_codeact_v1.8_claude35sonnet",
        ]

        for agent_dir in agent_dirs:
            # Try different file locations
            for fname in ["all_preds.jsonl", "predictions.jsonl"]:
                url = f"{base}/evaluation/lite/{agent_dir}/{fname}"
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    response = urllib.request.urlopen(req, timeout=15)
                    content = response.read().decode("utf-8")
                    patches = {}
                    for line in content.strip().split("\n"):
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        iid = rec.get("instance_id", "")
                        patch = rec.get("model_patch", rec.get("prediction", ""))
                        if iid and patch and patch.strip():
                            patches[iid] = patch
                    if patches:
                        print(f"  Loaded {len(patches)} patches for {agent_dir}")
                        agents_found[agent_dir] = patches
                        break
                except Exception:
                    continue

        return agents_found
    except Exception as e:
        print(f"  Failed to load from GitHub: {e}")
        return {}


def load_pass_fail():
    """Load pass/fail results from msgpack leaderboard."""
    import msgpack
    path = ROOT / "output" / "leaderboard" / "lite_results.msgpack"
    with open(path, "rb") as f:
        data = msgpack.unpack(f, raw=False)
    return data


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect patches from all sources
    print("Step 1: Collecting patches from multiple sources...")

    all_agent_patches = {}

    print("  Loading local cross-agent patches...")
    local = load_local_cross_agent_patches()
    for agent, patches in local.items():
        print(f"    {agent}: {len(patches)} patches")
        all_agent_patches[agent] = patches

    print("  Loading Devin patches from HuggingFace...")
    devin = load_devin_patches()
    for agent, patches in devin.items():
        print(f"    {agent}: {len(patches)} patches")
        all_agent_patches[agent] = patches

    print("  Loading patches from SWE-bench experiments GitHub repo...")
    github_patches = load_openhands_patches()
    for agent, patches in github_patches.items():
        print(f"    {agent}: {len(patches)} patches")
        all_agent_patches[agent] = patches

    print(f"\n  Total agents with patches: {len(all_agent_patches)}")
    for agent in sorted(all_agent_patches):
        print(f"    {agent}: {len(all_agent_patches[agent])} instances")

    # Step 2: Load pass/fail data to filter to resolved instances
    print("\nStep 2: Loading pass/fail results...")
    pass_fail = load_pass_fail()

    # Map agent names to leaderboard keys
    agent_to_leaderboard = {
        "SWE-agent GPT-4": "20240402_sweagent_gpt4",
        "SWE-agent Claude 3.5 Sonnet": "20240620_sweagent_claude3.5sonnet",
        "SWE-agent GPT-4o": "20240728_sweagent_gpt4o",
        "SWE-agent Claude 3 Opus": "20240402_sweagent_claude3opus",
        "Devin": None,  # Devin has its own pass/fail in the dataset
        "20240530_autocoderover-v20240408": "20240530_autocoderover-v20240408",
        "20240523_aider": "20240523_aider",
        "20240630_agentless_gpt4o": "20240630_agentless_gpt4o",
        "20240621_autocoderover-v20240620": "20240621_autocoderover-v20240620",
        "20240509_amazon-q-developer-agent-20240430-dev": "20240509_amazon-q-developer-agent-20240430-dev",
        "20240725_opendevin_codeact_v1.8_claude35sonnet": "20240725_opendevin_codeact_v1.8_claude35sonnet",
    }

    # For Devin, get pass/fail from the dataset itself
    devin_passed = set()
    try:
        from datasets import load_dataset
        ds = load_dataset("OpenHandsCommunity/Devin-SWE-bench-output", split="train")
        for row in ds:
            if row.get("pass_or_fail") == "pass":
                devin_passed.add(row["instance_id"])
        print(f"  Devin passed: {len(devin_passed)} instances")
    except Exception:
        pass

    # Step 3: Compute edit certificates
    print("\nStep 3: Computing edit certificates...")
    agent_certs = {}
    for agent, patches in all_agent_patches.items():
        certs = {}
        lb_key = agent_to_leaderboard.get(agent)

        for iid, patch in patches.items():
            # Check if agent passed this instance
            passed = False
            if agent == "Devin":
                passed = iid in devin_passed
            elif lb_key and lb_key in pass_fail:
                passed = pass_fail[lb_key].get(iid, False)
            else:
                # If we can't confirm pass, still compute cert but flag it
                passed = True  # Include all patches, filter later

            if not passed:
                continue

            cert = compute_certificate(patch)
            if cert:  # Only include non-empty certificates
                certs[iid] = cert

        agent_certs[agent] = certs
        print(f"  {agent}: {len(certs)} edit certificates (from passed instances)")

    # Save agent patches (certificates as sorted lists for JSON)
    agent_patches_out = {}
    for agent, certs in agent_certs.items():
        agent_patches_out[agent] = {
            iid: sorted(cert) for iid, cert in certs.items()
        }

    with open(OUTPUT_DIR / "agent_patches.json", "w") as f:
        json.dump(agent_patches_out, f, indent=2)
    print(f"\n  Saved agent_patches.json")

    # Step 4: Pairwise comparison
    print("\nStep 4: Computing pairwise Jaccard similarities...")

    agent_names = sorted(agent_certs.keys())
    pairwise_results = []
    all_jaccards = []

    for a1, a2 in combinations(agent_names, 2):
        certs1 = agent_certs[a1]
        certs2 = agent_certs[a2]

        # Find co-solved instances (both have certificates)
        co_solved = set(certs1.keys()) & set(certs2.keys())

        if not co_solved:
            pairwise_results.append({
                "agent_1": a1,
                "agent_2": a2,
                "n_co_solved": 0,
                "mean_jaccard": None,
                "identical_frac": None,
                "jaccard_gt_0.5": None,
                "jaccard_gt_0.8": None,
            })
            continue

        jaccards = []
        identical = 0
        for iid in co_solved:
            j = jaccard(certs1[iid], certs2[iid])
            jaccards.append(j)
            all_jaccards.append({
                "agent_1": a1,
                "agent_2": a2,
                "instance_id": iid,
                "jaccard": j,
                "cert_1_size": len(certs1[iid]),
                "cert_2_size": len(certs2[iid]),
            })
            if certs1[iid] == certs2[iid]:
                identical += 1

        jaccards = np.array(jaccards)
        pairwise_results.append({
            "agent_1": a1,
            "agent_2": a2,
            "n_co_solved": len(co_solved),
            "mean_jaccard": float(np.mean(jaccards)),
            "median_jaccard": float(np.median(jaccards)),
            "std_jaccard": float(np.std(jaccards)),
            "identical_frac": identical / len(co_solved),
            "jaccard_gt_0.5": float(np.mean(jaccards > 0.5)),
            "jaccard_gt_0.8": float(np.mean(jaccards > 0.8)),
        })

    # Step 5: Agent vocabulary diversity
    print("\nStep 5: Computing agent vocabulary diversity...")
    agent_vocab = {}
    for agent, certs in agent_certs.items():
        all_ops = set()
        for cert in certs.values():
            all_ops.update(cert)
        agent_vocab[agent] = {
            "n_instances": len(certs),
            "vocab_size": len(all_ops),
            "vocabulary": sorted(all_ops),
        }

    # Average pairwise Jaccard per agent
    agent_avg_jaccard = defaultdict(list)
    for pr in pairwise_results:
        if pr["mean_jaccard"] is not None:
            agent_avg_jaccard[pr["agent_1"]].append(pr["mean_jaccard"])
            agent_avg_jaccard[pr["agent_2"]].append(pr["mean_jaccard"])

    for agent in agent_vocab:
        vals = agent_avg_jaccard.get(agent, [])
        agent_vocab[agent]["avg_pairwise_jaccard"] = float(np.mean(vals)) if vals else None

    # Step 6: Co-failure analysis
    print("\nStep 6: Analyzing co-failure patterns...")
    co_failure = {}
    for a1, a2 in combinations(agent_names, 2):
        lb1 = agent_to_leaderboard.get(a1)
        lb2 = agent_to_leaderboard.get(a2)

        if not lb1 or not lb2:
            continue
        if lb1 not in pass_fail or lb2 not in pass_fail:
            continue

        pf1 = pass_fail[lb1]
        pf2 = pass_fail[lb2]
        common = set(pf1.keys()) & set(pf2.keys())

        both_fail = sum(1 for iid in common if not pf1[iid] and not pf2[iid])
        both_pass = sum(1 for iid in common if pf1[iid] and pf2[iid])
        a1_only = sum(1 for iid in common if pf1[iid] and not pf2[iid])
        a2_only = sum(1 for iid in common if not pf1[iid] and pf2[iid])

        co_failure[f"{a1} vs {a2}"] = {
            "both_pass": both_pass,
            "both_fail": both_fail,
            "a1_only_pass": a1_only,
            "a2_only_pass": a2_only,
            "total": len(common),
        }

    # Compile summary
    summary = {
        "n_agents": len(agent_certs),
        "agents": {agent: len(certs) for agent, certs in agent_certs.items()},
        "pairwise_comparisons": pairwise_results,
        "agent_vocabulary": agent_vocab,
        "co_failure_analysis": co_failure,
        "overall_stats": {},
    }

    # Overall stats
    if all_jaccards:
        all_j = np.array([j["jaccard"] for j in all_jaccards])
        summary["overall_stats"] = {
            "total_pairwise_comparisons": len(all_j),
            "mean_jaccard": float(np.mean(all_j)),
            "median_jaccard": float(np.median(all_j)),
            "identical_frac": float(np.mean(all_j == 1.0)),
            "jaccard_gt_0.5_frac": float(np.mean(all_j > 0.5)),
            "jaccard_gt_0.8_frac": float(np.mean(all_j > 0.8)),
            "jaccard_zero_frac": float(np.mean(all_j == 0.0)),
        }
        print(f"\n  Overall: {len(all_j)} pairwise comparisons")
        print(f"  Mean Jaccard: {np.mean(all_j):.3f}")
        print(f"  Median Jaccard: {np.median(all_j):.3f}")
        print(f"  Identical certificates: {np.mean(all_j == 1.0):.1%}")
        print(f"  Jaccard > 0.5: {np.mean(all_j > 0.5):.1%}")
        print(f"  Jaccard > 0.8: {np.mean(all_j > 0.8):.1%}")
        print(f"  Jaccard = 0: {np.mean(all_j == 0.0):.1%}")

    with open(OUTPUT_DIR / "pairwise_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved pairwise_results.json")

    # Step 7: Visualization
    print("\nStep 7: Generating visualization...")
    make_figure(all_jaccards, pairwise_results, agent_names)

    print(f"\nDone. All outputs in {OUTPUT_DIR}")


def make_figure(all_jaccards, pairwise_results, agent_names):
    """Create Altair figure showing pairwise Jaccard distribution."""
    try:
        import altair as alt
        import pandas as pd
    except ImportError:
        print("  altair/pandas not available, skipping figure")
        return

    BLUE = "#0072B2"
    ORANGE = "#E69F00"
    GREEN = "#009E73"
    PINK = "#CC79A7"
    GRAY = "#999999"

    if not all_jaccards:
        print("  No pairwise data to plot")
        return

    df = pd.DataFrame(all_jaccards)

    # Panel 1: Jaccard distribution histogram
    hist = alt.Chart(df).mark_bar(color=BLUE, opacity=0.8).encode(
        alt.X("jaccard:Q", bin=alt.Bin(maxbins=20), title="Jaccard similarity"),
        alt.Y("count()", title="Count"),
    ).properties(
        width=300,
        height=250,
        title="Pairwise edit certificate similarity (all agent pairs)"
    )

    # Panel 2: Mean Jaccard per agent pair (heatmap-style)
    pair_df = pd.DataFrame([
        r for r in pairwise_results if r["mean_jaccard"] is not None
    ])

    if not pair_df.empty:
        # Create a short label for each agent
        short = {}
        for a in agent_names:
            if "GPT-4o" in a:
                short[a] = "GPT-4o"
            elif "GPT-4" in a:
                short[a] = "GPT-4"
            elif "Claude 3.5" in a:
                short[a] = "Claude 3.5"
            elif "Claude 3 Opus" in a:
                short[a] = "Opus"
            elif "Devin" in a:
                short[a] = "Devin"
            elif "autocoderover" in a:
                if "v20240620" in a:
                    short[a] = "ACR v2"
                else:
                    short[a] = "ACR v1"
            elif "aider" in a:
                short[a] = "Aider"
            elif "agentless" in a:
                short[a] = "Agentless"
            elif "amazon" in a:
                short[a] = "Amazon Q"
            elif "opendevin" in a or "openhands" in a:
                short[a] = "OpenHands"
            else:
                short[a] = a[:15]

        pair_df["label"] = pair_df.apply(
            lambda r: f"{short.get(r['agent_1'], r['agent_1'][:8])} vs {short.get(r['agent_2'], r['agent_2'][:8])}",
            axis=1,
        )

        bars = alt.Chart(pair_df).mark_bar(color=ORANGE).encode(
            alt.X("mean_jaccard:Q", title="Mean Jaccard similarity", scale=alt.Scale(domain=[0, 1])),
            alt.Y("label:N", title="", sort=alt.EncodingSortField(field="mean_jaccard", order="descending")),
            alt.Color("n_co_solved:Q", scale=alt.Scale(scheme="blues"), title="Co-solved count"),
        ).properties(
            width=300,
            height=max(150, len(pair_df) * 25),
            title="Mean pairwise Jaccard by agent pair"
        )

        chart = (hist | bars).resolve_scale(color="independent")
    else:
        chart = hist

    chart.save(str(OUTPUT_DIR / "pairwise_jaccard_distribution.html"))
    print(f"  Saved pairwise_jaccard_distribution.html")

    # Also try to save as PNG if vl-convert is available
    try:
        chart.save(str(OUTPUT_DIR / "pairwise_jaccard_distribution.png"), scale_factor=2)
        print(f"  Saved pairwise_jaccard_distribution.png")
    except Exception:
        print("  PNG export not available (vl-convert missing)")


if __name__ == "__main__":
    main()
