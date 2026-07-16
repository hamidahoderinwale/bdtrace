#!/usr/bin/env python3
"""
Fetch SWE-bench Lite agent results from the experiments repo via GitHub API.

Uses `gh api` (no clone needed) to fetch results.json per agent — much cheaper
than cloning the full repo. Results are stored as msgpack (compact binary) rather
than JSON to minimize storage.

Output:
  output/leaderboard/lite_results.msgpack  — {agent_id: {instance_id: bool}}
  output/leaderboard/lite_results.json     — human-readable fallback

Usage:
  uv run python scripts/fetch_leaderboard_results.py
  uv run python scripts/fetch_leaderboard_results.py --agents sweagent moatless agentless
"""

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "leaderboard"

REPO = "SWE-bench/experiments"
SPLIT = "lite"


def gh_api(path: str) -> dict | list | None:
    result = subprocess.run(
        ["gh", "api", path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def list_agents() -> list[str]:
    data = gh_api(f"repos/{REPO}/contents/evaluation/{SPLIT}")
    if not data:
        return []
    return [item["name"] for item in data if item["type"] == "dir"]


def fetch_results_json(agent_id: str) -> dict | None:
    """Fetch results/results.json for one agent via GitHub API (base64 encoded)."""
    path = f"repos/{REPO}/contents/evaluation/{SPLIT}/{agent_id}/results/results.json"
    data = gh_api(path)
    if not data or "content" not in data:
        return None
    try:
        content = base64.b64decode(data["content"]).decode("utf-8")
        return json.loads(content)
    except Exception:
        return None


def parse_resolved(results_json: dict, all_instance_ids: set[str]) -> dict[str, bool]:
    """Convert results.json to {instance_id: bool} pass/fail map."""
    resolved = set(results_json.get("resolved", []))
    return {iid: (iid in resolved) for iid in all_instance_ids}


def load_msgpack(path: Path) -> dict:
    import msgpack
    with open(path, "rb") as f:
        return msgpack.unpack(f, raw=False)


def save_msgpack(data: dict, path: Path) -> None:
    import msgpack
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        msgpack.pack(data, f, use_bin_type=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agents", nargs="*", default=None,
        help="Filter agents by substring (e.g. sweagent moatless). Default: all."
    )
    parser.add_argument("--resume", action="store_true", help="Skip already fetched agents")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    msgpack_path = OUTPUT_DIR / "lite_results.msgpack"
    json_path = OUTPUT_DIR / "lite_results.json"

    # Load existing if resuming
    existing: dict[str, dict[str, bool]] = {}
    if args.resume and msgpack_path.exists():
        try:
            existing = load_msgpack(msgpack_path)
            print(f"Resuming: {len(existing)} agents already fetched")
        except Exception:
            pass

    # Load instance IDs from existing agent results (already have lite full list)
    all_ids: set[str] = set()
    for p in (ROOT / "output" / "swebench_results_lite_agents").glob("*.json"):
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, list):
            all_ids.update(r["instance_id"] for r in data)
    print(f"Reference instance IDs: {len(all_ids)}")

    # List agents
    print("Listing agents from GitHub API...")
    all_agents = list_agents()
    print(f"Found {len(all_agents)} agents")

    # Filter if requested
    if args.agents:
        all_agents = [a for a in all_agents
                      if any(f in a for f in args.agents)]
        print(f"Filtered to {len(all_agents)} agents")

    # Fetch results
    results: dict[str, dict[str, bool]] = dict(existing)
    n_fetched = 0
    n_failed = 0

    for agent_id in all_agents:
        if args.resume and agent_id in existing:
            continue
        raw = fetch_results_json(agent_id)
        if raw is None:
            n_failed += 1
            print(f"  SKIP  {agent_id}  (no results.json)")
            continue
        pass_fail = parse_resolved(raw, all_ids)
        results[agent_id] = pass_fail
        n_fetched += 1
        n_pass = sum(pass_fail.values())
        n_total = len(pass_fail)
        print(f"  OK    {agent_id}  ({n_pass}/{n_total} = {100*n_pass/max(n_total,1):.1f}%)")

        # Save incrementally
        if n_fetched % 5 == 0:
            save_msgpack(results, msgpack_path)

    save_msgpack(results, msgpack_path)
    # Also save human-readable JSON (smaller since it's just True/False per instance)
    with open(json_path, "w") as f:
        json.dump({
            agent: {iid: v for iid, v in pf.items() if v}  # only resolved instances
            for agent, pf in results.items()
        }, f, separators=(",", ":"))

    print(f"\nFetched {n_fetched} agents, {n_failed} skipped")
    print(f"Saved to {msgpack_path} ({msgpack_path.stat().st_size / 1024:.1f} KB)")
    print(f"Saved to {json_path} ({json_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
