"""Fetch pass/fail (resolved instance lists) for the extended-corpus
submissions from SWE-bench/experiments.

Each lite submission folder hosts results/results.json which contains a
"resolved" list of instance_ids. We cache this list per submission so MI
decomposition (Phase 7) and failure-mode analysis (Phase 8) can attach
pass/fail labels to the new agents.

Output:
    output/paper2_pilot/extended_pass_fail.json
        {
          "<submission>": {
            "resolved": [iid, ...],
            "no_generation": [...],
            "no_logs": [...],
          }, ...
        }

Usage:
    python -m analysis.preferences.fetch_extended_results
"""

from __future__ import annotations

import base64
import json
import sys
import time
import urllib.request as ur
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "output" / "paper2_pilot"
GH = "https://api.github.com/repos/SWE-bench/experiments/contents/evaluation/lite"

SUBMISSIONS = [
    "20240402_sweagent_claude3opus",
    "20240402_sweagent_gpt4",
    "20240620_sweagent_claude3.5sonnet",
    "20240728_sweagent_gpt4o",
    "20250226_sweagent_claude-3-7-sonnet-20250219",
    "20250205_dars_agent_claude_3.5_sonnet_deepseek_r1",
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022",
    "20250111_moatless_deepseek_v3",
]


def fetch_results_json(submission: str, retries: int = 3) -> dict | None:
    url = f"{GH}/{submission}/results/results.json"
    for attempt in range(retries):
        try:
            with ur.urlopen(url, timeout=20) as r:
                d = json.load(r)
            text = base64.b64decode(d["content"]).decode()
            return json.loads(text)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1 + attempt)
                continue
            print(f"  [error] {submission}: {e}")
            return None
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "extended_pass_fail.json"

    if out_path.exists():
        cached = json.loads(out_path.read_text())
    else:
        cached = {}

    summary = []
    for sub in SUBMISSIONS:
        if sub in cached and "resolved" in cached[sub]:
            n = len(cached[sub]["resolved"])
            print(f"  {sub:55s}  resolved={n}  (cached)")
            summary.append((sub, n, "cached"))
            continue
        print(f"  fetching {sub}...")
        r = fetch_results_json(sub)
        if r is None:
            print(f"    failed")
            continue
        cached[sub] = {
            "resolved": r.get("resolved", []),
            "no_generation": r.get("no_generation", []),
            "no_logs": r.get("no_logs", []),
        }
        n = len(cached[sub]["resolved"])
        print(f"    resolved={n}")
        summary.append((sub, n, "fetched"))
        out_path.write_text(json.dumps(cached, indent=2))

    print("\n=== Summary ===")
    for sub, n, status in summary:
        print(f"  {sub:55s}  resolved={n}  ({status})")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
