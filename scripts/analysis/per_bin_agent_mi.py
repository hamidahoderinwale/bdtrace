"""Per-difficulty-bin distributional view of agent pass rates.

Within each n_resolved bucket (0/9 through 9/9), record:
  * pass rate per agent (the distributional primitive)
  * range across agents (max - min)
  * std across agents
  * MI(agent; pass/fail) and pct = MI / H(Y) for backwards compatibility
    with anyone consuming the old fields

The figure consumes the per-agent rates directly; the MI / pct fields
are retained as scalar summaries for prose references.

Reads:
    output/paper2_pilot/bpe_sequences_extended.jsonl
    output/paper2_pilot/extended_pass_fail.json

Writes:
    output/paper2_pilot/per_bin_agent_mi.json
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "output" / "paper2_pilot"
OUT_JSON = OUT / "per_bin_agent_mi.json"

SUBMISSION_TO_AGENT = {
    "20240402_sweagent_claude3opus":                "Claude-3",
    "20240402_sweagent_gpt4":                       "GPT-4",
    "20240620_sweagent_claude3.5sonnet":            "Claude-3.5",
    "20240728_sweagent_gpt4o":                      "GPT-4o",
    "20250226_sweagent_claude-3-7-sonnet-20250219": "Claude-3.7-thinking",
    "20250526_sweagent_claude-4-sonnet-20250514":   "Claude-4",
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022": "Agentless+Claude-3.5",
    "20250205_dars_agent_claude_3.5_sonnet_deepseek_r1": "DARS+R1",
    "20250111_moatless_deepseek_v3":                "Moatless+V3",
}


def entropy_bits(probs) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)


def main() -> int:
    pf = json.loads((OUT / "extended_pass_fail.json").read_text())
    resolved_by_agent: dict[str, set[str]] = {
        a: set(pf.get(sub, {}).get("resolved", []))
        for sub, a in SUBMISSION_TO_AGENT.items()
    }

    # Per (agent, instance) pass/fail rows.
    rows: list[dict] = []
    with (OUT / "bpe_sequences_extended.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({
                "agent": r["agent"],
                "instance_id": r["instance_id"],
                "passed": r["instance_id"] in resolved_by_agent.get(r["agent"], set()),
            })

    # n_resolved per instance.
    pass_count_by_iid: Counter = Counter()
    for r in rows:
        if r["passed"]:
            pass_count_by_iid[r["instance_id"]] += 1

    # Bucket into bins by n_resolved.
    buckets: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        nr = pass_count_by_iid.get(r["instance_id"], 0)
        buckets[nr].append(r)

    per_bin: dict[str, dict] = {}
    for nr in range(10):
        bucket_rows = buckets.get(nr, [])
        n = len(bucket_rows)
        if n == 0:
            per_bin[str(nr)] = {
                "n": 0, "mi": 0.0, "pct": 0.0, "hy": 0.0,
                "per_agent": {}, "range": 0.0, "std": 0.0,
            }
            continue
        passes = [r["passed"] for r in bucket_rows]
        p_pass = sum(passes) / n
        hy = entropy_bits([p_pass, 1 - p_pass])

        # Per-agent pass rate within this bin: this IS the distribution.
        by_agent: dict[str, list[bool]] = defaultdict(list)
        for r in bucket_rows:
            by_agent[r["agent"]].append(r["passed"])
        per_agent_rates: dict[str, dict] = {}
        rates: list[float] = []
        for a, ys in by_agent.items():
            p_pass_a = sum(ys) / len(ys) if ys else 0.0
            per_agent_rates[a] = {
                "pass_rate": round(p_pass_a, 4),
                "n": len(ys),
                "passed": int(sum(ys)),
            }
            rates.append(p_pass_a)

        rng = max(rates) - min(rates) if rates else 0.0
        if len(rates) > 1:
            mean_r = sum(rates) / len(rates)
            std_r = (sum((x - mean_r) ** 2 for x in rates) / len(rates)) ** 0.5
        else:
            std_r = 0.0

        # MI machinery retained for backwards-compat consumers.
        h_y_given_a = 0.0
        for a, ys in by_agent.items():
            p_a = len(ys) / n
            p_pass_a = sum(ys) / len(ys) if ys else 0.0
            h_a = entropy_bits([p_pass_a, 1 - p_pass_a])
            h_y_given_a += p_a * h_a
        mi = max(hy - h_y_given_a, 0.0)
        pct = round(100 * mi / hy, 1) if hy > 0 else 0.0

        per_bin[str(nr)] = {
            "n": n,
            "mi": round(mi, 4),
            "pct": pct,
            "hy": round(hy, 4),
            "per_agent": per_agent_rates,
            "range": round(rng, 4),
            "std":   round(std_r, 4),
        }

    OUT_JSON.write_text(json.dumps(per_bin, indent=2))
    print(f"Saved {OUT_JSON}")
    print()
    print(f"{'bin':<6} {'n':>5} {'range':>7} {'std':>6} {'MI':>7} {'pct':>6}")
    for nr in range(10):
        b = per_bin[str(nr)]
        print(f"{nr}/9    {b['n']:>5d} {b['range']:>6.2f} {b['std']:>6.3f} "
              f"{b['mi']:>7.4f} {b['pct']:>5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
