"""Per-agent cost extraction across the 9-agent corpus.

Pulls cost from each scaffold's native location:
  - SWE-agent baseline (4 agents): info.model_stats inside the cached envelope's content
  - Claude-3.7-thinking, Claude-4: same place — info.model_stats
  - Moatless+V3: sum of completions.usage.completion_cost across the tree
  - DARS+R1: estimated (no embedded cost data) using Claude-3.5-Sonnet pricing
    against an estimated token-per-atom rate from the SWE-agent baseline
  - Agentless+Claude-3.5: estimated using the published $0.34-$0.70/issue range
    from the Agentless paper README; we use the midpoint $0.50

Reads:
    output/trajectories/.cache/<submission>/<iid>.json envelopes
    output/paper2_pilot/aggregate_metrics_extended.json (mean lengths)
    output/paper2_pilot/extended_pass_fail.json
Writes:
    output/paper2_pilot/cost_per_agent.json

Usage:
    uv run python scripts/analysis/extract_cost_per_agent.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "output" / "trajectories" / ".cache"
OUT_DAT = ROOT / "output" / "paper2_pilot"

SUBMISSIONS = {
    "20240402_sweagent_claude3opus":                          "Claude-3",
    "20240402_sweagent_gpt4":                                 "GPT-4",
    "20240620_sweagent_claude3.5sonnet":                      "Claude-3.5",
    "20240728_sweagent_gpt4o":                                "GPT-4o",
    "20250226_sweagent_claude-3-7-sonnet-20250219":           "Claude-3.7-thinking",
    "20250526_sweagent_claude-4-sonnet-20250514":             "Claude-4",
    "20250205_dars_agent_claude_3.5_sonnet_deepseek_r1":      "DARS+R1",
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022":      "Agentless+Claude-3.5",
    "20250111_moatless_deepseek_v3":                          "Moatless+V3",
}

# Public pricing per million tokens (input, output) at time of submission
PRICING_PER_MILLION = {
    "claude-3-5-sonnet": (3.0, 15.0),    # for DARS estimate
}


def _walk_for_cost(obj, results=None):
    """Recursively sum every completion_cost / response_cost we find."""
    if results is None:
        results = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in ("completion_cost", "response_cost") and isinstance(v, (int, float)):
                results.append(float(v))
            else:
                _walk_for_cost(v, results)
    elif isinstance(obj, list):
        for item in obj:
            _walk_for_cost(item, results)
    return results


def cost_swe_agent(envelope: dict) -> float | None:
    """Cost from info.model_stats.instance_cost (Claude/GPT SWE-agent format)."""
    content = envelope.get("content", envelope)
    info = content.get("info", {}) if isinstance(content, dict) else {}
    stats = info.get("model_stats", {})
    cost = stats.get("instance_cost")
    return float(cost) if cost is not None else None


def cost_moatless(envelope: dict) -> float | None:
    """Sum of every completion_cost across the Moatless tree."""
    content = envelope.get("content", envelope)
    if not isinstance(content, dict):
        return None
    costs = _walk_for_cost(content)
    if not costs:
        return None
    return sum(costs)


def per_agent_costs(submission_id: str, agent: str) -> dict:
    sub_dir = CACHE / submission_id
    if not sub_dir.is_dir():
        return {"agent": agent, "submission": submission_id, "error": "cache missing"}
    files = sorted(p for p in sub_dir.glob("*.json") if p.name != "manifest.json")
    if not files:
        return {"agent": agent, "submission": submission_id, "error": "no trajectories"}

    is_moatless = "moatless" in submission_id
    is_dars = "dars" in submission_id
    is_agentless = "agentless" in submission_id

    costs: list[float] = []
    extracted_n = 0
    for fp in files:
        try:
            envelope = json.loads(fp.read_text())
        except Exception:
            continue
        cost: float | None
        if is_moatless:
            cost = cost_moatless(envelope)
        elif is_dars or is_agentless:
            cost = None  # no embedded cost
        else:
            cost = cost_swe_agent(envelope)
        if cost is not None:
            costs.append(cost)
            extracted_n += 1

    out = {
        "agent": agent,
        "submission": submission_id,
        "n_trajectories": len(files),
        "n_with_cost": extracted_n,
    }

    if costs:
        out["cost_mean_usd"]   = round(mean(costs), 4)
        out["cost_median_usd"] = round(median(costs), 4)
        out["cost_total_usd"]  = round(sum(costs), 2)
        out["source"] = (
            "info.model_stats.instance_cost"
            if not is_moatless
            else "sum(completion_cost) across tree"
        )
    elif is_dars:
        # Estimate from Claude-3.5-Sonnet pricing × per-task token estimate
        # DARS uses Claude-3.5-Sonnet as trajectory backbone (per published paper).
        # Estimate: 1.5x Claude-3.5 SWE-agent token volume due to tree expansion.
        # Claude-3.5 SWE-agent baseline mean cost = ~$1.62/task (token_cost.json).
        # Tree expansion expansion factor estimate: 1.5x base cost.
        out["cost_mean_usd"] = round(1.62 * 1.5, 2)
        out["source"] = "estimate: 1.5x Claude-3.5 SWE-agent baseline (tree expansion); not embedded in traces"
        out["estimate_method"] = "Claude-3.5 SWE-agent base * 1.5 (tree-expansion multiplier)"
    elif is_agentless:
        # Documented cost in Agentless paper README: $0.34-$0.70 per issue (v1.5 with Claude 3.5 Sonnet).
        out["cost_mean_usd"] = 0.50
        out["source"] = "estimate: Agentless 1.5 paper README midpoint ($0.34-$0.70 per issue)"
        out["estimate_method"] = "Agentless paper documented range midpoint"

    return out


def main() -> None:
    rows = []
    for sub_id, agent in SUBMISSIONS.items():
        info = per_agent_costs(sub_id, agent)
        rows.append(info)

    # Add resolve rate to compute cost-per-resolved
    pf = json.loads((OUT_DAT / "extended_pass_fail.json").read_text())
    for r in rows:
        sub = r["submission"]
        n_resolved = len(set(pf.get(sub, {}).get("resolved", [])))
        r["n_resolved"] = n_resolved
        if "cost_mean_usd" in r and n_resolved > 0:
            r["cost_per_resolved_usd"] = round(r["cost_mean_usd"] * 300 / n_resolved, 2)

    out_path = OUT_DAT / "cost_per_agent.json"
    out_path.write_text(json.dumps(rows, indent=2, default=str))

    # Print summary table
    print(f"{'Agent':<24} {'Cost/task':>10} {'Cost/resolved':>14} {'Source':<60}")
    print("-" * 110)
    for r in rows:
        agent = r["agent"]
        cost = r.get("cost_mean_usd")
        cpr  = r.get("cost_per_resolved_usd")
        src  = r.get("source", "-")
        cost_s = f"${cost:.2f}" if cost is not None else "-"
        cpr_s  = f"${cpr:.2f}"  if cpr  is not None else "-"
        print(f"{agent:<24} {cost_s:>10} {cpr_s:>14} {src[:58]}")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
