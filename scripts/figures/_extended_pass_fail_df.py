"""Shared helper: build a 9-agent DataFrame mirroring the lite_all_models.parquet
schema (instance_id, model_id, agent, passed, n_resolved) from the JSONL +
JSON sources used elsewhere on the extended corpus.

Imported by figure scripts whose original parquet read was 4-agent. The schema
matches the column names downstream code expects, so the rest of each script
runs unchanged after the substitution.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

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


def load_extended_traj_pass_fail() -> pd.DataFrame:
    """Returns one row per (agent, instance_id) pair across the 9-agent corpus.

    Columns:
        instance_id  - SWE-bench instance string
        model_id     - SWE-bench submission ID (matches the parquet's column)
        agent        - short agent name from theme.AGENT_SHORT
        passed       - bool, whether the agent resolved that instance
        n_resolved   - count of agents (out of the 9) that resolved that instance
    """
    pf_path = ROOT / "output" / "paper2_pilot" / "extended_pass_fail.json"
    seq_path = ROOT / "output" / "paper2_pilot" / "bpe_sequences_extended.jsonl"
    pf = json.loads(pf_path.read_text())

    # Trajectory inventory: every (agent, instance) we have a bpe sequence for.
    rows: list[dict] = []
    with seq_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            agent = r["agent"]
            iid = r["instance_id"]
            # Reverse lookup: find the submission ID for this agent.
            sub = next(
                (s for s, a in SUBMISSION_TO_AGENT.items() if a == agent),
                None,
            )
            passed = bool(sub and iid in pf.get(sub, {}).get("resolved", []))
            rows.append({
                "instance_id": iid,
                "model_id":    sub or agent,
                "agent":       agent,
                "passed":      passed,
            })
    df = pd.DataFrame(rows)
    n_res = df.groupby("instance_id")["passed"].sum().rename("n_resolved")
    df = df.merge(n_res, on="instance_id")
    return df
