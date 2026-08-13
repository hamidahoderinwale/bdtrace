#!/usr/bin/env python3
"""Generate extensive per-agent cost breakdown LaTeX table.

For each agent: step% and token% per stage, total tokens, resolved count,
tokens-per-resolved. Agent rows color-coded by family using xcolor.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from _extended_pass_fail_df import SUBMISSION_TO_AGENT

# ── Data loading ──────────────────────────────────────────────────────────────

pf = json.loads((ROOT / "output/paper2_pilot/extended_pass_fail.json").read_text())
resolved_by_agent = {
    SUBMISSION_TO_AGENT[s]: set(d.get("resolved", []))
    for s, d in pf.items() if s in SUBMISSION_TO_AGENT
}

step_res   = json.loads((ROOT / "output/paper2_pilot/step_resources.json").read_text())
atoms_data = step_res["atoms"]

STAGES = ["Explore", "Browse", "Edit", "Test", "Shell", "Finish"]

def classify(atom: str) -> str:
    if atom.startswith("SEARCH"):               return "Explore"
    if atom.startswith(("OPEN","NAV","FIND")):  return "Browse"
    if atom.startswith(("EDIT","CREATE")):      return "Edit"
    if atom.startswith("RUN"):                  return "Test"
    if atom.startswith("SHELL_"):               return "Shell"
    if atom.startswith("SUBMIT"):               return "Finish"
    return "Other"

agent_step_counts: dict[str, Counter] = defaultdict(Counter)
agent_total_steps: dict[str, int]     = defaultdict(int)

seq_path = ROOT / "output/paper2_pilot/bpe_sequences_extended.jsonl"
with seq_path.open() as f:
    for line in f:
        r = json.loads(line)
        ag, atoms = r["agent"], r["canonical"]
        n = max(len(atoms), 1)
        agent_total_steps[ag] += n
        for a in atoms:
            agent_step_counts[ag][classify(a)] += 1

agent_stage_tok:    dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
agent_total_tokens: dict[str, float]            = defaultdict(float)

for atom, info in atoms_data.items():
    s = classify(atom)
    for ag, cnt in info.get("by_agent", {}).items():
        tok = cnt * info["mean_tokens_per_use"]
        agent_stage_tok[ag][s] += tok
        agent_total_tokens[ag] += tok

# Global cost per step
g_occ: dict[str, int]   = defaultdict(int)
g_tok: dict[str, float] = defaultdict(float)
for atom, info in atoms_data.items():
    s = classify(atom)
    g_occ[s] += info["occurrences"]
    g_tok[s] += info["occurrences"] * info["mean_tokens_per_use"]
mean_k = {s: g_tok[s] / g_occ[s] / 1000 if g_occ[s] else 0 for s in STAGES}

def sfrac(agent: str, stage: str) -> float:
    n = agent_total_steps.get(agent, 0)
    return agent_step_counts[agent].get(stage, 0) / n if n else 0.0

def tfrac(agent: str, stage: str) -> float:
    tot = agent_total_tokens.get(agent, 0)
    return agent_stage_tok[agent].get(stage, 0) / tot if tot else 0.0

# ── Agent metadata ────────────────────────────────────────────────────────────

SHORT = {
    "Claude-3.7-thinking":  r"Claude-3.7$^*$",
    "Agentless+Claude-3.5": "Agentless",
    "Moatless+V3":          "Moatless",
    "DARS+R1":              "DARS",
}

# xcolor definitions matching the project palette
# These need \usepackage[table]{xcolor} in the LaTeX preamble
FAMILY_COLORS = {
    "Claude":    "claude!10",   # light teal
    "GPT":       "gpt!10",      # light blue
    "Scaffolds": "scaffold!10", # light olive
}

# Raw hex for the color definitions block
COLOR_DEFS = r"""\definecolor{claude}{HTML}{20A380}
\definecolor{gpt}{HTML}{5692E5}
\definecolor{scaffold}{HTML}{585E53}"""

FAMILY_ORDER = [
    ("Claude",    ["Claude-3", "Claude-3.5", "Claude-3.7-thinking", "Claude-4"]),
    ("GPT",       ["GPT-4", "GPT-4o"]),
    ("Scaffolds", ["DARS+R1", "Agentless+Claude-3.5", "Moatless+V3"]),
]

# ── Build LaTeX ───────────────────────────────────────────────────────────────

# Table is wide: 2 + 2*6 + 3 = 17 columns
# Family | Agent | [S_step% S_tok%] × 6 | Resolved | Total tokens | Tok/resolved

n_stage_cols = 2 * len(STAGES)
col_spec = "ll" + "rr" * len(STAGES) + "rrr"

lines: list[str] = []

# Color definitions preamble note
lines.append("% Add to preamble: \\usepackage[table]{xcolor}")
lines.append("% " + COLOR_DEFS.replace("\n", "\n% "))
lines.append("")
lines.append(r"\begin{table}[p]")
lines.append(r"\centering\scriptsize\setlength{\tabcolsep}{3.5pt}")
lines.append(r"\begin{tabular}{" + col_spec + "}")
lines.append(r"\toprule")

# Header row 1: stage group headers
stage_header = " & ".join(
    r"\multicolumn{2}{c}{\textbf{" + s + "}}" for s in STAGES
)
lines.append(
    r"\textbf{Family} & \textbf{Agent} & "
    + stage_header
    + r" & \textbf{Res.} & \textbf{Tokens} & \textbf{Tok/res.} \\"
)

# Header row 2: S% / T% sub-headers under each stage
subheader = " & ".join(r"\textit{S\%} & \textit{T\%}" for _ in STAGES)
lines.append(r" &  & " + subheader + r" &  &  &  \\")
lines.append(r"\midrule")

for family, agents in FAMILY_ORDER:
    row_color = FAMILY_COLORS[family]
    first = True
    for agent in agents:
        if agent not in agent_total_steps:
            continue
        has_tok = agent_total_tokens.get(agent, 0) > 0
        resolved = len(resolved_by_agent.get(agent, set()))
        tot_tok  = agent_total_tokens.get(agent, 0)
        eff      = tot_tok / resolved if (resolved > 0 and has_tok) else None

        name = SHORT.get(agent, agent)
        fam  = family if first else ""
        first = False

        cells = []
        for s in STAGES:
            sf = f"{sfrac(agent, s)*100:.0f}\\%"
            tf = f"{tfrac(agent, s)*100:.0f}\\%" if has_tok else "n/a"
            cells.append(f"{sf} & {tf}")

        tok_str = f"{tot_tok/1e6:.0f}M" if has_tok else "---"
        eff_str = f"{eff/1e6:.2f}M"    if eff is not None else "---"

        row = (
            r"\rowcolor{" + row_color + "}"
            + f"{fam} & {name} & "
            + " & ".join(cells)
            + f" & {resolved} & {tok_str} & {eff_str} \\\\"
        )
        lines.append(row)
    lines.append(r"\midrule")

# Footer: cost per step
cost_row = (
    r"\multicolumn{2}{l}{\textit{Mean tokens / step (k)}} & "
    + " & ".join(f"\\multicolumn{{2}}{{c}}{{{mean_k[s]:.1f}k}}" for s in STAGES)
    + r" & & & \\"
)
lines.append(cost_row)
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")

lines.append(
    r"\caption{Per-agent stage allocation and token efficiency "
    r"(SWE-bench Verified, $n=2{,}639$). "
    r"For each stage, \textit{S\%} is the fraction of agent steps; "
    r"\textit{T\%} is the fraction of the total token budget. "
    r"The footer shows mean tokens per step per stage across all agents. "
    r"Shell steps cost 2.2k tokens on average, versus 11.6k for Edit and "
    r"18.3k for Explore, so \textit{S\%} and \textit{T\%} diverge most "
    r"for shell-heavy agents. "
    r"Claude-4 achieves the lowest tokens per resolved instance (0.37M) "
    r"despite its high Shell step fraction, because shell steps are cheap. "
    r"DARS and Agentless lack step-level token data (---). "
    r"$^*$Extended thinking. Row shading groups agents by model family.}"
)
lines.append(r"\label{tab:stage_cost_full}")
lines.append(r"\end{table}")

print("\n".join(lines))
