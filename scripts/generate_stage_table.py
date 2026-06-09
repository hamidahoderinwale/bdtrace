#!/usr/bin/env python3
"""Generate LaTeX stage allocation + cost tables for appendix."""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from _extended_pass_fail_df import SUBMISSION_TO_AGENT

pf = json.loads((ROOT / "output/paper2_pilot/extended_pass_fail.json").read_text())
resolved_by_agent = {
    SUBMISSION_TO_AGENT[s]: set(d.get("resolved", []))
    for s, d in pf.items() if s in SUBMISSION_TO_AGENT
}

# ── Stage classification ──────────────────────────────────────────────────────

STAGES = ["Explore", "Browse", "Edit", "Test", "Shell", "Finish"]

def classify(atom: str) -> str:
    if atom.startswith("SEARCH"):              return "Explore"
    if atom.startswith(("OPEN","NAV","FIND")): return "Browse"
    if atom.startswith(("EDIT","CREATE")):     return "Edit"
    if atom.startswith("RUN"):                 return "Test"
    if atom.startswith("SHELL_"):              return "Shell"
    if atom.startswith("SUBMIT"):              return "Finish"
    return "Other"

# ── Load trajectory stage fractions ──────────────────────────────────────────

trajs: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

seq_path = ROOT / "output/paper2_pilot/bpe_sequences_extended.jsonl"
with seq_path.open() as f:
    for line in f:
        r = json.loads(line)
        agent = r["agent"]
        passed = r["instance_id"] in resolved_by_agent.get(agent, set())
        atoms = r["canonical"]
        n = max(len(atoms), 1)
        counts = Counter(classify(a) for a in atoms)
        fracs = {s: counts.get(s, 0) / n for s in STAGES}
        trajs[agent]["pass" if passed else "fail"].append(fracs)

# ── Load step-level token costs ───────────────────────────────────────────────

step_res = json.loads(
    (ROOT / "output/paper2_pilot/step_resources.json").read_text()
)
atoms_data = step_res["atoms"]

stage_occ:    dict[str, int]   = defaultdict(int)
stage_tokens: dict[str, float] = defaultdict(float)
agent_total:  dict[str, float] = defaultdict(float)

for atom, info in atoms_data.items():
    stage = classify(atom)
    occ   = info["occurrences"]
    tok   = info["mean_tokens_per_use"]
    stage_occ[stage]    += occ
    stage_tokens[stage] += occ * tok
    for ag, cnt in info.get("by_agent", {}).items():
        agent_total[ag] += cnt * tok

mean_tok_per_step = {
    s: stage_tokens[s] / stage_occ[s] if stage_occ[s] else 0
    for s in STAGES
}

# ── Formatting helpers ────────────────────────────────────────────────────────

SHORT = {
    "Claude-3.7-thinking":  r"Claude-3.7$^*$",
    "Agentless+Claude-3.5": "Agentless",
    "Moatless+V3":          "Moatless",
    "DARS+R1":              "DARS",
}

FAMILY_ORDER = [
    ("Claude",    ["Claude-3", "Claude-3.5", "Claude-3.7-thinking", "Claude-4"]),
    ("GPT",       ["GPT-4", "GPT-4o"]),
    ("Scaffolds", ["DARS+R1", "Agentless+Claude-3.5", "Moatless+V3"]),
]

# ── Table 1: Stage allocation ─────────────────────────────────────────────────

lines: list[str] = []
lines.append(r"\begin{table}[h]")
lines.append(r"\centering\small")
lines.append(r"\begin{tabular}{llr" + "r" * len(STAGES) + "}")
lines.append(r"\toprule")
header = " & ".join(r"\textbf{" + s + "}" for s in STAGES)
lines.append(
    r"\textbf{Family} & \textbf{Agent} & \textbf{Pass\%} & "
    + header + r" \\"
)
lines.append(r"\midrule")

for family, agents in FAMILY_ORDER:
    first = True
    for agent in agents:
        if agent not in trajs:
            continue
        all_t  = trajs[agent]["pass"] + trajs[agent]["fail"]
        n_pass = len(trajs[agent]["pass"])
        n_tot  = n_pass + len(trajs[agent]["fail"])
        pass_r = n_pass / n_tot if n_tot else 0
        means  = {s: sum(t[s] for t in all_t) / len(all_t) for s in STAGES}
        name   = SHORT.get(agent, agent)
        fam    = family if first else ""
        first  = False
        cells  = " & ".join(f"{means[s]*100:.0f}\\%" for s in STAGES)
        lines.append(f"{fam} & {name} & {pass_r:.0%} & {cells} \\\\")
    lines.append(r"\midrule")

# Cost-per-step footer row
cost_cells = " & ".join(
    f"{mean_tok_per_step[s]/1000:.1f}k" for s in STAGES
)
lines.append(r"\multicolumn{3}{l}{\textit{Mean tokens / step}} & "
             + cost_cells + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(
    r"\caption{Stage allocation and token cost per step "
    r"(SWE-bench Verified, $n=2{,}639$ trajectories). "
    r"Explore=\textsc{search}; Browse=\textsc{open}/\textsc{nav}; "
    r"Edit=\textsc{edit}/\textsc{create}; Test=\textsc{run}; "
    r"Shell=\textsc{shell\_*}; Finish=\textsc{submit}. "
    r"\textit{Other} ($<$3\%) omitted. $^*$Extended thinking. "
    r"The footer row shows mean tokens per step for each stage: "
    r"shell steps cost 2.2k tokens on average versus 11.6k for edit "
    r"and 18.3k for explore, so step fraction overstates shell cost. "
    r"Moatless Test\,=\,0\% is a canonicalization artifact.}"
)
lines.append(r"\label{tab:stage_allocation}")
lines.append(r"\end{table}")
lines.append("")

# ── Table 2: Efficiency (tokens per resolved instance) ───────────────────────

lines.append(r"\begin{table}[h]")
lines.append(r"\centering\small")
lines.append(r"\begin{tabular}{llrrrr}")
lines.append(r"\toprule")
lines.append(
    r"\textbf{Family} & \textbf{Agent} & \textbf{Resolved} "
    r"& \textbf{Total tokens} & \textbf{Tokens / resolved} \\"
)
lines.append(r"\midrule")

for family, agents in FAMILY_ORDER:
    first = True
    for agent in agents:
        if agent not in trajs:
            continue
        resolved = len(resolved_by_agent.get(agent, set()))
        tokens   = agent_total.get(agent, 0)
        eff      = tokens / resolved if resolved else float("inf")
        name     = SHORT.get(agent, agent)
        fam      = family if first else ""
        first    = False
        tok_str  = f"{tokens/1e6:.0f}M" if tokens > 0 else "n/a"
        eff_str  = f"{eff/1e6:.2f}M" if tokens > 0 else "n/a"
        lines.append(
            f"{fam} & {name} & {resolved} & {tok_str} & {eff_str} \\\\"
        )
    lines.append(r"\midrule")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(
    r"\caption{Token efficiency by agent. "
    r"Total tokens estimated from per-atom mean context lengths; "
    r"agents without step-level token data (DARS, Agentless) omitted. "
    r"Claude-4 resolves more instances per token than any other agent "
    r"(0.37M tokens/resolved), despite its high shell-step fraction, "
    r"because shell steps cost 5.4$\times$ less per call than edit steps. "
    r"GPT-4o is the least efficient at 2.45M tokens per resolved instance.}"
)
lines.append(r"\label{tab:token_efficiency}")
lines.append(r"\end{table}")

print("\n".join(lines))
