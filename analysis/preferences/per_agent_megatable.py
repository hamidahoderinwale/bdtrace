"""Per-agent megatable on extended 8-submission corpus.

Combines:
  - aggregate_metrics_extended.json  (length, repertoire, compression, entropy)
  - failure_modes_extended.json      (Type A/B fractions, post-loc steps)
  - extended_pass_fail.json          (resolved sets for 4 new submissions)
  - lite_all_models.parquet          (resolved per instance for original 4)

Outputs:
  output/paper2_pilot/per_agent_megatable.json
  output/paper2_pilot/per_agent_megatable.tex
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "output" / "paper2_pilot"
LITE = ROOT / "output" / "trajectories" / "lite_all_models.parquet"

AGG = json.load((PILOT / "aggregate_metrics_extended.json").open())["metrics"]
FM  = json.load((PILOT / "failure_modes_extended.json").open())["by_agent"]
PF  = json.load((PILOT / "extended_pass_fail.json").open())

SUB_TO_AGENT = {
    "20240402_sweagent_claude3opus":                          "Claude-3",
    "20240402_sweagent_gpt4":                                 "GPT-4",
    "20240620_sweagent_claude3.5sonnet":                      "Claude-3.5",
    "20240728_sweagent_gpt4o":                                "GPT-4o",
    "20250226_sweagent_claude-3-7-sonnet-20250219":           "Claude-3.7-thinking",
    "20250205_dars_agent_claude_3.5_sonnet_deepseek_r1":      "DARS+R1",
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022":      "Agentless+Claude-3.5",
    "20250111_moatless_deepseek_v3":                          "Moatless+V3",
}
AGENT_META = {
    "Claude-3":              ("SWE-agent",  "Claude-3 (Opus)",        "RLHF dense"),
    "Claude-3.5":            ("SWE-agent",  "Claude-3.5 (Sonnet)",    "RLHF dense"),
    "Claude-3.7-thinking":   ("SWE-agent",  "Claude-3.7 (thinking)",  "Extended-thinking"),
    "GPT-4":                 ("SWE-agent",  "GPT-4",                  "RLHF dense"),
    "GPT-4o":                ("SWE-agent",  "GPT-4o",                 "RLHF dense"),
    "DARS+R1":               ("DARS",       "DeepSeek-R1",            "RL-only reasoning"),
    "Agentless+Claude-3.5":  ("Agentless",  "Claude-3.5 (Sonnet)",    "RLHF dense"),
    "Moatless+V3":           ("Moatless",   "DeepSeek-V3",            "MoE pretrain"),
}
ORDER = ["Claude-3", "Claude-3.5", "Claude-3.7-thinking", "GPT-4", "GPT-4o",
         "DARS+R1", "Agentless+Claude-3.5", "Moatless+V3"]
SCAFFOLD_ORDER = ["SWE-agent", "DARS", "Agentless", "Moatless"]


LITE_TOTAL = 300  # SWE-bench Lite size; standard leaderboard denominator


def resolve_rate_all() -> dict:
    """SWE-bench Lite resolve rate = n_resolved / 300 across all 8 submissions."""
    out = {}
    for sub_id, info in PF.items():
        agent = SUB_TO_AGENT.get(sub_id)
        if agent is None:
            continue
        n_resolved = len(set(info.get("resolved") or []))
        out[agent] = (n_resolved, LITE_TOTAL,
                      round(100 * n_resolved / LITE_TOTAL, 1))
    return out


def build_records():
    rr = resolve_rate_all()

    rows = []
    for a in ORDER:
        scaffold, backbone, paradigm = AGENT_META[a]
        agg = AGG.get(a, {})
        fm  = FM.get(a, {})
        n_pass, n_att, pct = rr.get(a, (None, None, None))

        if fm.get("applicable", True):
            type_a = round(100 * fm.get("frac_never", 0), 1)
            type_b = round(100 * fm.get("frac_reached", 0), 1)
            postloc = fm.get("steps_after_median")
        else:
            type_a = type_b = postloc = None

        rows.append({
            "agent": a,
            "scaffold": scaffold,
            "backbone": backbone,
            "paradigm": paradigm,
            "n_trajectories":   agg.get("n_trajectories"),
            "n_attempted":      n_att,
            "n_resolved":       n_pass,
            "resolve_rate_pct": pct,
            "mean_atoms":       agg.get("mean_canonical_length"),
            "repertoire_90":    agg.get("distinct_motifs_at_90pct"),
            "compression":      agg.get("mean_compression"),
            "entropy_bits":     agg.get("entropy_motifs_bits"),
            "type_a_pct":       type_a,
            "type_b_pct":       type_b,
            "postloc_median":   postloc,
        })
    return rows


def fmt(v, spec):
    if v is None:
        return "—"
    try:
        if isinstance(v, float) and v == 0:
            v = 0.0  # clip negative-zero from float rounding
        s = format(v, spec)
        return s
    except (TypeError, ValueError):
        return str(v)


def to_latex(rows) -> str:
    """Scaffold-grouped megatable. Bold for paper-headline cells."""
    HIGHLIGHT = {
        ("Agentless+Claude-3.5", "entropy_bits"),
        ("Agentless+Claude-3.5", "repertoire_90"),
        ("Agentless+Claude-3.5", "compression"),
        ("Claude-3.7-thinking",  "mean_atoms"),
        ("Claude-3.7-thinking",  "postloc_median"),
        ("Claude-3.7-thinking",  "type_a_pct"),
        ("DARS+R1",              "compression"),
        ("GPT-4o",               "repertoire_90"),
    }

    def cell(agent, key, val, spec):
        s = fmt(val, spec)
        if (agent, key) in HIGHLIGHT and val is not None:
            return r"\textbf{" + s + "}"
        return s

    SCAFFOLD_COLOR = {
        "SWE-agent":  r"\rowcolor{swecol}",
        "DARS":       r"\rowcolor{darscol}",
        "Agentless":  r"\rowcolor{agentlesscol}",
        "Moatless":   r"\rowcolor{moatlesscol}",
    }

    lines = []
    lines.append(r"% Megatable: per-agent fingerprint, extended 8-submission corpus.")
    lines.append(r"% Requires \usepackage{booktabs,xcolor,colortbl} in preamble.")
    lines.append(r"% Define palette colors before the table:")
    lines.append(r"%   \definecolor{swecol}{HTML}{F2F8F4}     % SWE-agent rows")
    lines.append(r"%   \definecolor{darscol}{HTML}{FBF1F5}    % DARS row")
    lines.append(r"%   \definecolor{agentlesscol}{HTML}{F0F5FB} % Agentless row")
    lines.append(r"%   \definecolor{moatlesscol}{HTML}{F7F6F0}  % Moatless row")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{llllrrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Scaffold} & \textbf{Backbone} & \textbf{Paradigm} & \textbf{Agent label} "
        r"& \textbf{$n$} & \textbf{Resolve} & \textbf{Mean} & \textbf{Repertoire} "
        r"& \textbf{Compr.} & \textbf{Entropy} & \textbf{Type A} & \textbf{Type B} "
        r"& \textbf{Post-loc} \\"
    )
    lines.append(
        r" & & & & & \textbf{rate} & \textbf{atoms} & \textbf{@90\%} "
        r"& \textbf{ratio} & \textbf{(bits)} & & & \textbf{median} \\"
    )
    lines.append(r"\midrule")

    by_scaffold: dict[str, list[dict]] = {s: [] for s in SCAFFOLD_ORDER}
    for r in rows:
        by_scaffold[r["scaffold"]].append(r)

    for i, scaffold in enumerate(SCAFFOLD_ORDER):
        if i > 0:
            lines.append(r"\midrule")
        for r in by_scaffold[scaffold]:
            a = r["agent"]
            lines.append(
                f"{SCAFFOLD_COLOR[scaffold]} "
                f"{r['scaffold']} & "
                f"{r['backbone']} & "
                f"{r['paradigm']} & "
                f"{a} & "
                f"{fmt(r['n_trajectories'], 'd')} & "
                f"{cell(a, 'resolve_rate_pct',   r['resolve_rate_pct'],   '.1f')}\\% & "
                f"{cell(a, 'mean_atoms',         r['mean_atoms'],         '.1f')} & "
                f"{cell(a, 'repertoire_90',      r['repertoire_90'],      'd')} & "
                f"{cell(a, 'compression',        r['compression'],        '.3f')} & "
                f"{cell(a, 'entropy_bits',       r['entropy_bits'],       '.2f')} & "
                f"{cell(a, 'type_a_pct',         r['type_a_pct'],         '.1f') + ('\\%' if r['type_a_pct'] is not None else '')} & "
                f"{cell(a, 'type_b_pct',         r['type_b_pct'],         '.1f') + ('\\%' if r['type_b_pct'] is not None else '')} & "
                f"{cell(a, 'postloc_median',     r['postloc_median'],     '.0f')} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Per-agent procedural fingerprint and outcome summary across the "
        r"extended 8-submission corpus (4 scaffolds, 7 backbones, 3 vendors, 4 training "
        r"paradigms). Rows grouped and shaded by scaffold. Bolded cells mark the three "
        r"headline observations: (i) Agentless's deterministic-pipeline behavior "
        r"(entropy 0, repertoire 1, extreme compression); (ii) Claude-3.7-thinking's "
        r"long-but-stuck fingerprint (longest mean atoms among SWE-agent, lowest Type A, "
        r"highest post-localization step median); (iii) within-scaffold backbone effects "
        r"that survive within-paradigm but break under paradigm change. "
        r"Resolve rate from \texttt{lite\_all\_models.parquet} (original 4) and "
        r"\texttt{extended\_pass\_fail.json} (4 new submissions). Type A/B and "
        r"post-localization metrics computed from canonicalized trajectories; "
        r"Agentless is a section-level deterministic pipeline so Type A/B classification "
        r"is not applicable. Repertoire@90\% is the number of distinct BPE motifs "
        r"required to cover 90\% of the agent's tokens. Entropy is computed over the "
        r"motif-only sub-vocabulary.}"
    )
    lines.append(r"\label{tab:per-agent-megatable}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main() -> int:
    rows = build_records()

    out_json = PILOT / "per_agent_megatable.json"
    out_json.write_text(json.dumps({
        "n_agents": len(rows),
        "scaffold_order": SCAFFOLD_ORDER,
        "agent_order": ORDER,
        "rows": rows,
    }, indent=2))

    out_tex = PILOT / "per_agent_megatable.tex"
    out_tex.write_text(to_latex(rows))

    print(f"\nSaved {out_json}")
    print(f"Saved {out_tex}")
    print()
    print("Per-agent summary:")
    for r in rows:
        print(f"  {r['agent']:24s}  scaffold={r['scaffold']:10s}  "
              f"resolve={r['resolve_rate_pct']}%  "
              f"len={r['mean_atoms']}  rep@90={r['repertoire_90']}  "
              f"H={r['entropy_bits']}  postloc={r['postloc_median']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
