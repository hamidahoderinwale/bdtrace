"""Ethnographic case studies: 4 tasks, thick description, side-by-side.

For each selected task, render a self-contained panel:
  - Header: instance_id, outcome badge, per-agent tokens/cost
  - Phase-map: three horizontal bands, one per agent, colored by procedural phase
  - Motif chip sequences: three columns, shared motifs tinted
  - Narrative cards: 2-3 sentence plain-English summary per agent
  - Key differences: table of motif-use divergences

Phase detection (heuristic state machine on canonical atoms):
  - EXPLORATION: FIND_FILE, SEARCH, SHELL_LS, SHELL_CD (before first OPEN/EDIT)
  - LOCALIZATION: OPEN_*, NAV_*, SEARCH after first OPEN (before first EDIT)
  - EDITING: EDIT_*, CREATE_*
  - VERIFICATION: RUN_PYTHON_*
  - CLEANUP_SUBMIT: SHELL_RM, SUBMIT

Outputs:
    output/paper2_pilot/case_studies.html  (self-contained + section-embeddable)

Usage:
    python -m analysis.preferences.case_studies
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE = PROJECT_ROOT / "output" / "trajectories" / ".cache"
OUT = PROJECT_ROOT / "output" / "paper2_pilot"
SEQ_PATH = OUT / "bpe_sequences.jsonl"
DIVERSITY_PATH = OUT / "task_diversity.csv"
PAIRS_PATH = OUT / "tied_outcome_pairs.csv"

AGENT_SHORT = {
    "20240402_sweagent_gpt4": "GPT-4",
    "20240620_sweagent_claude3.5sonnet": "Claude-3.5",
    "20240728_sweagent_gpt4o": "GPT-4o",
}
AGENT_ORDER = ["GPT-4", "Claude-3.5", "GPT-4o"]

PHASE_COLORS = {
    "exploration": "#7a9eca",
    "localization": "#3faea1",
    "editing": "#e89545",
    "verification": "#9e7acd",
    "cleanup_submit": "#888888",
}
PHASE_LABELS = {
    "exploration": "exploration",
    "localization": "localization",
    "editing": "editing",
    "verification": "verification",
    "cleanup_submit": "cleanup + submit",
}

SELECTED_TASKS = [
    ("django__django-13447", "short 3/3 — tight convergence", 3),
    ("django__django-11039", "medium 3/3 — productive divergence", 3),
    ("django__django-14855", "long 3/3 — GPT-4o verbosity signature", 3),
    ("django__django-13551", "medium 0/3 — what failure looks like", 0),
]


def classify_atom_phase(atom: str, seen_open: bool, seen_edit: bool) -> str:
    if atom == "SUBMIT" or atom.startswith("SHELL_RM"):
        return "cleanup_submit"
    if atom.startswith("RUN_PYTHON_") or atom.startswith("RUN_PYTEST"):
        return "verification"
    if atom.startswith("EDIT_") or atom.startswith("CREATE_"):
        return "editing"
    if atom.startswith("OPEN_") or atom.startswith("NAV_"):
        if seen_edit:
            return "editing"
        return "localization"
    if atom == "SEARCH" or atom.startswith("FIND_FILE"):
        if seen_edit:
            return "editing"
        if seen_open:
            return "localization"
        return "exploration"
    if atom.startswith("SHELL_"):
        if not seen_open and not seen_edit:
            return "exploration"
        return "editing"
    return "exploration"


def detect_phases(atoms: list[str]) -> list[tuple[str, int, int]]:
    """Return list of (phase, start_idx, end_idx_exclusive) segments."""
    if not atoms:
        return []
    seen_open = False
    seen_edit = False
    per_atom_phase: list[str] = []
    for a in atoms:
        phase = classify_atom_phase(a, seen_open, seen_edit)
        per_atom_phase.append(phase)
        if a.startswith("OPEN_"):
            seen_open = True
        if a.startswith("EDIT_") or a.startswith("CREATE_"):
            seen_edit = True

    segments = []
    start = 0
    for i in range(1, len(per_atom_phase)):
        if per_atom_phase[i] != per_atom_phase[i - 1]:
            segments.append((per_atom_phase[start], start, i))
            start = i
    segments.append((per_atom_phase[start], start, len(per_atom_phase)))
    return segments


def load_sequences() -> dict[tuple[str, str], dict]:
    out = {}
    with open(SEQ_PATH) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(r["agent"], r["instance_id"])] = r
    return out


def load_model_stats() -> dict[tuple[str, str], dict]:
    out = {}
    for agent_dir in sorted(CACHE.iterdir()):
        if not agent_dir.is_dir():
            continue
        short = AGENT_SHORT.get(agent_dir.name, agent_dir.name)
        for traj_file in sorted(agent_dir.glob("*.json")):
            with open(traj_file) as f:
                d = json.load(f)
            stats = (d.get("info") or {}).get("model_stats") or {}
            out[(short, traj_file.stem)] = {
                "tokens_sent": int(stats.get("tokens_sent", 0)),
                "api_calls": int(stats.get("api_calls", 0)),
                "instance_cost_usd": float(stats.get("instance_cost", 0)),
            }
    return out


def load_resolved() -> set[tuple[str, str]]:
    out = set()
    agent_map = {
        "Claude 3.5 Sonnet (SWE-agent)": "Claude-3.5",
        "GPT-4 (SWE-agent)": "GPT-4",
        "GPT-4o (SWE-agent)": "GPT-4o",
    }
    with open(PAIRS_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            out.add((agent_map.get(row["agent_a"]), row["instance_id"]))
            out.add((agent_map.get(row["agent_b"]), row["instance_id"]))
    return out


def render_phase_map(agent: str, segments: list[tuple[str, int, int]], total: int) -> str:
    segs_html = []
    for phase, start, end in segments:
        width_pct = (end - start) / total * 100
        color = PHASE_COLORS.get(phase, "#999")
        label = PHASE_LABELS.get(phase, phase)
        segs_html.append(
            f'<div style="background:{color}; color:white; font-size:10px; padding:3px 6px;'
            f' width:{width_pct:.2f}%; display:inline-block; text-align:center;'
            f' white-space:nowrap; overflow:hidden; text-overflow:ellipsis;"'
            f' title="{label} (steps {start}-{end - 1}, {end - start} atoms)">'
            f'{label if end - start >= 4 else ""}'
            f'</div>'
        )
    return (
        f'<div style="display:flex; align-items:center; margin:6px 0;">'
        f'  <div style="min-width:90px; font-weight:600; font-size:13px; color:#333;">{agent}</div>'
        f'  <div style="flex:1; display:flex; flex-wrap:nowrap; font-family:SF Mono,monospace;">'
        f'{"".join(segs_html)}'
        f'  </div>'
        f'  <div style="min-width:80px; text-align:right; color:#666; font-size:12px;">{total} steps</div>'
        f'</div>'
    )


def render_motif_chips(agent: str, motifs: list[str], shared_by_others: dict[str, set[str]]) -> str:
    other_agents = [a for a in shared_by_others if a != agent]

    def abbrev(m: str, maxl: int = 24) -> str:
        parts = m.split("+")
        if len(parts) <= 2:
            s = m.replace("+", "→")
        else:
            s = f"{parts[0]}→…→{parts[-1]} ({len(parts)})"
        return s if len(s) <= maxl else s[: maxl - 1] + "…"

    chips = []
    for m in motifs:
        n_other = sum(1 for a in other_agents if m in shared_by_others[a])
        if "+" not in m:
            bg, border = "#f3f3f3", "#bbb"
        elif n_other == 2:
            bg, border = "#c8e6c9", "#2e7d32"
        elif n_other == 1:
            bg, border = "#fff3c4", "#b58900"
        else:
            bg, border = "#ffffff", "#c03030"
        chips.append(
            f'<span style="display:inline-block; padding:2px 5px; margin:1px; '
            f'background:{bg}; border:1px solid {border}; border-radius:3px; '
            f'font-family:SF Mono,Menlo,monospace; font-size:10.5px;" title="{m}">'
            f'{abbrev(m)}</span>'
        )
    return (
        f'<div style="flex:1; padding:8px; background:#fdfdfd; border:1px solid #e8e8e8; border-radius:4px; margin:4px;">'
        f'<div style="font-weight:600; font-size:13px; margin-bottom:4px; color:#333;">{agent}</div>'
        f'<div style="line-height:1.7;">{"".join(chips)}</div>'
        f'</div>'
    )


def phase_vocab_name(phase: str) -> str:
    return {
        "exploration": "exploring (search/find)",
        "localization": "locating (open/nav)",
        "editing": "editing (edit/create)",
        "verification": "verifying (running)",
        "cleanup_submit": "submitting",
    }.get(phase, phase)


def narrative_from_phases(segments: list[tuple[str, int, int]], total: int) -> str:
    if not segments:
        return "Empty trajectory."
    # Collapse consecutive segments of the same phase
    counts: dict[str, int] = {}
    for phase, s, e in segments:
        counts[phase] = counts.get(phase, 0) + (e - s)
    # Find the dominant phase
    dominant_phase, dominant_count = max(counts.items(), key=lambda kv: kv[1])
    dominant_frac = dominant_count / total if total else 0

    opening = segments[0][0]
    closing = segments[-1][0]

    parts = []
    parts.append(f"Started with {phase_vocab_name(opening)}.")
    if dominant_frac > 0.4 and dominant_phase not in (opening, closing):
        parts.append(f"Spent most time {phase_vocab_name(dominant_phase)} ({int(dominant_frac * 100)}% of steps).")
    if closing == "cleanup_submit":
        parts.append("Cleaned up and submitted.")
    elif closing == "editing":
        parts.append("Ended in the edit loop without explicit cleanup.")
    elif closing == "verification":
        parts.append("Ended on verification.")
    else:
        parts.append(f"Ended in {phase_vocab_name(closing)}.")
    return " ".join(parts)


def render_case(
    instance_id: str,
    note: str,
    difficulty: int,
    seqs: dict,
    stats: dict,
    resolved: set,
) -> str:
    agents_present = [a for a in AGENT_ORDER if (a, instance_id) in seqs]
    max_atoms = max(seqs[(a, instance_id)]["canonical_length"] for a in agents_present)
    motif_sets = {a: set(seqs[(a, instance_id)]["bpe"]) for a in agents_present}

    badge_color = {0: "#c03030", 3: "#2e7d32"}.get(difficulty, "#b58900")
    badge_text = f"{difficulty}/3 resolved"

    header_meta = []
    for a in agents_present:
        st = stats.get((a, instance_id), {})
        tokens = st.get("tokens_sent", 0)
        cost = st.get("instance_cost_usd", 0)
        calls = st.get("api_calls", 0)
        was_resolved = (a, instance_id) in resolved
        check = "&#10003;" if was_resolved else "&#10007;"
        header_meta.append(
            f'<span style="margin-right:18px; font-size:12px; color:#444;">'
            f'<strong>{a}</strong> {check}&nbsp; '
            f'{tokens/1000:.0f}k tok &middot; {calls} calls &middot; ${cost:.2f}'
            f'</span>'
        )

    phase_rows = []
    for a in agents_present:
        atoms = seqs[(a, instance_id)]["canonical"]
        segments = detect_phases(atoms)
        phase_rows.append(render_phase_map(a, segments, len(atoms)))

    chip_cols = []
    for a in agents_present:
        motifs = seqs[(a, instance_id)]["bpe"]
        chip_cols.append(render_motif_chips(a, motifs, motif_sets))

    narrative_rows = []
    for a in agents_present:
        atoms = seqs[(a, instance_id)]["canonical"]
        segments = detect_phases(atoms)
        narr = narrative_from_phases(segments, len(atoms))
        narrative_rows.append(
            f'<div style="padding:6px 10px; background:#f8f8f8; border-left:3px solid #888; margin:4px 0; font-size:12.5px;">'
            f'<strong>{a}:</strong> {narr}'
            f'</div>'
        )

    unique_counts = {}
    for a in agents_present:
        others_union = set()
        for b in agents_present:
            if b != a:
                others_union |= motif_sets[b]
        unique = [m for m in seqs[(a, instance_id)]["bpe"] if "+" in m and m not in others_union]
        unique_counts[a] = unique

    diff_rows_html = []
    for a in agents_present:
        if unique_counts[a]:
            sample = ", ".join(m[:40] for m in unique_counts[a][:4])
            diff_rows_html.append(
                f'<tr><td style="padding:4px 8px; font-size:12px;"><strong>{a}</strong></td>'
                f'<td style="padding:4px 8px; font-size:12px;">{len(unique_counts[a])} unique motifs</td>'
                f'<td style="padding:4px 8px; font-size:11px; font-family:SF Mono,monospace; color:#555;">{sample}</td></tr>'
            )

    return f"""
<div style="margin: 28px 0; padding: 18px 22px; background: #fff; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
  <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
    <div>
      <h3 style="margin:0; font-size:16px; color:#111;">{instance_id}</h3>
      <div style="font-size:12px; color:#666;">{note}</div>
    </div>
    <span style="background:{badge_color}; color:white; padding:4px 10px; border-radius:12px; font-size:11px; font-weight:600;">{badge_text}</span>
  </div>
  <div style="margin:6px 0 10px 0;">{''.join(header_meta)}</div>

  <div style="margin:14px 0 4px; font-size:12px; color:#666; font-weight:600;">phase map (real step counts; widths proportional; x-axis matches within this card)</div>
  {''.join(phase_rows)}

  <div style="display:flex; margin-top:14px;">
    {''.join(chip_cols)}
  </div>

  <div style="margin:12px 0 4px; font-size:12px; color:#666; font-weight:600;">narrative</div>
  {''.join(narrative_rows)}

  {"<div style='margin:12px 0 4px; font-size:12px; color:#666; font-weight:600;'>agent-unique motifs</div><table style='border-collapse:collapse; width:100%;'>" + ''.join(diff_rows_html) + "</table>" if diff_rows_html else ''}
</div>
"""


def build_legend() -> str:
    legend = []
    for phase in ["exploration", "localization", "editing", "verification", "cleanup_submit"]:
        legend.append(
            f'<span style="display:inline-block; padding:3px 8px; margin:2px; '
            f'background:{PHASE_COLORS[phase]}; color:white; border-radius:3px; font-size:11px;">'
            f'{PHASE_LABELS[phase]}</span>'
        )
    motif_legend = (
        '<span style="padding:1px 5px; background:#c8e6c9; border:1px solid #2e7d32; border-radius:3px; font-size:11px;">shared by all three</span>&nbsp; '
        '<span style="padding:1px 5px; background:#fff3c4; border:1px solid #b58900; border-radius:3px; font-size:11px;">shared by two</span>&nbsp; '
        '<span style="padding:1px 5px; background:#ffffff; border:1px solid #c03030; border-radius:3px; font-size:11px;">unique to this agent</span>&nbsp; '
        '<span style="padding:1px 5px; background:#f3f3f3; border:1px solid #bbb; border-radius:3px; font-size:11px;">single atom</span>'
    )
    return (
        '<div style="background:#f9f9f9; padding:10px 14px; border-radius:6px; margin:10px 0 24px; font-size:12px;">'
        f'<strong>phases:</strong> {"".join(legend)}<br>'
        f'<strong style="margin-right:6px;">motifs:</strong> {motif_legend}'
        '</div>'
    )


def main() -> int:
    seqs = load_sequences()
    stats = load_model_stats()
    resolved = load_resolved()

    cases = []
    for inst, note, diff in SELECTED_TASKS:
        if any((a, inst) not in seqs for a in AGENT_ORDER):
            print(f"SKIP {inst}: missing sequences for one or more agents")
            continue
        print(f"rendering {inst}: {note}")
        cases.append(render_case(inst, note, diff, seqs, stats, resolved))

    header = """
<section id="case-studies">
<h2>Case studies: four tasks, three agents, side-by-side</h2>
<p>Ethnographic zoom-in on a handful of specific tasks. For each task: the outcome badge, per-agent token and cost summary, a <em>phase-map</em> showing how each agent's trajectory segments into procedural phases, the full motif sequence per agent with shared vs unique motifs tinted, a plain-English narrative per agent, and a summary of agent-unique motifs. The phase bars are drawn to real step counts within a card (GPT-4 may appear much shorter than GPT-4o when its trajectory is half the length).</p>
"""
    footer = "</section>\n"

    body = header + build_legend() + "\n".join(cases) + footer
    (OUT / "case_studies.html").write_text(body)
    print(f"\nSaved: {OUT / 'case_studies.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
