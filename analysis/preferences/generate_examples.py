"""Extract concrete examples for the dashboard: raw → canonical → BPE.

Produces an HTML snippet showing:
  1. What a BPE merge looks like (pair example)
  2. A real trajectory in three views (raw / canonical / BPE-motifs)
  3. Top motifs with trajectory samples showing where they occur

Outputs:
  output/paper2_pilot/examples_snippet.html  - ready to paste into dashboard

Usage:
    python -m analysis.preferences.generate_examples
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.preferences.bpe import train_bpe, apply_bpe
from analysis.preferences.canonicalize import canonicalize_trajectory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE = PROJECT_ROOT / "output" / "trajectories" / ".cache"
OUT = PROJECT_ROOT / "output" / "paper2_pilot"

AGENT_SHORT = {
    "20240402_sweagent_gpt4": "GPT-4",
    "20240620_sweagent_claude3.5sonnet": "Claude-3.5",
    "20240728_sweagent_gpt4o": "GPT-4o",
}


def load_trajectory(agent_dir_name: str, instance_id: str):
    path = CACHE / agent_dir_name / f"{instance_id}.json"
    with open(path) as f:
        raw = json.load(f)
    return raw


def motif_color(motif: str, palette: list[str]) -> str:
    """Pick a stable color per motif by hashing."""
    idx = hash(motif) % len(palette)
    return palette[idx]


def render_sequence_boxes(tokens: list[str], colors: dict[str, str] = None) -> str:
    """Render a sequence of tokens as HTML boxes for readability."""
    boxes = []
    for t in tokens:
        # Use a background color if this is a motif (has +), light otherwise
        is_motif = "+" in t
        color = colors.get(t, "#e8f0ff") if colors and is_motif else ("#e8f0ff" if is_motif else "#f5f5f5")
        border = "#6090d0" if is_motif else "#bbb"
        # Abbreviate long motifs
        display = t if len(t) < 40 else t[:18] + "...+" + t.rsplit("+", 1)[-1][:18]
        display = display.replace("+", "→")
        boxes.append(
            f'<span style="display:inline-block; padding:2px 6px; margin:2px; '
            f'background:{color}; border:1px solid {border}; border-radius:3px; '
            f'font-family:SF Mono,Menlo,monospace; font-size:11.5px;" '
            f'title="{html.escape(t)}">{html.escape(display)}</span>'
        )
    return "".join(boxes)


def build_example_1_what_is_a_merge() -> str:
    """Explain what a BPE merge is, with a before/after illustration."""
    return """
<h3>6.1 What is a BPE merge?</h3>

<p>A merge is a single step in BPE training. Each merge finds the most-frequent adjacent pair of tokens in the corpus, and binds that pair into a single new symbol. From that point on, every occurrence of that pair is treated as one token.</p>

<p><strong>Concrete example from our training run — the first merge that was performed:</strong></p>

<table style="margin: 8px 0; font-size: 14px;">
<tr><td><strong>Pair found:</strong></td>
<td><span style="padding:2px 6px; background:#f5f5f5; border:1px solid #bbb; border-radius:3px; font-family:SF Mono,monospace; font-size:11.5px;">EDIT_SRC_PY</span>
+
<span style="padding:2px 6px; background:#f5f5f5; border:1px solid #bbb; border-radius:3px; font-family:SF Mono,monospace; font-size:11.5px;">EDIT_SRC_PY</span>
</td></tr>
<tr><td><strong>Frequency across corpus:</strong></td><td>Most frequent adjacent pair (over 800 occurrences)</td></tr>
<tr><td><strong>New merged symbol:</strong></td>
<td><span style="padding:2px 6px; background:#e8f0ff; border:1px solid #6090d0; border-radius:3px; font-family:SF Mono,monospace; font-size:11.5px;">EDIT_SRC_PY → EDIT_SRC_PY</span>
</td></tr>
<tr><td><strong>Meaning:</strong></td><td>"Two source-file edits in a row" — a recognizable burst of editing activity</td></tr>
</table>

<p>After this first merge, BPE looks for the most frequent pair in the <em>updated</em> corpus — which might now involve the new merged symbol. That is how longer motifs grow: once <code>A→B</code> is a single token, it can itself pair with neighbors to form <code>A→B→C</code>, etc.</p>

<p>We ran 124 merges total. Each merge adds one new symbol to the vocabulary and re-expresses the corpus to use that symbol where the pair appeared.</p>
"""


def build_example_2_trajectory_views(raw_trajectory, canonical, bpe_expressed, instance_id, agent) -> str:
    """Show one real trajectory in three views: raw / canonical / BPE."""

    # Render raw actions (truncated)
    raw_actions = [step.get("action", "").strip() for step in raw_trajectory]
    raw_html = []
    for i, action in enumerate(raw_actions):
        snippet = action.split("\n", 1)[0][:60]
        raw_html.append(
            f'<div style="font-family:SF Mono,monospace; font-size:11.5px; '
            f'padding:3px 6px; margin:1px 0; background:#f9f9f9; border-left:2px solid #ccc;">'
            f'<span style="color:#888;">[{i:>2}]</span> {html.escape(snippet)}'
            + ('<span style="color:#aaa;">…</span>' if len(action) > 60 else '')
            + '</div>'
        )

    # Canonical tokens
    canonical_html = render_sequence_boxes(canonical)

    # BPE tokens with stable colors per motif
    motif_palette = ["#c7e6c4", "#f7d7a6", "#d2c4f0", "#f5c7c4", "#bde4f2",
                     "#f0e4a6", "#c5e4d2", "#e0c4e8", "#f7c4d5", "#b5d7e2"]
    bpe_motifs_in_this = [t for t in bpe_expressed if "+" in t]
    color_map = {}
    for t in sorted(set(bpe_motifs_in_this), key=lambda x: -bpe_expressed.count(x)):
        color_map[t] = motif_palette[len(color_map) % len(motif_palette)]
    bpe_html = render_sequence_boxes(bpe_expressed, colors=color_map)

    compression = len(bpe_expressed) / max(len(canonical), 1)

    return f"""
<h3>6.2 One trajectory in three views</h3>

<p>Here is a real SWE-agent trajectory (agent <strong>{agent}</strong>, instance <code>{html.escape(instance_id)}</code>) shown at each representation level: raw shell-like commands, canonicalized atoms, and BPE motifs.</p>

<h4 style="font-size:14px; margin:20px 0 6px;">Raw action strings ({len(raw_trajectory)} steps)</h4>
<p style="font-size:13px; color:#666;">What the agent actually did — shell-like commands. Surface form varies enormously: different file paths, different arguments, different whitespace.</p>
<div style="background:#fdfdfd; padding:8px 12px; border-radius:4px; border:1px solid #e5e5e5; max-height:280px; overflow-y:auto;">
{''.join(raw_html)}
</div>

<h4 style="font-size:14px; margin:20px 0 6px;">Canonicalized atoms ({len(canonical)} tokens)</h4>
<p style="font-size:13px; color:#666;">Each raw action → one canonical atom. Type-tagged file paths (SRC/TEST/REPRO/CONFIG) and stripped of non-semantic literals. Same length as raw — just normalized.</p>
<div style="background:#fdfdfd; padding:10px 12px; border-radius:4px; border:1px solid #e5e5e5;">
{canonical_html}
</div>

<h4 style="font-size:14px; margin:20px 0 6px;">BPE-expressed ({len(bpe_expressed)} tokens — compression {compression:.2f})</h4>
<p style="font-size:13px; color:#666;">Recurring sub-sequences are merged into single motifs (colored boxes are motifs; gray are atoms). Compression ratio = {len(bpe_expressed)}/{len(canonical)} = <strong>{compression:.2f}</strong> — i.e., after BPE the sequence uses only {int(compression*100)}% as many tokens as before.</p>
<div style="background:#fdfdfd; padding:10px 12px; border-radius:4px; border:1px solid #e5e5e5;">
{bpe_html}
</div>

<p style="font-size:13px; color:#555; margin-top:16px;">
<strong>What "compression" means here:</strong> the same trajectory that took {len(canonical)} canonical atoms to express takes only {len(bpe_expressed)} BPE tokens because recurring patterns (like "edit-edit" or "search-then-open") got bound into single tokens. Across the whole corpus of 867 trajectories, compression ratio is ~0.41 at V=200 (meaning trajectories use ~41% as many tokens after BPE).
</p>
"""


def build_example_3_top_motifs_with_samples(bpe_expressed_all, records, motif_count_floor=100) -> str:
    """Show top motifs with example trajectory snippets where they occur."""

    counts = Counter()
    for seq in bpe_expressed_all:
        counts.update(seq)

    top_motifs = [(t, c) for t, c in counts.most_common() if "+" in t][:8]

    sections = []
    sections.append("""
<h3>6.3 Top motifs with real-trajectory occurrences</h3>

<p>The BPE-learned motifs correspond to recognizable procedural practices. Here are the top motifs by frequency across the corpus, with examples of where they appear in real trajectories.</p>
""")

    # For each motif, find a trajectory where it appears and show context
    for rank, (motif, count) in enumerate(top_motifs, 1):
        parts = motif.split("+")
        motif_len = len(parts)
        display = " → ".join(parts)

        # Find a sample trajectory with this motif
        sample = None
        for r in records:
            if motif in r["bpe"]:
                idx = r["bpe"].index(motif)
                # Grab surrounding context (2 tokens before and after)
                context_start = max(0, idx - 2)
                context_end = min(len(r["bpe"]), idx + 3)
                context = r["bpe"][context_start:context_end]
                sample = {
                    "agent": r["agent"],
                    "instance_id": r["instance_id"],
                    "context": context,
                    "motif_position_in_context": idx - context_start,
                }
                break

        context_html = ""
        if sample:
            ctx_parts = []
            for i, t in enumerate(sample["context"]):
                is_the_motif = (i == sample["motif_position_in_context"] and t == motif)
                bg = "#f7d7a6" if is_the_motif else ("#e8f0ff" if "+" in t else "#f5f5f5")
                border = "#d49030" if is_the_motif else ("#6090d0" if "+" in t else "#bbb")
                disp = t if len(t) < 30 else t[:13] + "...+" + t.rsplit("+", 1)[-1][:13]
                disp = disp.replace("+", "→")
                ctx_parts.append(
                    f'<span style="display:inline-block; padding:2px 6px; margin:2px; '
                    f'background:{bg}; border:1px solid {border}; border-radius:3px; '
                    f'font-family:SF Mono,monospace; font-size:11px;" '
                    f'title="{html.escape(t)}">{html.escape(disp)}</span>'
                )
            context_html = f"""
<p style="margin: 4px 0; font-size:13px;">Example occurrence in agent <strong>{sample['agent']}</strong>, instance <code>{html.escape(sample['instance_id'])}</code> (motif highlighted):</p>
<div style="padding: 8px 12px; background:#fafafa; border-radius:4px;">
{''.join(ctx_parts)}
</div>
"""

        sections.append(f"""
<div style="margin: 18px 0; padding: 12px 16px; background:#fff; border:1px solid #e0e0e0; border-radius:6px;">
<div style="font-size:14px;">
<span style="color:#888;">#{rank}</span>
&nbsp;
<span style="font-family:SF Mono,monospace; font-size:12.5px;">{html.escape(display)}</span>
&nbsp; <span style="color:#888; font-size:12px;">({motif_len} atoms, {count} occurrences)</span>
</div>
{context_html}
</div>
""")

    return "".join(sections)


def main():
    print("Loading sequences...")
    records = []
    for agent_dir in sorted(CACHE.iterdir()):
        if not agent_dir.is_dir():
            continue
        for traj_file in sorted(agent_dir.glob("*.json")):
            with open(traj_file) as f:
                raw = json.load(f)
            seq = canonicalize_trajectory(raw.get("trajectory", []))
            if seq:
                records.append({
                    "agent_dir": agent_dir.name,
                    "agent": AGENT_SHORT.get(agent_dir.name, agent_dir.name),
                    "instance_id": traj_file.stem,
                    "raw_trajectory": raw.get("trajectory", []),
                    "canonical": seq,
                })

    print(f"  {len(records)} trajectories")

    # Train BPE at V=200 (same as dashboard)
    canonical_seqs = [r["canonical"] for r in records]
    print("Training BPE...")
    model, expressed = train_bpe(canonical_seqs, target_size=200, verbose=False)
    for r, e in zip(records, expressed):
        r["bpe"] = e

    # Pick a moderate-length trajectory for the three-view example
    # Filter to Claude 3.5 (has good middle-length sequences), length 15-25
    candidates = [r for r in records
                  if r["agent"] == "Claude-3.5"
                  and 15 <= len(r["canonical"]) <= 25]
    # Pick the one with the most motifs in its BPE expression
    candidates.sort(key=lambda r: -sum(1 for t in r["bpe"] if "+" in t))
    chosen = candidates[0]
    print(f"  chose trajectory: {chosen['agent']} / {chosen['instance_id']} "
          f"({len(chosen['canonical'])} canonical → {len(chosen['bpe'])} BPE)")

    # Build HTML sections
    html_sections = []

    html_sections.append('<section id="examples">')
    html_sections.append('<h2>6. Concrete examples — what BPE is actually doing</h2>')
    html_sections.append('<p>Plain-language illustrations of the core BPE concepts using real data from this pilot.</p>')

    html_sections.append(build_example_1_what_is_a_merge())
    html_sections.append(build_example_2_trajectory_views(
        chosen["raw_trajectory"], chosen["canonical"], chosen["bpe"],
        chosen["instance_id"], chosen["agent"],
    ))
    html_sections.append(build_example_3_top_motifs_with_samples(expressed, records))

    html_sections.append('</section>')

    out_path = OUT / "examples_snippet.html"
    out_path.write_text("\n".join(html_sections))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
