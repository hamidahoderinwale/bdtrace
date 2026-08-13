"""Embedding-based CoT-action alignment as robustness check on the regex matcher.

Replaces the keyword-regex matcher with sentence-transformer embeddings:
- Each canonical-atom verb gets a short natural-language description
- Each thought sentence is embedded and compared via cosine similarity
- Threshold sweep: 0.40 to 0.70 in steps of 0.05
- Same Jaccard/forward/reverse metrics as the regex script

Reads:
    output/trajectories/.cache/20250226_sweagent_claude-3-7-sonnet-20250219/
    output/trajectories/.cache/20250526_sweagent_claude-4-sonnet-20250514/
    output/paper2_pilot/cot_action_alignment.jsonl (existing regex results for comparison)

Writes:
    output/paper2_pilot/cot_action_alignment_embedding.jsonl
    output/paper2_pilot/cot_action_alignment_embedding_summary.json
    output/figures/fig_cot_alignment_robustness.png
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, GREEN, BLUE, MAGENTA, OLIVE
from analysis.preferences.canonicalize import canonicalize_trajectory
from analysis.preferences.canonicalize_extended import _rewrite_str_replace_editor

register()

CACHE = ROOT / "output" / "trajectories" / ".cache"
AGENTS = {
    "Claude-3.7-thinking": "20250226_sweagent_claude-3-7-sonnet-20250219",
    "Claude-4": "20250526_sweagent_claude-4-sonnet-20250514",
}
OUT_DIR = ROOT / "output" / "paper2_pilot"
OUT_FIG = ROOT / "output" / "figures"
REGEX_RESULTS = OUT_DIR / "cot_action_alignment.jsonl"

THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
PRIMARY_THRESHOLD = 0.55  # chosen post-sweep; we'll re-evaluate from the data

# Multiple descriptions per verb to mitigate single-phrasing bias. Each verb has
# 3-4 paraphrases written in different modalities (forward-looking vs reflective,
# first-person vs third-person). Final match is max similarity across all
# descriptions of a verb.
VERB_DESCRIPTIONS: dict[str, list[str]] = {
    "EDIT": [
        "The agent plans to edit, modify, change, update, or rewrite existing code.",
        "I should change this code to fix the issue.",
        "Let me modify this function.",
        "I need to update the implementation.",
    ],
    "OPEN": [
        "The agent plans to view, read, or inspect a file or its contents.",
        "Let me look at this file to understand it.",
        "I should examine the code here.",
        "I need to read the implementation.",
    ],
    "CREATE": [
        "The agent plans to create a new file or write new code from scratch.",
        "I'll create a new file for this.",
        "Let me write a new script.",
        "I need to make a reproducer file.",
    ],
    "SEARCH": [
        "The agent plans to search for code, text, or references in the codebase.",
        "Let me search for this pattern.",
        "I should grep for this term.",
        "I need to find references to this.",
    ],
    "FIND_FILE": [
        "The agent plans to find or locate a specific file by name.",
        "I need to find where this file is.",
        "Let me locate this file in the repository.",
    ],
    "NAV": [
        "The agent plans to navigate, scroll, or move to a specific location in a file.",
        "Let me scroll down to see more.",
        "I should go to line 100.",
    ],
    "RUN_PYTEST": [
        "The agent plans to run tests, execute the test suite, or verify behavior with tests.",
        "Let me run the tests to see if this works.",
        "I should execute pytest.",
        "I need to run the test suite.",
    ],
    "RUN_PYTHON": [
        "The agent plans to run a Python script to reproduce an issue or execute code.",
        "Let me run this Python script.",
        "I should execute the reproducer.",
        "Let me run python reproduce.py.",
    ],
    "RUN_PIP": [
        "The agent plans to install a Python package via pip.",
        "I need to install this package.",
        "Let me pip install this.",
    ],
    "RUN_LINT": [
        "The agent plans to run a linter or code style checker.",
        "Let me run flake8.",
        "I should check with mypy.",
    ],
    "SHELL_LS": [
        "The agent plans to list files in a directory.",
        "Let me list the directory contents.",
        "I'll do ls to see what's here.",
    ],
    "SUBMIT": [
        "The agent plans to submit, finalize, or complete the task.",
        "I'm done; let me submit.",
        "This looks good; I'll submit now.",
        "The fix is complete; submitting.",
    ],
}


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter — splits on .!? followed by whitespace or end."""
    if not text:
        return []
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]


def stated_atoms_embedding(thought: str, model: SentenceTransformer, verb_embs: np.ndarray, verbs: list[str], threshold: float) -> set[str]:
    """Embed each sentence in thought; for each, add any verb whose cos-sim exceeds threshold."""
    sentences = split_sentences(thought)
    if not sentences:
        return set()
    sent_embs = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    # verb_embs is already normalized
    sims = sent_embs @ verb_embs.T  # (n_sentences, n_verbs)
    matched = set()
    for verb_idx, verb in enumerate(verbs):
        if sims[:, verb_idx].max() >= threshold:
            matched.add(verb)
    return matched


def observed_atoms_verb(trajectory: list[dict]) -> set[str]:
    """Same as the regex script — verb-level observed atoms from canonicalizer."""
    atoms = canonicalize_trajectory(trajectory)
    out = set()
    for atom, step in zip(atoms, trajectory):
        if atom == "UNKNOWN_STR_REPLACE_EDITOR":
            rewritten = _rewrite_str_replace_editor(step.get("action") or "")
            atom = rewritten or "OTHER"
        if atom.startswith("UNKNOWN_") or atom in ("OTHER", "EMPTY", "COMMENT"):
            continue
        if atom.startswith("RUN_PYTEST"):       verb = "RUN_PYTEST"
        elif atom.startswith("RUN_PYTHON"):     verb = "RUN_PYTHON"
        elif atom.startswith("RUN_PIP"):        verb = "RUN_PIP"
        elif atom.startswith("RUN_LINT"):       verb = "RUN_LINT"
        elif atom.startswith("RUN_TEST_SCRIPT"): verb = "RUN_PYTEST"
        elif atom.startswith("EDIT"):           verb = "EDIT"
        elif atom.startswith("OPEN"):           verb = "OPEN"
        elif atom.startswith("CREATE"):         verb = "CREATE"
        elif atom.startswith("NAV"):            verb = "NAV"
        elif atom == "FIND_FILE":               verb = "FIND_FILE"
        elif atom == "SEARCH":                  verb = "SEARCH"
        elif atom == "SUBMIT":                  verb = "SUBMIT"
        elif atom.startswith("SHELL_LS"):       verb = "SHELL_LS"
        elif atom.startswith("SHELL_"):         continue
        else:                                    continue
        out.add(verb)
    return out


def alignment_metrics(stated: set, observed: set) -> dict:
    inter = stated & observed
    union = stated | observed
    return {
        "n_stated": len(stated),
        "n_observed": len(observed),
        "n_intersection": len(inter),
        "jaccard":          len(inter) / len(union)    if union   else None,
        "forward_coverage": len(inter) / len(stated)   if stated  else None,
        "reverse_coverage": len(inter) / len(observed) if observed else None,
    }


def per_agent_summary(records: list[dict], threshold: float) -> dict:
    df = pd.DataFrame([r for r in records if r["threshold"] == threshold])
    summary = {}
    for agent, sub in df.groupby("agent"):
        valid = sub.dropna(subset=["jaccard", "forward_coverage", "reverse_coverage"])
        summary[agent] = {
            "n_trajectories": int(len(valid)),
            "median_jaccard":          float(valid["jaccard"].median()),
            "median_forward_coverage": float(valid["forward_coverage"].median()),
            "median_reverse_coverage": float(valid["reverse_coverage"].median()),
            "iqr_forward_coverage":    [float(valid["forward_coverage"].quantile(0.25)), float(valid["forward_coverage"].quantile(0.75))],
            "iqr_reverse_coverage":    [float(valid["reverse_coverage"].quantile(0.25)), float(valid["reverse_coverage"].quantile(0.75))],
        }
    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    print("Loading sentence-transformer model ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    verbs = list(VERB_DESCRIPTIONS.keys())
    # Flatten descriptions across verbs; track which verb each description belongs to
    all_descriptions: list[str] = []
    verb_for_desc: list[str] = []
    for verb in verbs:
        for desc in VERB_DESCRIPTIONS[verb]:
            all_descriptions.append(desc)
            verb_for_desc.append(verb)
    print(f"Embedding {len(all_descriptions)} descriptions across {len(verbs)} verbs ...")
    desc_embs = model.encode(all_descriptions, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    # Build mask per verb for max-sim aggregation
    verb_to_desc_indices = {v: [i for i, vd in enumerate(verb_for_desc) if vd == v] for v in verbs}

    records = []
    for agent, subdir in AGENTS.items():
        base = CACHE / subdir
        files = sorted(base.glob("*.json"))
        print(f"\nProcessing {agent} ({len(files)} trajectories) ...")
        for i, f in enumerate(files):
            if i % 50 == 0:
                print(f"  {i}/{len(files)}")
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            content = d.get("content")
            if not isinstance(content, dict):
                continue
            trajectory = content.get("trajectory", [])
            if not trajectory:
                continue
            thoughts = [s.get("thought", "") for s in trajectory if isinstance(s, dict)]
            full_thought = "\n".join(t for t in thoughts if isinstance(t, str) and t.strip())
            if not full_thought:
                continue

            observed = observed_atoms_verb(trajectory)

            # Compute embedding-matched stated set at every threshold
            sentences = split_sentences(full_thought)
            if not sentences:
                continue
            sent_embs = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            sims = sent_embs @ desc_embs.T  # (n_sentences, n_descriptions)
            # Per-verb max sim = max over sentences × that verb's descriptions
            per_verb_max = {}
            for verb in verbs:
                desc_idx = verb_to_desc_indices[verb]
                per_verb_max[verb] = float(sims[:, desc_idx].max())
            max_sims_list = [per_verb_max[v] for v in verbs]

            for thresh in THRESHOLDS:
                stated = {v for v, s in zip(verbs, max_sims_list) if s >= thresh}
                metrics = alignment_metrics(stated, observed)
                records.append({
                    "agent": agent,
                    "instance_id": d.get("instance_id") or f.stem,
                    "threshold": thresh,
                    "stated": sorted(stated),
                    "observed": sorted(observed),
                    "max_sims": per_verb_max,
                    **metrics,
                })

    # Persist per-trajectory results
    jsonl_path = OUT_DIR / "cot_action_alignment_embedding.jsonl"
    with jsonl_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(records)} records ({len(records) // len(THRESHOLDS)} trajectories × {len(THRESHOLDS)} thresholds) to {jsonl_path}")

    # Per-threshold summary
    summary = {"thresholds": {}}
    for t in THRESHOLDS:
        summary["thresholds"][f"{t:.2f}"] = per_agent_summary(records, t)

    # Compare to regex results
    if REGEX_RESULTS.exists():
        regex_records = [json.loads(line) for line in REGEX_RESULTS.open()]
        regex_df = pd.DataFrame([r for r in regex_records if r["jaccard"] is not None])
        regex_summary = {}
        for agent, sub in regex_df.groupby("agent"):
            regex_summary[agent] = {
                "median_jaccard":          float(sub["jaccard"].median()),
                "median_forward_coverage": float(sub["forward_coverage"].median()),
                "median_reverse_coverage": float(sub["reverse_coverage"].median()),
            }
        summary["regex_baseline"] = regex_summary

    summary_path = OUT_DIR / "cot_action_alignment_embedding_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")
    print(json.dumps(summary, indent=2))

    # Figure: side-by-side comparison of regex (baseline) vs embedding at primary threshold.
    # Four bar groups (agent × metric), two bars each (matcher), shows whether the
    # asymmetric over-inclusion finding survives matcher choice.
    df_embed = pd.DataFrame([r for r in records if r["threshold"] == PRIMARY_THRESHOLD]).dropna(subset=["forward_coverage", "reverse_coverage"])
    embed_agg = df_embed.groupby("agent").agg(
        forward=("forward_coverage", "median"),
        reverse=("reverse_coverage", "median"),
    ).reset_index()
    embed_agg["matcher"] = "Embedding (paraphrase-tolerant)"

    if REGEX_RESULTS.exists():
        regex_records = [json.loads(line) for line in REGEX_RESULTS.open()]
        regex_df = pd.DataFrame([r for r in regex_records if r["jaccard"] is not None])
        regex_agg = regex_df.groupby("agent").agg(
            forward=("forward_coverage", "median"),
            reverse=("reverse_coverage", "median"),
        ).reset_index()
        regex_agg["matcher"] = "Regex (baseline)"
        combined = pd.concat([regex_agg, embed_agg], ignore_index=True)
    else:
        combined = embed_agg

    long_df = pd.melt(
        combined, id_vars=["agent", "matcher"],
        value_vars=["forward", "reverse"],
        var_name="metric", value_name="coverage",
    )
    long_df["metric"] = long_df["metric"].map({"forward": "Forward", "reverse": "Reverse"})

    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("matcher:N", title=None, axis=alt.Axis(labelAngle=0, labelFontSize=9)),
            y=alt.Y("coverage:Q", title="Median coverage", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "matcher:N",
                scale=alt.Scale(domain=["Regex (baseline)", "Embedding (paraphrase-tolerant)"], range=[OLIVE, BLUE]),
                legend=alt.Legend(title="Matcher", orient="bottom"),
            ),
            column=alt.Column("metric:N", title=None, header=alt.Header(labelFontSize=11)),
            row=alt.Row("agent:N", title=None, header=alt.Header(labelFontSize=11)),
        )
        .properties(width=140, height=120, title=alt.TitleParams(
            text=f"CoT-action coverage: regex baseline vs embedding (threshold {PRIMARY_THRESHOLD})",
            fontSize=12, color="#111111", anchor="start",
        ))
        .configure_view(strokeWidth=0)
    )
    chart.save(str(OUT_FIG / "fig_cot_alignment_robustness.png"), scale_factor=2)
    print(f"Saved {OUT_FIG / 'fig_cot_alignment_robustness.png'}")


if __name__ == "__main__":
    main()
