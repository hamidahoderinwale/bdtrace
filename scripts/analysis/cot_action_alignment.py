"""CoT-action alignment for the two extended-thinking agents.

For Claude-3.7-thinking and Claude-4 trajectories, compares the action types
the agent states in its `thought` field against the canonical atoms it
actually emits. Reports per-trajectory Jaccard overlap, forward coverage
(stated that occurred), and reverse coverage (occurred that were stated).

Reads:
    output/trajectories/.cache/20250226_sweagent_claude-3-7-sonnet-20250219/
    output/trajectories/.cache/20250526_sweagent_claude-4-sonnet-20250514/

Writes:
    output/paper2_pilot/cot_action_alignment.jsonl
    output/paper2_pilot/cot_action_alignment_summary.json
    output/figures/fig_cot_action_alignment.png

Usage:
    uv run python scripts/analysis/cot_action_alignment.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from collections import Counter

import altair as alt
import pandas as pd

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

# Verb-keyword to canonical-atom-verb mapping. Deliberately blunt: false
# positives are tolerated, the same rule applies uniformly across agents.
# Word boundaries enforced so "edit" doesn't fire inside "editor".
VERB_PATTERNS: dict[str, str] = {
    r"\b(?:edit|modify|change|update|fix|patch|alter|rewrite|replace)\b": "EDIT",
    r"\b(?:open|view|inspect|examine|look at|read)\b": "OPEN",
    r"\b(?:create|write a new|make a new|new file)\b": "CREATE",
    r"\b(?:search|grep|look for|find references|search for)\b": "SEARCH",
    r"\b(?:find file|locate|find the file|find a file)\b": "FIND_FILE",
    r"\b(?:navigate|scroll|go to|jump to)\b": "NAV",
    r"\b(?:run|execute|run the test|run tests|pytest|test it|run pytest)\b": "RUN_PYTEST",
    r"\b(?:run python|python script|reproduce|reproducer)\b": "RUN_PYTHON",
    r"\b(?:install|pip install|pip)\b": "RUN_PIP",
    r"\b(?:lint|flake8|mypy|ruff|black)\b": "RUN_LINT",
    r"\b(?:list (?:files|directory)|ls)\b": "SHELL_LS",
    r"\b(?:submit|finish|complete the task|wrap up)\b": "SUBMIT",
}
COMPILED = {re.compile(p, re.IGNORECASE): atom for p, atom in VERB_PATTERNS.items()}


def stated_atoms_from_thought(thought: str) -> set[str]:
    """Run keyword matcher over thought text. Returns set of stated atom verbs.

    We keep this at the verb level (no file-type tag) because thought text
    rarely names file types in the same syntactic position; matching at the
    verb level is the conservative comparison.
    """
    if not thought or not isinstance(thought, str):
        return set()
    out = set()
    for pat, atom in COMPILED.items():
        if pat.search(thought):
            out.add(atom)
    return out


def observed_atoms_verb(trajectory: list[dict]) -> set[str]:
    """Run the canonicalizer, then strip file-type suffixes to get verb-level set.

    Mirrors the granularity at which we extract from thought, so the comparison
    is verb-to-verb rather than full-atom-to-verb.
    """
    atoms = canonicalize_trajectory(trajectory)
    out = set()
    # Also handle Claude-3.7's str_replace_editor which canonicalize_trajectory
    # emits as UNKNOWN_STR_REPLACE_EDITOR. Rewrite those to OPEN/EDIT/CREATE.
    for atom, step in zip(atoms, trajectory):
        if atom == "UNKNOWN_STR_REPLACE_EDITOR":
            rewritten = _rewrite_str_replace_editor(step.get("action") or "")
            atom = rewritten or "OTHER"
        if atom.startswith("UNKNOWN_") or atom in ("OTHER", "EMPTY", "COMMENT"):
            continue
        # Strip file-type suffix (e.g. EDIT_SRC_PY -> EDIT, RUN_PYTEST_TEST_PY -> RUN_PYTEST)
        if atom.startswith("RUN_PYTEST"):
            verb = "RUN_PYTEST"
        elif atom.startswith("RUN_PYTHON"):
            verb = "RUN_PYTHON"
        elif atom.startswith("RUN_PIP"):
            verb = "RUN_PIP"
        elif atom.startswith("RUN_LINT"):
            verb = "RUN_LINT"
        elif atom.startswith("RUN_TEST_SCRIPT"):
            verb = "RUN_PYTEST"
        elif atom.startswith("EDIT"):
            verb = "EDIT"
        elif atom.startswith("OPEN"):
            verb = "OPEN"
        elif atom.startswith("CREATE"):
            verb = "CREATE"
        elif atom.startswith("NAV"):
            verb = "NAV"
        elif atom == "FIND_FILE":
            verb = "FIND_FILE"
        elif atom == "SEARCH":
            verb = "SEARCH"
        elif atom == "SUBMIT":
            verb = "SUBMIT"
        elif atom.startswith("SHELL_LS"):
            verb = "SHELL_LS"
        elif atom.startswith("SHELL_"):
            continue  # other shell verbs not in our stated taxonomy
        else:
            continue
        out.add(verb)
    return out


def alignment_metrics(stated: set, observed: set) -> dict:
    """Jaccard, forward coverage, reverse coverage."""
    inter = stated & observed
    union = stated | observed
    return {
        "n_stated": len(stated),
        "n_observed": len(observed),
        "n_intersection": len(inter),
        "jaccard": len(inter) / len(union) if union else None,
        "forward_coverage": len(inter) / len(stated) if stated else None,
        "reverse_coverage": len(inter) / len(observed) if observed else None,
        "stated_only": sorted(stated - observed),
        "observed_only": sorted(observed - stated),
        "intersection": sorted(inter),
    }


def process_agent(agent: str, subdir: str) -> list[dict]:
    """Iterate trace files; emit per-trajectory alignment records."""
    base = CACHE / subdir
    files = sorted(base.glob("*.json"))
    records = []
    for f in files:
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

        # Aggregate thought across all steps
        thoughts = [s.get("thought", "") for s in trajectory if isinstance(s, dict)]
        full_thought = "\n".join(t for t in thoughts if isinstance(t, str) and t.strip())

        stated = stated_atoms_from_thought(full_thought)
        observed = observed_atoms_verb(trajectory)
        metrics = alignment_metrics(stated, observed)

        records.append({
            "agent": agent,
            "instance_id": d.get("instance_id") or f.stem,
            "n_steps": len(trajectory),
            "n_thought_chars": len(full_thought),
            "stated": sorted(stated),
            "observed": sorted(observed),
            **metrics,
        })
    return records


def per_agent_summary(records: list[dict]) -> dict:
    """Median, IQR, n for each metric, per agent."""
    df = pd.DataFrame(records)
    summary = {}
    for agent, sub in df.groupby("agent"):
        # Drop trajectories where stated or observed is empty (metric undefined)
        valid = sub.dropna(subset=["jaccard", "forward_coverage", "reverse_coverage"])
        summary[agent] = {
            "n_trajectories_total": int(len(sub)),
            "n_trajectories_valid": int(len(valid)),
            "median_jaccard": float(valid["jaccard"].median()),
            "iqr_jaccard": [float(valid["jaccard"].quantile(0.25)), float(valid["jaccard"].quantile(0.75))],
            "median_forward_coverage": float(valid["forward_coverage"].median()),
            "iqr_forward_coverage": [float(valid["forward_coverage"].quantile(0.25)), float(valid["forward_coverage"].quantile(0.75))],
            "median_reverse_coverage": float(valid["reverse_coverage"].median()),
            "iqr_reverse_coverage": [float(valid["reverse_coverage"].quantile(0.25)), float(valid["reverse_coverage"].quantile(0.75))],
            "mean_n_stated": float(valid["n_stated"].mean()),
            "mean_n_observed": float(valid["n_observed"].mean()),
        }
    return summary


def make_figure(records: list[dict], out_path: Path) -> None:
    """Two-panel figure: per-trajectory scatter + Jaccard distribution."""
    df = pd.DataFrame(records)
    df = df.dropna(subset=["jaccard", "forward_coverage", "reverse_coverage"])

    color_map = {"Claude-3.7-thinking": GREEN, "Claude-4": BLUE}
    color_scale = alt.Scale(
        domain=list(color_map.keys()), range=list(color_map.values())
    )

    # Left: scatter of forward x reverse coverage
    scatter = (
        alt.Chart(df)
        .mark_circle(size=40, opacity=0.55)
        .encode(
            x=alt.X("forward_coverage:Q", title="Forward coverage", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("reverse_coverage:Q", title="Reverse coverage", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("agent:N", scale=color_scale, title="Agent"),
            tooltip=["agent", "instance_id", "jaccard", "forward_coverage", "reverse_coverage"],
        )
        .properties(width=300, height=300, title="Stated vs observed action types")
    )
    diag_data = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    diag = (
        alt.Chart(diag_data)
        .mark_line(color=OLIVE, strokeDash=[4, 4], opacity=0.5)
        .encode(x="x:Q", y="y:Q")
    )
    left = scatter + diag

    # Right: Jaccard density per agent
    right = (
        alt.Chart(df)
        .transform_density(
            "jaccard", as_=["jaccard", "density"], groupby=["agent"], extent=[0, 1]
        )
        .mark_area(opacity=0.55)
        .encode(
            x=alt.X("jaccard:Q", title="Jaccard overlap", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color("agent:N", scale=color_scale, legend=None),
        )
        .properties(width=300, height=300, title="Overlap distribution")
    )

    chart = (
        alt.hconcat(left, right)
        .resolve_scale(color="shared")
        .configure_view(strokeWidth=0)
        .configure_title(fontSize=12, color="#111111", anchor="start")
    )
    chart.save(str(out_path), scale_factor=2)
    print(f"Saved {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for agent, subdir in AGENTS.items():
        print(f"Processing {agent} from {subdir} ...")
        records = process_agent(agent, subdir)
        print(f"  {len(records)} trajectories")
        all_records.extend(records)

    # Per-trajectory JSONL
    jsonl_path = OUT_DIR / "cot_action_alignment.jsonl"
    with jsonl_path.open("w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(all_records)} records to {jsonl_path}")

    # Summary JSON
    summary = per_agent_summary(all_records)
    summary_path = OUT_DIR / "cot_action_alignment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")
    print(json.dumps(summary, indent=2))

    # Figure
    make_figure(all_records, OUT_FIG / "fig_cot_action_alignment.png")


if __name__ == "__main__":
    main()
