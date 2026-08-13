"""Export per-step data for the Live Programming viewer.

For each selected task, walks the three agents' raw .traj files and emits
one JSON file at docs/paper2_pilot/viewer_data/<instance_id>.json with:

  - canonical atom sequence
  - BPE motif sequence
  - per-step {atom, action, thought, observation_preview} for drill-down
  - phase segmentation
  - model_stats (tokens, cost)
  - resolution status

Usage:
    python -m analysis.preferences.build_viewer_data
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE = PROJECT_ROOT / "output" / "trajectories" / ".cache"
OUT_DIR = PROJECT_ROOT / "docs" / "paper2_pilot" / "viewer_data"
SEQ_PATH = PROJECT_ROOT / "output" / "paper2_pilot" / "bpe_sequences.jsonl"
DIVERSITY_PATH = PROJECT_ROOT / "output" / "paper2_pilot" / "task_diversity.csv"
PAIRS_PATH = PROJECT_ROOT / "output" / "paper2_pilot" / "tied_outcome_pairs.csv"

AGENT_DIR_TO_SHORT = {
    "20240402_sweagent_gpt4": "GPT-4",
    "20240620_sweagent_claude3.5sonnet": "Claude-3.5",
    "20240728_sweagent_gpt4o": "GPT-4o",
}
AGENTS = ["GPT-4", "Claude-3.5", "GPT-4o"]

SELECTED_TASKS = [
    ("django__django-13447", "short 3/3 (tight convergence)"),
    ("django__django-11039", "medium 3/3 (productive divergence)"),
    ("django__django-14855", "long 3/3 (GPT-4o verbosity signature)"),
    ("django__django-13551", "medium 0/3 (what failure looks like)"),
]


def classify_phase(atom: str, seen_open: bool, seen_edit: bool) -> str:
    if atom == "SUBMIT" or atom.startswith("SHELL_RM"):
        return "cleanup_submit"
    if atom.startswith("RUN_PYTHON_") or atom.startswith("RUN_PYTEST"):
        return "verification"
    if atom.startswith("EDIT_") or atom.startswith("CREATE_"):
        return "editing"
    if atom.startswith("OPEN_") or atom.startswith("NAV_"):
        return "editing" if seen_edit else "localization"
    if atom == "SEARCH" or atom.startswith("FIND_FILE"):
        if seen_edit:
            return "editing"
        if seen_open:
            return "localization"
        return "exploration"
    if atom.startswith("SHELL_"):
        return "exploration" if not seen_open and not seen_edit else "editing"
    return "exploration"


def phase_segments(atoms: list[str]) -> list[dict]:
    if not atoms:
        return []
    seen_open = seen_edit = False
    per_phase = []
    for a in atoms:
        p = classify_phase(a, seen_open, seen_edit)
        per_phase.append(p)
        if a.startswith("OPEN_"):
            seen_open = True
        if a.startswith("EDIT_") or a.startswith("CREATE_"):
            seen_edit = True
    segs = []
    start = 0
    for i in range(1, len(per_phase)):
        if per_phase[i] != per_phase[i - 1]:
            segs.append({"phase": per_phase[start], "start": start, "end": i})
            start = i
    segs.append({"phase": per_phase[start], "start": start, "end": len(per_phase)})
    return segs


def truncate(s: str | None, n: int) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def load_bpe_records() -> dict[tuple[str, str], dict]:
    out = {}
    with open(SEQ_PATH) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(r["agent"], r["instance_id"])] = r
    return out


def load_resolved() -> set[tuple[str, str]]:
    out = set()
    m = {
        "Claude 3.5 Sonnet (SWE-agent)": "Claude-3.5",
        "GPT-4 (SWE-agent)": "GPT-4",
        "GPT-4o (SWE-agent)": "GPT-4o",
    }
    with open(PAIRS_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            out.add((m.get(row["agent_a"]), row["instance_id"]))
            out.add((m.get(row["agent_b"]), row["instance_id"]))
    return out


def load_difficulty() -> dict[str, int]:
    out = {}
    with open(DIVERSITY_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["instance_id"]] = int(row["n_resolved"])
    return out


def load_traj_steps(agent: str, instance_id: str) -> tuple[list[dict], dict]:
    inv_map = {v: k for k, v in AGENT_DIR_TO_SHORT.items()}
    traj_path = CACHE / inv_map[agent] / f"{instance_id}.json"
    if not traj_path.exists():
        return [], {}
    with open(traj_path) as f:
        d = json.load(f)
    steps_raw = d.get("trajectory") or []
    stats = (d.get("info") or {}).get("model_stats") or {}
    steps = []
    for s in steps_raw:
        steps.append({
            "action": truncate(s.get("action"), 220),
            "thought": truncate(s.get("response"), 400),
            "observation_preview": truncate(s.get("observation"), 280),
        })
    return steps, stats


def build_for_task(instance_id: str, note: str, bpe: dict, resolved: set, difficulty: dict) -> dict:
    per_agent = {}
    max_atoms = 0
    for a in AGENTS:
        rec = bpe.get((a, instance_id))
        if rec is None:
            continue
        steps_raw, stats = load_traj_steps(a, instance_id)
        atoms = rec["canonical"]
        motifs = rec["bpe"]
        max_atoms = max(max_atoms, len(atoms))
        aligned_steps = []
        for i, atom in enumerate(atoms):
            if i < len(steps_raw):
                s = steps_raw[i]
            else:
                s = {"action": "", "thought": "", "observation_preview": ""}
            aligned_steps.append({"atom": atom, **s})
        per_agent[a] = {
            "canonical": atoms,
            "bpe": motifs,
            "canonical_length": len(atoms),
            "bpe_length": len(motifs),
            "steps": aligned_steps,
            "phases": phase_segments(atoms),
            "tokens_sent": int(stats.get("tokens_sent", 0)),
            "api_calls": int(stats.get("api_calls", 0)),
            "cost_usd": float(stats.get("instance_cost", 0)),
            "resolved": (a, instance_id) in resolved,
        }
    return {
        "instance_id": instance_id,
        "note": note,
        "n_resolved": difficulty.get(instance_id, -1),
        "max_atoms": max_atoms,
        "agents": per_agent,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bpe = load_bpe_records()
    resolved = load_resolved()
    difficulty = load_difficulty()

    index = []
    for inst, note in SELECTED_TASKS:
        task_data = build_for_task(inst, note, bpe, resolved, difficulty)
        if len(task_data["agents"]) < 3:
            print(f"skip {inst}: only {len(task_data['agents'])} agents have data")
            continue
        path = OUT_DIR / f"{inst}.json"
        path.write_text(json.dumps(task_data))
        size_kb = path.stat().st_size / 1024
        print(f"wrote {path.name} ({size_kb:.1f} KB)")
        index.append({
            "instance_id": inst,
            "note": note,
            "n_resolved": task_data["n_resolved"],
            "max_atoms": task_data["max_atoms"],
        })
    (OUT_DIR / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nwrote {OUT_DIR / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
