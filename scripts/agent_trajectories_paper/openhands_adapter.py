"""OpenHands -> procgrep coarse-atom adapter (prototype, no GPU/no spend).

Streams a sample of the public nvidia/SWE-Zero-openhands-trajectories corpus and
maps each assistant tool_call to a coarse atom (the same alphabet the behavioral
search uses), writing {id, agent, atoms, resolved} jsonl. This is what lets
procgrep_search.py run on a corpus *other than* our own traces -- the proof the
behavioral-search idea generalizes. First-pass heuristic mapping; refine as needed.

  distillation_run/.venv/bin/python openhands_adapter.py 300
"""
from __future__ import annotations
import json
import sys
from datasets import load_dataset

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
OUT = "/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces/output/paper2_pilot/openhands_sample.jsonl"
DS = "nvidia/SWE-Zero-openhands-trajectories"


def classify(name: str, raw_args) -> str:
    try:
        a = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
    except Exception:
        a = {}
    if name == "str_replace_editor":
        cmd = str(a.get("command", "")).lower()
        if cmd == "view":
            return "read_file"
        if cmd == "create":
            return "create_file"
        if cmd in ("str_replace", "insert", "edit"):
            return "edit"
        return "other"
    if name == "execute_bash":
        c = str(a.get("command", "")).lower()
        if any(k in c for k in ("pytest", "unittest", "tox", "nosetests")) or \
           ("python" in c and "test" in c):
            return "run_test"
        if any(k in c for k in ("grep", "find ", "rg ", "locate", "ls ", "glob")):
            return "search_repo"
        if any(k in c for k in ("cat ", "head ", "tail ", "less ")):
            return "read_file"
        return "other"
    if name in ("finish", "submit"):
        return "submit"
    return "other"


def atoms_of(traj) -> list[str]:
    out = []
    for m in traj:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", {}) or {}
            out.append(classify(fn.get("name", ""), fn.get("arguments", "")))
    return out


def main() -> int:
    ds = load_dataset(DS, split="train", streaming=True)
    n = 0
    with open(OUT, "w") as f:
        for ex in ds:
            atoms = atoms_of(ex.get("trajectory", []))
            if not atoms:
                continue
            f.write(json.dumps({"id": ex["instance_id"], "agent": "OpenHands-SWE-Zero",
                                "atoms": atoms, "resolved": None}) + "\n")
            n += 1
            if n >= N:
                break
    print(f"wrote {n} OpenHands trajectories -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
