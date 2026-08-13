"""Re-fetch OpenHands (SWE-Zero) trajectories WITH raw text + repo, for the
behavioral-retrieval eval (no spend). Emits {id, repo, agent, atoms, text}:
  - atoms: coarse procedural atoms (procgrep behavioral retrieval)
  - text : concatenated message contents (the BM25 keyword baseline indexes this)
  - repo : topical ground-truth label
Double-dissociation eval: fingerprint(atoms) should win same-behavior precision@k;
BM25(text) should win same-repo precision@k.

  distillation_run/.venv/bin/python fetch_openhands_rawtext.py 500
"""
from __future__ import annotations
import json
import sys
from datasets import load_dataset

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
OUT = "/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces/output/paper2_pilot/openhands_rawtext.jsonl"
DS = "nvidia/SWE-Zero-openhands-trajectories"
CAP = 6000  # cap raw text per trajectory (chars)


def classify(name, raw_args):
    try:
        a = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
    except Exception:
        a = {}
    if name == "str_replace_editor":
        cmd = str(a.get("command", "")).lower()
        return {"view": "read_file", "create": "create_file", "str_replace": "edit",
                "insert": "edit", "edit": "edit"}.get(cmd, "other")
    if name == "execute_bash":
        c = str(a.get("command", "")).lower()
        if any(k in c for k in ("pytest", "unittest", "tox", "nosetests")) or ("python" in c and "test" in c):
            return "run_test"
        if any(k in c for k in ("grep", "find ", "rg ", "locate", "ls ", "glob")):
            return "search_repo"
        if any(k in c for k in ("cat ", "head ", "tail ", "less ")):
            return "read_file"
        return "other"
    if name in ("finish", "submit"):
        return "submit"
    return "other"


def parse(traj):
    atoms, texts = [], []
    for m in traj:
        if m.get("content"):
            texts.append(str(m["content"]))
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {}) or {}
                atoms.append(classify(fn.get("name", ""), fn.get("arguments", "")))
                if fn.get("arguments"):
                    texts.append(str(fn["arguments"]))
    return atoms, (" ".join(texts))[:CAP]


def main() -> int:
    ds = load_dataset(DS, split="train", streaming=True)
    n = 0
    with open(OUT, "w") as f:
        for ex in ds:
            atoms, text = parse(ex.get("trajectory", []))
            if not atoms:
                continue
            f.write(json.dumps({"id": ex["instance_id"], "repo": ex.get("repo", "?"),
                                "agent": "OpenHands-SWE-Zero", "atoms": atoms, "text": text}) + "\n")
            n += 1
            if n >= N:
                break
    print(f"wrote {n} trajectories (atoms+text+repo) -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
