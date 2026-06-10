"""Augment the local 9-agent corpus with raw trajectory text (from the cached
.traj) so the retrieval eval can include a keyword (BM25) baseline alongside the
procedural fingerprint. Emits {id, agent, repo, atoms, text}. No spend.

text = serialized raw .traj (what a keyword index over the trace would see),
capped. Missing cache files -> empty text (excluded from BM25). Run from repo root.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path("/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces")
CACHE = ROOT / "output/trajectories/.cache"
OUT = ROOT / "output/paper2_pilot/local_rawtext.jsonl"
CAP = 8000


def to_canon(a):
    if a.startswith("EDIT"): return "edit"
    if a.startswith("CREATE"): return "create_file"
    if a.startswith("RUN"): return "run_test"
    if a.startswith(("OPEN", "NAV")): return "read_file"
    if a.startswith(("SEARCH", "FIND")): return "search_repo"
    if "SUBMIT" in a: return "submit"
    return "other"


def repo_of(iid):
    return re.sub(r"-\d+$", "", iid)


def main():
    rows = [json.loads(l) for l in open(ROOT / "output/paper2_pilot/bpe_sequences_extended.jsonl")]
    have = 0
    with open(OUT, "w") as f:
        for r in rows:
            iid, sub = r["instance_id"], r["submission"]
            p = CACHE / sub / f"{iid}.json"
            text = ""
            if p.exists():
                try:
                    text = json.dumps(json.load(open(p)))[:CAP]
                    have += 1
                except Exception:
                    pass
            f.write(json.dumps({"id": iid, "agent": r["agent"], "repo": repo_of(iid),
                                "atoms": [to_canon(a) for a in r["canonical"]], "text": text}) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT}  ({have} with raw text)")


if __name__ == "__main__":
    main()
