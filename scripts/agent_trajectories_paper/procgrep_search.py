"""ProcGrep behavioral trajectory search -- prototype.

Demonstrates the "search by behavior, not by task/outcome" idea on the local
2.6k-trajectory corpus (9 agents, with pass/fail), using procgrep for the
similarity half. Two query modes:

  structured  -- a procedural pattern over canonical atoms, e.g.
                 --seq search_repo,read_file,edit  --absent run_test
                 (ordered subsequence, gaps allowed; + presence/absence filters).
                 This is the thing keyword/metadata search over the raw trace
                 *cannot* express ("never ran tests", "edit-streak then submit").

  similarity  -- --like <Agent>: rank all trajectories by JSD (procgrep.jsd)
                 to that agent's mean fingerprint; return the nearest. Tests
                 "find trajectories that behave like X".

Run with the procgrep venv:
  /Users/hamidaho/learning-from-dev/procgrep/.venv/bin/python procgrep_search.py \
      --seq search_repo,read_file,edit --absent run_test --topk 8
  ...  --like Claude-4 --topk 8
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/Users/hamidaho/learning-from-dev/procgrep/src")
import numpy as np
from procgrep.jsd import jsd

ROOT = Path("/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces")
ROWS = ROOT / "output/paper2_pilot/bpe_sequences_extended.jsonl"
PF = ROOT / "output/paper2_pilot/extended_pass_fail.json"
COARSE = ["search_repo", "read_file", "edit", "create_file", "run_test", "submit", "other"]
CIX = {a: i for i, a in enumerate(COARSE)}


def to_canon(a: str) -> str:
    if a.startswith("EDIT"): return "edit"
    if a.startswith("CREATE"): return "create_file"
    if a.startswith("RUN"): return "run_test"
    if a.startswith(("OPEN", "NAV")): return "read_file"
    if a.startswith(("SEARCH", "FIND")): return "search_repo"
    if "SUBMIT" in a: return "submit"
    return "other"


def load():
    pf = json.load(open(PF))
    resolved = {k: set(v.get("resolved", [])) for k, v in pf.items()}
    out = []
    for line in open(ROWS):
        r = json.loads(line)
        atoms = [to_canon(a) for a in r["canonical"]]
        sub = r["submission"]
        out.append({"id": r["instance_id"], "agent": r["agent"], "atoms": atoms,
                    "resolved": r["instance_id"] in resolved.get(sub, set())})
    return out


def is_subseq(pattern, seq):
    it = iter(seq)
    return all(p in it for p in pattern)


def dist(atoms):
    v = np.zeros(len(COARSE))
    for a in atoms:
        v[CIX[a]] += 1
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", help="comma-separated ordered subsequence, e.g. search_repo,read_file,edit")
    ap.add_argument("--present", help="comma-separated atoms that must appear")
    ap.add_argument("--absent", help="comma-separated atoms that must NOT appear")
    ap.add_argument("--like", help="agent name: rank by JSD to its mean fingerprint")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--corpus", help="jsonl with {id,agent,atoms,resolved} to search instead of the local corpus")
    args = ap.parse_args()
    rows = ([json.loads(l) for l in open(args.corpus)] if args.corpus else load())
    print(f"corpus: {len(rows)} trajectories, {len(set(r['agent'] for r in rows))} agents\n")

    if args.like:
        members = [dist(r["atoms"]) for r in rows if r["agent"] == args.like]
        if not members:
            print(f"no agent '{args.like}'"); return 2
        centroid = np.mean(members, axis=0)
        ranked = sorted(rows, key=lambda r: jsd(dist(r["atoms"]), centroid))
        hits = ranked[:args.topk]
        same = sum(h["agent"] == args.like for h in hits)
        print(f"nearest {args.topk} to {args.like} fingerprint  ({same}/{args.topk} are {args.like}):")
        for h in hits:
            print(f"  {jsd(dist(h['atoms']), centroid):.3f}  {h['agent']:20s} {h['id']:32s} "
                  f"{'PASS' if h['resolved'] else 'fail'}")
        return 0

    seq = args.seq.split(",") if args.seq else []
    present = args.present.split(",") if args.present else []
    absent = args.absent.split(",") if args.absent else []
    hits = []
    for r in rows:
        s = set(r["atoms"])
        if seq and not is_subseq(seq, r["atoms"]): continue
        if present and not all(p in s for p in present): continue
        if absent and any(a in s for a in absent): continue
        hits.append(r)
    desc = []
    if seq: desc.append("→".join(seq))
    if present: desc.append("present:" + ",".join(present))
    if absent: desc.append("absent:" + ",".join(absent))
    lbl = lambda r: "PASS" if r is True else ("fail" if r is False else "?")
    n = len(hits); known = sum(1 for h in hits if h["resolved"] is not None)
    npass = sum(1 for h in hits if h["resolved"] is True)
    print(f"query [{' | '.join(desc)}] -> {n} trajectories"
          + (f", {npass}/{known} ({100*npass/known:.0f}%) passed" if known else ""))
    by_agent = Counter(h["agent"] for h in hits)
    for a, c in by_agent.most_common():
        print(f"  {a:22s} {c}")
    print("\nexamples:")
    for h in hits[:args.topk]:
        print(f"  {h['agent']:20s} {h['id']:32s} {lbl(h['resolved'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
