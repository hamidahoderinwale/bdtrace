"""Ground (or falsify) the stuck-reading in-task-monitoring claim from public data.

Uses the 499 public SWE-agent-LM-32B child trajectories + 284 parent trajectories
(no GPU rollout needed). Computes, for the parent-pass/child-fail set:
  - read-cycle prevalence (longest run of consecutive read_file w/o an edit)
  - onset step of the stuck-read loop
  - whether the child runs to exhaustion without ever editing
  - trajectory length distribution (tests the '150+ steps' claim)
  - aggregate tokens/step for stuck vs non-stuck child (proxy; per-step tokens
    are NOT stored, so the per-cycle token claim is reported as an estimate only)
Run from repo root.
"""
import json
import glob
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(".")
DR = ROOT / "distillation_run"
child = [json.loads(l) for l in open(DR / "fingerprints_child.jsonl")]
parent = [json.loads(l) for l in open(DR / "fingerprints_parent.jsonl")]

EDIT = {"edit", "create_file"}
READ = {"read_file"}


def max_read_run(seq):
    """Longest run of consecutive read_file with no edit interrupting; and its start index."""
    best, best_start = 0, None
    cur, start = 0, None
    for i, a in enumerate(seq):
        if a in READ:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best, best_start = cur, start
        elif a in EDIT:
            cur = 0
        # other/search/run/submit don't extend but don't break a read-dominated stretch hard;
        # keep strict: only edit resets. (search between reads still counts as interruption of the *run*)
        else:
            cur = 0
    return best, best_start


def first_edit(seq):
    for i, a in enumerate(seq):
        if a in EDIT:
            return i
    return None


# instance -> outcome
c_out = {r["trace_id"]: r["outcome"] for r in child}
p_out = {r["trace_id"]: r["outcome"] for r in parent}
c_seq = {r["trace_id"]: r["canonical"] for r in child}

pp_cf = [i for i in c_out if i in p_out and p_out[i] == "resolved" and c_out[i] == "unresolved"]
print(f"parent trajectories: {len(parent)}, child: {len(child)}")
print(f"shared instances: {len(set(c_out)&set(p_out))}")
print(f"parent-pass / child-fail instances: {len(pp_cf)}")
print()

K = 4  # threshold for a "stuck read run"
fires, onsets, no_edit, lengths = 0, [], 0, []
for i in pp_cf:
    s = c_seq[i]
    lengths.append(len(s))
    run, start = max_read_run(s)
    fe = first_edit(s)
    if run >= K:
        fires += 1
        onsets.append(start)
    if fe is None:
        no_edit += 1

print(f"=== Stuck-reading on parent-pass/child-fail (n={len(pp_cf)}), read-run threshold K={K} ===")
print(f"fires (read-run >= {K}): {fires}/{len(pp_cf)} = {fires/len(pp_cf):.0%}")
if onsets:
    onsets = np.array(onsets)
    print(f"onset step: median {np.median(onsets):.0f}, "
          f"<=12 in {(onsets<=12).mean():.0%}, <=8 in {(onsets<=8).mean():.0%}")
print(f"child never edits at all: {no_edit}/{len(pp_cf)} = {no_edit/len(pp_cf):.0%}")
lengths = np.array(lengths)
print(f"child length: median {np.median(lengths):.0f}, max {lengths.max()}, "
      f">=150: {(lengths>=150).sum()}, >=75: {(lengths>=75).sum()}")
print()

# sensitivity across K
print("sensitivity (fires / n):")
for k in (3, 4, 5, 6, 8):
    f = sum(1 for i in pp_cf if max_read_run(c_seq[i])[0] >= k)
    print(f"  K={k}: {f}/{len(pp_cf)} = {f/len(pp_cf):.0%}")
print()

# token proxy: aggregate tokens/step from raw trajs, stuck vs non-stuck child fails
def load_stats(iid):
    p = DR / "child_traj" / f"{iid}.traj"
    if not p.exists():
        return None
    o = json.load(open(p))
    ms = o.get("info", {}).get("model_stats", {})
    n = ms.get("api_calls") or (len(o.get("trajectory", [])) or 1)
    ts = ms.get("tokens_sent", 0)
    return ts / n if n else None

stuck = [i for i in pp_cf if max_read_run(c_seq[i])[0] >= K]
notstuck = [i for i in pp_cf if max_read_run(c_seq[i])[0] < K]
ts_stuck = [v for v in (load_stats(i) for i in stuck) if v]
ts_not = [v for v in (load_stats(i) for i in notstuck) if v]
print("=== aggregate tokens_sent / step (PROXY; per-step tokens not stored) ===")
if ts_stuck:
    print(f"stuck-reading child fails:   {np.mean(ts_stuck):.0f} tok/step (n={len(ts_stuck)})")
if ts_not:
    print(f"non-stuck child fails:       {np.mean(ts_not):.0f} tok/step (n={len(ts_not)})")
print("NOTE: this is mean context-sent per API call, not per-read-cycle cost; the")
print("'835 vs 210 per cycle' figure is NOT reconstructible from aggregate telemetry.")
