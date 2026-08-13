"""Distillation fingerprints from PUBLIC trajectories (no GPU, no rollout).

Child  = SWE-agent-LM-32B trajs, public at s3://swe-bench-submissions/verified/
         20250511_sweagent_lm_32b/trajs/ (anon https). Canonicalized with the
         same canonicalize_envelope that built the parent corpus.
Parent = Claude-3.7-thinking (the teacher: SWE-agent + Claude-3.7 Sonnet), n=284,
         already in output/paper2_pilot/bpe_sequences_extended.jsonl on the SAME
         SWE-bench Verified tasks -> same-task teacher-vs-student comparison.

Emits distillation_run/fingerprints_{parent,child}.jsonl with schema
  {trace_id, role, outcome, canonical:[coarse atoms], native:[fine atoms]}
then runs make_distillation_panels.py.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.preferences.canonicalize_extended import canonicalize_envelope  # noqa: E402

TRAJ_DIR = ROOT / "distillation_run" / "child_traj"
DATA = ROOT / "distillation_run"
S3 = "https://swe-bench-submissions.s3.amazonaws.com"
PREFIX = "verified/20250511_sweagent_lm_32b/trajs/"
CHILD_LABELS = ROOT / "data" / "distillation_child" / "child_results.json"
CORPUS = ROOT / "output" / "paper2_pilot" / "bpe_sequences_extended.jsonl"
PASSFAIL = ROOT / "output" / "paper2_pilot" / "extended_pass_fail.json"
PARENT_AGENT = "Claude-3.7-thinking"


def to_canon(a: str) -> str:
    if a.startswith("EDIT"): return "edit"
    if a.startswith("CREATE"): return "create_file"
    if a.startswith("RUN"): return "run_test"
    if a.startswith(("OPEN", "NAV")): return "read_file"
    if a.startswith(("SEARCH", "FIND")): return "search_repo"
    if "SUBMIT" in a: return "submit"
    return "other"


def list_child_keys() -> list[str]:
    keys, token = [], None
    while True:
        url = f"{S3}/?list-type=2&prefix={PREFIX}&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token, safe="")
        xml = urllib.request.urlopen(url, timeout=60).read().decode()
        keys += [k for k in re.findall(r"<Key>(.*?)</Key>", xml) if k.endswith(".traj")]
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if "<IsTruncated>true</IsTruncated>" in xml and m:
            token = m.group(1)
        else:
            break
    return keys


def download_child() -> None:
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    keys = list_child_keys()
    print(f"child trajs listed: {len(keys)}", flush=True)
    for i, key in enumerate(keys):
        dst = TRAJ_DIR / Path(key).name
        if dst.exists() and dst.stat().st_size > 0:
            continue
        try:
            urllib.request.urlretrieve(f"{S3}/{urllib.parse.quote(key)}", dst)
        except Exception as e:
            print(f"  [dl error] {key}: {e}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"  downloaded {i + 1}/{len(keys)}", flush=True)


def build_child() -> int:
    labels = json.load(open(CHILD_LABELS))
    resolved = set(labels.get("resolved", []))
    out = DATA / "fingerprints_child.jsonl"
    n = 0
    with out.open("w") as f:
        for traj in sorted(TRAJ_DIR.glob("*.traj")):
            iid = traj.stem
            try:
                raw = json.loads(traj.read_text())
                native = canonicalize_envelope({"format": "sweagent_traj_subdir", "content": raw})
            except Exception as e:
                print(f"  [canon error] {iid}: {e}", flush=True); continue
            if not native:
                continue
            f.write(json.dumps({
                "trace_id": iid, "role": "child",
                "outcome": "resolved" if iid in resolved else "unresolved",
                "canonical": [to_canon(a) for a in native], "native": native,
            }) + "\n")
            n += 1
    print(f"wrote fingerprints_child.jsonl ({n} trajectories)", flush=True)
    return n


def build_parent() -> int:
    pf = json.load(open(PASSFAIL))
    res = {k: set(v.get("resolved", [])) for k, v in pf.items()}
    out = DATA / "fingerprints_parent.jsonl"
    n = 0
    with out.open("w") as f:
        for line in open(CORPUS):
            r = json.loads(line)
            if r["agent"] != PARENT_AGENT:
                continue
            native = r["canonical"]            # corpus 'canonical' field = fine-grained atoms
            resolved = r["instance_id"] in res.get(r["submission"], set())
            f.write(json.dumps({
                "trace_id": r["instance_id"], "role": "parent",
                "outcome": "resolved" if resolved else "unresolved",
                "canonical": [to_canon(a) for a in native], "native": native,
            }) + "\n")
            n += 1
    print(f"wrote fingerprints_parent.jsonl ({n} trajectories, agent={PARENT_AGENT})", flush=True)
    return n


def main() -> int:
    print("=== 1. download child trajs (public S3, no GPU) ===", flush=True)
    download_child()
    print("=== 2. canonicalize child ===", flush=True)
    nc = build_child()
    print("=== 3. parent from corpus ===", flush=True)
    npar = build_parent()
    if nc == 0 or npar == 0:
        print("ABORT: empty fingerprints"); return 1
    print("=== 4. render panels ===", flush=True)
    import subprocess
    rc = subprocess.call([sys.executable, str(ROOT / "distillation_run" / "make_distillation_panels.py")])
    print(f"panels exit={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
