"""Fetch trajectories for extended-corpus SWE-bench submissions.

Phase 1 of the cross-scaffold/cross-paradigm corpus extension. Reads
metadata.yaml for each submission to get the canonical S3 path,
discovers the per-instance file convention via S3 listing, and fetches
all instances into output/trajectories/.cache/<submission>/.

Per-submission file conventions discovered earlier in this session:
    sweagent_claude-3-7-sonnet:  trajs/{iid}/{iid}.traj
    dars_agent + R1:             trajs/{iid}.traj
    agentless-1.5 × Claude-3.5:  trajs/{iid}.log
    moatless + V3:               trajs/{iid}/trajectory.json
    SWE-Fixer + Qwen:            trajs/edit_trajs.jsonl (single file)

Per-instance content is saved as <instance_id>.json with a small
envelope so downstream canonicalizers can read a consistent format:
    {"submission": str, "instance_id": str, "format": str, "content": ...}

Reads:
    GitHub repos/SWE-bench/experiments/contents/evaluation/lite/<sub>/metadata.yaml
    S3 swe-bench-submissions.s3.amazonaws.com (public)
Writes:
    output/trajectories/.cache/<submission>/<instance_id>.json (one file each)
    output/trajectories/.cache/<submission>/manifest.json (per-fetch summary)
"""
from __future__ import annotations
import base64
import json
import sys
import time
import urllib.request as ur
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[2]
CACHE = ROOT / "output" / "trajectories" / ".cache"
S3    = "https://swe-bench-submissions.s3.amazonaws.com"
GH    = "https://api.github.com/repos/SWE-bench/experiments/contents/evaluation/lite"
NS    = "{http://s3.amazonaws.com/doc/2006-03-01/}"

SUBMISSIONS = [
    "20250226_sweagent_claude-3-7-sonnet-20250219",
    "20250205_dars_agent_claude_3.5_sonnet_deepseek_r1",
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022",
    "20250111_moatless_deepseek_v3",
    "20241128_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor_20241128",
]


def get_metadata(submission: str) -> dict:
    url = f"{GH}/{submission}/metadata.yaml"
    with ur.urlopen(url, timeout=15) as r:
        d = json.load(r)
    text = base64.b64decode(d["content"]).decode()
    out = {}
    for line in text.splitlines():
        ls = line.strip()
        for k in ("logs", "trajs", "site", "os_model"):
            if ls.startswith(f"{k}:"):
                out[k] = ls.split(":", 1)[1].strip()
    return out


def list_s3(prefix: str, max_keys: int = 1000, retries: int = 3) -> list[str]:
    url = f"{S3}/?prefix={prefix}&max-keys={max_keys}"
    for attempt in range(retries):
        try:
            with ur.urlopen(url, timeout=20) as r:
                xml = r.read()
            root = ET.fromstring(xml)
            keys = [c.find(NS + "Key").text for c in root.findall(NS + "Contents")]
            return keys
        except Exception:
            if attempt < retries - 1:
                time.sleep(1 + attempt)
                continue
            return []
    return []


def detect_pattern(submission: str, keys: list[str]) -> tuple[str, str]:
    """Return (pattern_label, sample_key)."""
    if any(k.endswith("/edit_trajs.jsonl") for k in keys):
        return "single_file_jsonl", next(k for k in keys if k.endswith("/edit_trajs.jsonl"))
    sub_keys = [k for k in keys if k.endswith(".traj") or k.endswith(".log") or k.endswith(".json")]
    if any("/trajectory.json" in k for k in sub_keys):
        return "instance_dir_json", "trajectory.json"
    if any(k.endswith(f".traj") and "/" in k.replace(f"lite/{submission}/trajs/", "")
           for k in sub_keys):
        return "instance_dir_traj", ".traj"
    if any(k.endswith(".log") for k in sub_keys):
        return "instance_log", ".log"
    if any(k.endswith(".traj") for k in sub_keys):
        return "instance_traj", ".traj"
    return "unknown", ""


def fetch_url(url: str, timeout: int = 30, retries: int = 3) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            with ur.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            if attempt < retries - 1:
                time.sleep(1 + attempt)
                continue
    raise RuntimeError(f"fetch failed: {url}: {last}")


def fetch_submission(submission: str) -> dict:
    sub_dir = CACHE / submission
    sub_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {submission} ===")
    md = get_metadata(submission)
    trajs_val = (md.get("trajs") or "").replace("s3://swe-bench-submissions/", "")
    logs_val  = (md.get("logs")  or "").replace("s3://swe-bench-submissions/", "")
    candidates = [v for v in (trajs_val, logs_val) if v and v != "null"]
    trajs_path = next((v for v in candidates if v.rstrip("/").endswith("/trajs")), None)
    if not trajs_path:
        trajs_path = candidates[0] if candidates else ""
    if not trajs_path:
        print(f"  no trajs/logs path in metadata; skipping")
        return {"submission": submission, "skipped": "no trajs path", "n_fetched": 0}
    print(f"  trajs path: {trajs_path}")

    prefix = trajs_path.rstrip("/") + "/"
    keys = list_s3(prefix)
    print(f"  S3 listing: {len(keys)} keys")
    if not keys:
        return {"submission": submission, "skipped": "empty S3 listing", "n_fetched": 0}
    pattern, _ = detect_pattern(submission, keys)
    print(f"  pattern: {pattern}")

    if pattern == "single_file_jsonl":
        url = f"{S3}/{prefix}edit_trajs.jsonl"
        body = fetch_url(url)
        n = 0
        for line in body.decode("utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                iid = rec.get("instance_id")
                if not iid:
                    continue
                out = {"submission": submission, "instance_id": iid,
                       "format": "swefixer_jsonl", "content": rec}
                (sub_dir / f"{iid}.json").write_text(json.dumps(out))
                n += 1
            except Exception:
                continue
        print(f"  fetched {n} instances from edit_trajs.jsonl")
        manifest = {"submission": submission, "pattern": pattern, "n_fetched": n,
                    "source_url": url}
        (sub_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest

    if pattern == "instance_dir_json":
        instances = sorted({
            k.replace(prefix, "").split("/")[0]
            for k in keys
            if "/trajectory.json" in k
        })
    elif pattern == "instance_dir_traj":
        instances = sorted({
            k.replace(prefix, "").split("/")[0]
            for k in keys
            if k.endswith(".traj")
        })
    elif pattern == "instance_log":
        instances = sorted({
            k.replace(prefix, "").rsplit(".log", 1)[0]
            for k in keys
            if k.endswith(".log")
        })
    elif pattern == "instance_traj":
        instances = sorted({
            k.replace(prefix, "").rsplit(".traj", 1)[0]
            for k in keys
            if k.endswith(".traj")
        })
    else:
        return {"submission": submission, "skipped": f"unknown pattern: {pattern}", "n_fetched": 0}

    print(f"  {len(instances)} instances inferred")

    def fetch_one(iid: str) -> tuple[str, bool, str]:
        out_path = sub_dir / f"{iid}.json"
        if out_path.exists():
            return iid, True, "cached"
        if pattern == "instance_dir_json":
            url = f"{S3}/{prefix}{iid}/trajectory.json"
            fmt = "moatless_trajectory_json"
        elif pattern == "instance_dir_traj":
            url = f"{S3}/{prefix}{iid}/{iid}.traj"
            fmt = "sweagent_traj_subdir"
        elif pattern == "instance_log":
            url = f"{S3}/{prefix}{iid}.log"
            fmt = "agentless_log_text"
        elif pattern == "instance_traj":
            url = f"{S3}/{prefix}{iid}.traj"
            fmt = "dars_traj_list"
        else:
            return iid, False, "unknown pattern"
        try:
            body = fetch_url(url, retries=2)
            try:
                content = json.loads(body)
            except Exception:
                content = body.decode("utf-8", errors="ignore")
            envelope = {"submission": submission, "instance_id": iid,
                        "format": fmt, "content": content}
            out_path.write_text(json.dumps(envelope))
            return iid, True, "fetched"
        except Exception as e:
            return iid, False, str(e)[:80]

    n_ok, n_fail = 0, 0
    failures = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(fetch_one, iid) for iid in instances]
        for f in as_completed(futures):
            iid, ok, info = f.result()
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                failures.append({"instance_id": iid, "info": info})
    print(f"  fetched: {n_ok}/{len(instances)} ({n_fail} failures)")

    manifest = {
        "submission": submission, "pattern": pattern, "n_instances": len(instances),
        "n_fetched": n_ok, "n_failures": n_fail, "failures": failures[:20],
        "trajs_path": trajs_path, "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (sub_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    summaries = []
    for sub in SUBMISSIONS:
        try:
            m = fetch_submission(sub)
            summaries.append(m)
        except Exception as e:
            print(f"  ERROR on {sub}: {e}")
            summaries.append({"submission": sub, "error": str(e)[:200]})

    print("\n=== Summary ===")
    for m in summaries:
        sub = m["submission"]
        if "error" in m:
            print(f"  {sub}  ERROR: {m['error'][:80]}")
        elif m.get("skipped"):
            print(f"  {sub}  skipped: {m['skipped']}")
        else:
            print(f"  {sub}  n={m.get('n_fetched',0)}/{m.get('n_instances','?')}  pattern={m.get('pattern','?')}")


if __name__ == "__main__":
    main()
