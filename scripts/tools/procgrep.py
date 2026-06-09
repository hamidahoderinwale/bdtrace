"""procgrep — procedural-grep prototype: stuck-edit-loop detector.

Operationalizes the §6.4 finding: long Type B failures show
EDIT_SRC_PY +0.053 / SUBMIT -0.063 across 7 scaffolds. This CLI takes a
SWE-agent trajectory and reports whether the stuck-edit-loop signature
is present.

Detection rule (single-trajectory):
  flag stuck_edit_loop = 1 if
    (post_localization_steps >= POSTLOC_THRESH) AND
    (EDIT_SRC_PY share >= EDIT_SHARE_THRESH) AND
    (SUBMIT share <= SUBMIT_SHARE_THRESH)
  else 0

Default thresholds chosen from the §6.4 long-Type-B signature numbers; see
`output/paper2_pilot/r10_postloc_motifs_extended.json` for the source data.

Usage:
    # single trajectory
    uv run python scripts/tools/procgrep.py path/to/trajectory.json
    # whole submission directory
    uv run python scripts/tools/procgrep.py output/trajectories/.cache/<submission>/

Outputs JSONL to stdout, one record per trajectory:
    {"path": "...", "instance_id": "...", "agent": "...",
     "n_atoms": int, "postloc_steps": int,
     "edit_src_share": float, "submit_share": float,
     "flag_stuck_edit_loop": 0|1, "verdict": "stuck-edit-loop"|"normal"|"not-applicable"}
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.preferences.canonicalize_extended import canonicalize_envelope

# --- Defaults from §6.4 long-Type-B signature ---
POSTLOC_THRESH        = 14       # median postloc-step boundary from R10
EDIT_SHARE_THRESH     = 0.30     # ~+0.053 above corpus mean (~0.25)
SUBMIT_SHARE_THRESH   = 0.05     # ~-0.063 below corpus mean (~0.10)


def first_localize_index(atoms: list[str]) -> int | None:
    """Earliest index where the agent appears to have localized: any EDIT_*
    or RUN_* atom signals localization is over and post-localization began.
    Returns None if no localization step found.
    """
    for i, a in enumerate(atoms):
        if a.startswith("EDIT_") or a.startswith("RUN_"):
            return i
    return None


def analyze(atoms: list[str]) -> dict:
    if not atoms:
        return {
            "n_atoms": 0,
            "postloc_steps": 0,
            "edit_src_share": 0.0,
            "submit_share": 0.0,
            "flag_stuck_edit_loop": 0,
            "verdict": "not-applicable",
        }
    n = len(atoms)
    loc_idx = first_localize_index(atoms)
    if loc_idx is None:
        return {
            "n_atoms": n,
            "postloc_steps": 0,
            "edit_src_share": 0.0,
            "submit_share": 0.0,
            "flag_stuck_edit_loop": 0,
            "verdict": "no-localization",
        }
    postloc = atoms[loc_idx:]
    postloc_n = len(postloc)
    cnt = Counter(postloc)
    edit_share = cnt.get("EDIT_SRC_PY", 0) / max(postloc_n, 1)
    submit_share = cnt.get("SUBMIT", 0) / max(postloc_n, 1)
    flag = int(
        postloc_n >= POSTLOC_THRESH
        and edit_share >= EDIT_SHARE_THRESH
        and submit_share <= SUBMIT_SHARE_THRESH
    )
    if flag:
        verdict = "stuck-edit-loop"
    elif postloc_n < POSTLOC_THRESH:
        verdict = "short-trajectory"
    else:
        verdict = "normal"
    return {
        "n_atoms": n,
        "postloc_steps": postloc_n,
        "edit_src_share": round(edit_share, 4),
        "submit_share": round(submit_share, 4),
        "flag_stuck_edit_loop": flag,
        "verdict": verdict,
    }


def process_one(path: Path) -> dict:
    try:
        envelope = json.loads(path.read_text())
    except Exception as e:
        return {"path": str(path), "error": str(e)[:80]}
    atoms = canonicalize_envelope(envelope)
    record = analyze(atoms)
    record["path"] = str(path)
    record["instance_id"] = envelope.get("instance_id") or path.stem
    record["agent"] = envelope.get("submission") or envelope.get("agent") or path.parent.name
    return record


def main(argv: list[str] | None = None) -> int:
    global POSTLOC_THRESH, EDIT_SHARE_THRESH, SUBMIT_SHARE_THRESH
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", help="Trajectory JSON file or directory of them")
    ap.add_argument("--postloc-thresh", type=int, default=POSTLOC_THRESH)
    ap.add_argument("--edit-share-thresh", type=float, default=EDIT_SHARE_THRESH)
    ap.add_argument("--submit-share-thresh", type=float, default=SUBMIT_SHARE_THRESH)
    ap.add_argument("--summary", action="store_true",
                    help="emit a one-line summary instead of per-trajectory JSONL")
    args = ap.parse_args(argv)

    POSTLOC_THRESH = args.postloc_thresh
    EDIT_SHARE_THRESH = args.edit_share_thresh
    SUBMIT_SHARE_THRESH = args.submit_share_thresh

    p = Path(args.path)
    if p.is_file():
        paths = [p]
    elif p.is_dir():
        paths = sorted(p.glob("*.json"))
        paths = [pp for pp in paths if pp.name not in ("manifest.json",)]
    else:
        print(f"path not found: {p}", file=sys.stderr)
        return 2

    n_total = 0
    n_flagged = 0
    n_normal = 0
    n_short = 0
    n_no_loc = 0
    for pp in paths:
        rec = process_one(pp)
        if "error" in rec:
            continue
        n_total += 1
        if rec["flag_stuck_edit_loop"] == 1:
            n_flagged += 1
        elif rec["verdict"] == "normal":
            n_normal += 1
        elif rec["verdict"] == "short-trajectory":
            n_short += 1
        elif rec["verdict"] == "no-localization":
            n_no_loc += 1
        if not args.summary:
            print(json.dumps(rec, default=str))

    if args.summary or len(paths) > 1:
        print(json.dumps({
            "n_total": n_total,
            "n_stuck_edit_loop": n_flagged,
            "n_normal": n_normal,
            "n_short": n_short,
            "n_no_localization": n_no_loc,
            "stuck_edit_loop_rate": round(n_flagged / max(n_total, 1), 4),
            "thresholds": {
                "postloc_thresh": POSTLOC_THRESH,
                "edit_share_thresh": EDIT_SHARE_THRESH,
                "submit_share_thresh": SUBMIT_SHARE_THRESH,
            },
        }, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
