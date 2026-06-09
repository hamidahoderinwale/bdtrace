"""Distillation child-trajectory rollout (Path B: native Modal via mini-swe-agent).

Thin wrapper around mini-swe-agent's maintained SWE-bench runner with its
``swerex_modal`` backend, so each SWE-bench-Verified environment runs on Modal
(native x86 -- no local Docker, no arm64 emulation), driven against our served
SWE-agent-LM-32B sglang endpoint. Captures one ``<instance>.traj.json`` per
instance (mini-swe-agent-1.1 format) -> canonicalizer -> fingerprints_child.jsonl.

Isolated: uses distillation_run/.venv, writes only under distillation_run/.
GPU/Modal env spins up only on --smoke/--full (the served model + per-instance
swerex_modal sandboxes). --dry-run prints the exact command and spends nothing.

Sequence:
  1. python distillation_run/rollout.py --dry-run                 # no GPU: print command
  2. modal serve distillation_run/modal_serve.py                  # serve 32B -> URL (sglang at <URL>/v1)
  3. python distillation_run/rollout.py --smoke --endpoint <URL>  # 3 instances on Modal
  4. python distillation_run/rollout.py --full  --endpoint <URL> --n 75

Prereqs (smoke/full): served endpoint URL; Modal auth (`modal setup`, configured);
mini-swe-agent[modal] + swe-rex in distillation_run/.venv (installed).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / "distillation_run" / ".venv" / "bin"
OUT = ROOT / "distillation_run" / "child_traj"
LABELS = ROOT / "data" / "distillation_child" / "child_results.json"

SMOKE_3 = ["astropy__astropy-13579", "astropy__astropy-14096", "astropy__astropy-14309"]
SERVED_MODEL = "SWE-agent-LM-32B"   # must match sglang --served-model-name on the endpoint


def load_instances(n: int) -> list[str]:
    return json.load(open(LABELS)).get("resolved", [])[:n]


def filter_regex(instance_ids: list[str]) -> str:
    """Exact-match alternation regex for --filter (anchored)."""
    import re
    return "^(" + "|".join(re.escape(i) for i in instance_ids) + ")$"


def build_cmd(instance_ids: list[str], endpoint: str) -> list[str]:
    """The mini-extra swebench invocation: native Modal env + our served model."""
    base = endpoint.rstrip("/")
    api_base = base if base.endswith("/v1") else base + "/v1"
    return [
        str(VENV / "mini-extra"), "swebench",
        "-c", "swebench", "-c", "swebench_modal",                 # base + Modal backend
        "-c", "model.model_class=litellm",
        "-c", f"model.model_name=openai/{SERVED_MODEL}",          # OpenAI-compatible (sglang)
        "-c", f"model.model_kwargs.api_base={api_base}",
        "-c", "model.model_kwargs.api_key=sk-none",
        "--subset", "verified", "--split", "test",
        "--filter", filter_regex(instance_ids),
        "-o", str(OUT),
        "-w", "1",                                                # serial; raise once smoke passes
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="print the command, spend nothing")
    g.add_argument("--smoke", action="store_true", help="run the 3 smoke instances on Modal")
    g.add_argument("--full", action="store_true", help="run --n resolved instances on Modal")
    ap.add_argument("--endpoint", help="served sglang URL (required for --smoke/--full)")
    ap.add_argument("--n", type=int, default=75)
    args = ap.parse_args()

    instances = SMOKE_3 if (args.smoke or args.dry_run) else load_instances(args.n)
    endpoint = args.endpoint or "<ENDPOINT>"
    cmd = build_cmd(instances, endpoint)

    if args.dry_run:
        print("=== DRY RUN (no GPU / no Modal) ===")
        print(f"instances ({len(instances)}): {instances if len(instances) <= 5 else instances[:5] + ['...']}")
        print(f"output dir: {OUT}")
        print("command:\n  " + " ".join(cmd))
        print("\nNotes:\n  - swerex_modal runs each env on Modal (native x86; no local Docker).")
        print("  - SERVED_MODEL must match sglang --served-model-name on the endpoint.")
        print("  - smoke first (3); raise -w / switch to --full only after it passes.")
        return 0

    if not args.endpoint:
        print("--endpoint required for --smoke/--full"); return 2
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"rolling out {len(instances)} instances on Modal against {args.endpoint}")
    print("  " + " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
