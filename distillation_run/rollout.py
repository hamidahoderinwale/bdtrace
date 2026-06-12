"""Steering-experiment rollout driver (mini-swe-agent v2.3.0).

Runs one model on a fixed SWE-bench-Verified subset under a chosen procedural
condition (baseline / test_driven / patch_driven), via mini-swe-agent's v2
`mini-extra swebench` runner with the `swebench_modal` backend (per-instance
envs on Modal; no local Docker). Each condition's `.traj` go to
child_traj/<condition>/, then are scored with intervention/score_intervention.py.

Providers:
  --provider openrouter : model = $ROLLOUT_MODEL (an OpenRouter id); needs
                          OPENROUTER_API_KEY in the env (run via run_smoke_api.sh,
                          which sources intervention/.env).
  --provider selfhost   : model = openai/SWE-agent-LM-32B against a served sglang
                          endpoint (--endpoint); use modal_serve.py to serve.

--dry-run prints the exact command and spends nothing. --smoke = 3 instances;
--full = --n resolved instances. Serial (-w 1) until smoke passes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "distillation_run"
VENV = HERE / ".venv" / "bin"
CONF = HERE / "intervention" / "conf"
OUT = HERE / "child_traj"
LABELS = ROOT / "data" / "distillation_child" / "child_results.json"

SMOKE_3 = ["astropy__astropy-13579", "astropy__astropy-14096", "astropy__astropy-14309"]
SELFHOST_MODEL = "SWE-agent-LM-32B"  # must match sglang --served-model-name
CONDITIONS = ("baseline", "test_driven", "patch_driven")


def load_instances(n: int) -> list[str]:
    return json.load(open(LABELS)).get("resolved", [])[:n]


def filter_regex(ids: list[str]) -> str:
    return "^(" + "|".join(re.escape(i) for i in ids) + ")$"


def build_cmd(condition: str, ids: list[str], provider: str, model: str, endpoint: str) -> list[str]:
    cond_cfg = CONF / f"{condition}.yaml"
    cmd = [
        str(VENV / "mini-extra"), "swebench",
        "-c", "swebench.yaml", "-c", "swebench_modal.yaml", "-c", str(cond_cfg),
    ]
    if provider == "openrouter":
        # Use the dedicated --model-class flag, not -c model.model_class=. The bundled
        # swebench_modal.yaml ships model_class: portkey, and a -c key=value override did
        # not beat the merged file config (the v2.3.0 failure was "Unknown model class:
        # portkey"). The typer flag sits at the highest-precedence config layer and wins.
        # litellm routes the openrouter/ prefix via OPENROUTER_API_KEY in the env; the
        # leftover portkey-only keys (provider) are ignored by LitellmModelConfig.
        cmd += ["-m", model, "--model-class", "litellm"]
    else:  # selfhost
        base = endpoint.rstrip("/")
        api_base = base if base.endswith("/v1") else base + "/v1"
        # -m + --model-class take precedence over swebench_modal.yaml; api_base/api_key
        # merge into model_kwargs. Selfhost path is unverified pending an sglang endpoint.
        cmd += [
            "-m", f"openai/{SELFHOST_MODEL}", "--model-class", "litellm",
            "-c", f"model.model_kwargs.api_base={api_base}",
            "-c", "model.model_kwargs.api_key=sk-none",
        ]
    cmd += [
        "--subset", "verified", "--split", "test",
        "--filter", filter_regex(ids),
        "-o", str(OUT / condition),
        "-w", "1",
    ]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the command, spend nothing")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="3 smoke instances")
    g.add_argument("--full", action="store_true", help="--n resolved instances")
    ap.add_argument("--condition", choices=CONDITIONS, default="baseline")
    ap.add_argument("--provider", choices=("openrouter", "selfhost"), default="openrouter")
    ap.add_argument("--endpoint", help="served sglang URL (selfhost only)")
    ap.add_argument("--model", default=os.environ.get("ROLLOUT_MODEL", "openrouter/qwen/qwen-2.5-coder-32b-instruct"))
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    if not (CONF / f"{args.condition}.yaml").exists():
        print(f"missing {CONF / (args.condition + '.yaml')} -- run intervention/make_condition_configs.py first")
        return 2
    if args.provider == "selfhost" and not (args.endpoint or args.dry_run):
        print("--endpoint required for --provider selfhost (except --dry-run)"); return 2

    ids = SMOKE_3 if args.smoke else load_instances(args.n)
    cmd = build_cmd(args.condition, ids, args.provider, args.model, args.endpoint or "<ENDPOINT>")

    if args.dry_run:
        print("=== DRY RUN (no spend) ===")
        print(f"condition: {args.condition} | provider: {args.provider} | model: {args.model}")
        print(f"instances ({len(ids)}): {ids if len(ids) <= 5 else ids[:5] + ['...']}")
        print(f"output: {OUT / args.condition}")
        print("command:\n  " + " ".join(cmd))
        if args.provider == "openrouter":
            print("\nnote: OPENROUTER_API_KEY must be in env (run via run_smoke_api.sh).")
        return 0

    (OUT / args.condition).mkdir(parents=True, exist_ok=True)
    print(f"[{args.condition}/{args.provider}] {len(ids)} instances -> {OUT / args.condition}")
    print("  " + " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
