# Distillation child-trajectory collection — plan

Goal: generate child (SWE-agent-LM-32B) trajectories to pair with the free
parent trajectories + free child labels, then produce the distillation panels.

## What's already free / done
- **Child pass/fail labels**: `data/distillation_child/child_results.json` — 201/500 resolved (40.2%). No need to grade.
- **Parent trajectories**: `SWE-bench/SWE-smith-trajectories` on HF (public) — sample ~75 to match child instances.

## Harness status (2026-06-08) — Path B (native Modal)
- **`rollout.py` rewritten as a thin wrapper over mini-swe-agent's maintained `mini-extra swebench` runner with its `swerex_modal` backend.** Each SWE-bench env runs on Modal (native x86 — no local Docker, no arm64 emulation). `--dry-run` validated.
- **Why Path B:** local route was blocked (Docker daemon off + arm64 host vs x86 SWE-bench images = qemu emulation). The native Modal backend sidesteps both.
- **Deps installed** in `distillation_run/.venv`: `mini-swe-agent[modal]` + `swe-rex` (`swerex_modal` imports OK), `modal`, `datasets`.
- **Consistency fix applied:** `modal_serve.py` now passes `--served-model-name SWE-agent-LM-32B` so it matches `rollout.py`'s `openai/SWE-agent-LM-32B`.
- **Remaining unknowns to confirm on smoke-3:** (a) sglang served-model-name actually resolves via litellm `openai/` + `api_base`; (b) `swerex_modal` per-instance image build time on first run (config allows 10 min startup); (c) where mini-swe-agent writes each `.traj.json` under `-o` (expected `<out>/<instance>.traj.json`).

## Fire sequence (user authorizes GPU/Modal spend)
1. **Serve** the 32B: `modal serve distillation_run/modal_serve.py` → prints URL (sglang at `<URL>/v1`), weights from Volume.
2. **Dry-run** (free): `distillation_run/.venv/bin/python distillation_run/rollout.py --dry-run` → prints the exact `mini-extra swebench` command.
3. **Smoke-3**: `... rollout.py --smoke --endpoint <URL>` → 3 envs on Modal, verify `.traj.json` land in `distillation_run/child_traj/`. Debug the 3 unknowns here.
4. **Scale**: `... rollout.py --full --endpoint <URL> --n 75` (raise `-w` once smoke passes).
5. Tear down: Ctrl-C the `modal serve` (scale-to-zero also handles idle).
   - Skip grading (labels already free) — only the action sequence is needed.
   - **Validate on smoke-3 before scaling.** Two Modal apps run: `swe-agent-lm-32b-serve` (GPU model) + the ephemeral `swerex_modal` per-instance sandboxes (CPU).

## Free / local (no Modal) — can run in parallel
3. **Parent fingerprints**: pull ~75 rows of `SWE-smith-trajectories`,
   canonicalize via the `mini-swe-agent` adapter → `fingerprints_parent.jsonl`.
4. **Child fingerprints**: canonicalize the collected `.traj.json` → join
   `child_results.json` labels → `fingerprints_child.jsonl`.
5. **Analysis**: `lineage_diff(parent, child)` across the four axes
   (vocabulary Jaccard, entropy shift, outcome-stratified overlap, conditional
   JSD) → the A/B/C panels in Altair → `docs/papers/figures/fig_distillation_*`.

## Cost guardrails (baked into modal_serve.py)
single A100, Volume-cached weights, scale-to-zero (120 s idle), container cap
= 1, hard per-call timeout. Smoke-3 before the ~75 run. Ephemeral `modal serve`
+ explicit teardown (Ctrl-C). Estimated ~$5–15 + a couple hours incl. debugging.
