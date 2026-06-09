"""Modal app: serve SWE-bench/SWE-agent-LM-32B via sglang for child-trajectory collection.

UNTESTED until `modal token new` is run — this is the serving half of the
distillation child-trajectory collection. Cost guardrails are baked in:
single A100, weights cached in a Volume (no re-download per cold start),
scale-to-zero idle timeout, hard per-call timeout, container cap = 1.

Usage (after `pip install modal && modal token new`):

    # one-time: cache the weights into the Volume
    modal run distillation_run/modal_serve.py::cache_weights

    # serve (ephemeral; Ctrl-C tears down) -> prints an OpenAI-compatible URL
    modal serve distillation_run/modal_serve.py

The rollout harness (mini-swe-agent against this endpoint on SWE-bench-Verified
instances, in Modal Sandboxes) is the separate, iteration-prone piece — see
ROLLOUT_PLAN.md. Grading is skipped: child pass/fail labels are already free
in data/distillation_child/child_results.json.
"""

from __future__ import annotations

import modal

MODEL = "SWE-bench/SWE-agent-LM-32B"
GPU = "A100-80GB"  # 33B bf16 ~66 GB; fits one 80 GB card. Right-sized, single GPU.

# --- cost guardrails ---------------------------------------------------------
IDLE_TIMEOUT = 120     # scale-to-zero after 2 min idle
CALL_TIMEOUT = 1800    # hard 30 min ceiling per serve container life-extension
MAX_CONTAINERS = 1     # never fan out GPUs

weights = modal.Volume.from_name("swe-agent-lm-32b-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"

# Light image for the download-only step (no CUDA / nvcc needed).
download_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub>=0.24", "hf_transfer>=0.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
# Serving uses the prebuilt sglang image (CUDA + sglang already compiled) —
# the same image the model card recommends. Avoids building flashinfer/nvcc.
serve_image = modal.Image.from_registry("lmsysorg/sglang:latest")

app = modal.App("swe-agent-lm-32b-serve")


@app.function(image=download_image, volumes={WEIGHTS_DIR: weights}, timeout=3600)
def cache_weights() -> None:
    """One-time: download the 33B weights into the Volume so serve cold-starts are fast."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL, local_dir=f"{WEIGHTS_DIR}/SWE-agent-LM-32B")
    weights.commit()
    print("weights cached to Volume")


@app.function(
    image=serve_image,
    gpu=GPU,
    volumes={WEIGHTS_DIR: weights},
    scaledown_window=IDLE_TIMEOUT,
    max_containers=MAX_CONTAINERS,
    timeout=CALL_TIMEOUT,
)
@modal.web_server(port=30000, startup_timeout=600)
def serve() -> None:
    """Launch an OpenAI-compatible sglang server for the cached 32B model."""
    import subprocess

    subprocess.Popen(
        [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            f"{WEIGHTS_DIR}/SWE-agent-LM-32B",
            "--served-model-name",
            "SWE-agent-LM-32B",  # must match rollout.py SERVED_MODEL (openai/SWE-agent-LM-32B)
            "--host",
            "0.0.0.0",
            "--port",
            "30000",
            "--tp",
            "1",
            "--enable-metrics",
        ]
    )
