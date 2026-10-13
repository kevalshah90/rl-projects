"""Step 9: serve Qwen3.5-9B with vLLM on Modal as an OpenAI-compatible endpoint.

Deploy (stays up until stopped):  uv run modal deploy scripts/s09_serve.py
Stop when done (stops billing):   uv run modal app stop -y kernelbook-vllm
URL: printed by `modal deploy`, ending in ...kernelbook-vllm-serve.modal.run; clients use <URL>/v1
with the API key from the Modal secret `kernelbook-vllm` (also in the local, gitignored .env).

Its own image: vLLM 0.31.0 needs torch 2.13, unrelated to the task image's torch 2.5.1.
"""

import subprocess

import modal

MODEL = "Qwen/Qwen3.5-9B"          # 9.65B dense, hybrid linear attention; bf16 weights ~19.3 GB.
                                 # Was Qwen3.5-4B: p = 0 on all 7 tasks tried in step 9
VLLM_VERSION = "0.31.0"          # latest vLLM (2026-10-05); Qwen3.5 support landed earlier in 2026
MAX_MODEL_LEN = 32768            # tokens per episode: prompt + all turns. The card suggests 128K+ for
                                 # long thinking; 32K keeps memory and latency sane for a pilot.

app = modal.App("kernelbook-vllm")
# vLLM's official image for this version: its tested environment, including the CUDA toolkit.
# A plain pip install on debian_slim failed at warm-up: FlashInfer JIT-compiles its sampling
# kernel and needs nvcc, which that image lacks ("Could not find nvcc", step 9 finding).
# entrypoint([]) clears the image's own `vllm serve` entrypoint so Modal can run our function.
# add_python: Modal couldn't detect the image's Python ("unable to determine the version of
# Python"), so Modal adds its own just to run serve(). Safe: serve() only launches the `vllm`
# command-line program, which keeps using the image's own Python and tested packages.
image = modal.Image.from_registry(f"vllm/vllm-openai:v{VLLM_VERSION}", add_python="3.12").entrypoint([])
# Cache the ~9 GB of weights in a Modal Volume, so restarts don't re-download them.
hf_cache = modal.Volume.from_name("kernelbook-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",                  # fast generation keeps episodes short; agent sandboxes wait on it
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("kernelbook-vllm"),    # VLLM_API_KEY: who may call the server
             modal.Secret.from_name("kernelbook-hf")],     # HF_TOKEN: authenticated, faster HF downloads
    timeout=60 * 60 * 4,         # hard cap: 4 h, even if we forget to stop it
    scaledown_window=10 * 60,    # shut down after 10 min with no requests
    max_containers=1,            # exactly one GPU for the model
)
@modal.concurrent(max_inputs=64)  # many episodes share one server, like a real RL rollout fleet
@modal.web_server(port=8000, startup_timeout=15 * 60)
def serve():
    import os

    # Flags from the Qwen3.5 model card's vLLM section:
    #   --reasoning-parser qwen3        split <think>...</think> out of the reply
    #   --enable-auto-tool-choice --tool-call-parser qwen3_coder   parse the model's tool calls
    #   --language-model-only           skip the vision encoder: more memory for the KV cache
    subprocess.Popen([
        "vllm", "serve", MODEL, "--port", "8000", "--served-model-name", MODEL,
        "--max-model-len", str(MAX_MODEL_LEN), "--language-model-only",
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--api-key", os.environ["VLLM_API_KEY"],   # only clients with the key can call it
    ])
