"""Step 9 check: is the vLLM server up, and does Qwen3.5-9B make a parsed tool call?

Run from the project root (after `modal deploy scripts/s09_serve.py`):
    uv run python scripts/s09_ping.py
The first request starts the server's container (cold start: weights download + load,
several minutes), so this waits on /health before sending the real request.

Sends one chat request that offers a `check` tool (same name and shape as our MCP tool)
and asks the model to use it. Passes when the reply carries a structured tool call,
i.e. vLLM's qwen3_coder parser understood the model's tool-call format.
"""

import os
import time
from pathlib import Path

import httpx
from openai import OpenAI

MODEL = "Qwen/Qwen3.5-9B"
URL = "https://kevalshah90--kernelbook-vllm-serve.modal.run"
# The key lives in the gitignored .env (VLLM_API_KEY=...), the same value as the Modal secret.
KEY = os.environ.get("VLLM_API_KEY") or next(
    line.split("=", 1)[1].strip() for line in Path(".env").read_text().splitlines()
    if line.startswith("VLLM_API_KEY="))

CHECK_TOOL = {"type": "function", "function": {
    "name": "check",
    "description": "Test /workspace/solution.py exactly like the final grader does.",
    "parameters": {"type": "object", "properties": {}}}}


def wait_for_health(timeout_s: int = 20 * 60) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            if httpx.get(f"{URL}/health", timeout=60).status_code == 200:
                print(f"healthy after {time.time() - start:.0f} s")
                return
        except httpx.HTTPError:
            pass                                    # container still starting
        time.sleep(15)
    raise TimeoutError("server not healthy after 20 min: check `modal app logs kernelbook-vllm`")


def main() -> None:
    wait_for_health()
    client = OpenAI(base_url=f"{URL}/v1", api_key=KEY)
    t = time.time()
    reply = client.chat.completions.create(
        model=MODEL, tools=[CHECK_TOOL], max_tokens=1024, temperature=1.0,
        messages=[{"role": "user", "content":
                   "I just wrote /workspace/solution.py. Use the check tool to test it."}])
    msg = reply.choices[0].message
    calls = [c.function.name for c in (msg.tool_calls or [])]
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    print(f"reply in {time.time() - t:.1f} s; tokens: {reply.usage.completion_tokens}")
    print(f"reasoning (first 150 chars): {(reasoning or '')[:150]!r}")
    print(f"content: {(msg.content or '')[:150]!r}")
    print(f"tool calls: {calls}  ->  {'OK' if calls == ['check'] else 'BAD: no parsed check() call'}")


if __name__ == "__main__":
    main()
