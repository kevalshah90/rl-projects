# RL Projects

Practice projects for RL post-training of language models: building environments, graders and training loops from scratch.

## Projects

| Project | What it is | Status |
| --- | --- | --- |
| [kernelbook-rl](kernelbook-rl/) | Turns [GPUMODE/KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook) rows into Harbor-format Triton-kernel tasks (with a GPU grader and an MCP `check` tool), then trains Qwen on them with GRPO, first with our own loop (`rl_scratch/`), then with prime-rl. | Steps 0–10 done: 436 validated tasks (train 382 / dev 54), vLLM serving and baseline evaluation working, GRPO loss implemented and tested. |

See each project's README for setup, the step-by-step plan and run commands.

## Layout

- `kernelbook-rl/`: the project above. Run its commands from inside that folder (`uv sync` first).
- `.github/workflows/`: CI that builds the kernelbook-rl base Docker image.

## Reviews

- [Mercor APEX-Agents + SkyRL Recipe — Source Review](https://claude.ai/artifact/FJKpkFqpP9xuqJf6ACXhY5): walkthrough of Mercor's SkyRL training recipe for APEX-Agents (TITO token handling, failure masking, DPPO + `prompt_mean` loss, ablations, eval harness, and the APEX-Agents dataset).
