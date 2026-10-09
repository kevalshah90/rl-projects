# kernelbook-rl

Turns [GPUMODE/KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook) rows into Harbor-format
Triton-kernel tasks, then trains Qwen on them with GRPO — first with our own loop (`rl_scratch/`),
then with prime-rl (`rl_prime/`).

Plan: https://claude.ai/code/artifact/915e76e0-6253-4b37-b9cc-29745e1cbff2

## Where things run

- **Mac CPU:** everything that needs no GPU — dataset profiling, filtering, and the F5 execution check
  (`kernel_env/exec_worker.py` runs rows in timeout-guarded, network-blocked subprocesses), plus all
  Mac-only tests.
- **Modal:** only work that needs a GPU — the base-image smoke test, grading Triton kernels, task
  validation, the vLLM server, agent and grader sandboxes, and the RL loops (steps 3+).
- **Mac, during evaluations (step 9+):** `vf-eval` orchestrates the run and relays every model call
  (see [Running an evaluation](#running-an-evaluation-step-9)), so the Mac must stay online.

## Steps

Each step adds one component and ends with a check. Run commands from this folder.

| Step | What it adds | Runs on | Check |
| --- | --- | --- | --- |
| 0 | Project skeleton | Mac | `uv sync` then `uv run python -c "import modal, kernel_env"` and `uv run modal token info` (login check only, starts nothing) |
| 1 | `kernel_env/data.py` (loading), `scripts/s01_profile.py` | Mac | `uv run python scripts/s01_profile.py` writes `work.nosync/data/profile_report.md` |
| 2 | `kernel_env/data.py` (filters), `kernel_env/exec_worker.py`, `scripts/s02_filter.py`, `tests/test_data.py` | Mac CPU | `uv run pytest tests/test_data.py`, then `uv run python scripts/s02_filter.py` writes `work.nosync/data/kept.jsonl` + `dropped.jsonl` |
| 3 | `docker/base.Dockerfile`, `../.github/workflows/kernel-env-base-image.yml`, `kernel_env/config.py`, `scripts/s03_smoke_test.py` | GitHub Actions (build), Modal L4 (test) | Image `<user>/kernel-env-base:torch2.5.1-r2` on Docker Hub; `uv run modal run scripts/s03_smoke_test.py` writes `work.nosync/data/smoke_s03.jsonl` |
| 4 | `kernel_env/guard.py`, `kernel_env/triton_hook.py`, `kernel_env/grade.py`, `tests/test_grade_gpu.py` | Modal L4 | `uv run modal run tests/test_grade_gpu.py` ends with `7/7 as expected` |
| 5 | `templates/instruction.md`, `templates/test.sh`, `scripts/s05_make_tasks.py` | Mac | `uv run python scripts/s05_make_tasks.py --limit 20` (or no limit for all) writes `work.nosync/tasks/<id>/` + `registry.json` |
| 6 | `scripts/s06_validate.py` | Modal L4s (`--mode direct`, default) or Harbor trials (`--mode harbor`) | `uv run python scripts/s06_validate.py` grades oracle (must score 1) and nop (must score 0) on every task; writes `work.nosync/data/validation.jsonl` |
| 7 | `kernel_env/mcp_tools.py` (agent's `check` tool), `s05` adds `mcp_servers` + `.kernel_tools/`, `tests/test_mcp_gpu.py` | Modal L4 | `uv run modal run tests/test_mcp_gpu.py` ends with `3/3 as expected` |
| 8 | `scripts/s08_split.py` | Mac | `uv run python scripts/s08_split.py` splits kept tasks by module group (sha256 of the name, ~10% dev); writes `work.nosync/data/splits/{train,dev}.txt` + `work.nosync/tasks/registry-{train,dev}.json` |
| 9 | `scripts/s09_serve.py` (vLLM), `scripts/s09_ping.py`, `kernelbook_taskset.py`, `configs/baseline_train.toml`, `configs/eval_dev.toml`, `scripts/s09_baseline.py` | Modal H100 (vLLM) + L4s (agents, graders); Mac (orchestrator + relay) | `uv run python scripts/s09_ping.py` ends with `tool calls: ['check'] -> OK`; then see [Running an evaluation](#running-an-evaluation-step-9) |
| 10 | `rl_scratch/grpo.py` (group advantages, informative-group filter, clipped policy loss with token / sequence / prompt aggregation), `tests/test_grpo.py` | Mac CPU | `uv run pytest tests/test_grpo.py`: 12 hand-computed checks pass |

## Status (2026-10-08)

Steps 0–8 done: 500 tasks generated, 436 validated, split into train 382 / dev 54. Step 9: vLLM
serving works, and full agent episodes run end to end (sandbox → tunnel → vLLM → MCP `check` →
separate grader).

- **Qwen3.5-4B: p = 0 on all 7 tasks tried** (two 3-task dev pilots, one 4-task train test). The
  first pilot also showed the model never found the `check` tool (fixed: `instruction.md` names
  it `kernel-tools_check`); after that, failures were Triton skill (invented APIs, kernels called
  like functions, compile errors). No 0 < p < 1 means no GRPO signal.
- **Switched to Qwen3.5-9B** (same family, parsers and template; still RL-trainable on our GPUs).
  Same 4 train tasks × 2 attempts: **3/8 passes**, verified as real Triton kernels with an exact
  match: basicmodel3 p = 1, smoothl1loss p = 0.5 (trainable), avgpoolpad and residualblock p = 0.
- Modal GPU limit is 10, so `max_concurrent = 4` (≈ 6 GPUs at peak).

Next: the 30-task train baseline (`-n 30`, 4 attempts) to estimate how many train tasks are
trainable, then the dev evaluation (`configs/eval_dev.toml`), then the RL loop (steps 10+).

## Running an evaluation (step 9)

Every run needs three variables from `.env` (gitignored): `VLLM_API_KEY` (our vLLM server),
`PRIME_API_KEY` (the tunnel that lets sandboxes reach the relay on the Mac) and `SSL_CERT_FILE`
(certifi's CA bundle; the python.org Python has none, and sandbox connections fail without it).
Warm the server before a run, so the first model calls don't hit a cold start:

```bash
set -a; source .env; set +a
uv run python scripts/s09_ping.py                       # waits until vLLM is healthy
uv run vf-eval @ configs/baseline_train.toml -n 4 -r 2 --env.agent.max-turns 30
uv run python scripts/s09_baseline.py                   # pass rates for the newest run
```

| Part | Sets | Effect |
| --- | --- | --- |
| `@ configs/baseline_train.toml` | everything | model, vLLM URL, train tasks, bash harness, Modal runtime, `max_concurrent = 4`; flags after it override single settings |
| `-n 4` | `select.limit` | the first 4 tasks of the fixed shuffle (`shuffle = true, seed = 0`): always the same 4 |
| `-r 2` | `num_rollouts` | 2 attempts per task → 8 episodes |
| `--env.agent.max-turns 30` | `env.agent.max_turns` | at most 30 model replies per episode (~5 min). The configs already use 30; the flag is shown as an example override |

`--dry-run` writes the final settings to `<run>/configs/resolved/eval.json` without running anything.
Train vs dev: only train runs may choose what we train on (`s09_baseline.py` writes
`splits/trainable.txt` only for train-only runs). `eval_dev.toml` is the held-out score, run
unchanged before and after RL.

### 1. Where things run

```
 YOUR MAC                                    MODAL (cloud)
┌────────────────────────────────┐          ┌──────────────────────────────────────┐
│ vf-eval (orchestrator)         │          │ kernelbook-vllm  (1× H100)           │
│  • reads config + tasks        │          │   Qwen3.5-9B, OpenAI-compatible API  │
│  • schedules episodes          │          └──────────────▲───────────────────────┘
│  • writes traces.jsonl         │                         │ HTTPS (VLLM_API_KEY)
│                                │                         │
│ interception server ───────────┼─────────────────────────┘
│  (127.0.0.1:port) records every│
│  model call: tokens, logprobs  │          ┌──────────────────────────────────────┐
│         ▲                      │          │ agent sandboxes (L4 each, ≤4 at once)│
│         │ Prime tunnel (HTTPS) │◄─────────┤   bash harness + kernel-tools MCP    │
└─────────┼──────────────────────┘          ├──────────────────────────────────────┤
          └── PRIME_API_KEY                 │ grader sandboxes (L4 each, brief)    │
                                            │   tests/test.sh → reward.json        │
                                            └──────────────────────────────────────┘
```

The agent never calls vLLM directly. Each model call goes **sandbox → tunnel → Mac → vLLM** and
back. The stop on the Mac is how verifiers records every token for the trace that RL later trains
on, which is also why the Mac must stay awake and online during a run.

### 2. What the verifiers library does

[verifiers](https://github.com/PrimeIntellect-ai/verifiers) (Prime Intellect) turns our task folders
into **episodes** and records them as **traces**. It is the same library prime-rl uses to generate
rollouts during training, so evaluation episodes run exactly like RL episodes will.

| Piece | Ours | verifiers |
| --- | --- | --- |
| What the task is | `task.toml`, `instruction.md`, `reference.py`, `tests/test.sh`, `kernel_env/grade.py` | — |
| The `check` tool | `kernel_env/mcp_tools.py` (MCP server) | starts it in the sandbox, offers it to the model |
| Loading tasks | `kernelbook_taskset.py` (~10 lines: read our registry) | `HarborTask` parses `task.toml` (GPU, timeouts, MCP, artifacts, network) |
| Running episodes | — | everything below |

```
vf-eval (CLI)            reads baseline_train.toml + flags, picks the tasks (e.g. -n 30) × attempts,
   │                     keeps max_concurrent (4) running at once, draws the progress screen
   ▼
Env: HarborEnv           one episode = solve in one box, grade in a FRESH box
   │                     (because task.toml says environment_mode = "separate")
   ├─► Runtime: Modal    creates the sandbox (image, L4), uploads environment/ → /workspace,
   │                     applies the network policy (only the tunnel domain allowed)
   │
   ├─► Harness: bash     a small agent program it installs in the sandbox (.vf-venv/):
   │                     the loop  model → tool call → result → model …
   │                     tools: bash, edit, plus MCP tools from task.toml (kernel-tools_check),
   │                     max_turns, context compaction at 16K tokens remaining
   │
   ├─► Interception      a proxy on the Mac that every model call passes through:
   │     + Tunnel        records the prompt, every generated token and its logprob, and the
   │                     tool calls, then forwards to vLLM. The Prime tunnel lets the
   │                     sandbox reach it.
   │
   ├─► Finalize          copies /workspace/solution.py out (task.toml: artifacts)
   │
   └─► Scoring           new sandbox, stages tests/, runs tests/test.sh,
                         reads /logs/verifier/reward.json → reward + metrics
   ▼
Trace → traces.jsonl     one line per episode: messages, token ids, logprobs, rewards,
                         metrics, timings, errors
```

**Why the trace matters for RL:** every model call passes through the interception proxy, so a
trace holds **exactly the tokens the model sampled and their logprobs** (`nodes[*].token_ids`,
`mask` = which tokens were sampled and so trainable, `logprobs`). GRPO's loss is computed on
those. Re-tokenizing the text afterwards does not always give back the same tokens.

Three similar names:

| Name | What it is |
| --- | --- |
| **verifiers** | the library (Python package), used through `vf-eval` now and through prime-rl later |
| **`verifiers-v1`** | the Modal app it puts all its sandboxes under, agents and graders (see `modal container list`) |
| **Harbor "verifier"** | a task's grader (`tests/test.sh`), which verifiers runs during scoring |

### 3. Startup (Mac, ~5 s)

```
configs/baseline_train.toml ──┐
-n 4  -r 2  --max-turns 30 ───┴──► resolved config (<run>/configs/resolved/eval.json)

registry-train.json (382 tasks) ──► kernelbook_taskset.py loads each task folder
                                         │
                                         ▼
                     shuffle(seed=0) ──► take first 4  (-n 4)
                                         │
                                         ▼
                     × 2 attempts (-r 2) = 8 episodes queued
```

### 4. Scheduling: 8 episodes, at most 4 at once

```
time ─────────────────────────────────────────────────────────────►
       0 min            ~5–6 min           ~10–12 min
slot 1 [ task A #1 ─────────] [ task C #1 ─────────]
slot 2 [ task A #2 ─────────] [ task C #2 ─────────]
slot 3 [ task B #1 ───────]   [ task D #1 ───────────]
slot 4 [ task B #2 ──────────] [ task D #2 ────────]
         wave 1                 wave 2  (a slot refills as soon as its episode ends)

GPUs at peak: 1 H100 (vLLM) + 4 L4 (agents) + ~1 L4 (a grader) ≈ 6 of the 10-GPU Modal limit
```

### 5. One episode

```
┌─ BOOT ── ~5–15 s ───────────────────────────────────────────────────────┐
│ Modal creates a sandbox: image kernel-env-base:torch2.5.1-r2, 1× L4     │
└─────────────────────────────────────────────────────────────────────────┘
          │
┌─ SETUP ── ~30 s ────────────────────────────────────────────────────────┐
│ • uploads environment/ → /workspace                                     │
│     reference.py               the PyTorch module to rewrite            │
│     .kernel_tools/kernel_env/  grader code for the check tool           │
│ • installs the bash harness, starts the MCP server (mcp_tools.py)       │
│ • opens the tunnel, then locks the network: only the relay is reachable │
└─────────────────────────────────────────────────────────────────────────┘
          │
┌─ ROLLOUT ── up to max_turns, 900 s maximum (task.toml) ─────────────────┐
│                                                                         │
│   prompt = system prompt + instruction.md + tools [bash, edit,          │
│            kernel-tools_check]                                          │
│      │                                                                  │
│      ▼                                                                  │
│   ┌──────────────┐  tool call   ┌────────────────────────────────────┐  │
│   │ model turn   │ ───────────► │ bash:  cat reference.py / python … │  │
│   │ <think>…     │              │ edit:  change solution.py          │  │
│   │ </think>     │ ◄─────────── │ check: grade solution.py now       │  │
│   │ + tool call  │  tool result │   → {"failed_gate": "runs", …}     │  │
│   └──────────────┘              └────────────────────────────────────┘  │
│      │  repeat: each loop = 1 turn                                      │
│      ▼                                                                  │
│   ends when: model stops (agent_completed) │ max_turns                  │
│              │ 900 s (timeout) │ an error                               │
│   context nearly full (16K left)? → compaction: summarize, continue as  │
│   a new branch                                                          │
└─────────────────────────────────────────────────────────────────────────┘
          │
┌─ FINALIZE ── ~1 s ──────────────────────────────────────────────────────┐
│ copies /workspace/solution.py out (task.toml: artifacts); agent box     │
│ released                                                                │
└─────────────────────────────────────────────────────────────────────────┘
          │
┌─ SCORING ── ~1 min, in a FRESH sandbox (separate grader) ───────────────┐
│ new L4 box, same image  ← solution.py copied in, tests/ staged          │
│ tests/test.sh → python -m kernel_env.grade, with 7 gates:               │
│   exists → guard → parse → weights → runs → uses_triton → correct       │
│ writes /logs/verifier/reward.json = {"reward": 0 or 1, gate metrics…}   │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
  one JSON line appended to traces.jsonl
```

### 6. One model turn on the wire

```
sandbox                  tunnel        Mac (interception)               vLLM (H100)
  │ POST /v1/chat/completions │                 │                           │
  ├──────────────────────────►├────────────────►│ record request ──────────►│
  │                           │                 │                           │ generate
  │                           │                 │◄────── streamed tokens ───┤ (thinking +
  │◄──────────────────────────┤◄────────────────┤ record tokens, logprobs   │  tool call)
  │ run the tool call         │                 │                           │
```

### 7. Outputs

```
work.nosync/outputs/baseline_train/<run>/
├── configs/
│   ├── eval.toml              ← the TOML, copied as-is
│   └── resolved/eval.json     ← final settings, including the -n/-r/--max-turns overrides
├── logs/latest/eval.log       ← timestamps: sandboxes up, rollouts done, warnings
└── traces.jsonl               ← one line per episode:
      task, ok, stop_condition, rewards, gate metrics,
      every message (nodes), every model call, timing per phase
```

`s09_baseline.py` reduces these to per-task pass rates (p ∈ {0, ½, 1} with 2 attempts) and writes
`splits/trainable.txt`: the tasks with 0 < p < 1, the ones that can teach GRPO something.

### Monitoring a run

| What | Command |
| --- | --- |
| Containers (vLLM `Pending` = waiting for a GPU; `verifiers-v1` = sandboxes) | `uv run modal container list` |
| vLLM logs (`Running` / `Waiting` reqs, throughput) | `uv run modal app logs kernelbook-vllm \| grep -v "Route:"` |
| The run's own log | `tail -f work.nosync/outputs/baseline_train/<run>/logs/latest/eval.log` |
| Finished episodes | `wc -l work.nosync/outputs/baseline_train/<run>/traces.jsonl` |

Don't `modal app stop` the vLLM app between runs: it undeploys it. When idle, it scales to zero
GPUs after 10 minutes on its own.

## Known data issues

- **Step 6 drops ~20% of tasks** (4 of the first 20). Most are a bug in KernelBook's answer keys,
  not in our grader: for modules with 2+ same-shaped weights, the oracle's `forward()` wrapper
  passes the weights to the compiled kernel in the wrong order (e.g. weight and bias swapped in
  ScalarBiasScale, uuid 25; swapping them back gives an exact match). The task is valid, but with
  no working answer key it can't be proven solvable, so it's dropped. See `scripts/s06_validate.py`.
- Weights created with `torch.Tensor(n)` / `torch.empty` are uninitialized memory (NaN or huge on a
  GPU). `kernel_env/grade.py` replaces such values with small seeded random ones before copying
  the reference's weights into the solution.

## Setup notes

- `.venv` is a symlink to `~/.venvs/kernelbook-rl`. iCloud-synced folders mark `.pth` files hidden,
  which Python skips, breaking the editable install. Recreate with
  `UV_PROJECT_ENVIRONMENT=~/.venvs/kernelbook-rl uv sync && ln -s ~/.venvs/kernelbook-rl .venv`.
- `.python-version` pins an ARM64 Python 3.12: torch 2.5.1 has no wheels for Intel-Mac Python.
- Generated files live in `work.nosync/{data,tasks,outputs}`; every script gets these paths from
  `kernel_env/config.py` (`DATA`, `TASKS`, `OUTPUTS`). iCloud skips folders ending in `.nosync`;
  otherwise it creates empty "name 2" duplicates whenever a synced folder is deleted and
  recreated (as `s05` does to `tasks/`). Scripts create the folders they write to.
