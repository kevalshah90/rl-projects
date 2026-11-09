# KernelBook RL: lab notebook

Running record for the eventual report / paper: what we ran, the numbers, what broke, and
the decisions we made. Newest results go in the tables; the log is chronological. Each entry
should say **what**, **numbers**, **where the data is**, and **why it matters**.

---

## 1. Setup (fixed unless noted)

| Item | Value |
|---|---|
| Policy | Qwen/Qwen3.5-9B, untrained (no SFT) |
| Inference | vLLM on one Modal H100 (`kernelbook-vllm`), 32K context |
| Sampling | temperature 1.0, max_tokens 8192 per turn |
| Agent | verifiers `bash` harness: bash + edit + MCP `kernel-tools_check`; max 30 turns, 900 s |
| Sandboxes | Modal; agent box and a **fresh** grader box per episode (`environment_mode = "separate"`) |
| GPU per box | A10G (since 2026-11-05; was L4, see log) |
| Reward | binary: 1 only if all 7 gates pass (exists, guard, parse, weights, runs, uses_triton, correct) |
| Correctness | `allclose` vs PyTorch reference on 3 seeds (0,1,2), rtol 1e-3, atol 1e-5 |
| Data | GPUMODE/KernelBook -> 11,162 rows kept by filter -> 500 Harbor tasks -> 436 pass oracle/nop validation |
| Split | by module family (no near-duplicates across splits): train 382, dev 54 (378 groups) |
| GRPO (from scratch, `rl_scratch/grpo.py`) | Dr. GRPO default: advantage = r - group mean (no std); token-mean loss; clip (0.2, 0.28) |

## 2. Results

### 2.1 Train baseline (untrained policy) -> trainable set

Purpose: find tasks with 0 < pass rate < 1 (the only ones GRPO learns from). Not a before/after number.

| Run | Tasks | Episodes | Clean | pass@1 | pass@4 | p=0 | 0<p<1 | p=1 |
|---|---|---|---|---|---|---|---|---|
| batch 1 (`...--b5a0244c`) | 30 | 120 | 120 | 25% (30/120 episodes) | | 18 | 8 | 4 |
| batch 1 + 2 pooled (`...--b5a0244c`, `...--9ca93226`) | 200 | 800 | 798 | **21.8%** | **45.5%** | 109 | **81** | 10 |

- Trainable 81 by pass rate: 45 at 1/4, 19 at 2/4, 17 at 3/4.
- Stop reasons (800): max_turns 643, agent_completed 154, agent_timeout 2, harness_error 1.
- Mean per episode: 27.3 turns, ~17.8K output tokens, 218 s.
- Batch 2 tasks were harder than batch 1 early on (~12% vs 25% pass in the first ~130 episodes); pooled 21.8%.
- 182 of 382 train tasks not yet baselined.
- Data: `work.nosync/data/baseline.jsonl`, `work.nosync/data/splits/trainable.txt` (81).

### 2.2 Dev eval (the before/after comparison)

| Run | Policy | Tasks | pass@1 | pass@4 | Notes |
|---|---|---|---|---|---|
| E0 | untrained | 54 dev | _not run yet_ | | |

Rule of thumb: with 54 tasks x 4 at ~20%, only trust differences of roughly 8+ points.

## 3. Log

### 2026-10-07 to 10-13: steps 1-10
Profile/filter KernelBook, base image + GPU smoke test (20/21 oracles match), grader (7 gates)
+ guard, Harbor task generation, oracle/nop validation (436/500 kept; most drops are a
KernelBook weight-order bug in the oracle wrappers), MCP check tool, family-grouped split,
vLLM serving + vf-eval, GRPO math with unit tests. Commits 7e0ef97 .. 560443a.

### 2026-11-05: train baseline attempt 1 hung (L4)
- 30-task run; only 2 of 4 sandboxes ever started; no progress after 16:44.
- Cause: verifiers' `ModalRuntime.start()` calls `modal.Sandbox.create` inside `run_shielded`
  with **no timeout**, before the setup deadline starts. An L4 capacity shortage hangs the rollout forever.
- Fix: `gpu_types = ["A10G", "L4"]`. verifiers passes only `gpu_types[0]` (Modal takes one GPU
  type), so this means A10G. Commit ffc4a09.
- Not re-validated: oracle/nop ran on L4; A10G assumed equivalent (both have TF32; fp32 semantics same).

### 2026-11-05: attempt 2 partial (Mac slept)
- 74/120 episodes `SandboxError` after the Mac slept at 12:46 (DNS failed: tunnel and Modal
  network-policy calls). Interception server + tunnel run on the Mac -> the laptop is a single point of failure.
- Fix: run under `caffeinate -dims`.

### 2026-11-05: misleading log reward
- `eval.log` "rollout done ... reward=0.000" is logged **before** the separate grader scores
  (`verifiers/v1/rollout.py:654`), so it is always 0. True reward is in `traces.jsonl`.
  Monitor from traces, never from the log line.

### 2026-11-05: attempt 3 (batch 1) clean
- 120/120 clean, 25% pass, 8 trainable. Run `...--b5a0244c`.

### 2026-11-05/06: batch 2 (tasks 31-200, `skip = 30`, `limit = 170`)
- Stopped at 23:56 by **Modal workspace spend limit** (~145 instant SandboxErrors); vLLM dropped too.
- 2026-11-06 09:55: limit raised; resumed with `vf-eval ... --resume` (re-runs only missing/errored
  rollouts; kept 140 clean). Finished 18:52, 0 sandbox errors after resume.
- `s09_baseline.py` now pools several run dirs (commit 6886277).

### 2026-11-09: metrics
- `s09_baseline.py` prints pass@1 (mean of per-task pass rates) and pass@k (share of tasks
  solved at least once). Commit edf4a46.

## 4. Decisions (and why)

| Decision | Why |
|---|---|
| Separate grader sandbox per episode | Agent can't tamper with grading; only `solution.py` crosses over |
| Binary reward | Simple, hard to game; partial credit for gates could reward non-solutions |
| Split by module family | Avoid near-duplicate leakage between train and dev |
| Train only on 0<p<1 tasks | Unanimous groups have zero advantage under GRPO |
| Dr. GRPO (no std) by default | Std normalization over-weights lucky passes in lopsided groups |
| 4 rollouts per task for baseline | Cost; pass rates are coarse (steps of 1/4) |
| A10G over L4 | L4 shortage hung sandbox creation (no timeout in verifiers) |
| Before/after on dev (E0), not train | Train tasks get trained on (memorization) and were selected by noisy scores (regression to the mean) |

## 5. Threats to validity / caveats for the write-up

- **Pass-rate noise:** 4 samples per task; "trainable" selection is partly luck (regression to the mean).
- **Turn limit dominates:** 80% of baseline episodes hit 30 turns; results are specific to this budget.
- **Grader gaps:** guard is a heuristic static scan; seeds (0,1,2) are readable in the agent box
  (lookup-table hack possible); edits to the agent's own `reference.py` aren't tracked.
- **Hardware change:** validation on L4, rollouts on A10G.
- **Token data:** eval traces (vf-eval `EvalClient`) have no token ids / logprobs; training must use the prime-rl client.
- **Infra failures excluded:** errored episodes are dropped from pass rates, not counted as fails.

## 6. Ideas for the report / paper

- Infra lessons for agentic RL: untimed sandbox creation, laptop-hosted interception as SPOF,
  spend limits, log lines that report reward before grading.
- pass@1 vs pass@4 gap (21.8% vs 45.5%): RL as reliability vs new capability; track both.
- Effect of std normalization (GRPO vs Dr. GRPO) and loss aggregation (token / sequence / prompt).
- Turn-budget behavior: does RL reduce max_turns endings?

## 7. Next

1. Dev eval E0 (54 x 4, ~3.5-4 h).
2. Training-loop smoke test on the 81 trainable tasks (~10-20 steps).
3. First real training run; dev eval after; compare with E0.
