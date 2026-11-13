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
| batch 1 + 2 pooled (`...--b5a0244c`, `...--9ca93226`) | 200 | 800 | 798 | 21.8% | 45.5% | 109 | 81 | 10 |
| batch 3 (`...--0ff88a70`, tasks 201-382) | 182 | 728 | 727 | 18.1% | 36.8% | 115 | 57 | 10 |
| **all 3 pooled (full train split)** | **382** | 1528 | 1525 | **20.0%** | **41.4%** | 224 | **138** | 20 |

- Trainable 81 by pass rate: 45 at 1/4, 19 at 2/4, 17 at 3/4.
- Stop reasons (800): max_turns 643, agent_completed 154, agent_timeout 2, harness_error 1.
- Mean per episode: 27.3 turns, ~17.8K output tokens, 218 s.
- Batch 2 tasks were harder than batch 1 early on (~12% vs 25% pass in the first ~130 episodes); pooled 21.8%.
- Full pool (1528): stop reasons max_turns 1223, agent_completed 299, agent_timeout 4, harness_error 1,
  sandbox_error 1; 3 errored, 3 tasks with < 4 clean attempts. Mean 27.1 turns, ~17.9K output tokens, 243 s.
- Batch 3 is a bit harder (18.1% vs 21.8%), within noise for 4 samples/task; train pass@1 20.0% vs dev E0 21.9%.
- Data: `work.nosync/data/baseline.jsonl` (382), `work.nosync/data/splits/trainable.txt` (**138**).

### 2.2 Dev eval (the before/after comparison)

| Run | Policy | Tasks | pass@1 | pass@4 | Notes |
|---|---|---|---|---|---|
| E0 (`eval_dev/...--6d73e277`) | untrained | 54 dev | **21.9%** | **42.6%** | 216 episodes, 212 clean; p=0 31, 0<p<1 22, p=1 1 |

Rule of thumb: with 54 tasks x 4 at ~20%, only trust differences of roughly 8+ points.

- E0 stop reasons (216): max_turns 173, agent_completed 38, harness_error 3, agent_timeout 1, sandbox_error 1.
- Mean per episode: 27.0 turns, ~17.3K output tokens, 226 s. 3 tasks had < 4 clean attempts.
- Dev matches train (21.9% vs 21.8% pass@1; 42.6% vs 45.5% pass@4): dev is not easier or harder.
- Data: `work.nosync/data/eval_dev.jsonl` (per-task rows).

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

### 2026-11-09: dev eval E0 started
- `configs/eval_dev.toml`, identical to the train baseline except registry (dev), output dir,
  and no `[select]` (all 54 tasks). 54 x 4 = 216 episodes, under caffeinate, A10G.
- Run dir `work.nosync/outputs/eval_dev/kernelbook_taskset--qwen--qwen3.5-9b--bash--6d73e277`.

### 2026-11-09: dev eval E0 done
- Finished 17:36. pass@1 **21.9%**, pass@4 **42.6%** over 54 dev tasks; 4/216 errored
  (3 harness_error, 1 sandbox_error), excluded. This is the "before" number.
- Incident: `s09_baseline.py` wrote dev per-task rows over `data/baseline.jsonl` (the 200 train
  tasks). Restored from a backup; s09 now writes dev rows to `data/eval_dev.jsonl` and only train
  runs touch `baseline.jsonl` / `trainable.txt`.

### 2026-11-09: prime-rl GRPO settings checked (commit 389eacb, read from source)
| Setting | prime-rl default | Our `rl_scratch/grpo.py` |
|---|---|---|
| Advantage | reward - group mean, **no std** (`orchestrator/algo/grpo.py`) | same (Dr. GRPO) |
| Baseline | plain mean; `length_weighted_baseline = false` | plain mean |
| Length penalty | off (`length_penalty = None`) | none |
| Loss aggregation | `token` (sum / total loss tokens in batch); `prompt` optional | `token` default; sequence/prompt optional |
| Ratio | trainer vs **inference (vLLM) logprobs**, no separate old-policy pass | new vs old logprobs |
| Trust region | `ipo` loss: mask tokens whose \|p_train - p_infer\| > 0.3, cap ratio at 1e4 | PPO clip (0.2, 0.28) |
| KL to reference | none by default | none |
| Zero-advantage groups | **dropped** by a built-in filter and replaced (`constant_trainer_batch_size = true`) | `informative_groups` drops them |
| Off-policy lag | async; tokens older than `max_staleness = 8` steps masked | on-policy |
- Implications: the std question is settled (both no std). Main differences are IPO masking vs PPO clip
  and async staleness.
- Correction (same day): I first wrote that unanimous groups are kept. `docs/algorithms.md` says the
  orchestrator filters samples with zero advantage on all RL tokens and samples replacements, so the
  `AdvRangeGate` is not needed for that.
- For the smoke test: keep defaults; consider a lower `max_staleness` for a closer match to on-policy math.

### 2026-11-11: train baseline batch 3 started (the remaining 182 train tasks)
- `configs/baseline_train.toml` `[select]` skip = 200, limit = 182 (same seed-0 shuffle), so batches
  1-3 cover all 382 train tasks. 728 episodes, concurrency 4 (10-GPU Modal limit), caffeinate, A10G.
- Run dir `work.nosync/outputs/baseline_train/kernelbook_taskset--qwen--qwen3.5-9b--bash--0ff88a70`.
- Goal: grow the trainable pool from 81 to ~150 (at the ~40% rate of batches 1-2).
- 11:00-11:30: Modal A10G **capacity shortage**. 2 episodes failed with "Insufficient capacity to
  create sandbox" after ~21 min each in `boot` (sandbox creation blocks, then errors); throughput fell to
  ~8 episodes / 50 min. A probe at 11:55 got an A10G in 2 s (T4 4 s; L4/L40S/A100 not up in 180 s, possibly
  our 10-GPU workspace cap), so the shortage was transient.
- Own mistake: my first probes ran without `source .env` and failed on SSL (`SSL_CERT_FILE`), which
  looked like "no GPU available". Always load `.env` before any Modal/vLLM script.
- 12:17: stopped and resumed (`--resume`, logs/attempt_2): kept 11 clean episodes, 717 owed.
- 12:22: paused at the user's request (moving location). 11 clean episodes saved (5 passed); backup of
  traces in the scratchpad; no leftover Modal sandboxes. Resume with the same `--resume` command.
- 13:34: resumed (logs/attempt_3), 717 owed. Mac on battery at start (81%); watcher alerts below 20%.
- 17:39: paused again (user closing the lid): 171 traces saved (169 clean), no leftover sandboxes. Batch 3 looks harder so far (16% pass, 6/35 finished tasks trainable).
- 22:02-22:35: vLLM would not restart after its idle scale-down: `kernelbook-vllm` function had
  13 calls **pending with 0 runners** (`get_current_stats()`), though H100 sandboxes started in 1 s.
  Plain `modal deploy` was skipped ("no changes"). Fix: `modal app stop` + `modal deploy` (same URL);
  runner up in seconds, ping OK 22:38. Lesson: if vLLM calls sit pending, check backlog/runners first.
- 22:38: resumed (logs/attempt_4), 559 owed, 169 clean kept.
- 23:58-01:09: 3 isolated SandboxErrors (2x runtime killed exit 137, likely agent code OOM; 1x
  "Sandbox is unavailable") + 1 harness error, out of ~130 episodes; pace 129 episodes/h. Watcher now
  alerts on >= 3 sandbox errors within 30 min instead of 3 total.
- 03:43-07:12 (Nov 12): **local DNS outage** on the Mac (`[Errno 8] nodename nor servname provided`,
  tunnel `lookup s0.tunnel.pinfra.io: i/o timeout`). Every new sandbox failed instantly, so vf-eval
  burned through the remaining tasks: run "finished" at 728/728 with **310 errored** (304 DNS), 76 of
  182 tasks with no clean attempt. The watcher had exited on its 03:4x alert, so nobody saw it.
  Errored episodes are excluded from scores (not counted as fails), so no data was corrupted.
  Lesson: a fast-failing infra error looks like progress; the watcher must alert on error *bursts*
  (now: >= 3 DNS errors -> alert) and keep watching rather than exit.
- 08:04: DNS and vLLM OK; resumed (logs/attempt_5): 418 clean kept, 310 owed (81 tasks).
  Pace 25-46 episodes/h (boot 134 s early, ~25 s later). Correction: the "129 episodes/h" above was
  wrong; the real pre-outage pace was ~54/h. 7 new errors (2 DNS blips at 10:45, 5 harness exit 1 at
  11:21-11:28).
- ~12:20: paused (user): 575 traces (568 clean), 2 leftover sandboxes terminated, 0 live. 153 owed.
- 18:51: resumed (logs/attempt_6): 568 clean kept, 160 owed (153 + the 7 errored), 42 tasks.
- 19:00-23:20: **tunnel degraded**: only 21 episodes after 20:00 and 37 of 64 new episodes errored.
  Mostly `harness 'bash' exited 1` with `openai.NotFoundError: 404` on the agent's model call (the
  sandbox reaches vLLM through vf-eval's local tunnel), plus `Tunnel is inactive`, `send Login request
  to plugin error`, DNS lookups for s0.tunnel.pinfra.io failing, and Modal exec/write errors. The
  watcher only knew DNS-error text, so it missed this. Fix: stop vf-eval, restart (new tunnel).
- 23:28: resumed (logs/attempt_7): 595 clean kept, 133 owed (38 tasks). Watcher now also alerts on
  tunnel errors and on >= 6 errored rollouts in the attempt.
- 23:28-23:33: every new episode failed at once with `TunnelError: Failed to create tunnel: Internal
  Server Error` (17 episodes). Even `list_tunnels` returned 500: Prime Intellect's tunnel API was down
  (server side, not our setup). Watcher caught it in 4 min; stopped vf-eval: 595 clean kept, 0 live
  sandboxes. Polling the tunnel API every 5 min; resume when it answers.
  Dependency note for the report: vf-eval's sandboxes reach our vLLM through Prime's tunnel, an external
  single point of failure. For training, consider a direct route (Modal tunnel or vLLM URL directly).
- 23:33-00:07: API flapped (OK, then 500 a minute later); auto-resume waited for 3 OK checks 2 min apart.
- 00:07 (Nov 13): resumed (logs/attempt_8), 133 owed; finished ~02:20 with 728/728, **727 clean**
  (1 sandbox_error left; not re-run). Pooled with batches 1-2 -> section 2.1.
- Recurring `models call failed` warnings (vLLM calls retried after 30 s, then succeed); not yet causing failures.

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

1. ~~Dev eval E0~~ done: 21.9% / 42.6%.
2. ~~Check prime-rl GRPO settings~~ done: no std, token-mean, IPO loss (see log).
3. ~~Baseline the remaining 182 train tasks~~ done: 382 tasks, 20.0% / 41.4%, 138 trainable.
4. Training-loop smoke test on the 138 trainable tasks (~10-20 steps).
5. First real training run; dev eval after (same config); compare with E0.
