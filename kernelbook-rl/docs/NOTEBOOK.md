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
| Policies / KL | Ratio pi_theta / mu against vLLM log probs (off-policy correction); **no KL penalty to a reference** (beta = 0). See log 2026-11-16 |

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

### 2026-11-13: host side moved to Modal (no Prime tunnel, no laptop)
- Why: the agent's model calls go to vf-eval's *interception server*, which forwards them to vLLM
  and records each turn into the trace (training needs this), so sandboxes can't call vLLM directly.
  On the Mac, sandboxes reached that server only via Prime's tunnel service, which failed twice in
  the baseline; the Mac also lost DNS once and needed 3 pauses.
- `scripts/s10_modal_eval.py`: runs vf-eval in a Modal container (CPU, 24 h cap, `--detach`), publishes
  the interception port with `modal.forward`, and sets verifiers' `custom` tunnel
  (`interception.type = server`, `tunnel.type = custom`, url = the forward URL). Outputs go to the
  Volume `kernelbook-runs` (committed every 2 min). Image: locked deps minus torch/CUDA (host never
  runs kernels). Registries hold absolute Mac paths, so a container copy is written to /tmp.
- `--resume` rejects any config change, and the forward URL is new per container: the script rewrites
  the URL in the run's saved config (transport only, not an experimental setting).
- Trade-off: one interception server (`server`) instead of the elastic pool (32 rollouts each); fine
  for our 4-32 concurrent rollouts.
- Smoke test (`modal_smoke/...--4d6651da`, task 00456-smoothl1loss, Mac baseline 4/4): 4/4 episodes
  clean, 3/4 pass, 4 min 16 s, boot 20 s. Traces have the same structure as Mac runs (nodes with
  token_ids/logprobs/mask fields; empty for both, since the eval client doesn't request them).

### 2026-11-16: concepts for the write-up: the policies, on- vs off-policy, KL
Facts about prime-rl below come from the 11-09 source read (commit 389eacb); the rest is
standard RL background, written down so the report uses the terms consistently.

**The policies (three copies of the model can matter)**

| Name | Symbol | Where it lives | What it is used for |
|---|---|---|---|
| Behavior policy (sampler) | mu, "pi_old", "pi_vLLM" | vLLM on the H100, serving adapter version v_k | Generates every agent turn. Its log prob for each sampled token is recorded by the interception server into the trace (training client only; eval traces leave these fields empty). |
| Learner (target) policy | pi_theta | Trainer (FSDP, base + LoRA adapter) | Recomputes log probs of the same tokens in its forward pass. The gradient flows only through these; AdamW updates only the adapter. |
| Reference policy | pi_ref | Not loaded (frozen Qwen3.5-9B) | Only needed for a KL penalty, which we don't use (see below). With LoRA it would cost no extra memory: same weights with the adapter switched off. |

The loop: mu samples, the grader scores, the orchestrator computes A = r - group mean, and the
trainer moves pi_theta. The new adapter is then sent to vLLM, so pi_theta becomes the next mu.
RL is "on-policy" when mu = pi_theta at the moment of the gradient step, and "off-policy" otherwise.

**Where the setup is on-policy**
- The data is always the model's own behavior, never a dataset or another model's outputs. In
  that broad sense the whole thing is on-policy RL, as opposed to SFT or offline RL.
- Rewards and advantages come straight from mu's rollouts and are never reweighted.
- If training were fully synchronous (`max_staleness = 0`, one optimizer step per batch), mu and
  pi_theta would hold the same weights, and the ratio would be 1 up to numerical noise.

**Where it becomes off-policy (four sources, all ending up in the same per-token ratio)**
1. **Async staleness (prime-rl):** batch B is sampled with v_{k-s} while the trainer is at v_k.
   Tokens with s > `max_staleness` are masked (default 8; we plan 1 for the smoke test).
2. **Mid-episode weight swaps:** an agent episode is up to 30 turns and 900 s long. If vLLM loads a
   new adapter partway through, early turns come from one mu and later turns from another. This
   is open unknown #5; check per-turn policy versions in the first training traces.
3. **Engine mismatch:** even with identical weights, vLLM and the FSDP trainer compute slightly
   different probabilities (different kernels, numeric precision, batching). This is the
   "training-inference mismatch". It may be larger for Qwen3.5's hybrid linear attention, since
   vLLM and the trainer use different implementations of those layers.
4. **Several optimizer steps per batch (PPO-style; only if a loop around our scratch GRPO did it):**
   after the first step, `old_logp` is out of date. This is PPO's classic off-policy part.
   `grpo.py` today is the math of one update, so this doesn't apply yet.

**How each side corrects for it**
- Both use a per-token importance ratio, ratio = pi_theta(token) / mu(token), computed against the
  **vLLM log probs**. Neither does a separate old-policy forward pass, so one ratio absorbs all
  four sources above.
- prime-rl (IPO loss): drops tokens where |p_train - p_infer| > 0.3 (a gap in probability, not in
  ratio) and caps the ratio at 1e4.
- Our `rl_scratch/grpo.py`: PPO clip (0.2, 0.28). Outside that range the clipped side carries no gradient.
- Only token probabilities get the correction. Advantages still come from mu's rollouts. That's the
  usual PPO/GRPO approximation, and it holds only while mu and pi_theta stay close. Hence the masks
  and a small `max_staleness`.
- To log in the smoke test: share of tokens masked by IPO, mean |ratio - 1|, a staleness
  histogram, and a mismatch-KL estimate KL(mu || pi_theta) on the sampled tokens. That KL is a
  diagnostic, not a loss term.

**KL: two different uses**
- (a) A **KL penalty to a reference** keeps pi_theta close to pi_ref. It's a regularizer
  inherited from RLHF: maximize E[r] - beta * KL(pi_theta || pi_ref).
- (b) The **behavior-to-learner KL** (mu vs pi_theta) measures how off-policy a batch is. It's
  what PPO's clip, TRPO's constraint and IPO's mask keep small.
- (b) is always present in effect, through the ratio/mask. (a) is a choice, discussed next.

**Do we use a KL penalty? No (beta = 0 in both prime-rl's default and `grpo.py`).** Why:
- *The reward is verifiable, not learned.* In RLHF the KL protects against over-optimizing a
  reward model that is only an approximation. Ours is a sandboxed binary grader: there's no
  learned model to over-optimize. Its real gaps (heuristic guard, readable seeds) are better
  fixed in the grader. A KL term would only slow hacking down, not block it.
- *The policy needs to move.* Pass@1 is 20%. A pull back toward the base model works against the
  goal, and recent RL-with-verifiable-reward recipes (DAPO, Dr. GRPO) dropped it for that reason.
- *There are already brakes:* LoRA rank 16 limits how far the weights can move, plus the IPO mask
  or clip, a small learning rate, and short runs (15-100 steps).
- *Cost:* a reference forward pass per batch. That's cheap with LoRA (adapter off) but an extra
  ~18 GB weight copy for a full fine-tune.
- **When to add it:** entropy collapse, broken tool-call or format behavior, a drop on dev or on
  general ability, or as a paper ablation (beta in {0, 0.001, 0.01}).

**Forward vs reverse KL** (pi = the learner, ref = the reference)
- **Forward KL(ref || pi)** = E_{x~ref}[log ref(x)/pi(x)]. It's *mass-covering*: pi pays a lot
  wherever ref has mass and pi doesn't, so pi spreads over everything ref does. The expectation is
  over **ref's samples**, which we don't have. Minimizing it on such samples is just SFT or
  distillation (cross-entropy).
- **Reverse KL(pi || ref)** = E_{x~pi}[log pi(x)/ref(x)]. It's *mode-seeking* (zero-forcing): pi
  can ignore parts of ref and focus on a few good modes, but can't put mass where ref has almost
  none. The expectation is over **the policy's own samples**, which are exactly our rollouts, so
  it can be computed for free. This is the RLHF one. The penalized objective has the closed-form
  optimum pi* proportional to ref * exp(r / beta): the reference reweighted by reward.
- Why reverse suits RL: its sampling distribution matches the one we already have, and
  mode-seeking is what we want (concentrate on kernels that pass). A side effect is lower
  diversity, which can show up as pass@k rising less than pass@1. Track both (see section 6).
- Per-token estimators on sampled tokens, with q = ref(token) / pi(token) (Schulman's k1/k2/k3):
  - k1 = -log q: unbiased, high variance, can be negative.
  - k2 = 0.5 (log q)^2: biased but low variance.
  - k3 = (q - 1) - log q: unbiased, never negative, low variance. GRPO (DeepSeekMath) adds this to the loss.
  - Subtlety to verify before citing: an estimator that is unbiased for the KL *value* doesn't
    automatically give the right *gradient* when used as a loss. k3-as-loss has been reported to
    be biased for reverse KL, and with off-policy samples (from mu, not pi) it also needs the
    importance ratio.
  - It matters whether the KL goes in the reward (PPO-RLHF: per-token -beta log(pi/ref)) or in the
    loss (GRPO). Only relevant if we turn KL on.

### 2026-11-17: part 1, check 1: prime-rl Modal image builds (`scripts/s11_prime_rl_image.py`)
- **What:** a Modal image for prime-rl's three processes (inference, orchestrator, trainer). It's leaner
  than prime-rl's `Dockerfile.cuda`, which also builds UCX, NIXL and AWS EFA for multi-node use.
  Layers:
  - `nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04` + Python 3.12 + uv 0.11.9;
  - prime-rl @ `389eacb` with its submodules;
  - `uv sync --frozen --no-dev --extra gpu --extra flash-attn`;
  - `modal==1.5.1`;
  - our `kernel_env/`, `kernelbook_taskset.py`, configs and tasks.
- **Result** (a check run on CPU in the image):
  - versions: prime-rl 0.10.0, torch 2.14.0+cu130, vLLM 0.33.0, transformers 5.20.2,
    flash-linear-attention 0.6.0, flash-attn 2.8.3, harbor 0.24.0, verifiers 0.4.2.dev20 (`a12316c`);
  - all imports work (`prime_rl`, `verifiers`, `renderers`, `vllm`, `fla`), and `rl --help` exits 0;
  - a full build takes ~5-10 min; a change to our own code only rebuilds the top layer.
- **Issue 1: `--locked` fails.** prime-rl's lock uses a rolling cooldown
  (`exclude-newer-span = "P7D"`), so re-resolving on a later day differs from the lock. Used
  `--frozen` instead: it installs exactly the lock's versions without re-resolving.
- **Issue 2: `modal` missing.** prime-rl depends on `verifiers[harbor]` without the modal extra. Our
  orchestrator creates the agent and grader sandboxes on Modal, so I added `modal==1.5.1`, the
  version in prime-rl's lock.
- **Found for check 2:** our `kernelbook_taskset.py` doesn't import against prime-rl's verifiers.
  - `HarborEnv` was removed in `269ff7b` ("Own task workspaces and Compose services through
    deployments").
  - prime-rl's verifiers (`a12316c`) is 183 commits ahead of ours (`da75527`, an ancestor), and 13 of
    those commits touch the Harbor taskset.
  - `HarborTaskset`, `HarborTask` and `parse_task` still exist.
- **Not tested yet** (needs a GPU, check 4): Modal's driver vs CUDA 13, and loading Qwen3.5-9B in the trainer.

### 2026-11-17: part 1, check 2: our taskset vs prime-rl's verifiers (`da75527` -> `a12316c`)
- **Diff:** `a12316c` is 183 commits ahead of ours. 13 of them touch the Harbor taskset: `env.py`
  (`HarborEnv`) and `compose.py` were deleted and `taskset.py` was rewritten. Grading moved into
  the common rollout code (`580b173`, `269ff7b`), driven by each task's `scoring.mode`.
- **Same behavior (read from code, then parsed in the s11 image):**
  - all 382 train + 54 dev tasks parse;
  - `environment_mode = "separate"` -> `scoring.mode = "isolated"`: a fresh box with the same image
    (`fresh_copy`), network `allow = []` (from `no-network`), and only `/workspace/solution.py` crosses;
  - `reward_key = "reward"`, other `reward.json` keys become metrics;
  - the `ignore_timeouts` default is unchanged (we set it to false).
- **Fixed:** `kernelbook_taskset.py` exports `HarborEnv` only if it exists, so one file works on
  both versions (eval/E0 on `da75527`, training on `a12316c`).
- **Renamed:** `vf-eval` -> `vf-run` in the new verifiers.
- **Blocker:** `682e8dfb1` (2026-10-19) removed the built-in Modal runtime. Built-ins are now
  `subprocess`, `docker` and `prime`; other providers load as plugins (`type = "custom"`,
  `import_path` = a `RuntimeBackend` subclass), checked with `tests/unit/test_runtimes.py --runtime`.
- **Decision:** port Modal as our own runtime plugin rather than switch to Prime sandboxes, pin an
  older prime-rl, or force our verifiers into prime-rl. Why:
  - it keeps the same A10G hardware and images as the baseline and E0, so before/after numbers
    stay comparable;
  - it uses the supported extension point and its test suite.

### 2026-11-17: check 2 (cont.): Modal runtime plugin (`modal_runtime.py`) passes verifiers' runtime suite
- **What:** verifiers' Modal runtime as of `da75527` (the code that ran the baseline and E0), moved
  onto the new plugin contract (`RuntimeBackend` + `CustomRuntimeConfig`, `type = "custom"`,
  `import_path = "modal_runtime.ModalBackend"`). It declares task images + port exposure; no user
  switching, no Compose (same as before). Modal client 1.6.1 (the baseline's version) is in its own
  image layer.
- **Result:** `tests/unit/test_runtimes.py --runtime <plugin>`: **22/22 passed** in 8.7 s:
  - run, exec, background processes, files, task images, port exposure and network policies all work;
  - user switching and Compose are refused up front;
  - 0 leftover sandboxes.
- **Bug found and fixed:** `run` left stdin open, so a stdin-reading command (bare `cat`, `python`)
  hung forever. The old Modal runtime had the same latent bug. Now it is wrapped as
  `sh -c 'exec "$@" < /dev/null'`, as the base class does.
- **Incident:** the first suite run (not detached) stalled ~27 min on that hang. Then the Mac lost
  DNS (~14:03), Modal stopped the app, and one test sandbox was orphaned (its owner was hard-killed,
  so teardown never ran). I found it with `modal.Sandbox.list` and terminated it (~30 min CPU).
- **Changes from the incident:**
  - checks run with `modal run --detach`, live `-v` output, and `pytest-timeout` at 300 s;
  - new plugin field `sandbox_timeout` (default 2 h, was a fixed 24 h) caps the bill for orphans.
- **To do before training:** an orphan sweep that terminates `verifiers-v1` sandboxes older than the
  longest episode. A hard-killed orchestrator would orphan every in-flight sandbox.
- **Next (finishes check 2):** one real KernelBook episode through prime-rl's verifiers (`vf-run`)
  with the plugin: A10G agent box -> vLLM -> isolated grader box -> reward. Compare its trace and
  reward with the da75527 smoke test (00456-smoothl1loss).

### 2026-11-17: check 2 done: end to end on prime-rl's verifiers + Modal plugin (`configs/e2e_prime_rl.toml`)
- **Setup:** `vf-run` (renamed from `vf-eval`) in the s11 image, run detached on Modal. The
  interception server is behind `modal.forward` (same transport as s10). Config = baseline settings,
  `runtime type = "custom"` (our plugin), and `push` removed (`vf-run` has no upload option).
  Task: 00456-smoothl1loss x 4 (as in the s10 smoke test).
- **Result:** 4/4 clean, **3/4 pass** (s10 on old verifiers: 3/4).
  - Passes: 4, 15 and 30 turns.
  - Fail: 30 turns, stopped at the `runs` gate.
  - Per-gate metrics, `kernel_launches` and `max_abs_diff` come through as before; 0 leftover sandboxes.
  - Run: `work.nosync/outputs/e2e_prime_rl/...--960c9af1` (Volume `kernelbook-runs`).
- **Isolation proven** (1-episode rerun, `e2e_grader_check`, with the plugin's sandbox log made visible):
  - agent box `sb-7LW2...`: kernel-env-base, A10G:1, egress = only the interception host;
  - then a separate grader box `sb-x56J...`: same image, A10G:1, egress `domains=[] cidrs=[]` (none);
  - reward 1. Traces under both verifiers record only the agent's sandbox id, so this log line is
    the evidence.
- **Trace format:** a superset of the old one (new: `attempt`, `num_turns`, `usage`, `num_branches`,
  `is_truncated`, ...). `token_ids`/`logprobs` are empty because the eval client doesn't request them.
- **Finding: context compaction in ~80% of episodes (old and new verifiers alike).**
  - The `bash` harness compacts by default when 16K tokens remain below the context window (our
    32K server -> at ~16K tokens). The model writes a "handoff summary", then the conversation
    restarts from it, so the trace has 2 branches.
  - Compacted share: baseline batches 75% / 79% / 79%, E0 174/216 = 81%, e2e 2/4.
  - Comparability: the baseline, E0 and training all use the same behavior, so before/after
    numbers are unaffected.
  - Training: a compacted episode is 2 token sequences (pre-summary and post-summary), not 1. How
    prime-rl turns branches into samples, and whether both get the episode's reward, is now a
    question for check 3. Credit assignment differs from a single trajectory.

### 2026-11-17: part 1, check 3: the qwen3.5 renderer (`scripts/s12_renderer_check.py`, run via s11 `--which renderer`)
- **What it is:** prime-rl's `renderers` library. It turns messages into exact token ids and parses
  sampled ids back into reasoning/content/tool calls, so the trainer scores exactly the tokens vLLM
  sampled. Eval instead used vLLM's chat endpoint (HF template + `qwen3_coder` parser, tokens
  hidden). `Qwen/Qwen3.5-9B` maps to `Qwen35Renderer`.
- **Test:** real conversations from the e2e run: 4 episodes, 6 branches (2 compacted), 79 assistant
  turns, 75 tool calls. Tools passed in the training client's nested `tool_to_wire` form.
- **Result (after the patch below):**
  - renderer == HF `apply_chat_template` token for token on 6/6 branches;
  - each turn's generation prompt is an exact prefix of the full render: 79/79;
  - the renderer's parser and vLLM's `qwen3_coder` parser both recover the original calls: 79/79.
- **Bug found and patched:** `renderers/parsing.py` (pinned `a9d1676`, still on upstream main `9f4943f`)
  does `arg_value = pm.group(2).strip()`, removing ALL leading/trailing whitespace from every tool
  argument. Before the patch, 10/79 calls differed.
  - Effect: `edit` `old_str`/`new_str` lose their indentation, so edits fail to match or write
    unindented code. Training would run a different environment than eval (vLLM keeps whitespace).
  - The training tokens themselves are unaffected.
  - Fix: a one-line patch in the s11 image (`pm.group(2)`, unstripped; the regex already removes
    the template's newline per side). The build asserts the target line exists.
  - renderers' own Qwen3.5/round-trip/parsing tests with the patch: **2005 passed**.
  - Upstream report: not filed yet (outward-facing; ask first).
- **Test-harness gotcha:** a site-packages dependency ships a top-level `tests` package, which shadows
  renderers' `tests`. Fixed in the throwaway test container (`__init__.py` + PYTHONPATH).
- **Compaction -> training samples** (read from prime-rl `orchestrator/trajectories.py`,
  `algo/routing.py`, verifiers `Trace.branches`):
  - each branch becomes **one training sample** (its own token sequence, each within `seq_len`), so
    a compacted episode gives 2 samples: branch 1 = the original context up to the model's summary;
    branch 2 = a fresh context starting from the summary (as a user message, masked) + the rest;
  - GRPO computes **one advantage per episode** (reward - group mean). `assign_advantages` copies it
    to every sampled token on every trainable branch, **including the summary the model wrote**;
  - a node shared by 2 branches is trained only once (in the first branch); the compaction branch
    is trainable because the harness resumed from it;
  - consequences: ~1.8 samples per episode at our compaction rate; summary-writing is reinforced by
    the episode outcome; with token-mean loss, long (compacted) episodes contribute more tokens.

### 2026-11-17: part 1, check 4: Qwen3.5-9B in prime-rl's trainer on one H100 (`scripts/s13_trainer_check.py`, s11 `--which trainer`)
- **4a, driver vs CUDA 13:** Modal H100 has driver 610.57.04; torch 2.14.0+cu130 sees the GPU, bf16 matmul
  runs, fla 0.6.0 + flash-attn 2.8.3 import. (The base image's "NVIDIA Driver was not detected" banner
  is its entrypoint script failing to find `ldconfig`; harmless.)
- **4b, the real trainer loop** (`torchrun -m prime_rl.trainer.rl.train @ configs/check4_trainer_fake.toml`):
  LoRA r16/a32, fake data (random packed docs), 2 micro-batches x 32K tokens per step, 3 steps.
  - step 1: 1m49s (compile); steps 2-3: **9.6 s/step, ~6,800 tok/s, MFU ~23%, peak 56.2 GiB of 80**.
  - Mismatch KL 12 / grad norm 0 in that log are artifacts: the fake loader's "vLLM logprobs" are
    random, so the IPO mask drops nearly every token. Gradient flow is checked in 4c on real data.
- **4c, the engine-mismatch gap, measured.** vLLM 0.33 (prime-rl's pin) sampled at T=1 from the 76
  distinct assistant-turn prompts of the e2e run (rendered by `Qwen35Renderer`, 876-15,807 tokens):
  36,331 sampled tokens with their decode-path logprobs. prime-rl's model stack (`setup_model`: FSDP,
  AC, compile, LoRA with B=0, i.e. the base policy) scored the same tokens.
  - log p_train - log p_vllm: mean -0.00055, mean |.| 0.0073, p99 |.| 0.116, max |.| 0.378;
  - **mismatch KL (prime-rl's metric) 0.00029; IPO-masked (|dp| > 0.3): 0 of 36,331 tokens**;
  - by context position: 0-8K 0.00032, 8-16K 0.00023 (no growth with length; none past 16K here,
    since compaction keeps contexts under ~16K);
  - vLLM loads the GDN `linear_attn.norm` weights (fp32 in the checkpoint) as bf16; the trainer
    keeps fp32 master weights. This is one known contributor; the total gap is still tiny.
  - Read: at step 0 the only off-policy source is the engines, and it is ~0. Any masked fraction
    or mismatch KL seen in training is staleness / policy change, not numerics.
- **Backward on real data:** 16K-token sequence: grads reach 128/128 `lora_B` (A has zero grad at
  step 0 by design, since B = 0), peak 44.6 GiB. Fresh model at 32K tokens: 52.7 GiB.
- **LoRA targets (decision pending):** prime-rl's default targets skip the GatedDeltaNet (linear
  attention) projections, i.e. attention in 24 of 32 layers:
  - default: **29.1M** trainable (MLPs 25.2M; q/k/v/o of the 8 full-attention layers 4.0M);
  - + `in_proj_qkv`, `in_proj_z`, `out_proj`: **40.1M** (+11M); backward at 32K OK, 200/200 B grads.
  - Open: whether vLLM's Qwen3.5 can load LoRA on those modules (weight sync to inference). Test in the smoke run.
  - **Decided (2026-11-17):** smoke test uses the 40.1M set (`configs/rl_smoke.toml`, `[trainer.model.lora]`).
- **Image fixes needed to get here** (all in s11, each its own layer, sync layer stays cached):
  - `prime-kernels` wheel: the trainer imports it unconditionally, but it is only in the `kernels`
    extra (not in `all`; upstream's Dockerfile adds it);
  - `flash_attn_3` wheel (pinned with hash from uv.lock): `attn = "auto"` picks FA3 on Hopper and
    refuses to start without it;
  - standalone trainer needs `--no-monitors.prime` (it otherwise logs to the `rl` launcher's run).
- **Incidents:** vLLM's offline engine kept the process alive after finishing (H100 idle ~8 min, app
  stopped by hand); a bare `os._exit` then orphaned the engine subprocess holding ~68 GiB (gpu_memory_utilization 0.85), and the
  trainer OOMed at load. Fix: kill child processes before exiting, and print GPU memory before training.

### 2026-11-17: smoke config written; `rl --dry-run` passes (`configs/rl_smoke.toml`, s11 `--which smoke --dry-run`)
- `scripts/s14_trainable_registry.py` -> `registry-trainable.json` (dataset `kernelbook-trainable`, 138
  tasks; asserts every task dir exists and is in the train split).
- Config: 1 node, 2 H100 (1 vLLM + 1 trainer); vLLM `max_model_len` 32768 (default would be 262K);
  agent as baseline/E0/check 2 (bash, 30 turns, T=1, 8192 tokens/turn, task timeouts kept, Modal
  plugin); interception `server` + custom tunnel (s11 fills the `modal.forward` URL), 1 env worker;
  `max_inflight` 32 episodes; `max_staleness` 1; LoRA r16/a32 with the linear-attention targets.
- **`batch_size` counts samples (branches), not episodes.** With ~80% of episodes compacting into 2
  samples, 56 samples ~= 8 tasks x 4 rollouts (32 would have been ~4-5 groups).
- Dry run = parse + validate + resolve + write per-process configs, then exit (no GPU / model /
  sandboxes). Resolved configs (Volume `checks/rl_smoke_dry/`) confirm:
  - trainer: 10 LoRA targets, seq_len 32768, IPO loss (eps 0.3), AdamW, filesystem weight broadcast;
  - inference: `enable_lora` true, `max_lora_rank` 16 (auto-set from the trainer's LoRA), 32K context;
  - orchestrator: batch 56, group 4, staleness 1, renderer `qwen3.5`, GRPO token-mean, our taskset
    + `modal_runtime.ModalBackend`, 30 turns.
- **Not set by us, needs a decision: learning rate.** The trainer default is `lr = 1e-6` (sized for
  full fine-tuning); LoRA usually needs ~10-50x more (prime-rl's own LoRA example uses 5e-5).
  **Decided: `lr = 2e-5`** for the smoke run (`[trainer.optim]`).
- **Prime platform monitor is on by default** (`monitors.prime` in trainer and orchestrator): a real
  launch would register a run on Prime's platform. Turned off at launch (`--no-*.monitors.prime`);
  metrics stay local in the file monitor (`metrics.jsonl` under the run dir). Rerun dry run with the
  flags: `prime: None` in trainer + orchestrator, trainer `lr` 2e-05.
- **Launch wired:** s11 `smoke_run` (`--which smoke`, no `--dry-run`): one container with 2 H100s,
  `rl` runs trainer + vLLM + orchestrator + env worker; interception via `modal.forward(8765)`; run
  dir on local disk (vLLM reads weight broadcasts from it), logs/configs/metrics mirrored to the
  Volume `rl_smoke/` every minute (weights excluded); HF cache Volume for the model; 8 h timeout.

### 2026-11-18: smoke launch stopped before start (GPU capacity question); docs updated
- `--which smoke` launched (app `ap-RdYM1X7BjpI6ZC3MTIt3r1`); after ~10 min Modal had still not
  scheduled the 2x H100 container (0 tasks, nothing billed). Stopped it.
- Open question before relaunching: the Modal workspace GPU limit (10 GPUs in Oct; the reason the
  baseline ran 4 episodes at once). The smoke config holds 2 H100 + up to 32 agent A10Gs + graders.
  Either raise the limit or cap `max_inflight` (~3 at 10 GPUs, which makes 15 steps take many hours).
- **Decided (user):** the cap stays at 10 GPUs; minimal smoke run: `max_inflight` 4 (2 H100 + 4 x
  (agent + grader A10G) = 10 worst case, as the baseline), `batch_size` 16 samples (~2 groups),
  `max_steps` 5, launch timeout 6 h. The real run needs ~56 samples/step, so a higher GPU limit
  (or a ~15 h run) then. `rl --dry-run` re-run: resolves with these values.
- **Smoke attempt 1** (`ap-7Yij8wCtK4u6Yl3JSHOhWp`): container got 2 H100s in ~1 min this time; env
  worker started (taskset loaded, interception server on 8765). Failed in 1 min: `inference` always
  starts `vllm-router` in front of the vLLM engine (even single node), and the binary is in the
  `disagg` extra our lean image skips -> `FileNotFoundError: vllm-router`; trainer then got SIGTERM
  from the launcher. Fix: lockfile-pinned `vllm-router` 0.2.4 wheel (with hash) as its own layer.
- Architecture page (`docs/rl-architecture.html`) and README brought up to date (Modal-hosted
  orchestrator, prime-rl stack, A10G, smoke config); pushed with the check 2-4 commits.

### 2026-11-18: correction: `batch_size` counts traces (episodes), not branch samples
- Read `orchestrator/train_sink.py`: episodes wait in `pending_groups` until the whole group is graded;
  then `score_group` computes advantages over all 4; zero-advantage (unanimous) groups are dropped;
  each surviving trace goes to `pending_batch` with ALL its samples (1-2 branches). A batch is the
  first `batch_size` queued **traces** (`list(pending_batch.items())[: batch_size]`).
- So the config docstring "Samples to train on per step" is misleading. The earlier entry's
  "56 samples ~= 8 groups" is wrong: 56 would be 14 groups. Smoke `batch_size = 16` = 4 groups.
  Real run for 8 groups: `batch_size = 32`.
- A group can be split across batches (queue at 14 + a group of 4 -> 16 cut, 2 wait), but its
  advantages were computed on the whole group first; the remainder just trains one step later
  (one version staler; may be stale-masked at `max_staleness = 1`). With size-4 groups and a
  multiple-of-4 batch, splits only follow dropped traces (errors, stale).
- Effect on the smoke run: ~24-28 episodes per step (unanimous groups are dropped), ~50-70 min per
  step at 4 in flight; the 6 h timeout may end it at step 4-5. Weight sync is proven by step 1-2.

### 2026-11-18: smoke attempt 2 running; step 1 trained; compaction silently OFF under `rl`
- Run `rl_smoke/kernelbook--qwen3.5-9b--d36ba7f0` (app `ap-uhkFllfgCPRHoCCUx1buZ2`): vLLM up behind
  the router in 3m21s; v0 adapter synced; 4 episodes in flight; A10G sandboxes, tunnel-only egress,
  isolated grading all as in check 2.
- vLLM LoRA: 110 modules "will be ignored", **all `visual.*`** (unused vision tower); none of the
  language model's projections, so LoRA wraps the linear-attention (GDN) layers too.
- Orchestrator step 1: 19m19s, ~7 groups rolled for 16 trainable traces (unanimous groups
  dropped), reward 0.4375, staleness 0, Turns 22.6, **Branches 1.0**, truncation 62.5%.
- Trainer step 1: loss 0.0085, entropy 0.48, **mismatch KL 0.0002** (check 4: 0.0003), grad norm
  0.018, peak 53.5 GiB; first step includes TileLang/GDN kernel compile.
- **Finding: no compaction.** A trace failed: vLLM 400 "decoder prompt (length 36001) is longer than
  the maximum model length". verifiers' bash harness compacts at (context window - 16,384) when it
  can discover the window via `/v1/models` (`max_model_len` on the card whose id == model);
  otherwise compaction is OFF. Under `rl` discovery evidently fails (the router / adapter-named
  model), and vLLM's error wording isn't in `CONTEXT_OVERFLOW_MARKERS`, so overflow ends the trace
  instead of compacting. Baseline/E0 (vLLM serve directly) discovered 32,768 -> threshold 16,384.
  - Comparability: training episodes would run without compaction while E0 ran with it.
  - Fix for the next run: `env.agent.harness.compaction.summarize_at_tokens = 16384` (pinned,
    same threshold). Not applied to the running smoke run.

### 2026-11-18: smoke attempt 2 cancelled at 08:22 (client side); launch now uses `.spawn()`
- Logs stop at 08:23:42 with every component healthy (step-2 batch at 7/16, vLLM serving,
  sandboxes starting); no traceback, no final mirror. Modal app log: `08:22:46 Received a
  cancellation signal while processing input ... Successfully canceled input`.
- Cause: `smoke_run.remote()` ties the function call to the local `modal run` client. The Mac slept
  and the client later died (`AuthError: Jwt is expired`); Modal treated that as a cancel of the
  input. `--detach` keeps the app, not a `.remote()` call whose caller is gone.
- Fix: the entrypoint `spawn()`s `smoke_run` and returns; nothing on the Mac must stay connected.
- Proven before the cancel: step 1 end to end (rollouts -> trainer update, mismatch KL 0.0002,
  53.5 GiB); episodes kept running after the trainer's v1 broadcast. Not yet proven: steps 2-5.
- Relaunch also carries the compaction fix (`summarize_at_tokens = 16384`).

### 2026-11-18: smoke attempt 3 (spawned): steps 1-2 complete, then Modal preemption
- App `ap-FhzJfqylaHFm2eGpOJgeyF`, run `rl_smoke/kernelbook--qwen3.5-9b--56a9c1c6`, compaction pinned.
- Orchestrator: step 1 28m33s, reward 0.31, Branches 2.1, Turns 25.8, staleness 0, truncation 69%;
  step 2 19m28s, reward 0.38, Branches 2.2, Turns 25.1, **staleness 0 => all step-2 tokens sampled by v1**.
- Trainer: step 1 loss 0.0051, entropy 0.47, mismatch KL 0.0002, grad norm 0.017, 53.5 GiB;
  step 2 loss 0.0086, entropy 0.49, **mismatch KL 0.0003 with the trained v1 adapter**, grad norm
  0.018, 48.3 GiB, 4,550 tok/s (step 1 included kernel compiles).
- **Proven:** full loop twice; vLLM serves a trained LoRA adapter that includes the linear-attention
  (GDN) projections (no fallback to default targets needed); vLLM/trainer agreement holds after
  updates; memory stable; trainer ~1-2 min per step, rollouts are the bottleneck.
- ~17:21: `Container terminated due to preemption. Your Function will be restarted with the same
  input.` Modal re-ran `smoke_run` from scratch on a new container (run `...c7cf0b91`, 17:22:56):
  no checkpoint, so it restarts at step 0.
- **Lesson for the real run:** must be resumable: `[ckpt] interval`, a fixed `run.name` (resume needs
  a predictable run dir), and checkpoints on the Volume, not the container's local disk.
- Open (from this run): episodes compact twice far more often than baseline/E0 (57% vs 12%) with the
  same 16,384 threshold and the same ~620 output tokens per turn; cause is in how the rl path
  (interception -> renderer -> /inference/v1/generate) builds or reports context. To diagnose
  before the real run (comparability with E0).

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
| Host side (vf-eval / orchestrator) on Modal, not the Mac | Removes Prime's tunnel and laptop sleep/DNS/lid from the loop; same place training's orchestrator must run |
| No KL penalty to the reference (beta = 0) | Verifiable reward (no reward model to over-optimize); policy must move far from 20% pass@1; LoRA + IPO/clip already limit drift. Revisit on entropy collapse / format drift |
| Importance ratio against vLLM log probs, small `max_staleness` | One ratio corrects async lag, mid-episode swaps and engine mismatch; keeping lag small keeps the advantage approximation valid |
| Lean prime-rl image (gpu + flash-attn extras), `uv sync --frozen` | Single node, 2 H100s needs no UCX/NIXL/EFA; `--frozen` = prime-rl's exact tested versions (`--locked` fails on its rolling cooldown) |
| Modal sandboxes via our own verifiers runtime plugin (training) | prime-rl's verifiers dropped built-in Modal; a plugin keeps A10G + same images as baseline/E0 (comparability) |
| Patch renderers' Qwen3.5 tool-arg `.strip()` in our image | Keeps tool arguments byte-identical to vLLM's qwen3_coder parser (baseline/E0); otherwise training-time edits lose indentation |
| LoRA also on the linear-attention projections (`in_proj_qkv`, `in_proj_z`, `out_proj`) | Default targets leave attention in 24 of 32 layers frozen; +11M params (40.1M), trainer fits at 32K. Fallback to default if vLLM can't load LoRA on those modules |
| Add `prime-kernels` + `flash_attn_3` (lockfile-pinned wheels) to the lean image | The trainer needs both on H100; same versions upstream's Dockerfile installs |
| Before/after on dev (E0), not train | Train tasks get trained on (memorization) and were selected by noisy scores (regression to the mean) |

## 5. Threats to validity / caveats for the write-up

- **Pass-rate noise:** 4 samples per task; "trainable" selection is partly luck (regression to the mean).
- **Turn limit dominates:** 80% of baseline episodes hit 30 turns; results are specific to this budget.
- **Grader gaps:** guard is a heuristic static scan; seeds (0,1,2) are readable in the agent box
  (lookup-table hack possible); edits to the agent's own `reference.py` aren't tracked.
- **Hardware change:** validation on L4, rollouts on A10G.
- **Token data:** eval traces (vf-eval `EvalClient`) have no token ids / logprobs; training must use the prime-rl client.
- **Off-policy data:** training batches are up to `max_staleness` versions old and carry vLLM-vs-trainer numeric
  mismatch; report the IPO-masked fraction and mismatch KL so readers can judge how on-policy the run was.
- **Context compaction:** ~80% of episodes (baseline, E0) hit the harness's compaction at ~16K tokens and
  continue from a model-written summary. Results are for this agent setup (32K context, 30 turns, compaction on).
- **Infra failures excluded:** errored episodes are dropped from pass rates, not counted as fails.

## 6. Ideas for the report / paper

- Figures: [`visuals/`](visuals/README.md) has the animated prime-rl loop (async + staleness) and the
  trainer-step zoom (ratio, IPO mask, loss, LoRA backward, AdamW, adapter sync). Self-contained HTML.

- Infra lessons for agentic RL: untimed sandbox creation, laptop-hosted interception as SPOF,
  spend limits, log lines that report reward before grading.
- pass@1 vs pass@4 gap (21.8% vs 45.5%): RL as reliability vs new capability; track both.
- Effect of std normalization (GRPO vs Dr. GRPO) and loss aggregation (token / sequence / prompt).
- Turn-budget behavior: does RL reduce max_turns endings?

## 7. Next

1. ~~Dev eval E0~~ done: 21.9% / 42.6%.
2. ~~Check prime-rl GRPO settings~~ done: no std, token-mean, IPO loss (see log).
3. ~~Baseline the remaining 182 train tasks~~ done: 382 tasks, 20.0% / 41.4%, 138 trainable.
4. ~~Move host side off the Mac / Prime tunnel~~ done: `s10_modal_eval.py` (smoke 4/4 clean).
5. Training-loop smoke test on the 138 trainable tasks (~10-20 steps). Log: IPO-masked fraction,
   mean |ratio - 1|, staleness histogram, mismatch KL(mu || pi_theta), entropy, per-turn policy version.
6. First real training run; dev eval after (same config); compare with E0.
   - Comparison: only the weights may differ from E0. Plan: merge the LoRA adapter into the base
     weights, serve with `s09_serve.py`, rerun `configs/eval_dev.toml` unchanged (old stack, like
     E0); or rerun E0 on the new stack as the control. Report dev pass@1/pass@4 (headline), the
     train reward curve vs the train baseline, and turns, compaction rate and tool errors per episode.
