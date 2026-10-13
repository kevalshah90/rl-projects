"""Step 10: the math of one GRPO update, small enough to check by hand on a CPU.

Where it sits in one training step (the loop itself is step 11):
    vLLM samples G completions per prompt, with each sampled token's logprob  -> old_logp
    each completion is graded                                                 -> reward in {0, 1}
    group_advantages(rewards)          how much better each attempt did than its siblings
    informative_groups(rewards)        drop groups where every attempt scored the same
    trainer recomputes logprobs under the current weights                     -> new_logp
    policy_loss(new_logp, old_logp, ...)  clipped surrogate, aggregated by token/sequence/prompt
    backward, optimizer step, copy the new weights into vLLM

Shapes used throughout:
    G          attempts (completions) per prompt; completions are ordered group by group
    rewards    [num_prompts * G]
    adv        [B]      one advantage per completion (B = number of completions)
    *_logp     [B, T]   per-token logprobs, padded to the longest completion T
    mask       [B, T]   1.0 on real completion tokens, 0.0 on padding
    group_ids  [B]      which prompt each completion belongs to
"""

import torch


def group_advantages(rewards: torch.Tensor, group_size: int, normalize_std: bool = False) -> torch.Tensor:
    """Advantage of each completion = its reward minus the mean reward of its group.

    Default is Dr. GRPO: center only. Classic GRPO also divides by the group's standard
    deviation (normalize_std=True), which inflates near-unanimous groups: one pass among 8
    fails gets a large push. Kept as an option for the step 11 exercise.

    Example (G = 4, two prompts):
        rewards = [1, 0, 0, 1,   0, 0, 0, 0]
        -> [+0.5, -0.5, -0.5, +0.5,   0, 0, 0, 0]        (second group: no signal)
        normalize_std=True: first group's std = 0.577 -> [+0.866, -0.866, -0.866, +0.866, 0, 0, 0, 0]
    """
    r = rewards.view(-1, group_size)                 # [num_prompts, G]
    adv = r - r.mean(dim=1, keepdim=True)
    if normalize_std:
        adv = adv / (r.std(dim=1, keepdim=True) + 1e-6)   # all-equal group: 0 / eps = 0
    return adv.view(-1)


def informative_groups(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """Per completion: True if its group's rewards are NOT all equal (the group can teach).

    A group where every attempt scored the same has all advantages 0: training on it
    costs compute and changes nothing. Step 9's "trainable" tasks are the ones whose
    groups are informative.

    Example (G = 4): rewards = [1, 0, 0, 1,  0, 0, 0, 0,  1, 1, 1, 1]
        -> [True]*4 + [False]*4 + [False]*4
    """
    r = rewards.view(-1, group_size)
    keep = r.max(dim=1).values != r.min(dim=1).values    # [num_prompts]
    return keep.repeat_interleave(group_size)            # [num_prompts * G]


def policy_loss(new_logp: torch.Tensor, old_logp: torch.Tensor, adv: torch.Tensor,
                mask: torch.Tensor, group_ids: torch.Tensor,
                clip: tuple[float, float] = (0.2, 0.28), agg: str = "token") -> torch.Tensor:
    """PPO-style clipped surrogate (no critic, no KL term), averaged over real tokens.

    Per token:
        ratio     = exp(new_logp - old_logp)      how far the policy moved since sampling
        surrogate = min(ratio * A, clamp(ratio, 1 - clip[0], 1 + clip[1]) * A)
        loss      = -surrogate
    The min makes the update pessimistic: once ratio leaves the clip range in the direction
    the advantage pushes, the gradient is 0, so one step can't move a token too far.
    clip = (0.2, 0.28) is DAPO's "clip-higher": a bit more room to raise rare good tokens.
    old_logp comes from vLLM at sampling time, so ratio != 1 on the first step also measures
    the train/inference numeric mismatch.

    agg decides who gets a vote. Example: prompt A has completions of 100 and 10 tokens,
    prompt B one of 10 tokens. Each completion's share of the loss:
        "token"     each token equal:       100/120, 10/120, 10/120   (long answers dominate)
        "sequence"  each completion equal:  1/3, 1/3, 1/3             (classic GRPO's 1/|o_i|)
        "prompt"    each prompt equal:      100/220, 10/220, 1/2      (Mercor's prompt_mean)
    "prompt" = a token-mean within each prompt, then the mean over prompts. Each task gets one
    vote, and inside a task every token weighs the same, so there is no length bias: averaging
    each completion over its own length first would give 1/4, 1/4, 1/2 and penalize a long
    wrong answer less per token than a short one (classic GRPO's length bias).
    """
    ratio = torch.exp(new_logp - old_logp)                          # [B, T]
    a = adv[:, None]                                                # [B, 1], broadcast over T
    clipped = torch.clamp(ratio, 1 - clip[0], 1 + clip[1])
    per_token = -torch.minimum(ratio * a, clipped * a) * mask       # [B, T], 0 on padding

    if agg == "token":
        return per_token.sum() / mask.sum()
    if agg == "sequence":
        return (per_token.sum(dim=1) / mask.sum(dim=1)).mean()     # token-mean per completion
    if agg == "prompt":                                             # token-mean per prompt group
        return torch.stack([per_token[group_ids == g].sum() / mask[group_ids == g].sum()
                            for g in group_ids.unique()]).mean()
    raise ValueError(f"agg must be 'token', 'sequence' or 'prompt', got {agg!r}")
