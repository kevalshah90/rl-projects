"""Step 10 check: rl_scratch/grpo.py against values worked out by hand. CPU only, under a second.

Run:  uv run pytest tests/test_grpo.py
"""

import pytest
import torch

from rl_scratch.grpo import group_advantages, informative_groups, policy_loss


# ---- advantages --------------------------------------------------------------------------

def test_advantages_center_on_group_mean():
    rewards = torch.tensor([1., 0., 0., 1.,   0., 0., 0., 0.])
    expected = torch.tensor([.5, -.5, -.5, .5,   0., 0., 0., 0.])
    assert torch.allclose(group_advantages(rewards, group_size=4), expected)


def test_advantages_divide_by_std_when_asked():
    # Group 1: mean 0.5, unbiased std sqrt(4 * 0.25 / 3) = 0.5774 -> 0.5 / 0.5774 = 0.8660.
    rewards = torch.tensor([1., 0., 0., 1.,   0., 0., 0., 0.])
    expected = torch.tensor([.8660, -.8660, -.8660, .8660,   0., 0., 0., 0.])
    assert torch.allclose(group_advantages(rewards, 4, normalize_std=True), expected, atol=1e-4)


def test_informative_groups_drop_unanimous_groups():
    rewards = torch.tensor([1., 0., 0., 1.,   0., 0., 0., 0.,   1., 1., 1., 1.])
    expected = torch.tensor([True] * 4 + [False] * 8)
    assert torch.equal(informative_groups(rewards, 4), expected)


# ---- clipping (one token, so every aggregation gives the same number) --------------------

def one_token_loss(ratio: float, adv: float) -> torch.Tensor:
    new = torch.tensor([[torch.log(torch.tensor(ratio))]], requires_grad=True)
    loss = policy_loss(new, torch.zeros(1, 1), torch.tensor([adv]), torch.ones(1, 1),
                       torch.tensor([0]))
    loss.backward()
    return loss, new.grad


@pytest.mark.parametrize("ratio, adv, expected_loss, clipped", [
    (1.5,  1.0, -1.28, True),    # A > 0: gain capped at 1 + 0.28, no further push up
    (0.5,  1.0, -0.50, False),   # A > 0, ratio fell: pessimistic min keeps 0.5, still pushed up
    (0.5, -1.0,  0.80, True),    # A < 0: cap at 1 - 0.2, no further push down
    (1.5, -1.0,  1.50, False),   # A < 0, ratio rose: min keeps -1.5, still pushed down
])
def test_clipping(ratio, adv, expected_loss, clipped):
    loss, grad = one_token_loss(ratio, adv)
    assert loss.item() == pytest.approx(expected_loss, abs=1e-6)
    assert (grad.abs().item() == 0) == clipped     # clipped region: zero gradient


# ---- aggregation -------------------------------------------------------------------------

def three_completions():
    """Prompt 0: completions of 100 and 10 tokens; prompt 1: one of 10 tokens.
    ratio = 1 (new = old), so each token's loss is just -A, and each completion's
    token-mean is -A. Advantages 1, 2, 4 make each completion's weight visible."""
    T = 100
    mask = torch.zeros(3, T)
    mask[0, :100] = 1
    mask[1, :10] = 1
    mask[2, :10] = 1
    logp = torch.zeros(3, T)
    return logp, logp.clone(), torch.tensor([1., 2., 4.]), mask, torch.tensor([0, 0, 1])


@pytest.mark.parametrize("agg, expected", [
    ("token",    -(100 * 1 + 10 * 2 + 10 * 4) / 120),   # -1.3333: weights 100/120, 10/120, 10/120
    ("sequence", -(1 + 2 + 4) / 3),                      # -2.3333: weights 1/3 each
    ("prompt",   -((100 * 1 + 10 * 2) / 110 + 4) / 2),   # -2.5455: weights 100/220, 10/220, 1/2
])
def test_aggregation_weights(agg, expected):
    new, old, adv, mask, groups = three_completions()
    assert policy_loss(new, old, adv, mask, groups, agg=agg).item() == pytest.approx(expected)


def test_padding_is_ignored():
    # Garbage logprobs on padded positions must not change the loss.
    new, old, adv, mask, groups = three_completions()
    noisy = new.clone()
    noisy[mask == 0] = 5.0
    for agg in ("token", "sequence", "prompt"):
        assert policy_loss(noisy, old, adv, mask, groups, agg=agg).item() == \
            pytest.approx(policy_loss(new, old, adv, mask, groups, agg=agg).item())


def test_unknown_aggregation_raises():
    new, old, adv, mask, groups = three_completions()
    with pytest.raises(ValueError):
        policy_loss(new, old, adv, mask, groups, agg="mean")
