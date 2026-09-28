import pytest

from echolm.rl.grpo import advantages
from echolm.rl.grpo import trim

torch = pytest.importorskip("torch")

from echolm.rl.config import GrpoConfig  # noqa: E402
from echolm.rl.grpo import Rollout  # noqa: E402
from echolm.rl.grpo import clipped_loss  # noqa: E402
from echolm.rl.grpo import completion_logprobs  # noqa: E402
from echolm.rl.grpo import update  # noqa: E402
from echolm.rl.policy import Policy  # noqa: E402


def test_group_advantages():
    adv = advantages([1.0, 2.0, 3.0], "group")
    assert adv[1] == 0 and adv[0] == pytest.approx(-adv[2]) and adv[2] == pytest.approx(1.2246, abs=1e-3)
    assert advantages([0.5, 0.5], "group") == [0.0, 0.0]
    assert advantages([1.0, 3.0], "none") == [-1.0, 1.0]


def test_trim_keeps_first_end_token():
    assert trim([5, 6, 2, 0, 0], {2}) == [5, 6, 2]
    assert trim([5, 6, 7], {2}) == [5, 6, 7]


def test_clipped_loss_gradient():
    logp = torch.zeros(3, requires_grad=True)
    clipped_loss(logp, logp.detach(), 2.0, 0.2).backward()
    # on-policy the ratio is 1, so this is REINFORCE: d(-A * logp)/d logp = -A for every token
    assert torch.allclose(logp.grad, torch.full((3,), -2.0))


def test_clipping_stops_the_gradient_outside_the_trust_region():
    old = torch.zeros(2)
    logp = torch.tensor([0.5, -0.5], requires_grad=True)
    clipped_loss(logp, old, 1.0, 0.2).backward()
    # token 0: ratio e^0.5 > 1.2 with positive advantage is clipped; token 1 is below the region
    assert logp.grad[0] == 0 and logp.grad[1] != 0


def test_update_raises_the_likelihood_of_the_better_completion():
    from tests.tiny import build_model

    torch.manual_seed(0)
    model = build_model(40)
    policy = Policy(model, None, lambda m: m.eval(), lambda m: m.train())
    prompt, good, bad = [1, 5, 6], [7, 8, 2], [9, 10, 2]
    before = completion_logprobs(model, prompt, good).sum().item()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    runs = [Rollout(prompt, good, "g", 1.0, 1.0), Rollout(prompt, bad, "b", 0.0, -1.0)]
    stats = update(policy, runs, GrpoConfig(max_grad_norm=10.0), opt)
    assert stats["active"] == 2
    assert completion_logprobs(model, prompt, good).sum().item() > before


def test_update_skips_groups_without_signal():
    policy = Policy(None, None, None, None)
    runs = [Rollout([1], [2], "x", 0.5, 0.0)]
    assert update(policy, runs, GrpoConfig(), None)["active"] == 0
