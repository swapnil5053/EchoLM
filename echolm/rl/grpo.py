import logging
import statistics
from dataclasses import dataclass

from echolm.rl.config import GrpoConfig
from echolm.rl.policy import Policy
from echolm.rl.policy import eos_ids

log = logging.getLogger(__name__)


@dataclass
class Rollout:
    prompt_ids: list[int]
    completion_ids: list[int]
    text: str
    reward: float
    advantage: float
    old_logp: object = None


def advantages(rewards: list[float], scale: str) -> list[float]:
    mean = statistics.fmean(rewards)
    centered = [r - mean for r in rewards]
    if scale == "none":
        return centered
    std = statistics.pstdev(rewards)
    if std < 1e-6:
        return [0.0] * len(rewards)
    return [c / (std + 1e-4) for c in centered]


def trim(ids: list[int], eos: set[int]) -> list[int]:
    # keep the first end token so the model is also trained on when to stop; generate() only pads
    # after an end token, so a completion without one ran to max_new_tokens and is kept whole
    for i, t in enumerate(ids):
        if t in eos:
            return ids[:i + 1]
    return ids


def sample_group(policy: Policy, prompt_ids: list[int], n: int, max_new: int,
                 temperature: float, top_p: float) -> list[list[int]]:
    import torch

    model, tok = policy.model, policy.tok
    ids = torch.tensor([prompt_ids], device=model.device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                             temperature=temperature, top_p=top_p, max_new_tokens=max_new,
                             num_return_sequences=n, pad_token_id=pad)
    eos = eos_ids(policy)
    return [trim(row[len(prompt_ids):].tolist(), eos) for row in out]


def completion_logprobs(model, prompt_ids: list[int], completion_ids: list[int]):
    import torch

    ids = torch.tensor([prompt_ids + completion_ids], device=model.device)
    logits = model(input_ids=ids).logits[0, len(prompt_ids) - 1:-1].float()
    target = torch.tensor(completion_ids, device=model.device).unsqueeze(-1)
    return torch.log_softmax(logits, dim=-1).gather(-1, target).squeeze(-1)


def clipped_loss(logp, old_logp, adv: float, clip_eps: float):
    import torch

    ratio = torch.exp(logp - old_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    return -torch.min(unclipped, clipped).sum()


def update(policy: Policy, rollouts: list[Rollout], cfg: GrpoConfig, optimizer) -> dict:
    import torch

    active = [r for r in rollouts if r.advantage != 0 and r.completion_ids]
    n_tokens = sum(len(r.completion_ids) for r in active)
    if not active:
        return {"loss": 0.0, "grad_norm": 0.0, "active": 0}
    params = [p for p in policy.model.parameters() if p.requires_grad]
    policy.to_training(policy.model)
    total, norm = 0.0, 0.0
    for _ in range(cfg.num_iterations):
        for r in active:
            logp = completion_logprobs(policy.model, r.prompt_ids, r.completion_ids)
            if r.old_logp is None:
                r.old_logp = logp.detach()
            # token-level normalization: every generated token weighs the same across the step
            loss = clipped_loss(logp, r.old_logp, r.advantage, cfg.clip_eps) / n_tokens
            loss.backward()
            total += loss.item()
        norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {"loss": total / cfg.num_iterations, "grad_norm": norm, "active": len(active)}
