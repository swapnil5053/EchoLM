# ADR-0001: EchoLM architecture

- Status: accepted (revised 2026-09-26, see Revisions)
- Date: 2026-09-25
- Context: WeClone audit (`AUDIT-weclone.md`)

## Context

EchoLM fine-tunes a small LLM to reply the way one specific person writes in chat. Inputs are that person's WhatsApp (`.txt`) and Telegram (`result.json`) exports. The training target is a single laptop: RTX 4060 Laptop (8 GB VRAM), 16 GB RAM, Windows.

WeClone shows the concept works, but delegates everything to LLaMA-Factory, has no evaluation, and has a label-leak in its conversation-start samples. EchoLM keeps the concept and the time-window grouping idea. All code is new.

## Decision 1: Data flow

```
raw export (.txt / .json)
  │  parse/whatsapp.py, parse/telegram.py
  ▼
Msg(ts, sender, text, kind, chat_id, is_me)          # one record per message
  │  clean.py: drop system/media stubs, PII scrub (mask, don't drop), dedup
  ▼
sessions                                              # split on gap > 30 min
  │  window.py: merge consecutive same-sender messages → turns
  │             for each of MY turns: context = previous turns in session,
  │             newest first, up to k=8 turns / 512 tokens
  ▼
windows: {chat_id, ts, context[], target}
  │  split.py: split by time — last 10% of sessions per chat → test,
  │            previous 5% → val; never split within a session
  ├──────────────► sft.jsonl   (chat-template messages; loss on target only)
  ├──────────────► grpo.jsonl  (prompt = context; target kept as reference for rewards)
  └──────────────► profile.json (my stylometric fingerprint, computed on train only)
  ▼
train_sft (Unsloth + TRL SFTTrainer) ─► lora-sft/
  ▼
train_grpo (TRL GRPOTrainer, starts from lora-sft) ─► lora-grpo/
  ▼
evaluate: base vs sft vs sft+grpo on the test windows ─► report.json + W&B
  ▼
demo (Gradio, model switcher) · model card generator
```

Parsing decisions:
- **WhatsApp:** the timestamp format depends on phone locale (`dd/mm/yy` vs `m/d/yy`, 12 vs 24 hour, with or without seconds, `[...]` brackets on iOS). Detect the format from the first ~50 lines instead of guessing per line. Lines without a timestamp continue the previous message (keep the newlines — they are a style signal). Drop `<Media omitted>`, "This message was deleted", encryption notice, and join/leave lines. The user picks "me" by sender name via a CLI flag.
- **Telegram:** `from_id` identifies "me". `text` may be a list of entity fragments — join them. Keep `reply_to_message_id`, because a quoted reply is a better context anchor than "previous message".
- **PII:** mask in place (`<PHONE>`, `<EMAIL>`, `<URL>`) instead of dropping the message. Dropping silently breaks turn pairing (a WeClone bug).
- **No leaked targets:** conversation-start turns (I spoke first) get an empty/"(new conversation)" context, never the answer in the prompt.
- **Group chats:** off by default. When on, context turns are prefixed with the speaker's name.
- **Contact balancing:** cap windows per chat (e.g. 30% of the dataset) so one chat doesn't define "my style".

## Decision 2: Qwen 2.5 1.5B Instruct, 4-bit, LoRA r=64

**Primary reason: it fits.** Estimated budget for GRPO, which is the tighter stage:

| Item | Approx. |
|---|---|
| 1.5B weights in 4-bit | ~1.2 GB |
| LoRA r=64 on all 7 linear projections (~74M params) + Adam states | ~1.0 GB |
| vLLM generation (weights + KV cache, `gpu_memory_utilization≈0.35`) | ~2.5 GB |
| Activations, 4 × (512 prompt + 128 completion) tokens, gradient checkpointing | ~1–1.5 GB |
| **Total** | **~6–6.5 GB of 8** |

3B in 4-bit roughly doubles weights and LoRA and leaves no room for vLLM plus a group of 4. 7B only fits SFT, and only with small batches. These are estimates; session 3 measures real peak memory before any config is committed.

**Secondary reason (the one worth putting in the README):** style is mostly a surface-level target — length, casing, punctuation, emoji, code-switching, recurring phrases. A 1.5B model has the capacity for that. What it lacks is slack: it can't smooth over noisy data, so improvements have to come from curation and reward design, which can be measured. A large model that sounds like you may just be memorizing; the memorization metric in Decision 6 is what makes this claim checkable rather than rhetorical.

Stated limits: weaker world knowledge and multi-turn coherence than 7B. This will show in the demo.

**On r=64:** a few thousand personal windows plus r=64 on all projections can overfit or memorize. Default is r=64, alpha=64, dropout 0, one ablation run at r=16. Keep r=64 only if memorization rate and val loss support it.

## Decision 3: SFT first, then GRPO

SFT teaches the output distribution: reply length, register, language mix, what "a reply" looks like in this chat. GRPO without that starts from an assistant-style policy, samples almost never look like the user, rewards across the group barely differ, and the group-relative advantage is near zero. So there's nothing to learn from.

After SFT, the samples in a group already vary in how "me" they are, so the rewards separate them, and GRPO can push toward the tail of the style distribution that SFT's averaging loss washes out (SFT tends toward a bland, most-common reply).

GRPO starts from the SFT LoRA. The KL term (`beta`, start 0.04) is measured against the SFT policy, not the base model.

## Decision 4: GRPO over DPO

- DPO needs chosen/rejected pairs. There are no human preference labels for "sounds like me", and labeling them yourself doesn't scale.
- GRPO samples G completions per prompt and ranks them within the group with a programmatic reward. Style consistency can be computed automatically, so GRPO fits this problem.
- GRPO is on-policy: it corrects the SFT model's own failure modes (e.g. drifting into assistant tone) rather than a fixed pairwise snapshot.

**Considered alternative, kept as a fallback:** DPO with synthetic pairs — chosen = the real reply, rejected = the SFT model's reply (or the base model's). It's cheap, offline and stable. The weakness is that "rejected" is sometimes as good as or better than the real reply, which adds label noise. If GRPO turns out unstable or too slow on 8 GB, this is the plan B, and the same eval harness compares them.

Config start: `num_generations=4` (raise to 6/8 if peak VRAM < 6.5 GB), `max_prompt_length=512`, `max_completion_length=128`, temperature 0.9, lr 5e-6, beta 0.04, Unsloth `fast_inference=True` (vLLM).

**Platform constraint:** vLLM doesn't run natively on Windows. Without it, GRPO generation goes through HF `generate` and runs several times slower. Decision: run training in **WSL2 (Ubuntu)** with CUDA passthrough, with WSL's memory set to ~12 GB in `.wslconfig`. The parsing, eval and demo code stays OS-agnostic.

## Decision 5: What "style consistency" means operationally

A **style profile** is computed from my train-split target messages only. Features per reply:

| Group | Features |
|---|---|
| Length | chars, words, messages-per-burst (newline count) |
| Casing | lowercase-start rate, all-caps word rate, "i" vs "I" |
| Punctuation | terminal punctuation rate, `!`/`?`/`...` rate, repeated punctuation (`!!`, `??`) |
| Emoji/emoticon | per-char emoji rate, top-k emoji set, `:)`/`xd`-style emoticons |
| Lexical | signature lexicon: tokens with high log-odds of me vs the people I talk to (slang, fillers, abbreviations like "ngl", "bro", "haan") |
| Formality | contractions, abbreviations, and a small formal-marker list ("regards", "kindly", full sentences with capitalization) |

**Rewards** (each scaled to [0, 1], weighted sum):

| Reward | Definition | Start weight |
|---|---|---|
| `style` | 1 − normalized distance between the completion's feature vector and the profile (z-scored per feature, clipped) | 0.40 |
| `length` | `exp(-|log(len_c+1) − log(len_ref+1)|)` against the real reply to this context | 0.20 |
| `lexicon` | fraction of completion tokens in my signature lexicon, capped so repeating one word doesn't pay off; plus a small term for overlap with the reference reply | 0.20 |
| `consistency` | penalty when a small NLI model (DeBERTa-v3 MNLI, on **CPU**) says the reply contradicts the context | 0.20, off in first run |

Guards (subtracted, not weighted): repeated n-gram penalty, "assistant-isms" penalty ("As an AI", "I'd be happy to help", "Certainly!"), empty or truncated completion.

**Factual consistency, defined narrowly:** don't contradict what's in the context window. There's no ground truth for facts about your life, so "factual" can't mean more than that. The NLI model runs on CPU so it doesn't use VRAM, but it slows each step, so it's switched on only after the cheap rewards are stable.

**Response latency:** kept out of the rewards. A model produces text, not timing, so latency can't be rewarded. It is used in the data (burst merging and session splits) and reported as dataset statistics.

**Reward hacking** is the main risk: length and lexicon rewards alone can be gamed by padding and keyword stuffing. Mitigations: the caps above, the KL term, guards, and logging 5 sample completions to W&B every 50 steps to read them.

## Decision 6: Evaluation — does it sound like me, or just like a chatbot?

All three models (base with the same system prompt, SFT, SFT+GRPO) generate replies to the same held-out **test windows** (later in time than any training data), same sampling settings, 3 seeds.

| Metric | Measures | Why it's there |
|---|---|---|
| **Authorship classifier** | % of generated replies a held-out classifier attributes to me | The headline number. Char n-gram logistic regression trained on real messages (me vs the people I talk to), on a split disjoint from GRPO data. Its features differ from the reward features so GRPO isn't graded on its own reward |
| **Stylometric divergence** | Jensen–Shannon divergence per feature distribution (length, emoji rate, punctuation, casing) between generated and real test replies | Shows *which* style dimension improved |
| **Held-out NLL / perplexity** | Likelihood of my real test replies under each model | Standard, cheap, catches overfitting |
| **Assistant-ism rate** | % replies containing chatbot phrases | Directly measures "just a chatbot" |
| **Memorization rate** | % replies sharing an 8-gram with any training target | Checks copy vs generalize; also a privacy check |
| **Context relevance** | Embedding similarity between reply and last context turn, vs the real reply's similarity | Catches a model that sounds like me but ignores the message |
| **Blind human test** | 2–3 friends see context + two replies (real vs SFT+GRPO), pick the real one; ~50 items | Only real ground truth. 50% = indistinguishable |

A win is: authorship ↑ and divergence ↓ from base → SFT → SFT+GRPO, while assistant-ism and memorization stay low and relevance doesn't drop. Report all rows, including ones where GRPO doesn't help.

## Decision 7: Tooling

- **Package:** `echolm/` with `parse/`, `data/`, `train/`, `rewards/`, `eval/`, `demo/`; Click CLI `echolm parse | format | train sft | train grpo | eval | demo | card`. Plain YAML config and dataclasses, no Pydantic (per style guide).
- **W&B:** every run logs config, dataset hash, git SHA, peak VRAM, per-reward means, and sample completions. `WANDB_MODE=offline` is supported for runs without network.
- **Demo:** Gradio with base / SFT / SFT+GRPO switching — one 4-bit base loaded once, LoRA adapters swapped, so it fits in 8 GB. Bound to `127.0.0.1`, `share=False` (WeClone defaults to a public tunnel).
- **Model card:** generated from the eval report and training config — metrics table, data description (counts only), limitations, intended use.

## Decision 8: Privacy

The prompts contain other people's messages, and a model trained on them can reproduce them.
- Exports, processed data and adapters trained on real chats are git-ignored and never pushed.
- The public repo ships a small **synthetic** demo dataset and results from it. Real-data results can go in the README as aggregate numbers only.
- The model card generator refuses to reference an adapter trained on data marked `private: true`, unless an explicit override flag is passed.

## Consequences

- EchoLM is more code than WeClone because it doesn't delegate to LLaMA-Factory. That's the point of the rebuild, but it means the parsers, windowing and rewards need their own unit tests (`tests/` mirrors `echolm/`).
- WSL2 is required for GRPO at useful speed.
- The claims in the README (small model, style not memorization, GRPO helps) are falsifiable by the eval table. If GRPO doesn't beat SFT, the ADR's fallback (synthetic-pair DPO) gets tested and the result is reported either way.

## Revisions

**2026-09-26, answers to the open questions**
1. Chats are heavily Hinglish (romanized Hindi mixed with English). All text is one vocabulary: no language detection, no language-specific tokenization, no case folding, no filtering by language. The "language mix" feature is dropped from the style profile; code-switching is captured by the signature lexicon instead ("acha", "bhai", "kya" are lexicon tokens like any other).
2. v1 uses 1:1 chats only. WhatsApp exports with more than two senders and Telegram chats that aren't `personal_chat` are skipped with a warning. Group chats are deferred.
3. Dataset size: still to be measured on the real export (`stats.json` reports it). The r=64 vs r=16 decision in Decision 2 waits on that number.

**2026-09-26, changes made while building the data pipeline**
- **Turn merging:** all consecutive messages from the same sender within a session form one turn, instead of merging only under a 2-minute gap. This keeps user/assistant turns strictly alternating for the chat template, and the 30-minute session gap already bounds how far apart they can be.
- **Deleted and media messages are kept as placeholders** (`[deleted]`, `[media]`, `[forwarded] …`) rather than dropped. Dropping them silently merges turns that were really separated by the other person (the same class of bug found in WeClone's PII handling).
- **Contact cap is off by default** (`max_chat_share: null`). With one dominant chat, a 30% cap would discard most of the data. `stats.json` reports per-chat counts so the cap can be switched on deliberately.
- **Context budget is in characters** (1500, roughly 512 Qwen tokens for romanized Hinglish) so the data step has no model dependency. The training session checks the real token lengths.
- **SFT uses TRL's conversational prompt/completion format**, so loss is only on the target reply. Your earlier messages in the context are not trained on again in every window they appear in.
