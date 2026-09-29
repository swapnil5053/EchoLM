# ADR-0001: EchoLM architecture

- Status: accepted
- First draft: 2026-09-25, revised after the first real SFT run and the move to a public benchmark (2026-09-29)

## Context

EchoLM fine-tunes a small LLM to reply the way one specific person texts, from their WhatsApp and Telegram exports. The target machine is one Windows laptop: RTX 4060 Laptop GPU (8 GB), 16 GB RAM. The chats are heavily code-switched (romanized Hindi and English). WeClone showed the idea works but delegates everything to LLaMA-Factory, has no evaluation, and leaks the target reply into the prompt for conversation openers. EchoLM keeps the idea and its time-gap grouping; everything else is new.

Several decisions below changed after measuring the first real run. Each says what was measured.

## 1. Data

- **Parsing.** WhatsApp's text format depends on phone locale, so day/month order is inferred from the whole file (any first field > 12 decides), with a `--date-order` override. Lines without a timestamp continue the previous message; newlines are kept because burst texting is style. Telegram is read per chat, `personal_chat` only.
- **One vocabulary.** No language detection, case folding, stemming or filtering: Hinglish and English tokens are the same stream.
- **Masking, not dropping.** URLs, emails, phone numbers and OTPs are replaced in place. Deleted, media and forwarded messages remain as placeholders. Dropping them silently glues together turns the other person interrupted, the same class of bug as WeClone's.
- **Duplicates.** Only messages with the same platform id are duplicates. WhatsApp timestamps have minute resolution, so two identical "haha"s in one minute are two real messages; an earlier version deleted one of them.
- **Sessions: 180-minute gap** (was 30). Measured: median reply gap 1 min, but 24% of replies come after more than 30 min and 10% after 3.7 h. At 30 min, 28% of windows were "openers" answering nothing; at 180 min, 14%.
- **Windows.** One per turn of yours: up to 8 previous turns / 1500 characters as context. Context turns over 600 characters (83 pasted documents in the real export, one of 63k characters) become `[long message]`. Replies containing a link are dropped: the first model learned to answer with a bare `<URL>`.
- **Split by time.** Per chat, the last 10% of sessions are test and the 5% before that val, never splitting a session. Test is always later than training.
- **No per-chat cap by default.** With one dominant chat, capping its share discards most of the data.

## 2. Model: Qwen2.5-1.5B-Instruct, 4-bit, LoRA rank 16

1.5B in 4-bit leaves room on 8 GB for LoRA training plus sampling 4 replies per prompt; measured SFT peak was 4.9 GB. Style is mostly a surface target (length, casing, code-switching, recurring phrases), which a 1.5B model can represent.

Rank 16 rather than the 64 of the first draft: the real export gives 813 training replies, and a larger adapter mostly memorizes.

## 3. SFT

- Loss on reply tokens only. Labels are built in our own code (`echolm/train/data.py`, prompt tokens set to -100) and trained with the plain transformers `Trainer`, so the loss mask does not depend on how a TRL or Unsloth version handles prompt-completion data.
- Windows longer than the sequence limit are dropped, never truncated: right truncation would cut the reply being trained on.
- **Checkpoint selection with a tolerance.** Measured on the first run: val loss 4.234 at step 80 and 4.124 at step 100 (2.6% better), while the share of replies copying a 6-word run from training went from 0.018 to 0.188 and distinct replies fell from 0.37 to 0.29. `echolm train select` takes the earliest checkpoint within 3% of the best val loss.

## 4. GRPO instead of DPO

There is no preference data for "sounds like me", but a reward can be computed for any sampled reply against the real reply to the same message. GRPO samples a group of replies per prompt and learns from their relative rewards, which needs no preference pairs and corrects the policy's own failure modes on-policy. DPO with synthetic pairs (real reply chosen, SFT reply rejected) stays the fallback if GRPO does not beat SFT.

**The measured failure is collapse, not style.** SFT removed assistant phrasing (63% → 0%) and closed 70% of the style gap, but only 29–37% of its sampled replies are unique, against 96% for the real replies. The training replies are 95% unique, so the repetition comes from the model. A reward that scores closeness to an average "style profile" would make this worse, so the reward set is built around it:

| reward | why |
|---|---|
| chrF to the real reply for that message | a stock reply matches almost none of the real replies, so it loses within its group; character n-grams work on Hinglish without any language tooling |
| style match against the real reply for that message | per-message, not per-population, so it doesn't pull every reply towards the most typical one |
| length match | stops padding and one-word collapse |
| duplicate within the group (−1) | direct pressure against collapse |
| 6-gram copy of a *different* training reply (−1) | memorization |
| assistant phrases, empty reply (−1) | keeps SFT's gains |

## 5. GRPO implementation

Written from scratch in `echolm/rl` rather than through TRL's `GRPOTrainer`, so that every piece is visible and tested, and because the pieces that matter here are small:

- 4 samples per prompt, 4 prompts per step; advantages normalized within each group; groups with no reward spread are skipped (logged as `zero_std_groups`).
- Clipped ratio objective (PPO-style), active when a batch is reused for several updates; on-policy (`num_iterations: 1`) it reduces exactly to REINFORCE with a group baseline, which a unit test checks on the gradient.
- Token-level normalization across the step, so short and long replies weigh per token, not per reply.
- **No KL term.** With LoRA, the frozen reference policy is the model with the adapter disabled, i.e. the base chatbot. A KL penalty would pull the policy back towards exactly what SFT removed. Instead: a small learning rate (5e-6), gradient norm clipped at 0.2, and reward terms that penalize assistant phrasing.
- Validation every 25 steps on the val split, including step 0 = the SFT model, on fixed random draws (the same seed for every checkpoint, 2 samples per prompt). The first IRC run validated on fresh single samples: its curve dipped 10% and recovered, and the final gap to step 0 (+0.028) was within the noise between two evals, so checkpoints could not be told apart. It also used lr 1e-5 with 10 warmup steps; now 5e-6 with 20. The final adapter is the checkpoint with the highest val reward; if GRPO never beats SFT, the SFT weights are what is kept.
- Two model backends: Unsloth (fast kernels) and plain transformers + PEFT. The Windows script falls back to the second if the first fails. No vLLM: it does not run on Windows, so generation uses `model.generate`.

## 6. Evaluation

- The same held-out prompts (later in time than all training data) for every model, 3 samples each at temperature 0.8. Greedy decoding was dropped because it hides collapse behind one most-likely reply.
- **Detection AUC** is the headline metric: a cross-validated character n-gram classifier trying to separate the model's replies from the real ones (0.5 = indistinguishable). No reward optimizes it. A first "authorship" classifier (you vs the other person) was replaced after measurement: it gave the untrained chatbot a higher score (0.60) than it gave the real replies (0.61), because assistant text is neither person's.
- **Floor for detection:** real test replies vs real training replies. Your own style drifts over time, so this, not 0.5, is the realistic target.
- chrF (with a 95% bootstrap interval over prompts), style gap, perplexity of the real replies, distinct replies, 6-gram copying and assistant phrases complete the table. chrF, style, copying and assistant phrases are also GRPO rewards on the training split and are flagged as such in the report.

## 7. Platform and operations

- Windows native, PowerShell, Python 3.12. `scripts/run.ps1` runs every stage with a smoke test before each training run and a fallback path for the Windows-specific failure it is most likely to hit.
- Unattended runs: sleep is blocked while training (`SetThreadExecutionState`), W&B falls back to offline mode instead of prompting for a login, logs go to `train.log`, SFT resumes from checkpoints.
- Reproducibility: every run records its config, data file hashes, git commit, val history and peak VRAM in `run_info.json`.

## 8. Public benchmark: Ubuntu IRC

The first real chat cannot be published, not even its model. Results need data anyone can check, from one real person, informal, with replies to specific messages. Rejected: synthetic Hinglish dialogue sets (generated by an LLM, so the model would learn another model's style), Discord and Ubuntu Dialogue Corpus dumps without stable author ids, and tweet corpora (no replies). Chosen: the Ubuntu IRC logs (public domain, 2004–2025, real nicks), with 1:1 threads rebuilt for one prolific user.

- **Rules, not a model.** A line joins the exchange with the person it names (`nick:` / `nick,`), or it is the first unprefixed line after being named, or it continues the speaker's own attached line, within one minute. Everything else is dropped: a missing reply costs less than training on a reply to the wrong message.
- **Measured against human labels** (irc-disentanglement, ACL 2019). Window chosen on dev (1, 2, 5 minutes: precision 0.931 / 0.917 / 0.904, recall 0.682 / 0.710 / 0.725; precision matters more). On test: 0.933 precision, 0.646 recall of reply links; prefix-only 0.964 / 0.449. Two test logs have labels shifted by 6 and 3 lines against the published raw text; they are reported separately rather than silently dropped.
- **Split globally by time.** Hundreds of short chats, most with fewer than three sessions, so a per-chat split would put nearly all of them in train. `split_by: time` holds out the latest sessions across all chats.
- **Capped at the most recent 2000 of the user's lines**, which keeps SFT near an hour on the laptop GPU.
- **Anonymized** in the export: the user becomes "Alex", partners `user0001`…, and their nicks are replaced inside message text.

## 9. Privacy

Exports, processed data, adapters and generations are git-ignored. W&B receives metrics only unless `wandb_samples: true`. The demo binds to 127.0.0.1. A model trained on a real chat has read the other person's messages, so the weights stay private; the public repository ships a synthetic dataset.
