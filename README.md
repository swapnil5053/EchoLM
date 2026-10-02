# EchoLM

Fine-tune a 1.5B language model to reply the way one person texts, then check how close it got on messages it never saw. Runs on an 8 GB laptop GPU.

The pipeline takes a WhatsApp or Telegram export, builds a dataset from it, trains Qwen2.5-1.5B-Instruct with LoRA, and then trains it further with GRPO, a reinforcement learning step I wrote from scratch. Every model is tested on replies written after everything it trained on.

My own chats stay private, so the published results use a public stand-in: one active helper from the Ubuntu IRC logs, whose one-to-one conversations EchoLM rebuilds from the group channel.

## Results

Ubuntu IRC, one user, tested on the most recent 10% of their conversations.

<!-- results:start -->
80 held-out test replies (later in time than all training data), 3 sampled replies each at temperature 0.8. † = also a GRPO reward on the training split.

| model | detect AUC ↓ | chrF ↑ † | style gap ↓ † | reply ppl ↓ | distinct ↑ | 6-gram copy ↓ † | chatbot ↓ † | median words |
|---|---|---|---|---|---|---|---|---|
| base | 0.965 ±0.010 | 0.171 (0.157–0.184) | 1.199 | 51.9 | 1.000 | 0.009 | 0.292 | 46.3 |
| sft | 0.838 ±0.022 | 0.144 (0.134–0.154) | 0.237 | 19.7 | 0.992 | 0.000 | 0.009 | 19 |
| grpo | 0.829 ±0.032 | 0.141 (0.131–0.151) | 0.196 | 19.9 | 0.992 | 0.000 | 0.017 | 15 |
| your real replies | 0.588 | – | 0.000 | – | 0.988 | 0.000 | 0.000 | 11 |
<!-- results:end -->

How to read it:

- **detect AUC** is the main number. A classifier tries to tell the model's replies from the real ones. 1.0 means it always can, 0.5 means it can't. No reward trains on it. The last row compares the person's newer replies with their older ones, which is the best a model could realistically reach.
- **chrF** is character overlap with what the person actually replied, with a 95% interval. **style gap** compares nine style features such as length, casing and punctuation. **reply ppl** is how surprised the model is by the real replies.
- **distinct**, **6-gram copy** and **chatbot** catch the usual failures: repeating stock replies, copying training text, and sounding like an assistant.
- † marks metrics that GRPO was also rewarded on, so a gain there is expected rather than impressive.

What I take from it:

- SFT does most of the work. Detection drops from 0.965 to 0.838, the real replies become 2.6x less surprising to the model, assistant phrasing goes from 29% of replies to 1%, and replies shrink from 46 words to 19.
- GRPO does not beat SFT. Over three seeds (lr 5e-6, fixed-draw validation) detection is 0.841 ±0.005 against SFT's 0.838, chrF 0.145 against 0.144 and perplexity the same. It only moves what it is rewarded for: style gap 0.206 ±0.028 against 0.237, and slightly shorter replies. In one of the three seeds no GRPO checkpoint beat step 0 on validation, so the run kept the SFT weights, which is why grpo-s1 matches the sft row exactly.
- People can't tell either. In blind pairs the real reply was picked 45% of the time over 122 rounds against the GRPO model, about what guessing gives.
- The base model has the best chrF because chrF favours long answers that happen to cover more of the reference. That is why chrF is always paired with a length reward.
- The gap the classifier sees is still large: 0.84 against the 0.59 floor. With 80 test replies, differences of a few hundredths are noise.

<p align="center">
  <img src="docs/figures/sft_val_loss.svg" alt="SFT validation loss by step, with the selected checkpoint circled" width="49%">
  <img src="docs/figures/grpo_val_reward.svg" alt="GRPO validation reward by step, against the SFT starting point" width="49%">
</p>

The single `grpo` row and the curve above come from the first run (lr 1e-5, one fresh sample per validation prompt), which dipped before recovering and could not tell its checkpoints apart. The seeded runs use half the learning rate and validate every checkpoint on the same random samples (`scripts\run.ps1 -Stages seeds`). My reading: with rewards computed against one reference reply, GRPO on 2000 replies sharpens surface style but has nothing more to teach about what this person would say.

**On my own chat.** The first run was on a private Hinglish chat (813 training replies; only these totals are shared). SFT dropped assistant phrasing from 63% to 0%, but only a third of its replies were unique against 96% of mine. The checkpoint with the lowest validation loss copied training text 10x more often than one from 20 steps earlier with almost the same loss. So EchoLM picks the earliest checkpoint within 3% of the best loss, and GRPO has explicit penalties for duplicate and copied replies.

## How it works

**Data.** WhatsApp and Telegram exports are parsed into messages. Links, emails, phone numbers and OTPs are masked. A gap of 3 hours starts a new conversation, and back-to-back messages from one person form one turn. Each of your turns becomes a training example: up to 8 earlier turns as context, your turn as the answer. The newest conversations go to validation and test, so the model is always tested on the future. Hindi written in English letters and English are treated as one language, with no filtering.

**SFT.** Qwen2.5-1.5B-Instruct in 4-bit with Unsloth and LoRA rank 16. Only reply tokens count towards the loss.

**GRPO** (`echolm/rl`, no RL library). For each prompt the model writes 4 replies, each gets a score, and the model is pushed towards the ones that beat the group average. The score adds up:

| reward | what it checks |
|---|---|
| chrF | character overlap with the real reply to that message |
| style match | the nine style features, against the real reply to that message |
| length match | word count against the real reply |
| duplicate, copy | penalty for repeating another sample or copying a training reply |
| chatbot, empty | penalty for assistant phrases and empty replies |

There is no KL penalty. With LoRA the reference model would be the base chatbot, so the penalty would pull the model back towards what SFT removed. Small steps and gradient clipping keep it close instead. Step 0 is validated too, so if GRPO never beats SFT, the SFT weights are kept.

**Ubuntu IRC.** IRC is one shared room, so `echolm/irc` works out who each line was meant for: lines starting with `name:`, the first reply after being addressed, and follow-up lines within a minute. I checked these rules against the hand-labelled reply links of [Kummerfeld et al. (ACL 2019)](https://github.com/jkkummerfeld/irc-disentanglement): 93% of the lines it attaches go to the right person, and it recovers 65% of all replies. The user is renamed "Alex" and everyone else `user0001` and so on.

More detail on each decision and the numbers behind it: [docs/adr/0001-architecture.md](docs/adr/0001-architecture.md).

## Running it

Windows, an NVIDIA GPU with 8 GB, and Python 3.12. `scripts\run.ps1` sets up `.venv` and runs every stage, with a short smoke test before each training run.

Public benchmark (downloads 6 GB of logs once, about 4 hours in total):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run.ps1 -Dataset ubuntu
.venv\Scripts\echolm demo --outputs outputs/ubuntu --data data/ubuntu/processed
```

Your own chat: put the export in `exports\` and run

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run.ps1 -Me "your name as it appears in the export"
.venv\Scripts\echolm demo
```

The demo runs on your machine only. You can chat with the base, SFT and GRPO models, compare them on one message, or play **Real or model?**, where you pick the real reply out of a pair and your hit rate goes into the report.

Main commands (`echolm --help` lists all of them):

| command | does |
|---|---|
| `echolm parse EXPORT --me NAME` / `echolm format` | build the dataset |
| `echolm irc fetch`, `users`, `export`, `validate` | build and check the Ubuntu IRC dataset |
| `echolm train sft` / `select` / `grpo` | train, pick the SFT checkpoint, run GRPO |
| `echolm eval run` / `report` / `plot` | score a model, write the table, draw the curves |
| `echolm demo [--guess-only]` | local chat UI and the guessing game |
| `echolm card` / `push` | model card, and upload of the IRC model to Hugging Face |

## Privacy

Exports, datasets, adapters and generated replies are git-ignored. The demo only listens on 127.0.0.1. `echolm push` refuses any adapter that was not trained on the public IRC data. This repo only contains synthetic example chats (`examples/synthetic`).

## Project layout

```
echolm/     parse, data, irc, train, rl, eval, demo
configs/    data, SFT, GRPO and eval settings
scripts/    run.ps1, the whole pipeline on Windows
tests/      one file per module; SFT, GRPO, eval and the demo run end to end on CPU with a tiny model
docs/       design notes and figures
examples/   two synthetic Hinglish chats and the dataset built from them
```

```powershell
pip install -e ".[dev,eval,demo]"
pytest
```

Inspired by [WeClone](https://github.com/xming521/WeClone) ([ATTRIBUTION.md](ATTRIBUTION.md)). MIT license.
