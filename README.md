# EchoLM

Fine-tune a small language model (Qwen2.5-1.5B) to reply the way one specific person texts, then check how close it got on messages it never saw. Everything runs on an 8 GB laptop GPU.

## Why I built this

I wanted to know whether a model small enough to train on my laptop could pick up how I actually text: short, mostly lowercase, Hindi and English mixed in one sentence. Other projects already fine-tune on chat history, but they stop at "here's the model". They never measure whether the replies sound like you, or whether the model just learned a few stock lines.

So EchoLM does the training and the measuring. The model is always tested on replies written after everything it trained on, a classifier tries to tell its replies from the real ones, and a small game lets people try the same thing.

My own chats stay private, so the published numbers use a public stand-in: one active helper from the Ubuntu IRC logs, whose one-to-one conversations EchoLM rebuilds from the shared channel.

## Results

Ubuntu IRC, one user, tested on the most recent 10% of their conversations.

<!-- results:start -->
80 held-out test replies (later in time than all training data), 3 sampled replies each at temperature 0.8. † = also a GRPO reward on the training split.

| model | detect AUC ↓ | chrF ↑ † | style gap ↓ † | reply ppl ↓ | distinct ↑ | 6-gram copy ↓ † | chatbot ↓ † | median words |
|---|---|---|---|---|---|---|---|---|
| base | 0.965 ±0.010 | 0.171 (0.157–0.184) | 1.199 | 51.9 | 1.000 | 0.009 | 0.292 | 46.3 |
| sft | 0.838 ±0.022 | 0.144 (0.134–0.154) | 0.237 | 19.7 | 0.992 | 0.000 | 0.009 | 19 |
| grpo | 0.829 ±0.032 | 0.141 (0.131–0.151) | 0.196 | 19.9 | 0.992 | 0.000 | 0.017 | 15 |
| grpo (mean of 3 seeds) | 0.841 ±0.005 | 0.145 ±0.002 | 0.206 ±0.028 | 19.7 ±0.0 | 0.992 ±0.000 | 0.003 ±0.002 | 0.009 ±0.004 | 17.7 ±2.0 |
| grpo-s1 | 0.838 ±0.022 | 0.144 (0.134–0.154) | 0.237 | 19.7 | 0.992 | 0.000 | 0.009 | 19 |
| grpo-s2 | 0.839 ±0.023 | 0.143 (0.133–0.153) | 0.181 | 19.8 | 0.992 | 0.004 | 0.013 | 15.3 |
| grpo-s3 | 0.847 ±0.006 | 0.147 (0.136–0.157) | 0.200 | 19.7 | 0.992 | 0.004 | 0.004 | 18.7 |
| your real replies | 0.588 | – | 0.000 | – | 0.988 | 0.000 | 0.000 | 11 |

Human judge, blind A/B in the demo's "Real or model?" tab (50% = cannot tell): sft: real reply spotted in 3 of 6 rounds (50%); grpo: real reply spotted in 55 of 122 rounds (45%).

<!-- results:end -->

How to read it: **detect AUC** is the main number. A classifier tries to tell the model's replies from the real ones; 1.0 means it always can, 0.5 means it can't, and no reward trains on it. The last row compares the person's newer replies with their older ones, which is the best a model could realistically reach. † marks metrics GRPO was also rewarded on, so gains there are expected.

What I learned:

- **SFT does most of the work.** Detection drops from 0.965 to 0.838, the real replies become 2.6x less surprising to the model, and assistant-style phrasing goes from 29% of replies to 1%.
- **GRPO doesn't beat SFT.** Over three seeds, detection is 0.841 ±0.005 against 0.838, and chrF and perplexity don't move. It only improves what it's rewarded for (style gap 0.206 against 0.237). In one seed no GRPO checkpoint beat the starting point on validation, so that run kept the SFT weights, which is why `grpo-s1` matches `sft`.
- **People can't tell either.** In blind pairs the real reply was picked 45% of the time over 122 rounds, about what guessing gives.
- **chrF on its own is misleading.** The untrained model scores best because its long answers happen to overlap more with the reference, so chrF is always paired with a length reward.

<p align="center">
  <img src="docs/figures/sft_val_loss.svg" alt="SFT validation loss by step, with the selected checkpoint circled" width="49%">
  <img src="docs/figures/grpo_val_reward.svg" alt="GRPO validation reward by step, against the SFT starting point" width="49%">
</p>

The single `grpo` row and the right-hand curve are from the first GRPO run (lr 1e-5). The three seeded runs use half that learning rate.

On my own chat (813 training replies, only totals shared), SFT dropped assistant phrasing from 63% to 0%, but the checkpoint with the lowest validation loss copied training text 10x more often than one 20 steps earlier with almost the same loss. That's why EchoLM picks the earliest checkpoint within 3% of the best loss.

## Features

- **Chat parsing** for WhatsApp `.txt` and Telegram `.json` exports. Links, emails, phone numbers and OTPs are masked. Hinglish is kept as is, with no language filtering.
- **Time-based split**, so the test set is always later than the training data.
- **SFT** with LoRA on Qwen2.5-1.5B in 4-bit. Only reply tokens count towards the loss.
- **GRPO written from scratch** (`echolm/rl`): 4 samples per prompt, group-relative advantages, rewards for character overlap, style and length, and penalties for duplicates, copying and chatbot phrases. No KL term, because with LoRA the reference model would be the base chatbot that SFT just moved away from.
- **Evaluation** on later messages with a detection classifier, chrF with bootstrap intervals, style and copying metrics, and charts of the training curves.
- **Ubuntu IRC benchmark**: rebuilds one person's 1:1 threads from a group channel. Checked against hand-labelled reply links ([Kummerfeld et al., ACL 2019](https://github.com/jkkummerfeld/irc-disentanglement)): 93% of attached lines go to the right person.
- **Local demo** (Gradio): chat with the base, SFT and GRPO models, compare them side by side, or play **Real or model?** and pick the real reply out of a pair.

## Tech stack

Python 3.12, PyTorch, Hugging Face Transformers and PEFT, Unsloth, scikit-learn, Gradio, Click, pytest and ruff, GitHub Actions (Windows).

## Quick start

You need Windows, an NVIDIA GPU with 8 GB, and Python 3.12. The script sets up `.venv` and runs every stage.

Try it on the public IRC data (downloads about 6 GB of logs once, then takes about 4 hours):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run.ps1 -Dataset ubuntu
.venv\Scripts\echolm demo --outputs outputs/ubuntu --data data/ubuntu/processed
```

Or on your own chat: put the export in `exports\`, then

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run.ps1 -Me "your name as it appears in the export"
.venv\Scripts\echolm demo
```

Without a GPU you can still run the tests and the data steps on the synthetic example chats:

```powershell
pip install -e ".[dev,eval,demo]"
pytest
echolm parse examples/synthetic/whatsapp_rohan.txt --me Kabir
echolm format
```

`echolm --help` lists every command.

## Privacy

Exports, datasets, model weights and generated replies are git-ignored. The demo only listens on 127.0.0.1, and `echolm push` refuses to upload any model that wasn't trained on the public IRC data.

## License

MIT
