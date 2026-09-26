# EchoLM

Fine-tune a small LLM (Qwen 2.5 1.5B) to text like you, from your own WhatsApp and Telegram exports. SFT first, then GRPO with a style-consistency reward, evaluated against the base model on held-out chats. Built for an 8 GB laptop GPU.

Status: data pipeline and SFT stage done. GRPO, evaluation and demo are in progress. See [docs/adr/0001-architecture.md](docs/adr/0001-architecture.md) for the design.

## Quickstart (synthetic data, no GPU needed)

```bash
pip install -e ".[dev]"
echolm parse examples/synthetic/whatsapp_rohan.txt --me Kabir
echolm parse examples/synthetic/telegram_meera.json --me Kabir
echolm format --config configs/default.yaml
```

This writes to `data/processed/`:

| File | Contents |
|---|---|
| `windows.jsonl` | every context window with its split label |
| `train_sft.jsonl`, `val_sft.jsonl` | TRL conversational prompt/completion format |
| `grpo.jsonl` | train prompts plus the real reply as `reference` for reward functions |
| `test.jsonl` | held-out prompts plus `reference`, used only by evaluation |
| `stats.json` | counts per split and chat, openers, quotes, reply lengths |

The synthetic chats are two fictional Hinglish 1:1 conversations (Kabir with Rohan on WhatsApp, with Meera on Telegram) and produce ~56 windows. `examples/synthetic/processed/` has the expected output. `echolm synth` regenerates the raw exports.

## Your own data

- **WhatsApp:** open the chat → More → Export chat → Without media. Pass your name exactly as it appears in the file. Date order (day/month vs month/day) is detected automatically; use `--date-order` if the whole export is ambiguous.
- **Telegram:** Telegram Desktop → Settings → Advanced → Export Telegram data → JSON, or export a single chat. `--me` takes your display name or your `user…` id. Only personal (1:1) chats are used.

Put exports in `exports/` and parsed/processed files stay in `data/`; both are git-ignored. Never commit real chats or adapters trained on them.

## How the data is shaped

1. Messages are parsed; media, deleted and forwarded messages become placeholders (`[media]`, `[deleted]`, `[forwarded] …`) so the turn structure survives.
2. URLs, emails, phone numbers and OTPs are masked. Text is not lowercased, language-filtered or normalised: Hinglish and English are one vocabulary.
3. A gap over 3 hours starts a new session. Consecutive messages from the same person become one turn, joined by newlines (burst texting is part of style).
4. Each of your turns becomes a window: up to 8 previous turns / 1500 characters of context, your turn as the target. Context turns over 600 characters (pasted documents) are shown as `[long message]`. Telegram quote-replies to a message outside the context are prepended as `> quoted`.
5. The last 10% of each chat's sessions is test, the 5% before that is val. Splits never cut through a session, so test is strictly later in time than training.

## Training (SFT)

Needs an NVIDIA GPU. Tested target: RTX 4060 Laptop, 8 GB, Windows or Linux, Python 3.12.

```bash
pip install "torch>=2.8,<2.13" torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[train]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
wandb login          # or set report_to: none in configs/sft.yaml
echolm train sft
```

`configs/sft.yaml` holds the settings: Qwen2.5-1.5B-Instruct in 4-bit, LoRA rank 16 on all projections, 3 epochs, loss only on your reply. Before training it runs one eval on the base model so the first val loss and sample replies are the baseline. Every 20 steps it evaluates, prints greedy replies to 5 fixed val prompts next to what you actually said, and keeps the checkpoint with the lowest val loss.

Each run goes to `outputs/sft/<run name>/`: `adapter/` (LoRA weights + tokenizer) and `run_info.json` (config, data hashes, git commit, token lengths, best val loss, peak VRAM). Close browsers and other GPU apps first; the budget assumes most of the 8 GB is free.

## Development

```bash
pytest
ruff check .
```

See [ATTRIBUTION.md](ATTRIBUTION.md) for credits.
