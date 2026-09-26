# EchoLM

Fine-tune a small LLM (Qwen 2.5 1.5B) to text like you, from your own WhatsApp and Telegram exports. SFT first, then GRPO with a style-consistency reward, evaluated against the base model on held-out chats. Built for an 8 GB laptop GPU.

Status: data pipeline done. Training, rewards, evaluation and demo are in progress. See [docs/adr/0001-architecture.md](docs/adr/0001-architecture.md) for the design.

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
3. A gap over 30 minutes starts a new session. Consecutive messages from the same person become one turn, joined by newlines (burst texting is part of style).
4. Each of your turns becomes a window: up to 8 previous turns / 1500 characters of context, your turn as the target. Telegram quote-replies to a message outside the context are prepended as `> quoted`.
5. The last 10% of each chat's sessions is test, the 5% before that is val. Splits never cut through a session, so test is strictly later in time than training.

## Development

```bash
pytest
ruff check .
```

See [ATTRIBUTION.md](ATTRIBUTION.md) for credits.
