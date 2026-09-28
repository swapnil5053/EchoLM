# EchoLM

Fine-tune a small LLM (Qwen 2.5 1.5B) to text like you, from your own WhatsApp and Telegram exports. SFT first, then GRPO with a style-consistency reward, evaluated against the base model on held-out chats. Built for an 8 GB laptop GPU.

Status: data pipeline and SFT stage done (Windows). GRPO, evaluation and demo are in progress. See [docs/adr/0001-architecture.md](docs/adr/0001-architecture.md) for the design.

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

The synthetic chats are two fictional Hinglish 1:1 conversations (Kabir with Rohan on WhatsApp, with Meera on Telegram) and produce ~55 windows. `examples/synthetic/processed/` has the expected output. `echolm synth` regenerates the raw exports.

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

EchoLM targets Windows with an NVIDIA GPU (built on an RTX 4060 Laptop, 8 GB), Python 3.12 and PowerShell. Put your exports in `exports/`, plug in the charger, close browsers and other GPU apps, then from the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_windows.ps1 -Me "your name as it appears in the export"
```

The script creates `.venv` if needed, installs a CUDA build of torch plus Unsloth, re-parses every export, rebuilds the dataset, runs `echolm train check`, does a 10-step smoke run of the whole loop, then the full run. It stops at the first failing step and says which one. Add `-SkipInstall` on later runs.

The same steps by hand:

```powershell
.\.venv\Scripts\Activate.ps1
echolm train check               # GPU, packages, data, disk, W&B
echolm train sft --max-steps 10  # smoke test
echolm train sft                 # full run
echolm train sft --resume outputs\sft\<run name>   # continue after a crash
```

`configs/sft.yaml` holds the settings: Qwen2.5-1.5B-Instruct in 4-bit, LoRA rank 16 on all projections, 3 epochs, batch 4 x 4 accumulation. Loss is computed only on your reply tokens. Training starts with one eval of the untouched model as the baseline; every 20 steps it evaluates, saves a checkpoint, and prints greedy replies to 5 fixed val prompts next to what you actually said. The checkpoint with the lowest val loss is the one kept as the final adapter.

Each run goes to `outputs/sft/<run name>/`: `adapter/` (LoRA weights + tokenizer), `checkpoint-*/`, `train.log` and `run_info.json` (config, data hashes, git commit, token lengths, best val loss, peak VRAM). While it runs, Windows is kept from sleeping; closing the lid still follows your power settings. Without a W&B login it logs offline to `wandb/`, which `wandb sync` uploads later.

## Development

```bash
pytest
ruff check .
```

See [ATTRIBUTION.md](ATTRIBUTION.md) for credits.
