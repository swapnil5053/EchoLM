# Attribution

EchoLM was inspired by WeClone (https://github.com/xming521/WeClone, MIT license). The concept of fine-tuning LLMs on personal chat history, and the time-gap approach to grouping messages into conversations, originate from that project. The data pipeline, training approach (SFT + GRPO), reward design, evaluation harness, and all code were built independently.

WeClone is built around WeChat exports and also supports Telegram JSON exports, and it trains through LLaMA-Factory. EchoLM targets WhatsApp and Telegram exports with first-class support for code-switched (Hinglish) chat, trains with Unsloth + TRL, and adds a reinforcement-learning alignment stage and a style-evaluation harness.
