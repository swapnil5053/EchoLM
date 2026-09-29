# Attribution

EchoLM was inspired by WeClone (https://github.com/xming521/WeClone, MIT license). The concept of fine-tuning LLMs on personal chat history, and the time-gap approach to grouping messages into conversations, originate from that project. The data pipeline, training approach (SFT + GRPO), reward design, evaluation harness, and all code were built independently.

WeClone is built around WeChat exports and also supports Telegram JSON exports, and it trains through LLaMA-Factory. EchoLM targets WhatsApp and Telegram exports with first-class support for code-switched (Hinglish) chat, trains with Unsloth and its own GRPO implementation, and adds a reinforcement-learning alignment stage and a style-evaluation harness.

## Data

- The public benchmark uses the Ubuntu IRC logs as packaged in [common-pile/ubuntu_irc](https://huggingface.co/datasets/common-pile/ubuntu_irc) (public domain). They are downloaded at run time, not redistributed here.
- Thread rebuilding is checked against the reply annotations of [irc-disentanglement](https://github.com/jkkummerfeld/irc-disentanglement): Jonathan K. Kummerfeld et al., "A Large-Scale Corpus for Conversation Disentanglement", ACL 2019. The annotations are cloned at run time, not redistributed.
