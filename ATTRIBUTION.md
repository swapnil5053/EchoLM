# Attribution

EchoLM was inspired by [WeClone](https://github.com/xming521/WeClone) (MIT license). The idea of fine-tuning an LLM on your own chat history, and of splitting messages into conversations by time gaps, comes from that project. The code, the data pipeline, the training setup (SFT and GRPO), the rewards and the evaluation were written separately.

WeClone works with WeChat and Telegram exports and trains through LLaMA-Factory. EchoLM works with WhatsApp and Telegram exports, handles Hinglish chat, trains with Unsloth plus its own GRPO code, and measures the result on later messages.

## Data

- The public benchmark uses the Ubuntu IRC logs as packaged in [common-pile/ubuntu_irc](https://huggingface.co/datasets/common-pile/ubuntu_irc) (public domain). They are downloaded when you run the pipeline, not stored here.
- The thread rebuilding is checked against the reply annotations from [irc-disentanglement](https://github.com/jkkummerfeld/irc-disentanglement): Jonathan K. Kummerfeld et al., "A Large-Scale Corpus for Conversation Disentanglement", ACL 2019. They are also downloaded at run time.
