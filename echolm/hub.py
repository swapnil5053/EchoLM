import logging
from pathlib import Path

log = logging.getLogger(__name__)

PUBLIC_MARK = "datasets: [common-pile/ubuntu_irc]"
DEFAULT_NAME = "echolm-ubuntu-irc-qwen2.5-1.5b"


def check_public(adapter: Path) -> None:
    """Only adapters whose card says they were trained on the public IRC benchmark may leave the machine."""
    card = adapter / "README.md"
    if not (adapter / "adapter_config.json").exists():
        raise ValueError(f"{adapter} is not a LoRA adapter folder (no adapter_config.json)")
    if not card.exists() or PUBLIC_MARK not in card.read_text(encoding="utf-8"):
        raise ValueError(f"{adapter} has no Ubuntu IRC model card. Adapters trained on your own chats "
                         "never get uploaded; for the benchmark run `echolm card --irc` first")


def repo_id(api, name: str) -> str:
    return name if "/" in name else f"{api.whoami()['name']}/{name}"


def push(adapter: Path, name: str = DEFAULT_NAME, private: bool = False, api=None) -> str:
    check_public(adapter)
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    repo = repo_id(api, name)
    card = adapter / "README.md"
    card.write_text(card.read_text(encoding="utf-8").replace('"REPO"', f'"{repo}"'), encoding="utf-8")
    api.create_repo(repo, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(adapter), repo_id=repo, commit_message="upload echolm adapter")
    url = f"https://huggingface.co/{repo}"
    log.info("uploaded %s to %s", adapter, url)
    return url
