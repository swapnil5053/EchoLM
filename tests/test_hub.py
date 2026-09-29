import pytest

from echolm.hub import PUBLIC_MARK
from echolm.hub import push


class FakeApi:
    def __init__(self):
        self.calls = []

    def whoami(self):
        return {"name": "swapnil"}

    def create_repo(self, repo, private, exist_ok):
        self.calls.append(("create", repo, private))

    def upload_folder(self, folder_path, repo_id, commit_message):
        self.calls.append(("upload", folder_path, repo_id))


def adapter(tmp_path, card):
    folder = tmp_path / "adapter"
    folder.mkdir()
    (folder / "adapter_config.json").write_text("{}", encoding="utf-8")
    if card is not None:
        (folder / "README.md").write_text(card, encoding="utf-8")
    return folder


def test_push_uploads_the_public_adapter_and_fills_in_the_repo(tmp_path):
    folder = adapter(tmp_path, f"---\n{PUBLIC_MARK}\n---\nPeftModel.from_pretrained(base, \"REPO\")\n")
    api = FakeApi()
    url = push(folder, "echolm-test", api=api)
    assert url == "https://huggingface.co/swapnil/echolm-test"
    repo = "swapnil/echolm-test"
    assert api.calls == [("create", repo, False), ("upload", str(folder), repo)]
    assert '"swapnil/echolm-test"' in (folder / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("card", [None, "---\ntags: [hinglish]\n---\nprivate chat adapter\n"])
def test_push_refuses_adapters_from_private_chats(tmp_path, card):
    api = FakeApi()
    with pytest.raises(ValueError, match="never get uploaded"):
        push(adapter(tmp_path, card), "x", api=api)
    assert api.calls == []


def test_push_needs_an_adapter_folder(tmp_path):
    with pytest.raises(ValueError, match="adapter_config.json"):
        push(tmp_path, "x", api=FakeApi())
