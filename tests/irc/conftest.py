import json

import pytest

# a morning in a support channel: two helpers, three askers, a bot and a join line
DAY = """=== carol [~c@host] has joined #ubuntu
[09:00] <dave> anyone know why wifi drops after suspend?
[09:01] <alice> dave: which card? run lspci | grep -i net
[09:01] <dave> alice: intel 7260
[09:02] <alice> known bug, try the backport driver
[09:02] <alice> sudo apt install backport-iwlwifi-dkms
[09:03] <ubottu> dave: please see https://help.ubuntu.com/community/WifiDocs
[09:03] <dave> thanks alice, that worked
[09:04] <bob> !paste
[09:05] <erin> bob: is 22.04 still supported?
[09:05] <bob> erin, yes until 2027
[09:20] <alice> random thought
[09:30] <erin> alice: can i ask you too
[09:30] <alice> erin: sure
[09:31] <erin> alice: how do i reinstall grub
[09:31] <alice> boot a live usb and chroot in
"""


@pytest.fixture
def day_text():
    return DAY


@pytest.fixture
def logs(tmp_path):
    p = tmp_path / "logs.jsonl"
    rows = [{"day": d, "channel": "#ubuntu", "text": DAY} for d in ("2024-03-01", "2024-03-02")]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p
