from echolm.rl.tracker import Tracker


def test_tracker_without_wandb_logs_to_console(tmp_path, caplog):
    caplog.set_level("INFO")
    t = Tracker("none", "echolm", "run", {}, tmp_path)
    t.log({"reward": 0.5, "other": 1}, 3, "train")
    t.finish()
    assert "train step 3 reward=0.500" in caplog.text
