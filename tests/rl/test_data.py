from echolm.rl.data import batches
from echolm.rl.data import reward_context
from echolm.rl.data import val_rows
from tests.tiny import DATA


def test_batches_cycle_through_every_example_before_repeating():
    seen = [x for b in batches(list(range(5)), 2, 5, 0) for x in b]
    assert len(seen) == 10
    assert sorted(seen[:5]) == [0, 1, 2, 3, 4]


def test_val_rows_and_reward_context():
    rows = val_rows(DATA)
    assert rows and {"id", "prompt", "reference"} <= set(rows[0])
    ctx = reward_context(DATA)
    assert ctx.scales and ctx.index.owners
