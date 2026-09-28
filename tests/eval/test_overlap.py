from echolm.eval.overlap import copy_rates
from echolm.eval.overlap import repetition


def test_copy_rates_ignore_short_replies():
    train = ["haan", "baby I miss you so much im sorry"]
    outs = ["haan", "baby i miss you so much im sorry", "something new and different to say"]
    rates = copy_rates(outs, train)
    assert rates["exact_copy"] == 0.5
    assert rates["ngram_copy"] == 0.5
    assert rates["n_long"] == 2


def test_copy_rates_empty():
    assert copy_rates(["ok"], [])["ngram_copy"] == 0


def test_repetition():
    rep = repetition(["baby", "Baby", "haan", "ok"])
    assert rep["distinct"] == 0.75
    assert rep["top_share"] == 0.5
