from echolm.eval.style import chatbot_rate
from echolm.eval.style import features
from echolm.eval.style import strip_placeholders
from echolm.eval.style import style_gap


def test_features_on_hinglish_burst():
    f = features("haan yaar\nbas assignment 😂")
    assert f["lines"] == 2 and f["words"] == 5
    assert f["emoji"] == 1.0 and f["lowercase_start"] == 1.0
    assert f["ends_with_punct"] == 0.0


def test_features_formal_reply():
    f = features("Certainly! I would be HAPPY to help.")
    assert f["lowercase_start"] == 0.0 and f["ends_with_punct"] == 1.0
    assert f["caps_words"] > 0


def test_elongation():
    assert features("thank youuu")["elongated"] == 1.0
    assert features("thank you")["elongated"] == 0.0


def test_style_gap_zero_for_identical_and_larger_for_chatbot():
    real = ["haan bhai", "kuch nahi yaar\nbas", "acha 😂", "chal theek hai"]
    assert style_gap(real, real)["mean"] == 0
    bot = ["Hello! How can I assist you today?", "I'm sorry, but I can't help with that."] * 2
    assert style_gap(bot, real)["mean"] > 1


def test_chatbot_rate():
    assert chatbot_rate(["I'm sorry, but I can't access links", "haan", "How can I help you?", "ok"]) == 0.5


def test_strip_placeholders():
    assert strip_placeholders("dekh <URL> [media] ye") == "dekh ye"
    assert strip_placeholders("[long message]") == ""
