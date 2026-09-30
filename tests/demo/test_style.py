from echolm.demo.style import CSS
from echolm.demo.style import persona
from echolm.demo.style import scoreboard
from echolm.demo.style import transcript


def test_css_ships_with_the_package():
    assert "button.slip" in CSS and ".dark" in CSS


def test_persona_comes_from_the_system_prompt():
    assert persona("You are Alex. Reply in the chat exactly the way Alex writes.") == "Alex"
    assert persona("something else") == "reply"


def test_transcript_escapes_and_labels_turns():
    turns = [{"role": "user", "content": "<b>hi</b>"}, {"role": "assistant", "content": "yo"}]
    html = transcript(turns, "Alex")
    assert "&lt;b&gt;hi&lt;/b&gt;" in html and "<b>hi" not in html
    assert html.index("them") < html.index(">Alex<")
    assert "Which reply did Alex send?" in html
    assert "opened the conversation" in transcript([], "Alex")


def test_scoreboard():
    assert "no answers yet" in scoreboard([0, 0])
    board = scoreboard([27, 64], "Not this time", good=False)
    assert "27<small> / 64" in board and "width:42%" in board
    assert "42% spotted" in board and "verdict bad" in board
