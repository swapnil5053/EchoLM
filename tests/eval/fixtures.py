MINE = ["haan bhai", "acha yaar", "bas bhai kuch nahi", "chal bhai", "haan yaar theek", "bhai 😂",
        "kya scene hai", "nahi yaar", "acha acha", "haan chal", "bhai sun", "theek hai bhai"]
THEIRS = ["Okay, see you tomorrow.", "Did you finish the report?", "Please call me back.",
          "That sounds good.", "What time works for you?", "Sure, no problem.", "Thanks a lot.",
          "I will check and tell you.", "Are you coming today?", "Let me know.", "Good night.", "See you."]


def windows(split):
    return [{"id": f"{split}{i}", "split": split, "target": m,
             "context": [{"role": "them", "text": t}, {"role": "me", "text": "x"}]}
            for i, (m, t) in enumerate(zip(MINE, THEIRS, strict=True))]
