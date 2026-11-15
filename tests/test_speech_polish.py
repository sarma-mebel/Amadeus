"""Tool-turn speech polish: markdown stripping + the tagged-reply repair pass."""

from typing import List

from amadeus.providers.llm.base import ChatMessage, LLMProvider
from amadeus.web.streaming import (
    ensure_tagged_reply,
    parse_complete_reply,
    strip_markdown_for_speech,
)


def test_strip_markdown_flattens_lists_and_emphasis():
    text = (
        "Here's what landed:\n"
        "* **Wat3R: Underwater 3D Geometry:** light scattering is hard.\n"
        "- *Hockey stick* divergences with `math`.\n"
        "1. [Ideas Have Genomes](https://arxiv.org/abs/x) benchmark.\n"
        "## Heading\n"
        "Plain closing sentence."
    )
    cleaned = strip_markdown_for_speech(text)
    for marker in ("**", "* ", "`", "](", "##", "- "):
        assert marker not in cleaned, f"{marker!r} survived: {cleaned}"
    assert "Wat3R: Underwater 3D Geometry" in cleaned
    assert "Ideas Have Genomes" in cleaned
    assert "Plain closing sentence." in cleaned


def test_strip_markdown_leaves_prose_alone():
    text = "Hmph. 2 * 3 is 6, obviously. Ask something harder."
    # A lone asterisk (not paired emphasis) must survive — it's math, not markdown.
    assert strip_markdown_for_speech(text) == text


class _ScriptedLLM(LLMProvider):
    name = "scripted"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: List[List[ChatMessage]] = []

    def generate(self, messages: List[ChatMessage], **kwargs) -> str:
        self.calls.append(list(messages))
        return self.reply


def test_ensure_tagged_reply_repairs_untagged_draft():
    tagged = "[happy] Five new papers landed today. || 今日は新しい論文が5本出たわ。"
    llm = _ScriptedLLM(tagged)
    draft = "* **Paper one** is about X.\n* **Paper two** is about Y."

    result = ensure_tagged_reply(llm, "SYSTEM PROMPT WITH FORMAT RULES", draft)
    assert result == tagged
    assert parse_complete_reply(result)
    # The rewrite ran against the same system prompt and included the draft.
    assert llm.calls[0][0].role == "system"
    assert "DRAFT:" in llm.calls[0][1].content
    assert "Paper one" in llm.calls[0][1].content


def test_ensure_tagged_reply_keeps_draft_when_repair_fails():
    llm = _ScriptedLLM("still not tagged output")
    draft = "plain answer"
    assert ensure_tagged_reply(llm, "sys", draft) == draft


def test_ensure_tagged_reply_survives_provider_error():
    class _Boom(LLMProvider):
        name = "boom"

        def generate(self, messages, **kwargs):
            raise RuntimeError("provider down")

    assert ensure_tagged_reply(_Boom(), "sys", "draft") == "draft"
