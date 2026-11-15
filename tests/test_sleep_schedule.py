"""Sleep schedule: the window math, presence suppression, groggy registers,
and the reach-out hold. The pet's visual doze is client-side (petDoze.ts);
this covers the server half."""

import datetime as dt

from amadeus.config import Config
from amadeus.engine import ChatEngine
from amadeus.memory.profile import greeting_register_facts
from amadeus.presence import idle_chatter, poke_reaction
from amadeus.providers.llm.base import ChatMessage, DummyProvider
from amadeus.sleep_schedule import is_sleep_hours, sleep_status


def _at(hour: int) -> dt.datetime:
    return dt.datetime(2026, 8, 3, hour, 30)


def test_window_math_including_midnight_wrap():
    config = Config()
    config.sleep_schedule.start_hour = 1
    config.sleep_schedule.end_hour = 8
    assert is_sleep_hours(config, _at(3)) is True
    assert is_sleep_hours(config, _at(1)) is True  # start inclusive
    assert is_sleep_hours(config, _at(8)) is False  # end exclusive
    assert is_sleep_hours(config, _at(23)) is False

    config.sleep_schedule.start_hour = 23
    config.sleep_schedule.end_hour = 7
    assert is_sleep_hours(config, _at(23)) is True
    assert is_sleep_hours(config, _at(2)) is True
    assert is_sleep_hours(config, _at(12)) is False


def test_window_disabled_states():
    config = Config()
    config.sleep_schedule.enabled = False
    assert is_sleep_hours(config, _at(3)) is False

    config.sleep_schedule.enabled = True
    config.sleep_schedule.start_hour = 5
    config.sleep_schedule.end_hour = 5  # start == end disables
    assert is_sleep_hours(config, _at(5)) is False

    config.sleep_schedule.start_hour = "junk"  # degrade to awake, never raise
    assert is_sleep_hours(config, _at(3)) is False


def test_sleep_status_payload():
    config = Config()
    config.sleep_schedule.start_hour = 1
    config.sleep_schedule.end_hour = 8
    payload = sleep_status(config, _at(3))
    assert payload == {
        "enabled": True,
        "start_hour": 1,
        "end_hour": 8,
        "asleep_now": True,
    }


class _RecordingProvider:
    name = "recording"

    def __init__(self, reply: str = "Mm... five more minutes.") -> None:
        self.reply = reply
        self.calls: list[list[ChatMessage]] = []

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        return self.reply


def _presence_engine(tmp_path, hour: int) -> ChatEngine:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.enable_idle_chatter = True
    config.memory.enable_poke_reactions = True
    config.sleep_schedule.start_hour = hour
    config.sleep_schedule.end_hour = (hour + 2) % 24
    engine = ChatEngine(config, DummyProvider())
    engine.summary_provider = _RecordingProvider()
    return engine


def test_idle_chatter_suppressed_during_sleep(tmp_path, monkeypatch):
    now_hour = dt.datetime.now().hour
    engine = _presence_engine(tmp_path, now_hour)  # asleep right now
    engine.memory.add_message("user", "hello")
    engine.memory.add_message("assistant", "hi")
    assert idle_chatter(engine) is None
    from amadeus import telemetry

    assert telemetry.snapshot().get("presence.sleep_suppressed", 0) >= 1


def test_poke_gets_groggy_register_during_sleep(tmp_path):
    now_hour = dt.datetime.now().hour
    engine = _presence_engine(tmp_path, now_hour)
    text = poke_reaction(engine)
    assert text is not None
    system = engine.summary_provider.calls[0][0].content
    assert "you were" in system and "ASLEEP" in system
    assert "groggy" in system

    # Awake window: the groggy register must NOT leak into daytime pokes.
    awake = _presence_engine(tmp_path / "b", (now_hour + 6) % 24)
    assert poke_reaction(awake) is not None
    assert "ASLEEP" not in awake.summary_provider.calls[0][0].content


def test_greeting_register_carries_woken_beat():
    now = dt.datetime(2026, 8, 3, 3, 0)
    facts = greeting_register_facts(now, now, None, asleep=True)
    joined = "\n".join(facts)
    assert "you were asleep" in joined
    assert "Never lecture" in joined
    assert not any(
        "you were asleep" in f
        for f in greeting_register_facts(now, now, None, asleep=False)
    )
