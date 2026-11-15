"""Canned in-character replies for provider failures.

Extracted from engine.py (line-budget pass, 2026-07-26). Two voices per
failure kind: built-in Kurisu keeps her original strings; custom character
packs get deliberately plain character-neutral variants — a guest character
must not answer a provider failure in Kurisu's register ("Tch",
「〜じゃないわ」). Tagged format so they ride the normal emotion/voice
pipeline.
"""

from __future__ import annotations

# Spoken when the provider actively REJECTS our key (401/403) mid-turn —
# distinct from DummyProvider.NO_KEY_REPLY, which covers a key that was never
# set.
INVALID_KEY_REPLY = (
    "[angry] Tch — the provider is rejecting my API key, so I can't think "
    "straight right now. It's set, but it's wrong. || APIキーが拒否されてる"
    "のよ。設定はされてるけど、間違ってるみたい。\n"
    "[neutral] Open Settings and re-enter a valid key — then I'll be myself "
    "again. || 設定から正しいキーを入れ直して。そうすれば元に戻るわ。"
)

# Spoken when the provider refuses for quota/rate-limit reasons (429 /
# ResourceExhausted) mid-turn — the key is fine, the free-tier window (or
# paid budget) is just spent for now. Deliberately distinct from
# INVALID_KEY_REPLY's register: this isn't broken or misconfigured, so no
# "angry at myself" framing — just a matter-of-fact "out of capacity" with
# one actionable pointer (wait, or switch provider/key in Settings), no
# guilt, no apology loop. See provider_health.classify_and_record for the
# shared rate-limit/auth-error detection both replies key off of.
RATE_LIMITED_REPLY = (
    "[neutral] I've hit my thinking limit for now — the provider's quota is "
    "tapped out, not my key. || 今は思考の上限に達したところ。プロバイダーの"
    "クォータが尽きただけで、キーの問題じゃないわ。\n"
    "[neutral] Give it a few minutes for the quota window to reset, or swap "
    "to a different provider or key in Settings. || 数分待てばクォータの"
    "ウィンドウがリセットされる。急ぐなら設定で別のプロバイダーかキーに"
    "切り替えて。"
)

# A confused tool loop (iteration cap, refusals) with nothing to say.
LOST_TOOL_REPLY = (
    "Tch... I got lost in the middle of that and ran out of steps. "
    "Ask me again — and be more specific this time."
)

_NEUTRAL_INVALID_KEY_REPLY = (
    "[sad] Something is wrong with my connection — the provider is rejecting "
    "the API key. || プロバイダーがAPIキーを拒否しています。\n"
    "[neutral] Open Settings and re-enter a valid key, and I'll be back to "
    "normal. || 設定から正しいキーを入れ直してください。"
)
_NEUTRAL_RATE_LIMITED_REPLY = (
    "[neutral] I've hit the provider's usage limit for now — nothing is "
    "broken. || プロバイダーの利用上限に達しただけです。\n"
    "[neutral] Wait a few minutes for the window to reset, or switch the "
    "provider or key in Settings. || 数分待つか、設定でプロバイダーか"
    "キーを切り替えてください。"
)
_NEUTRAL_LOST_TOOL_REPLY = (
    "I lost the thread partway through that and ran out of steps. "
    "Ask me again, a bit more specifically."
)

_BY_KIND = {
    False: {
        "invalid_key": INVALID_KEY_REPLY,
        "rate_limited": RATE_LIMITED_REPLY,
        "lost": LOST_TOOL_REPLY,
    },
    True: {
        "invalid_key": _NEUTRAL_INVALID_KEY_REPLY,
        "rate_limited": _NEUTRAL_RATE_LIMITED_REPLY,
        "lost": _NEUTRAL_LOST_TOOL_REPLY,
    },
}


def canned_failure_reply(kind: str, *, is_custom_character: bool) -> str:
    """The canned provider-failure reply in the right voice."""
    return _BY_KIND[bool(is_custom_character)][kind]
