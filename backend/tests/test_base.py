"""Tests for backend/agents/base.py - extract_json is pure Python, fully testable.

call_claude / call_claude_stream require ANTHROPIC_API_KEY so they are
integration-gated; only extract_json is tested here.
"""

import pytest

from backend.agents.base import _response_text, extract_json


def test_extract_json_bare_object():
    assert extract_json('{"key": "value"}') == {"key": "value"}


def test_extract_json_fenced_json():
    text = '```json\n{"a": 1}\n```'
    assert extract_json(text) == {"a": 1}


def test_extract_json_fenced_no_lang():
    text = '```\n{"b": 2}\n```'
    assert extract_json(text) == {"b": 2}


def test_extract_json_embedded_in_prose():
    text = 'Here is the result: {"score": 0.95} - done.'
    result = extract_json(text)
    assert result == {"score": 0.95}


def test_extract_json_nested():
    text = '{"outer": {"inner": [1, 2, 3]}}'
    assert extract_json(text) == {"outer": {"inner": [1, 2, 3]}}


def test_extract_json_returns_fallback_on_invalid():
    result = extract_json("this is not json", fallback={"error": True})
    assert result == {"error": True}


def test_extract_json_returns_empty_dict_by_default_on_failure():
    result = extract_json("not json at all")
    assert result == {}


def test_extract_json_ignores_leading_trailing_whitespace():
    result = extract_json('\n\n  {"x": 1}  \n\n')
    assert result == {"x": 1}


def test_extract_json_handles_unicode():
    result = extract_json('{"name": "données"}')
    assert result == {"name": "données"}


def test_extract_json_array_response_returns_fallback():
    # Arrays are not dicts - should not be returned as-is
    result = extract_json("[1, 2, 3]", fallback={"error": "not a dict"})
    assert result == {"error": "not a dict"}


def test_extract_json_empty_string_returns_fallback():
    assert extract_json("", fallback={"empty": True}) == {"empty": True}


def test_extract_json_prefers_fenced_over_prose():
    text = 'Prose says {"wrong": true} but ```json\n{"right": true}\n``` is fenced.'
    result = extract_json(text)
    assert result == {"right": True}


# ── Truncated-JSON recovery (response cut off by max_tokens) ─────────────────


def test_extract_json_recovers_truncated_mid_value():
    # Output cut off inside the last column's reason string.
    text = (
        '{"columns": {'
        '"host": {"action": "keep", "reason": "categorical default"}, '
        '"clade": {"action": "drop", "reason": "leakage ri'
    )
    result = extract_json(text)
    # The complete "host" column survives; the incomplete "clade" is dropped.
    assert "host" in result["columns"]
    assert result["columns"]["host"]["action"] == "keep"
    assert "clade" not in result["columns"]


def test_extract_json_recovers_truncated_after_complete_column():
    text = (
        '{"columns": {'
        '"a": {"action": "keep"}, '
        '"b": {"action": "drop"}, '
        '"c": {"action":'
    )
    result = extract_json(text)
    assert set(result["columns"]) == {"a", "b"}


def test_extract_json_truncation_preserves_brace_in_string():
    # A '}' inside a string value must not be mistaken for a real closer.
    text = '{"columns": {"x": {"reason": "drop if > 50% null}"}, "y": {"action": "ke'
    result = extract_json(text)
    assert result["columns"]["x"]["reason"] == "drop if > 50% null}"
    assert "y" not in result["columns"]


def test_extract_json_unrecoverable_truncation_returns_fallback():
    # Nothing complete before the cut → fallback.
    result = extract_json('{"columns": {"x": {"action', fallback={"f": 1})
    assert result == {"f": 1}


# ── _response_text: never IndexError on empty content ───────────────────────


class _FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


def test_response_text_joins_text_blocks():
    resp = _FakeResponse([_FakeBlock("hello "), _FakeBlock("world")])
    assert _response_text(resp, 4096) == "hello world"


def test_response_text_empty_content_returns_empty_string():
    # The bug: content[0] on [] raised "list index out of range".
    resp = _FakeResponse([], stop_reason="max_tokens")
    assert _response_text(resp, 4096) == ""


def test_response_text_none_content_returns_empty_string():
    resp = _FakeResponse(None)
    assert _response_text(resp, 4096) == ""


# ── call_claude_stream: re-roll empty completions before surfacing "" ────────


class _FakeStreamCM:
    """Mimics `client.messages.stream(...)` - an async CM yielding `self`."""

    def __init__(self, chunks, stop_reason="end_turn"):
        self._chunks = chunks
        self._stop_reason = stop_reason

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        async def _gen():
            for c in self._chunks:
                yield c
        return _gen()

    async def get_final_message(self):
        return _FakeResponse([], stop_reason=self._stop_reason)


class _FakeClientCM:
    """Mimics `_make_client()` - an async CM yielding a client with .messages."""

    def __init__(self, stream_cm):
        from unittest.mock import MagicMock
        self.messages = MagicMock()
        self.messages.stream = MagicMock(return_value=stream_cm)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


async def test_call_claude_stream_rerolls_empty_then_succeeds():
    """An empty completion is re-rolled; the next non-empty attempt is returned.
    Because the empty attempt emitted nothing, on_chunk sees only the real text."""
    from unittest.mock import AsyncMock, patch

    from backend.agents.base import call_claude_stream

    clients = [
        _FakeClientCM(_FakeStreamCM([])),                 # attempt 1: empty
        _FakeClientCM(_FakeStreamCM(["real ", "answer"])),  # attempt 2: real text
    ]
    chunks: list[str] = []

    async def on_chunk(text: str) -> None:
        chunks.append(text)

    with (
        patch("backend.agents.base._make_client", side_effect=clients) as mk,
        patch("backend.agents.base.asyncio.sleep", new=AsyncMock()),
    ):
        out = await call_claude_stream(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-6",
            on_chunk=on_chunk,
        )

    assert out == "real answer"
    assert chunks == ["real ", "answer"], "the empty attempt must not emit chunks"
    assert mk.call_count == 2, "an empty completion must trigger exactly one re-roll here"


async def test_call_claude_stream_returns_empty_after_all_rerolls():
    """If every attempt is empty, "" is surfaced for the caller's fallback."""
    from unittest.mock import AsyncMock, patch

    from backend.agents.base import _MAX_ATTEMPTS, call_claude_stream

    clients = [_FakeClientCM(_FakeStreamCM([])) for _ in range(_MAX_ATTEMPTS)]

    with (
        patch("backend.agents.base._make_client", side_effect=clients) as mk,
        patch("backend.agents.base.asyncio.sleep", new=AsyncMock()),
    ):
        out = await call_claude_stream(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-6",
        )

    assert out == ""
    assert mk.call_count == _MAX_ATTEMPTS
