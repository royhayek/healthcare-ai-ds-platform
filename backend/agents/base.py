"""Shared model client for all agents (§20, §28).

Every agent imports from here. Do not instantiate anthropic.AsyncAnthropic
directly in agent files - that scatters model string handling and retry logic.

Retry policy: 3 attempts with 5 / 10 / 20 s delays on RateLimitError or
APIStatusError. Other exceptions propagate immediately.

Prompt caching (§28): all calls mark system prompts with cache_control so
repeated calls sharing the same system text (e.g., EDA → preprocessing →
model selection, which all start with the same operator instructions before
the profile context) hit the Anthropic cache rather than re-tokenising.
The minimum cacheable block is 1024 tokens for Sonnet/Opus and 2048 for Haiku;
smaller blocks silently bypass caching without error.

on_chunk in call_claude_stream:
  A callable that receives each text delta. May be sync or async - the
  dispatcher awaits coroutines automatically. Pass None for no streaming
  side-effects (the full text is still returned).
"""

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

import anthropic

from backend.core.config import settings

logger = logging.getLogger(__name__)

_RETRY_DELAYS = (5, 10, 20)
_MAX_ATTEMPTS = 3


def _make_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)


def _response_text(response: Any, max_tokens: int) -> str:
    """Safely extract text from a non-streaming Messages response.

    The Anthropic SDK can return an empty content list (e.g. when the model
    is cut off by max_tokens before emitting any text block, or on a paused
    turn). Indexing content[0] in that case raises "list index out of range"
    and crashes the calling agent. We instead join every text block and
    return "" when there are none, leaving JSON parsing / safe fallbacks to
    the caller. Truncation is logged so wide-dataset failures are diagnosable.
    """
    blocks = getattr(response, "content", None) or []
    text = "".join(
        getattr(b, "text", "") for b in blocks if getattr(b, "type", None) == "text"
    )
    stop_reason = getattr(response, "stop_reason", None)
    if not text:
        logger.error(
            "the model returned no text content (stop_reason=%s, n_blocks=%d)",
            stop_reason, len(blocks),
        )
    elif stop_reason == "max_tokens":
        logger.warning(
            "model response truncated at max_tokens=%d (%d chars). "
            "JSON may be incomplete; raise max_tokens on the caller.",
            max_tokens, len(text),
        )
    return text


def _cached_system(system: str | list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Wrap a string system prompt in a cache_control block (§28).

    Passing a list of blocks is a pass-through - callers that already build
    structured system content can control caching themselves.
    Returns None when system is None so callers can use `if cached_system`.
    """
    if system is None:
        return None
    if isinstance(system, list):
        return system
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


async def call_claude(
    messages: list[dict[str, Any]],
    model: str,
    system: str | list[dict[str, Any]] | None = None,
    max_tokens: int = 4096,
) -> str:
    """Non-streaming model call. Returns the text of the first content block.

    System prompts are automatically wrapped in a prompt-caching block (§28).
    Expected JSON response schema is documented in each caller's docstring;
    parse with extract_json() after this call.
    """
    kwargs: dict[str, Any] = dict(model=model, messages=messages, max_tokens=max_tokens)
    cached_sys = _cached_system(system)
    if cached_sys:
        kwargs["system"] = cached_sys

    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            async with _make_client() as client:
                response = await client.messages.create(**kwargs)
            return _response_text(response, max_tokens)
        except (anthropic.RateLimitError, anthropic.APIStatusError) as exc:
            last_exc = exc
            if attempt < _MAX_ATTEMPTS - 1:
                delay = _RETRY_DELAYS[attempt]
                logger.warning("model API error (attempt %d/%d), retrying in %ds: %s",
                               attempt + 1, _MAX_ATTEMPTS, delay, exc)
                await asyncio.sleep(delay)

    exc_detail = f"{type(last_exc).__name__}: {last_exc}" if last_exc else "unknown"
    raise RuntimeError(f"model call failed after {_MAX_ATTEMPTS} attempts - {exc_detail}") from last_exc


async def call_claude_stream(
    messages: list[dict[str, Any]],
    model: str,
    system: str | list[dict[str, Any]] | None = None,
    max_tokens: int = 4096,
    on_chunk: Callable[[str], None | Awaitable[None]] | None = None,
) -> str:
    """Streaming model call. Returns the full accumulated text.

    System prompts are automatically wrapped in a prompt-caching block (§28).
    on_chunk is called for each text delta. Sync and async callables are both
    supported. Use it to forward chunks to a Redis progress channel or SSE stream.
    """
    kwargs: dict[str, Any] = dict(model=model, messages=messages, max_tokens=max_tokens)
    cached_sys = _cached_system(system)
    if cached_sys:
        kwargs["system"] = cached_sys

    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            full_text = ""
            async with _make_client() as client:
                async with client.messages.stream(**kwargs) as stream:
                    async for text in stream.text_stream:
                        full_text += text
                        if on_chunk is not None:
                            result = on_chunk(text)
                            if asyncio.iscoroutine(result):
                                await result
                    final = await stream.get_final_message()
                if final.stop_reason == "max_tokens":
                    raise RuntimeError(
                        f"model response truncated at {len(full_text)} chars "
                        f"(max_tokens={max_tokens}). Raise max_tokens on the caller."
                    )
            if full_text.strip():
                return full_text
            # Empty completion (zero text blocks, stop_reason e.g. "end_turn") is
            # not an exception - the stream simply yielded nothing. Because no text
            # was emitted, nothing reached on_chunk, so a re-roll is safe (no
            # double-emit). Retry before surfacing "" to the caller's fallback;
            # an empty turn is usually transient. Use a short delay - this is a
            # re-roll, not rate-limit backoff.
            if attempt < _MAX_ATTEMPTS - 1:
                logger.warning(
                    "model stream returned no text (stop_reason=%s), re-rolling "
                    "(attempt %d/%d)", final.stop_reason, attempt + 1, _MAX_ATTEMPTS,
                )
                await asyncio.sleep(1)
                continue
            logger.error(
                "the model returned no text content from stream after %d attempts "
                "(stop_reason=%s)", _MAX_ATTEMPTS, final.stop_reason,
            )
            return full_text
        except (anthropic.RateLimitError, anthropic.APIStatusError) as exc:
            last_exc = exc
            if attempt < _MAX_ATTEMPTS - 1:
                delay = _RETRY_DELAYS[attempt]
                logger.warning("model stream error (attempt %d/%d), retrying in %ds: %s",
                               attempt + 1, _MAX_ATTEMPTS, delay, exc)
                await asyncio.sleep(delay)

    exc_detail = f"{type(last_exc).__name__}: {last_exc}" if last_exc else "unknown"
    raise RuntimeError(f"model stream failed after {_MAX_ATTEMPTS} attempts - {exc_detail}") from last_exc


def extract_json(text: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract a JSON object from model output.

    Handles:
    - Bare JSON objects
    - ```json ... ``` fences
    - ``` ... ``` fences
    - JSON object embedded in surrounding prose

    Returns fallback (default: {}) if all extraction attempts fail.
    Never raises - pipeline must not crash on a model parse error.
    """
    text = text.strip()

    # Try fenced code block: strip markers from edges rather than scanning
    # across the content with a non-greedy regex - the latter breaks when
    # the model embeds triple-backtick sequences inside JSON string values.
    fence_stripped = re.sub(r"^```(?:json)?\s*\n?", "", text)
    fence_stripped = re.sub(r"\n?```\s*$", "", fence_stripped).strip()
    if fence_stripped != text:
        try:
            result = json.loads(fence_stripped)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Try a fenced block sitting inside surrounding prose, e.g.
    # "Here is the strategy:\n```json\n{...}\n```". Edge-stripping above only
    # catches fences flush against the text boundaries.
    inner_fence = re.search(r"```(?:json)?\s*\n?(\{[\s\S]*?\})\s*\n?```", text)
    if inner_fence:
        try:
            result = json.loads(inner_fence.group(1))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Try the full text as JSON
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try to extract the outermost JSON object from mixed text
    brace_match = re.search(r"\{[\s\S]+\}", text)
    if brace_match:
        try:
            result = json.loads(brace_match.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Last resort: repair a truncated object (response cut off by max_tokens).
    # Recover whatever complete keys/values precede the cut rather than losing
    # the entire strategy to a safe fallback.
    repaired = _repair_truncated_json(text)
    if repaired is not None:
        logger.warning("extract_json: recovered %d keys from truncated JSON", len(repaired))
        return repaired

    logger.warning("extract_json: could not parse JSON from model output (len=%d)", len(text))
    return fallback if fallback is not None else {}


def _repair_truncated_json(text: str) -> dict[str, Any] | None:
    """Best-effort recovery of a JSON object truncated mid-output.

    Scans from the first '{', tracking string state and the bracket stack,
    and cuts at the last balanced-able closing bracket, appending the closers
    still owed. Returns the parsed dict, or None if nothing usable survives.
    Any keys/values after the cut point are dropped - by construction they
    were incomplete.
    """
    start = text.find("{")
    if start == -1:
        return None
    s = text[start:]

    stack: list[str] = []
    in_str = False
    escape = False
    best_cut = -1
    best_closers = ""

    for i, ch in enumerate(s):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack:
                stack.pop()
            best_cut = i + 1
            best_closers = "".join(reversed(stack))

    if best_cut == -1:
        return None

    candidate = s[:best_cut].rstrip().rstrip(",") + best_closers
    try:
        result = json.loads(candidate)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        return None
