#!/usr/bin/env python3
"""Ad-hoc integration tests for the WebSocket provider.

Tests:
    1. previous_response_not_found recovery: inject a fake response_id and
       verify the retry logic recovers.
    2. WebSocket model config: verify temperature/top_p rejection, and that
       tools and reasoning_effort are accepted by the real API.

Usage:
    OPENAI_API_KEY=... uv run python tests/real_api_test_websocket_provider.py
"""

import asyncio
import os
import re
import sys
from typing import Annotated

from loguru import logger

from line.llm_agent.config import LlmConfig, _normalize_config
from line.llm_agent.provider import Message
from line.llm_agent.tools.decorators import loopback_tool
from line.llm_agent.websocket_provider import _WebSocketProvider


@loopback_tool
async def get_weather(ctx, city: Annotated[str, "City name"]) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


logger.enable("line")

_WEBSOCKET_MODELS = ["gpt-5.2", "gpt-5.4"]
_MIN_MESSAGE = [Message(role="user", content="Say 'hello' and nothing else.")]


def _looks_like_expected_api_rejection(exc: BaseException) -> bool:
    """True if ``exc`` plausibly means the API rejected the request, not infra noise.

    We match on specific patterns rather than substring hints to avoid false
    positives (e.g. "401" matching inside a response_id).
    """
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc)

    # Clean WebSocket close with a normal close code → API rejected the request.
    m = re.search(r"websocket closed \(code=(\d+)\)", msg, re.IGNORECASE)
    if m:
        code = int(m.group(1))
        # 1000 = normal close (API chose to close), 1008 = policy violation.
        # 1006 = abnormal closure (no close frame — network drop), not a rejection.
        return code in (1000, 1008)

    # Structured OpenAI API error from the stream parser.
    if re.search(r"openai api error", msg, re.IGNORECASE):
        return True

    return False


# ---------------------------------------------------------------------------
# Helpers to reduce per-test boilerplate
# ---------------------------------------------------------------------------


async def _collect_text(provider, messages, *, config, tools=None):
    """Run a chat and return the concatenated text."""
    text = ""
    async for chunk in provider.chat(messages, tools=tools, config=config):
        if chunk.text:
            text += chunk.text
    return text.strip()


async def _collect_tool_calls(provider, messages, *, config, tools):
    """Run a chat and return whether any tool calls were received."""
    async for chunk in provider.chat(messages, tools=tools, config=config):
        if chunk.tool_calls:
            return True
    return False


async def _run_expect_rejection(name, api_key, config, *, model):
    """Run a chat that should be rejected by the WebSocket API.

    Returns None on expected rejection, or an error string on failure.
    """
    provider = _WebSocketProvider(model=model, api_key=api_key, default_reasoning_effort=None)
    try:
        text = await _collect_text(provider, _MIN_MESSAGE, config=config)
        if text:
            return f"expected rejection but got response: {text!r}"
        return "empty response instead of error"
    except RuntimeError as e:
        if _looks_like_expected_api_rejection(e):
            print(f"  rejected as expected: {e}")
            return None
        return f"unexpected RuntimeError (not treated as API rejection): {e}"
    except Exception as e:
        return f"unexpected {type(e).__name__}: {e}"
    finally:
        await provider.aclose()


async def _run_expect_success(name, api_key, config, *, model, **kwargs):
    """Run a chat that should succeed. Returns None on success, or an error string."""
    provider = _WebSocketProvider(
        model=model,
        api_key=api_key,
        default_reasoning_effort=kwargs.get("default_reasoning_effort"),
    )
    try:
        if kwargs.get("expect_tool_call"):
            got_tool = await _collect_tool_calls(
                provider, kwargs["messages"], config=config, tools=kwargs["tools"]
            )
            if got_tool:
                return None
            return "no tool call received"
        else:
            text = await _collect_text(provider, _MIN_MESSAGE, config=config)
            if text:
                print(f"  response: {text!r}")
                return None
            return "empty response"
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    finally:
        await provider.aclose()


# ---------------------------------------------------------------------------
# Test: previous_response_not_found recovery
# ---------------------------------------------------------------------------


async def test_response_not_found_recovery():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return 1

    model = "gpt-5-nano"
    config = _normalize_config(LlmConfig(system_prompt="You are a helpful assistant. Be brief."))
    provider = _WebSocketProvider(model=model, api_key=api_key, default_reasoning_effort=None)

    try:
        messages = [Message(role="user", content="Say 'hello' and nothing else.")]
        text = await _collect_text(provider, messages, config=config)
        print(f"Turn 1 response: {text!r}")

        # Corrupt history so next request sends a fake previous_response_id
        provider._history = [
            (identity, "resp_FAKE_000" if rid else None) for identity, rid in provider._history
        ]
        print("Corrupted history — all response_ids replaced with 'resp_FAKE_000'")

        messages.append(Message(role="assistant", content=text))
        messages.append(Message(role="user", content="Now say 'goodbye' and nothing else."))

        text2 = await _collect_text(provider, messages, config=config)
        print(f"Turn 2 response: {text2!r}")

        if text2:
            print("PASS: retry after previous_response_not_found succeeded")
            return 0
        else:
            print("FAIL: no response text on turn 2")
            return 1
    finally:
        await provider.aclose()


# ---------------------------------------------------------------------------
# Test: WebSocket model config (temperature, top_p, reasoning, tools)
# ---------------------------------------------------------------------------


async def test_websocket_model_config():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return 1

    failures = []

    rejection_tests = [
        ("temperature=0.7", LlmConfig(system_prompt="Be brief.", temperature=0.7)),
        ("temperature=0", LlmConfig(system_prompt="Be brief.", temperature=0.0)),
        ("top_p=0.9", LlmConfig(system_prompt="Be brief.", top_p=0.9)),
    ]

    for model in _WEBSOCKET_MODELS:
        print(f"\n{'#' * 60}")
        print(f"# Model: {model}")
        print("#" * 60)

        # Tests that should be rejected by the WebSocket API
        for name, llm_config in rejection_tests:
            label = f"{model}/{name}"
            print(f"\n{'=' * 60}")
            print(f"Test: WebSocket rejects {name} ({model})")
            print("=" * 60)
            err = await _run_expect_rejection(name, api_key, _normalize_config(llm_config), model=model)
            if err:
                print(f"FAIL: {err}")
                failures.append(label)
            else:
                print(f"PASS: {name} correctly rejected")

        # Tests that should succeed
        label = f"{model}/reasoning_effort=low"
        print(f"\n{'=' * 60}")
        print(f"Test: WebSocket completion with reasoning_effort=low ({model})")
        print("=" * 60)
        err = await _run_expect_success(
            "reasoning_effort=low",
            api_key,
            _normalize_config(LlmConfig(system_prompt="Be brief.", reasoning_effort="low")),
            model=model,
            default_reasoning_effort="low",
        )
        if err:
            print(f"FAIL: {err}")
            failures.append(label)
        else:
            print("PASS: reasoning_effort=low accepted")

        label = f"{model}/tools"
        print(f"\n{'=' * 60}")
        print(f"Test: WebSocket completion with tools ({model})")
        print("=" * 60)
        err = await _run_expect_success(
            "tools",
            api_key,
            _normalize_config(
                LlmConfig(
                    system_prompt="Use the get_weather tool when asked about weather.",
                )
            ),
            model=model,
            messages=[Message(role="user", content="What's the weather in Tokyo?")],
            tools=[get_weather],
            expect_tool_call=True,
        )
        if err:
            print(f"FAIL: {err}")
            failures.append(label)
        else:
            print("PASS: tool call received via WebSocket")

    # Summary
    print(f"\n{'=' * 60}")
    if failures:
        print(f"DONE — {len(failures)} failure(s): {', '.join(failures)}")
        return 1
    print("All WebSocket model config tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(test_response_not_found_recovery())
    if exit_code == 0:
        exit_code = asyncio.run(test_websocket_model_config())
    sys.exit(exit_code)
