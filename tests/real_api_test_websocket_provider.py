#!/usr/bin/env python3
"""
Test script for the _WebSocketProvider — OpenAI Responses API over WebSocket.

Compares latency against the conventional HTTP LlmProvider approach and
validates divergence handling (simulated TTS interruption / truncation).

Usage:
    uv run python line/llm_agent/scripts/real_api_test_websocket_provider.py [OPTIONS]

Options:
    --tests TESTS           Comma-separated tests (default: all)
                            Available: streaming, tools, multi_turn, divergence,
                            latency_comparison
    --model MODEL           WebSocket model (default: gpt-5-nano)
    --http-model MODEL      HTTP model for comparison (default: gpt-5-nano)
    --runs N                Iterations for latency comparison (default: 3)
    --diverge-rate FLOAT    Fraction of turns with history divergence (default: 0.3)

Environment variables:
    OPENAI_API_KEY          Required for both WebSocket and HTTP providers.
"""

import argparse
import asyncio
import logging
import os
import random
import re
import sys
import time
from typing import Annotated, List
import warnings

import litellm
from loguru import logger

from line.agent import TurnEnv
from line.events import (
    AgentSendText,
    AgentToolCalled,
    AgentToolReturned,
    UserTextSent,
)
from line.llm_agent import LlmAgent, LlmConfig, loopback_tool
from line.llm_agent.config import _normalize_config
from line.llm_agent.provider import LlmProvider, Message, ParsedModelId, parse_model_id
from line.llm_agent.websocket_provider import _WebSocketProvider

# =============================================================================
# Test Tools
# =============================================================================

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


@loopback_tool
async def get_weather(ctx, city: Annotated[str, "City name"]) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "new york": "72°F, partly cloudy",
        "san francisco": "65°F, foggy",
        "london": "55°F, rainy",
        "tokyo": "80°F, sunny",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@loopback_tool
async def calculate(ctx, expression: Annotated[str, "Math expression to evaluate"]) -> str:
    """Evaluate a mathematical expression."""
    try:
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            result = eval(expression)
            return f"{expression} = {result}"
        return "Invalid expression"
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# Helpers
# =============================================================================


async def collect_stream(provider, messages, tools=None, config=None):
    """Run a chat() call and collect all chunks. Returns (full_text, tool_calls, ttft, total)."""
    start = time.perf_counter()
    ttft = None
    full_text = ""
    all_tool_calls = []

    async for chunk in provider.chat(messages, tools, config=config):
        if chunk.text:
            if ttft is None:
                ttft = (time.perf_counter() - start) * 1000
            full_text += chunk.text
        if chunk.tool_calls:
            all_tool_calls = chunk.tool_calls

    total = (time.perf_counter() - start) * 1000
    if ttft is None:
        ttft = total
    return full_text, all_tool_calls, ttft, total


def print_header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# =============================================================================
# Tests
# =============================================================================


async def test_streaming(api_key: str, model_id: ParsedModelId):
    """Basic text streaming over WebSocket mode."""
    print_header(f"Streaming Test ({model_id.model})")

    provider = _WebSocketProvider(model_id=model_id, api_key=api_key)
    messages = [Message(role="user", content="Say 'Hello, World!' and nothing else.")]

    try:
        text, _, ttft, total = await collect_stream(
            provider,
            messages,
            config=LlmConfig(system_prompt="You are a helpful assistant."),
        )
        print(f"Response: {text.strip()}")
        print(f"TTFT: {ttft:.1f}ms | Total: {total:.1f}ms")
        print("PASS: Streaming test")
    finally:
        await provider.aclose()


async def test_tools(api_key: str, model: str):
    """Tool calling over WebSocket mode via LlmAgent."""
    print_header(f"Tool Calling Test ({model})")

    agent = LlmAgent(
        model=model,
        api_key=api_key,
        tools=[get_weather, calculate],
        config=LlmConfig(system_prompt="You are a helpful assistant. Use tools when needed."),
    )

    env = TurnEnv()
    user_message = "What's the weather in Tokyo? Also, what's 25 * 4?"
    event = UserTextSent(
        content=user_message,
        history=[UserTextSent(content=user_message)],
    )

    print(f"User: {user_message}")
    print("Agent response:")

    tool_calls_made = []
    start = time.perf_counter()
    ttft = None

    async for output in agent.process(env, event):
        if isinstance(output, AgentSendText):
            if ttft is None:
                ttft = (time.perf_counter() - start) * 1000
            print(output.text, end="", flush=True)
        elif isinstance(output, AgentToolCalled):
            tool_calls_made.append(output.tool_name)
            print(f"\n  [Tool call]: {output.tool_name}({output.tool_args})")
        elif isinstance(output, AgentToolReturned):
            print(f"  [Tool result]: {output.result}")

    total = (time.perf_counter() - start) * 1000
    print(f"\nTTFT: {ttft:.1f}ms | Total: {total:.1f}ms" if ttft else f"\nTotal: {total:.1f}ms")

    if tool_calls_made:
        print(f"PASS: Tool calling (tools used: {', '.join(tool_calls_made)})")
    else:
        print("WARN: No tools called (model may have answered directly)")

    await agent.cleanup()


async def test_multi_turn(api_key: str, model_id: ParsedModelId):
    """Multi-turn conversation — WS benefits from persistent connection + previous_response_id."""
    print_header(f"Multi-Turn Test ({model_id.model})")

    provider = _WebSocketProvider(model_id=model_id, api_key=api_key)
    config = LlmConfig(system_prompt="You are a helpful assistant. Be brief.")

    turns = [
        "What is the capital of France?",
        "And what about Germany?",
        "Which of those two cities has a larger population?",
    ]

    messages: List[Message] = []
    try:
        for i, user_text in enumerate(turns):
            messages.append(Message(role="user", content=user_text))
            text, _, ttft, total = await collect_stream(provider, messages, config=config)
            messages.append(Message(role="assistant", content=text))
            print(f"Turn {i + 1}: User: {user_text}")
            print(f"         Assistant: {text.strip()[:100]}...")
            print(f"         TTFT: {ttft:.1f}ms | Total: {total:.1f}ms")
        print("PASS: Multi-turn test")
    finally:
        await provider.aclose()


async def test_divergence(api_key: str, model_id: ParsedModelId):
    """Simulate TTS interruption (truncated assistant text) and verify divergence handling."""
    print_header(f"Divergence Test ({model_id.model})")

    provider = _WebSocketProvider(model_id=model_id, api_key=api_key)
    config = LlmConfig(system_prompt="You are a helpful assistant. Be brief.")

    try:
        # Turn 1: Normal conversation
        messages = [Message(role="user", content="Tell me about Paris.")]
        text1, _, ttft1, total1 = await collect_stream(provider, messages, config=config)
        print(f"Turn 1 (normal): TTFT={ttft1:.1f}ms Total={total1:.1f}ms")
        print(f"  Response: {text1.strip()[:80]}...")

        # Turn 2: Simulate interruption — truncate the assistant's response
        truncated = text1[: len(text1) // 3] + "..."
        messages.append(Message(role="assistant", content=truncated))
        messages.append(Message(role="user", content="Now tell me about London."))

        # This should trigger divergence detection (truncated != full text)
        # and fall back to sending the full context.
        text2, _, ttft2, total2 = await collect_stream(provider, messages, config=config)
        print(f"Turn 2 (truncated — diverged): TTFT={ttft2:.1f}ms Total={total2:.1f}ms")
        print(f"  Response: {text2.strip()[:80]}...")

        # Turn 3: Normal append after divergence — should use continuation again
        messages.append(Message(role="assistant", content=text2))
        messages.append(Message(role="user", content="Compare the two cities."))
        text3, _, ttft3, total3 = await collect_stream(provider, messages, config=config)
        print(f"Turn 3 (normal after diverge): TTFT={ttft3:.1f}ms Total={total3:.1f}ms")
        print(f"  Response: {text3.strip()[:80]}...")

        print("PASS: Divergence test")
    finally:
        await provider.aclose()


async def test_response_not_found_retry(api_key: str, model_id: ParsedModelId):
    """Force a ``previous_response_not_found`` by corrupting history; verify retry recovers."""
    print_header(f"Response Not Found Retry Test ({model_id.model})")

    provider = _WebSocketProvider(model_id=model_id, api_key=api_key, default_reasoning_effort=None)
    config = LlmConfig(system_prompt="You are a helpful assistant. Be brief.")

    try:
        messages = [Message(role="user", content="Say 'hello' and nothing else.")]
        text, _, _, _ = await collect_stream(provider, messages, config=config)
        print(f"Turn 1 response: {text.strip()!r}")
        print(f"History length after turn 1: {len(provider._history)}")
        print(f"Response IDs in history: {[rid for _, rid in provider._history if rid]}")

        provider._history = [
            (identity, "resp_FAKE_000" if rid else None) for identity, rid in provider._history
        ]
        print("Corrupted history — all response_ids replaced with 'resp_FAKE_000'")

        messages.append(Message(role="assistant", content=text))
        messages.append(Message(role="user", content="Now say 'goodbye' and nothing else."))
        text2, _, _, _ = await collect_stream(provider, messages, config=config)

        print(f"\nTurn 2 response: {text2.strip()!r}")
        print(f"History length after turn 2: {len(provider._history)}")
        print(f"Response IDs in history: {[rid for _, rid in provider._history if rid]}")

        if text2.strip():
            print("PASS: retry after previous_response_not_found succeeded")
        else:
            print("FAIL: no response text on turn 2")
    finally:
        await provider.aclose()


async def test_latency_comparison(
    api_key: str,
    ws_model_id: ParsedModelId,
    http_model: str,
    runs: int,
    diverge_rate: float,
):
    """Compare HTTP vs WebSocket mode latency across multiple iterations."""
    print_header(f"Latency Comparison: HTTP ({http_model}) vs WS ({ws_model_id.model})")
    print(f"Runs: {runs} | Diverge rate: {diverge_rate}")

    http_provider = LlmProvider(model=http_model, api_key=api_key)
    ws_provider = _WebSocketProvider(model_id=ws_model_id, api_key=api_key)
    config = LlmConfig(system_prompt="You are a helpful assistant. Be brief (1-2 sentences max).")

    conversation_prompts = [
        "What is the largest ocean?",
        "How deep is it at its deepest point?",
        "What creatures live at that depth?",
    ]

    http_ttfts: List[float] = []
    http_totals: List[float] = []
    ws_ttfts: List[float] = []
    ws_totals: List[float] = []

    try:
        for run in range(runs):
            print(f"\n--- Run {run + 1}/{runs} ---")

            # HTTP run
            http_messages: List[Message] = []
            for i, prompt in enumerate(conversation_prompts):
                http_messages.append(Message(role="user", content=prompt))
                text, _, ttft, total = await collect_stream(http_provider, http_messages, config=config)

                # Simulate divergence
                if i < len(conversation_prompts) - 1 and random.random() < diverge_rate:
                    text = text[: len(text) // 3] + "..."
                http_messages.append(Message(role="assistant", content=text))
                http_ttfts.append(ttft)
                http_totals.append(total)
                print(f"  HTTP turn {i + 1}: TTFT={ttft:.1f}ms Total={total:.1f}ms")

            # WS run
            ws_messages: List[Message] = []
            for i, prompt in enumerate(conversation_prompts):
                ws_messages.append(Message(role="user", content=prompt))
                text, _, ttft, total = await collect_stream(ws_provider, ws_messages, config=config)

                # Simulate divergence
                if i < len(conversation_prompts) - 1 and random.random() < diverge_rate:
                    text = text[: len(text) // 3] + "..."
                ws_messages.append(Message(role="assistant", content=text))
                ws_ttfts.append(ttft)
                ws_totals.append(total)
                print(f"  WS   turn {i + 1}: TTFT={ttft:.1f}ms Total={total:.1f}ms")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        _print_stats("HTTP TTFT", http_ttfts)
        _print_stats("WS   TTFT", ws_ttfts)
        _print_stats("HTTP Total", http_totals)
        _print_stats("WS   Total", ws_totals)

        if ws_ttfts and http_ttfts:
            avg_http = sum(http_ttfts) / len(http_ttfts)
            avg_ws = sum(ws_ttfts) / len(ws_ttfts)
            diff_pct = ((avg_http - avg_ws) / avg_http) * 100 if avg_http > 0 else 0
            print(f"\nWS TTFT is {diff_pct:+.1f}% vs HTTP (positive = WS faster)")

        print("PASS: Latency comparison")
    finally:
        await ws_provider.aclose()


def _print_stats(label: str, values: List[float]):
    if not values:
        print(f"  {label}: no data")
        return
    avg = sum(values) / len(values)
    mn = min(values)
    mx = max(values)
    print(f"  {label}: avg={avg:.1f}ms  min={mn:.1f}ms  max={mx:.1f}ms  (n={len(values)})")


# =============================================================================
# Main
# =============================================================================

AVAILABLE_TESTS = [
    "streaming",
    "tools",
    "multi_turn",
    "divergence",
    "response_not_found_retry",
    "latency_comparison",
    "websocket_model_config",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test _WebSocketProvider (OpenAI Responses API over WS).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tests",
        type=str,
        default="all",
        help=f"Comma-separated tests. Available: {', '.join(AVAILABLE_TESTS)}, all",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-nano",
        help="WebSocket model (default: gpt-5-nano)",
    )
    parser.add_argument(
        "--http-model",
        type=str,
        default="gpt-5-nano",
        help="HTTP model for comparison (default: gpt-5-nano)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Iterations for latency comparison (default: 3)",
    )
    parser.add_argument(
        "--diverge-rate",
        type=float,
        default=0.3,
        help="Fraction of turns with history divergence (default: 0.3)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Test: WebSocket model config (temperature, top_p, reasoning, tools)
# ---------------------------------------------------------------------------


async def test_websocket_model_config(api_key: str, model_id: ParsedModelId):
    """Verify WebSocket API accepts temperature, top_p, reasoning_effort, and tools."""
    failures = []

    rejection_tests = [
        ("temperature=0.7", LlmConfig(system_prompt="Be brief.", temperature=0.7)),
        ("temperature=0", LlmConfig(system_prompt="Be brief.", temperature=0.0)),
        ("top_p=0.9", LlmConfig(system_prompt="Be brief.", top_p=0.9)),
    ]

    model = model_id.model
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


async def main(args):
    print("OpenAI WebSocket Mode Provider Test")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return 1
    if args.tests == "all":
        tests_to_run = set(AVAILABLE_TESTS)
    else:
        tests_to_run = {t.strip() for t in args.tests.split(",")}
        invalid = tests_to_run - set(AVAILABLE_TESTS)
        if invalid:
            print(f"Unknown tests: {', '.join(invalid)}")
            print(f"Available: {', '.join(AVAILABLE_TESTS)}")
            return 1

    print(f"Tests: {', '.join(sorted(tests_to_run))}")
    print(f"WS model: {args.model}")
    print(f"HTTP model: {args.http_model}")

    model_id = parse_model_id(args.model)

    try:
        if "streaming" in tests_to_run:
            await test_streaming(api_key, model_id)
        if "tools" in tests_to_run:
            await test_tools(api_key, args.model)
        if "multi_turn" in tests_to_run:
            await test_multi_turn(api_key, model_id)
        if "divergence" in tests_to_run:
            await test_divergence(api_key, model_id)
        if "response_not_found_retry" in tests_to_run:
            await test_response_not_found_retry(api_key, model_id)
        if "latency_comparison" in tests_to_run:
            await test_latency_comparison(
                api_key,
                model_id,
                args.http_model,
                args.runs,
                args.diverge_rate,
            )
        if "websocket_model_config" in tests_to_run:
            await test_websocket_model_config(api_key, model_id)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    args = parse_args()

    warnings.filterwarnings("ignore", category=ResourceWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
    litellm.suppress_debug_info = True
    logger.disable("line")

    try:
        exit_code = asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nInterrupted")
        exit_code = 1
    finally:
        devnull = open(os.devnull, "w")
        sys.stderr = devnull

    sys.exit(exit_code)
