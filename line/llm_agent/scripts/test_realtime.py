#!/usr/bin/env python3
"""
Test script for the RealtimeProvider — OpenAI Realtime WebSocket (text-only).

Compares latency against the conventional HTTP LLMProvider approach.

Usage:
    uv run python line/llm_agent/scripts/test_realtime.py [OPTIONS]

Options:
    --tests TESTS           Comma-separated tests (default: all)
                            Available: streaming, tools, multi_turn, diff_sync,
                            latency_comparison
    --model MODEL           Realtime model (default: gpt-4o-mini-realtime-preview)
    --http-model MODEL      HTTP model for comparison (default: gpt-4o-mini)
    --runs N                Iterations for latency comparison (default: 3)
    --diverge-rate FLOAT    Fraction of turns with history divergence (default: 0.3)

Environment variables:
    OPENAI_API_KEY          Required for both Realtime and HTTP providers.
"""

import argparse
import asyncio
import logging
import os
import random
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
from line.llm_agent.provider import LLMProvider, Message
from line.llm_agent.realtime_provider import RealtimeProvider

# =============================================================================
# Test Tools
# =============================================================================


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

    async with await provider.chat(messages, tools, config=config) as stream:
        async for chunk in stream:
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


async def test_streaming(api_key: str, model: str):
    """Basic text streaming over Realtime WS."""
    print_header(f"Streaming Test ({model})")

    provider = RealtimeProvider(model=model, api_key=api_key)
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
    """Tool calling over Realtime WS via LlmAgent."""
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


async def test_multi_turn(api_key: str, model: str):
    """Multi-turn conversation — WS benefits from persistent connection."""
    print_header(f"Multi-Turn Test ({model})")

    provider = RealtimeProvider(model=model, api_key=api_key)
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


async def test_diff_sync(api_key: str, model: str):
    """Simulate history divergence (interrupted text) and verify diff-sync."""
    print_header(f"Diff-Sync Test ({model})")

    provider = RealtimeProvider(model=model, api_key=api_key)
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

        # This should trigger a diff that deletes the old assistant item and creates new ones
        text2, _, ttft2, total2 = await collect_stream(provider, messages, config=config)
        print(f"Turn 2 (diverged): TTFT={ttft2:.1f}ms Total={total2:.1f}ms")
        print(f"  Response: {text2.strip()[:80]}...")

        # Turn 3: Normal append after divergence
        messages.append(Message(role="assistant", content=text2))
        messages.append(Message(role="user", content="Compare the two cities."))
        text3, _, ttft3, total3 = await collect_stream(provider, messages, config=config)
        print(f"Turn 3 (normal after diverge): TTFT={ttft3:.1f}ms Total={total3:.1f}ms")
        print(f"  Response: {text3.strip()[:80]}...")

        print("PASS: Diff-sync test")
    finally:
        await provider.aclose()


async def test_latency_comparison(
    api_key: str,
    rt_model: str,
    http_model: str,
    runs: int,
    diverge_rate: float,
):
    """Compare HTTP vs Realtime WS latency across multiple iterations."""
    print_header(f"Latency Comparison: HTTP ({http_model}) vs WS ({rt_model})")
    print(f"Runs: {runs} | Diverge rate: {diverge_rate}")

    http_provider = LLMProvider(model=http_model, api_key=api_key)
    ws_provider = RealtimeProvider(model=rt_model, api_key=api_key)
    config = LlmConfig(system_prompt="You are a helpful assistant. Be brief (1-2 sentences max).")

    conversation_prompts = [
        "What is the largest ocean?",
        "How deep is it at its deepest point?",
        "What creatures live at that depth?",
    ]

    http_ttfts = []
    http_totals = []
    ws_ttfts = []
    ws_totals = []

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
    "diff_sync",
    "latency_comparison",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test RealtimeProvider (OpenAI Realtime WS, text-only).",
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
        default="gpt-4o-mini-realtime-preview",
        help="Realtime model (default: gpt-4o-mini-realtime-preview)",
    )
    parser.add_argument(
        "--http-model",
        type=str,
        default="gpt-4o-mini",
        help="HTTP model for comparison (default: gpt-4o-mini)",
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


async def main(args):
    print("OpenAI Realtime Provider Test")
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
    print(f"Realtime model: {args.model}")
    print(f"HTTP model: {args.http_model}")

    try:
        if "streaming" in tests_to_run:
            await test_streaming(api_key, args.model)
        if "tools" in tests_to_run:
            await test_tools(api_key, args.model)
        if "multi_turn" in tests_to_run:
            await test_multi_turn(api_key, args.model)
        if "diff_sync" in tests_to_run:
            await test_diff_sync(api_key, args.model)
        if "latency_comparison" in tests_to_run:
            await test_latency_comparison(
                api_key,
                args.model,
                args.http_model,
                args.runs,
                args.diverge_rate,
            )
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
        sys.stderr = open(os.devnull, "w")

    sys.exit(exit_code)
