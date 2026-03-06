#!/usr/bin/env python3
"""
Benchmark average latency + stddev for LLM models via LlmProvider.

Usage:
    uv run python line/llm_agent/scripts/bench_latency.py [OPTIONS]

Options:
    --runs N            Number of conversations per model (default: 20)
    --model MODEL       Only test specific model (e.g., "openai/gpt-5-nano")
    --pause SECONDS     Pause between conversations (default: 0.0)

Environment variables:
    OPENAI_API_KEY      - For OpenAI models (openai/gpt-5.2, gpt-5-mini, gpt-5-nano)
    ANTHROPIC_API_KEY   - For Anthropic models (anthropic/claude-haiku-4-5)
    GEMINI_API_KEY      - For Google models (gemini/gemini-2.5-flash, etc.)

The script will test whichever providers have API keys set.
"""

import argparse
import asyncio
from dataclasses import dataclass
import logging
import os
import statistics
import sys
import time
from typing import Optional
import uuid
import warnings

import litellm
from loguru import logger

from line.llm_agent.config import LlmConfig, _normalize_config
from line.llm_agent.provider import LlmProvider, Message

# =============================================================================
# Config
# =============================================================================

SYSTEM_PROMPT = """\
You are a friendly voice assistant built with Cartesia, designed for natural, open-ended conversation.

# Personality

Warm, curious, genuine, lighthearted. Knowledgeable but not showy.

# Voice and tone

Speak like a thoughtful friend, not a formal assistant or customer service bot.
Use contractions and casual phrasing—the way people actually talk.
Match the caller's energy: playful if they're playful, grounded if they're serious.
Show genuine interest: "Oh that's interesting" or "Hmm, let me think about that."

# Response style

Keep responses to 1-2 sentences for most exchanges. This is a conversation, not a lecture.
For complex topics, break information into digestible pieces and check in with the caller.
Never use lists, bullet points, or structured formatting—speak in natural prose.
Never say "Great question!" or other hollow affirmations.

# Tools

## web_search
Use when you genuinely don't know something or need current information. Don't overuse it.

Before searching, acknowledge naturally:
- "Let me look that up"
- "Good question, let me check"
- "Hmm, I'm not sure—give me a sec"

After searching, synthesize into a brief conversational answer. Never read search results verbatim.

## end_call
Use when the conversation has clearly concluded—goodbye, thanks, that's all, etc.

Process:
1. Say a natural goodbye first: "Take care!" or "Nice chatting with you!"
2. Then call end_call

Never use for brief pauses or "hold on" moments.

# About Cartesia (share when asked or naturally relevant)
Cartesia is a voice AI company making voice agents that feel natural and responsive. Your voice comes from
Sonic, their text-to-speech model with ultra-low latency—under 90ms to first audio. You hear through Ink,
their speech-to-text model optimized for real-world noise. This agent runs on Line, Cartesia's open-source
voice agent framework. For building voice agents: docs.cartesia.ai

# Handling common situations
Didn't catch something: "Sorry, I didn't catch that—could you say that again?"
Don't know the answer: "I'm not sure about that. Want me to look it up?"
Caller seems frustrated: Acknowledge it, try a different approach
Off-topic or unusual request: Roll with it—you can chat about anything

# Topics you can discuss
Anything the caller wants: their day, current events, science, culture, philosophy, personal decisions,
interesting ideas. Help think through problems by asking clarifying questions. Use light, natural humor when
appropriate."""

MODELS = [
    {"model": "openai/gpt-5.2", "reasoning_effort": "none"},
    # {"model": "gemini/gemini-3.1-flash-lite-preview"},
    # {"model": "gemini/gemini-3-flash-preview"},
    # {"model": "gemini/gemini-2.5-flash"},
    # {"model": "anthropic/claude-haiku-4-5", "reasoning_effort": None},
    # {"model": "openai/gpt-5-mini", "reasoning_effort": None},
    # {"model": "openai/gpt-5-nano", "reasoning_effort": None},
]

PROMPT_1 = "How's your day going?"
PROMPT_2 = "What's the weather like today?"

# =============================================================================
# Env-var helpers
# =============================================================================

_ENV_VAR_MAP = {
    "anthropic/": "ANTHROPIC_API_KEY",
    "gemini/": "GEMINI_API_KEY",
    "openai/": "OPENAI_API_KEY",
}


def _env_var_for_model(model: str) -> str:
    for prefix, var in _ENV_VAR_MAP.items():
        if model.startswith(prefix):
            return var
    return "OPENAI_API_KEY"


def _has_api_key(model: str) -> bool:
    return bool(os.getenv(_env_var_for_model(model)))


# =============================================================================
# Benchmark core
# =============================================================================


@dataclass
class TurnResult:
    ttft_ms: float
    total_ms: float
    text: str


@dataclass
class ConversationResult:
    turn1: TurnResult
    turn2: TurnResult


@dataclass
class ModelStats:
    model: str
    reasoning_effort: Optional[str]
    ttft1s: list
    ttft2s: list
    errors: int


async def stream_turn(
    provider: LlmProvider,
    messages: list[Message],
    config: LlmConfig,
) -> TurnResult:
    """Stream a single turn via LlmProvider. Returns timing info."""
    t0 = time.perf_counter()
    ttft = None
    text_parts: list[str] = []

    async for chunk in provider.chat(messages, config=config):
        if chunk.text:
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            text_parts.append(chunk.text)

    total = (time.perf_counter() - t0) * 1000
    return TurnResult(
        ttft_ms=ttft or total,
        total_ms=total,
        text="".join(text_parts),
    )


async def measure_conversation(
    provider: LlmProvider,
    config_kwargs: dict,
) -> ConversationResult:
    """Run a 2-turn conversation through LlmProvider."""
    # Nonce in the system prompt ensures we never hit a provider-side cache.
    nonce = uuid.uuid4().hex[:12]
    config = _normalize_config(LlmConfig(**{**config_kwargs, "system_prompt": f"[{nonce}] {SYSTEM_PROMPT}"}))

    messages = [Message(role="user", content=PROMPT_1)]
    turn1 = await stream_turn(provider, messages, config)

    messages.append(Message(role="assistant", content=turn1.text))
    messages.append(Message(role="user", content=PROMPT_2))
    turn2 = await stream_turn(provider, messages, config)

    return ConversationResult(turn1=turn1, turn2=turn2)


def _print_stats(label: str, values: list[float]) -> None:
    avg = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0
    print(
        f"  {label:14s} — avg: {avg:7.0f} ms  stddev: {sd:7.0f} ms"
        f"  (min {min(values):.0f} / max {max(values):.0f})"
    )


async def bench_model(
    model: str,
    reasoning_effort: Optional[str],
    n: int,
    pause: float,
) -> ModelStats:
    """Run n conversations for a single model config and print stats."""
    effort_str = reasoning_effort if reasoning_effort is not None else "default"
    label = f"{model}  (reasoning={effort_str}, {n} conversations)"
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    # Only pass reasoning_effort when explicitly set; otherwise leave it as
    # _UNSET so LlmProvider applies its own per-model default.
    config_kwargs: dict = {}
    if reasoning_effort is not None:
        config_kwargs["reasoning_effort"] = reasoning_effort
    api_key = os.getenv(_env_var_for_model(model))
    provider = LlmProvider(model=model, api_key=api_key)

    ttft1s: list[float] = []
    total1s: list[float] = []
    ttft2s: list[float] = []
    total2s: list[float] = []
    errors = 0

    for i in range(n):
        try:
            result = await measure_conversation(provider, config_kwargs)
            t1, t2 = result.turn1, result.turn2
            ttft1s.append(t1.ttft_ms)
            total1s.append(t1.total_ms)
            ttft2s.append(t2.ttft_ms)
            total2s.append(t2.total_ms)
            print(
                f"  [{i + 1:2d}/{n}]"
                f"  Turn1: TTFT {t1.ttft_ms:6.0f} ms, Total {t1.total_ms:6.0f} ms"
                f"  |  Turn2: TTFT {t2.ttft_ms:6.0f} ms, Total {t2.total_ms:6.0f} ms"
            )
        except Exception as e:
            errors += 1
            print(f"  [{i + 1:2d}/{n}]  ERROR: {e}")

        if i < n - 1:
            await asyncio.sleep(pause)

    if ttft1s:
        print()
        _print_stats("Turn1 TTFT", ttft1s)
        _print_stats("Turn1 Total", total1s)
        _print_stats("Turn2 TTFT", ttft2s)
        _print_stats("Turn2 Total", total2s)
    if errors:
        print(f"  Errors: {errors}/{n}")

    await provider.aclose()
    return ModelStats(
        model=model,
        reasoning_effort=reasoning_effort,
        ttft1s=ttft1s,
        ttft2s=ttft2s,
        errors=errors,
    )


# =============================================================================
# Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM latency via LlmProvider.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs", type=int, default=20, help="Conversations per model (default: 20)")
    parser.add_argument("--model", type=str, default=None, help="Only test a specific model name")
    parser.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="Seconds to wait between conversations (default: 0.0)",
    )
    return parser.parse_args()


async def main(args):
    # Filter to models the user asked for / has keys for.
    entries = []
    seen_skipped: set[str] = set()
    for entry in MODELS:
        model = entry["model"]
        if args.model and model != args.model:
            continue
        if not _has_api_key(model):
            env_var = _env_var_for_model(model)
            if env_var not in seen_skipped:
                print(f"  ✗ {env_var} not set — skipping {model}")
                seen_skipped.add(env_var)
            continue
        entries.append(entry)

    if not entries:
        print("\n⚠ No matching models with API keys found. Set at least one of:")
        for var in dict.fromkeys(_ENV_VAR_MAP.values()):
            print(f"  export {var}=your-key-here")
        return 1

    # Dedupe model names for the header.
    unique_models = list(dict.fromkeys(e["model"] for e in entries))
    print(
        f"\nBenchmarking {len(entries)} configs across {len(unique_models)} models"
        f" × {args.runs} conversations each"
    )
    print(f"  Turn 1: {PROMPT_1!r}")
    print(f"  Turn 2: {PROMPT_2!r}")
    print(f"  Pause:  {args.pause}s")
    for m in unique_models:
        print(f"  ✓ {m}")

    all_stats: list[ModelStats] = []
    for entry in entries:
        stats = await bench_model(
            model=entry["model"],
            reasoning_effort=entry.get("reasoning_effort"),
            n=args.runs,
            pause=args.pause,
        )
        all_stats.append(stats)

    # Summary table
    print(f"\n{'=' * 90}")
    print("  SUMMARY")
    print(f"{'=' * 90}")
    print(f"  {'Model':<40s} {'Reasoning':>10s} {'Turn1 TTFT':>18s} {'Turn2 TTFT':>18s}")
    print(f"  {'':40s} {'Effort':>10s} {'avg±std (ms)':>18s} {'avg±std (ms)':>18s}")
    print(f"  {'-' * 40} {'-' * 10} {'-' * 18} {'-' * 18}")

    for s in all_stats:
        effort = s.reasoning_effort if s.reasoning_effort is not None else "default"
        if s.ttft1s:
            avg1 = statistics.mean(s.ttft1s)
            sd1 = statistics.stdev(s.ttft1s) if len(s.ttft1s) > 1 else 0
            avg2 = statistics.mean(s.ttft2s)
            sd2 = statistics.stdev(s.ttft2s) if len(s.ttft2s) > 1 else 0
            t1 = f"{avg1:.0f}±{sd1:.0f}"
            t2 = f"{avg2:.0f}±{sd2:.0f}"
        else:
            t1 = t2 = "no data"
        err = f" ({s.errors} err)" if s.errors else ""
        print(f"  {s.model:<40s} {effort:>10s} {t1:>18s} {t2:>18s}{err}")

    print(f"\n{'=' * 90}")
    print("Done.")
    return 0


if __name__ == "__main__":
    args = parse_args()

    # Suppress noisy output from litellm / pydantic / asyncio.
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
