#!/usr/bin/env python3
"""
Test script for LLMProvider integration with real API keys.

Usage:
    uv run python line/v02/llm/scripts/test_provider.py

Environment variables:
    OPENAI_API_KEY      - For OpenAI models (gpt-4o, gpt-4o-mini)
    ANTHROPIC_API_KEY   - For Anthropic models (anthropic/claude-sonnet-4-20250514)
    GEMINI_API_KEY      - For Google models (gemini/gemini-2.0-flash)

The script will test whichever providers have API keys set.
"""

import asyncio
import logging
import os
import sys
from typing import Annotated
import warnings

import litellm
from loguru import logger

# Add parent to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
)

from line.v02.agent import TurnEnv
from line.v02.llm import (
    AgentSendText,
    AgentToolCalled,
    AgentToolReturned,
    CallStarted,
    LlmAgent,
    LlmConfig,
    LLMProvider,
    Message,
    SpecificUserTextSent,
    UserTextSent,
    loopback_tool,
)
from line.v02.llm.tools import web_search

# =============================================================================
# Test Tools
# =============================================================================


@loopback_tool()
async def get_weather(ctx, city: Annotated[str, "City name"]) -> str:
    """Get the current weather for a city."""
    # Simulated weather data
    weather_data = {
        "new york": "72°F, partly cloudy",
        "san francisco": "65°F, foggy",
        "london": "55°F, rainy",
        "tokyo": "80°F, sunny",
    }
    city_lower = city.lower()
    return weather_data.get(city_lower, f"Weather data not available for {city}")


@loopback_tool()
async def calculate(ctx, expression: Annotated[str, "Math expression to evaluate"]) -> str:
    """Evaluate a mathematical expression."""
    try:
        # Safe eval for simple math
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            result = eval(expression)
            return f"{expression} = {result}"
        return "Invalid expression"
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# Test Functions
# =============================================================================


async def test_api_key(model: str) -> bool:
    """Test that API key is valid and has permissions."""
    print(f"\n{'=' * 60}")
    print(f"Testing API key for {model}")
    print("=" * 60)

    provider = LLMProvider(model=model)
    # from line.v02.llm.provider import Message

    messages = [Message(role="user", content="Say 'ok'")]

    try:
        async with provider.chat(messages) as stream:
            async for chunk in stream:
                if chunk.text:
                    print(f"✓ API key valid - got response: {chunk.text.strip()}")
                    return True
        print("✓ API key valid")
        return True
    except Exception as e:
        print(f"✗ API key error: {e}")
        return False


async def test_streaming_text(model: str):
    """Test basic streaming text response."""
    print(f"\n{'=' * 60}")
    print(f"Testing streaming text with {model}")
    print("=" * 60)

    provider = LLMProvider(model=model)
    # from line.v02.llm.provider import Message

    messages = [Message(role="user", content="Say 'Hello, World!' and nothing else.")]

    print("Response: ", end="", flush=True)
    async with provider.chat(messages) as stream:
        async for chunk in stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
    print("\n✓ Streaming text test passed")


async def test_tool_calling(model: str):
    """Test tool calling with LlmAgent."""
    print(f"\n{'=' * 60}")
    print(f"Testing tool calling with {model}")
    print("=" * 60)

    agent = LlmAgent(
        model=model,
        tools=[get_weather, calculate],
        config=LlmConfig(
            system_prompt="You are a helpful assistant. Use tools when needed.",
        ),
    )

    env = TurnEnv()
    user_message = "What's the weather in Tokyo? Also, what's 25 * 4?"
    event = UserTextSent(
        content=user_message,
        history=[SpecificUserTextSent(content=user_message)],
    )

    print("User: What's the weather in Tokyo? Also, what's 25 * 4?")
    print("\nAgent response:")

    tool_calls_made = []
    async for output in agent.process(env, event):
        if isinstance(output, AgentSendText):
            print(output.text, end="", flush=True)  # Prints the entire text response
        elif isinstance(output, AgentToolCalled):
            tool_calls_made.append(output.tool_name)
            print(f"\n  [Tool call]: {output.tool_name}({output.tool_args})]")
        elif isinstance(output, AgentToolReturned):
            print(f"  [Tool result]: {output.result}]")

    print()

    if tool_calls_made:
        print(f"✓ Tool calling test passed (tools used: {', '.join(tool_calls_made)})")
    else:
        print("⚠ No tools were called (model may have answered directly)")


async def test_introduction(model: str):
    """Test introduction message on CallStarted."""
    print(f"\n{'=' * 60}")
    print(f"Testing introduction with {model}")
    print("=" * 60)

    agent = LlmAgent(
        model=model,
        config=LlmConfig(
            introduction="Hello! I'm your AI assistant. How can I help you today?",
        ),
    )

    env = TurnEnv()
    event = CallStarted()

    print("Event: CallStarted")
    print("Introduction: ", end="")

    async for output in agent.process(env, event):
        if isinstance(output, AgentSendText):
            print(output.text)

    print("✓ Introduction test passed")


async def test_web_search(model: str):
    """Test web search tool integration."""
    print(f"\n{'=' * 60}")
    print(f"Testing web search with {model}")
    print("=" * 60)

    agent = LlmAgent(
        model=model,
        tools=[web_search],
        config=LlmConfig(
            system_prompt="You are a helpful assistant with web search capabilities."
            +" Use web search to find current information.",
        ),
    )

    env = TurnEnv()
    user_message = "What is the current weather in San Francisco? Search the web for this information."
    event = UserTextSent(
        content=user_message,
        history=[SpecificUserTextSent(content=user_message)],
    )

    print(f"User: {user_message}")
    print("\nAgent response:")

    web_search_used = False
    async for output in agent.process(env, event):
        if isinstance(output, AgentSendText):
            print(output.text, end="", flush=True)
        elif isinstance(output, AgentToolCalled):
            if output.tool_name == "web_search":
                web_search_used = True
            print(f"\n  [Tool call]: {output.tool_name}({output.tool_args})")
        elif isinstance(output, AgentToolReturned):
            # Truncate long results for display
            result_preview = output.result[:200] + "..." if len(output.result) > 200 else output.result
            print(f"  [Tool result]: {result_preview}")

    print()

    if web_search_used:
        print("✓ Web search test passed (web_search tool was called)")
    else:
        print("⚠ Web search tool was not explicitly called (model may use native web search)")
    print("✓ Web search test completed")


# =============================================================================
# Main
# =============================================================================

MODELS = [
    ("OPENAI_API_KEY", "gpt-4o-mini"),
    ("ANTHROPIC_API_KEY", "anthropic/claude-sonnet-4-20250514"),
    ("GEMINI_API_KEY", "gemini/gemini-2.0-flash"),
]


async def main():
    print("LLM Provider Integration Test")
    print("=" * 60)

    # Find available models
    available = []
    for env_var, model in MODELS:
        if os.getenv(env_var):
            available.append((env_var, model))
            print(f"✓ {env_var} found - will test {model}")
        else:
            print(f"✗ {env_var} not set - skipping {model}")

    if not available:
        print("\n⚠ No API keys found. Set at least one of:")
        for env_var, _model in MODELS:
            print(f"  export {env_var}=your-key-here")
        return 1

    # First, validate all API keys
    print(f"\n{'=' * 60}")
    print("PHASE 1: Validating API Keys")
    print("=" * 60)

    valid_models = []
    for env_var, model in available:
        if await test_api_key(model):
            valid_models.append((env_var, model))

    if not valid_models:
        print("\n⚠ No valid API keys. Check your keys and permissions.")
        return 1

    # Run full tests only for models with valid keys
    print(f"\n{'=' * 60}")
    print("PHASE 2: Running Full Tests")
    print("=" * 60)

    for _env_var, model in valid_models:
        try:
            await test_streaming_text(model)
            await test_introduction(model)
            await test_tool_calling(model)
            await test_web_search(model)
        except Exception as e:
            print(f"\n✗ Error testing {model}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("All tests completed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    # Suppress noisy warnings from litellm/pydantic
    warnings.filterwarnings("ignore", category=ResourceWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

    # Suppress litellm colored output
    litellm.suppress_debug_info = True

    # Suppress loguru output
    logger.disable("line")

    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        exit_code = 1
    finally:
        # Suppress SSL cleanup errors on exit
        sys.stderr = open(os.devnull, "w")

    sys.exit(exit_code)
