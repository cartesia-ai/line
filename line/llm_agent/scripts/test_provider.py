#!/usr/bin/env python3
"""
Test script for LLMProvider integration with real API keys.

Usage:
    uv run python line/llm_agent/scripts/test_provider.py [OPTIONS]

Options:
    --tests TESTS       Comma-separated list of tests to run. Available tests:
                        streaming, introduction, tools, end_call, end_call_eval,
                        form_eval, web_search, all (default: all)
    --runs N            Number of iterations for eval tests (default: 3)
    --model MODEL       Only test specific model (e.g., "gpt-4o-mini")

Examples:
    # Run all tests
    uv run python line/llm_agent/scripts/test_provider.py

    # Run only end_call evaluation tests
    uv run python line/llm_agent/scripts/test_provider.py --tests end_call,end_call_eval,form_eval

    # Run form evaluation with more iterations
    uv run python line/llm_agent/scripts/test_provider.py --tests form_eval --runs 5

Environment variables:
    OPENAI_API_KEY      - For OpenAI models (gpt-4o, gpt-4o-mini)
    ANTHROPIC_API_KEY   - For Anthropic models (anthropic/claude-sonnet-4-20250514)
    GEMINI_API_KEY      - For Google models (gemini/gemini-2.5-flash-preview-09-2025)

The script will test whichever providers have API keys set.
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Annotated
import warnings

import litellm
from loguru import logger

from line.agent import TurnEnv
from line.events import (
    AgentSendText,
    AgentToolCalled,
    AgentToolReturned,
    CallStarted,
    UserTextSent,
)
from line.llm_agent import (
    LlmAgent,
    LlmConfig,
    end_call,
    loopback_tool,
    web_search,
)
from line.llm_agent.provider import LLMProvider, Message

# =============================================================================
# Test Tools
# =============================================================================


@loopback_tool
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


@loopback_tool
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


async def test_api_key(model: str, api_key: str) -> bool:
    """Test that API key is valid and has permissions."""
    print("\n" + "=" * 60)
    print(f"Testing API key for {model}")
    print("=" * 60)

    provider = LLMProvider(model=model, api_key=api_key)

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


async def test_streaming_text(model: str, api_key: str):
    """Test basic streaming text response."""
    print("\n" + "=" * 60)
    print(f"Testing streaming text with {model}")
    print("=" * 60)

    provider = LLMProvider(model=model, api_key=api_key)

    messages = [Message(role="user", content="Say 'Hello, World!' and nothing else.")]

    print("Response: ", end="", flush=True)
    async with provider.chat(messages) as stream:
        async for chunk in stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
    print("\n✓ Streaming text test passed")


async def test_tool_calling(model: str, api_key: str):
    """Test tool calling with LlmAgent."""
    print("\n" + "=" * 60)
    print(f"Testing tool calling with {model}")
    print("=" * 60)

    agent = LlmAgent(
        model=model,
        api_key=api_key,
        tools=[get_weather, calculate],
        config=LlmConfig(
            system_prompt="You are a helpful assistant. Use tools when needed.",
        ),
    )

    env = TurnEnv()
    user_message = "What's the weather in Tokyo? Also, what's 25 * 4?"
    event = UserTextSent(
        content=user_message,
        history=[UserTextSent(content=user_message)],
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


async def test_introduction(model: str, api_key: str):
    """Test introduction message on CallStarted."""
    print("\n" + "=" * 60)
    print(f"Testing introduction with {model}")
    print("=" * 60)

    agent = LlmAgent(
        model=model,
        api_key=api_key,
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


async def test_web_search(model: str, api_key: str, search_context_size: str = "medium"):
    """Test web search tool integration."""
    print("\n" + "=" * 60)
    if search_context_size != "medium":
        print(f"Testing web search with {model} (context_size={search_context_size}")
    else:
        print(f"Testing web search with {model}")
    print("=" * 60)

    agent = LlmAgent(
        model=model,
        api_key=api_key,
        tools=[web_search(search_context_size=search_context_size)],
        config=LlmConfig(
            system_prompt="You are a helpful assistant with web search capabilities."
            + " Use web search to find current information.",
        ),
    )

    env = TurnEnv()
    user_message = "What is the current weather in New York City? Search the web for this information."
    event = UserTextSent(
        content=user_message,
        history=[UserTextSent(content=user_message)],
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


async def test_function_tools_with_web_search(model: str, api_key: str):
    """Test combining function calling tools with web search.

    This reproduces the scenario from examples/basic_chat/main.py where
    both end_call (a function tool) and web_search are passed together.
    Some models (e.g. Gemini 3) don't support combining native web search
    with function calling tools in the same request.
    """
    print("\n" + "=" * 60)
    print(f"Testing function tools + web search with {model}")
    print("=" * 60)

    agent = LlmAgent(
        model=model,
        api_key=api_key,
        tools=[end_call, web_search],
        config=LlmConfig(
            system_prompt="You are a helpful assistant. Use web search when needed. "
            "Use end_call when the user says goodbye.",
        ),
    )

    env = TurnEnv()
    user_message = "Hi, how are you?"
    event = UserTextSent(
        content=user_message,
        history=[UserTextSent(content=user_message)],
    )

    print(f"User: {user_message}")
    print("\nAgent response:")

    async for output in agent.process(env, event):
        if isinstance(output, AgentSendText):
            print(output.text, end="", flush=True)
        elif isinstance(output, AgentToolCalled):
            print(f"\n  [Tool call]: {output.tool_name}({output.tool_args})")
        elif isinstance(output, AgentToolReturned):
            result_preview = output.result[:200] + "..." if len(output.result) > 200 else output.result
            print(f"  [Tool result]: {result_preview}")

    print()
    print("✓ Function tools + web search test passed")


# =============================================================================
# Main
# =============================================================================

MODELS = [
    ("OPENAI_API_KEY", "gpt-4o-mini"),
    ("ANTHROPIC_API_KEY", "anthropic/claude-haiku-4-5-20251001"),
    ("ANTHROPIC_API_KEY", "anthropic/claude-sonnet-4-20250514"),
    ("GEMINI_API_KEY", "gemini/gemini-2.5-flash"),
    ("GEMINI_API_KEY", "gemini/gemini-3-flash-preview"),
]

# Available test names
AVAILABLE_TESTS = [
    "streaming",  # test_streaming_text
    "introduction",  # test_introduction
    "tools",  # test_tool_calling
    "web_search",  # test_web_search
    "web_search_fn",  # test_function_tools_with_web_search
]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test LLMProvider integration with real API keys.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tests",
        type=str,
        default="all",
        help=f"Comma-separated tests. Available: {', '.join(AVAILABLE_TESTS)}, all",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of iterations for eval tests (default: 3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Only test specific model (e.g., 'gpt-4o-mini')",
    )
    return parser.parse_args()


async def main(args):
    print("LLM Provider Integration Test")
    print("=" * 60)

    # Parse which tests to run
    if args.tests == "all":
        tests_to_run = set(AVAILABLE_TESTS)
    else:
        tests_to_run = {t.strip() for t in args.tests.split(",")}
        invalid_tests = tests_to_run - set(AVAILABLE_TESTS)
        if invalid_tests:
            print(f"⚠ Unknown tests: {', '.join(invalid_tests)}")
            print(f"Available tests: {', '.join(AVAILABLE_TESTS)}")
            return 1

    print(f"Tests to run: {', '.join(sorted(tests_to_run))}")
    print(f"Eval iterations: {args.runs}")

    # Find available models
    available = []
    for env_var, model in MODELS:
        if args.model and model != args.model:
            continue
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
    print("\n" + "=" * 60)
    print("PHASE 1: Validating API Keys")
    print("=" * 60)

    valid_models = []
    for env_var, model in available:
        api_key = os.environ[env_var]
        if await test_api_key(model, api_key):
            valid_models.append((env_var, model))

    if not valid_models:
        print("\n⚠ No valid API keys. Check your keys and permissions.")
        return 1

    # Run full tests only for models with valid keys
    print("\n" + "=" * 60)
    print("PHASE 2: Running Full Tests")
    print("=" * 60)

    for env_var, model in valid_models:
        api_key = os.environ[env_var]
        try:
            if "streaming" in tests_to_run:
                await test_streaming_text(model, api_key)
            if "introduction" in tests_to_run:
                await test_introduction(model, api_key)
            if "tools" in tests_to_run:
                await test_tool_calling(model, api_key)
            if "web_search" in tests_to_run:
                await test_web_search(model, api_key)
                await test_web_search(model, api_key, search_context_size="high")
            if "web_search_fn" in tests_to_run:
                await test_function_tools_with_web_search(model, api_key)
        except Exception as e:
            print(f"\n✗ Error testing {model}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    # Parse arguments first (before suppressing output)
    args = parse_args()

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
        exit_code = asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nInterrupted")
        exit_code = 1
    finally:
        # Suppress SSL cleanup errors on exit
        sys.stderr = open(os.devnull, "w")

    sys.exit(exit_code)
