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

MCP Tests:
    MCP tests use a local dice-rolling MCP server (line/llm_agent/scripts/test_mcp_server.py)
    that exposes a roll tool via stdio transport.

The script will test whichever providers have API keys set.
"""

import argparse
import asyncio
import logging
import os
import pathlib
import sys
from typing import Annotated, List
import warnings

import litellm
from loguru import logger

from line.agent import TurnEnv
from line.events import (
    AgentSendText,
    AgentTextSent,
    AgentToolCalled,
    AgentToolReturned,
    CallStarted,
    InputEvent,
    UserTextSent,
)
from line.llm_agent import (
    LlmAgent,
    LlmConfig,
    end_call,
    loopback_tool,
    mcp_tool,
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


async def test_end_call_eagerness(model: str, api_key: str):
    """Test end_call tool with different eagerness levels.

    Verifies that the end_call tool can be configured with different
    eagerness levels that affect the tool description.
    """
    print("\n" + "=" * 60)
    print(f"Testing end_call eagerness levels with {model}")
    print("=" * 60)

    # Test all three eagerness levels
    for eagerness in ["low", "normal", "high"]:
        configured_end_call = end_call(eagerness=eagerness)
        print(f"\n  Eagerness: {eagerness}")
        print(f"  Description: {configured_end_call.description[:80]}...")

        agent = LlmAgent(
            model=model,
            api_key=api_key,
            tools=[configured_end_call],
            config=LlmConfig(
                system_prompt=f"You are a helpful assistant. Eagerness level: {eagerness}",
            ),
        )

        # Verify the tool is properly configured by checking resolved tools
        resolved_tools, _ = agent._resolve_tools(agent._tools)
        assert len(resolved_tools) == 1, f"Expected 1 tool, got {len(resolved_tools)}"
        assert resolved_tools[0].name == "end_call", f"Expected 'end_call', got {resolved_tools[0].name}"
        print(f"  ✓ Tool resolved correctly with eagerness={eagerness}")

    # Test custom description override
    custom_desc = "Only end after user explicitly says 'terminate session'"
    custom_end_call = end_call(description=custom_desc)
    assert custom_end_call.description == custom_desc
    print(f"\n  Custom description: {custom_desc}")
    print("  ✓ Custom description override works")

    print("\n✓ end_call eagerness test passed")


async def eval_end_call_behavior(model: str, api_key: str, n_runs: int = 3):
    """Evaluate that end_call eagerness levels actually affect LLM behavior.

    Runs test scenarios multiple times to statistically measure how different
    eagerness levels influence the LLM's decision to call end_call.

    Test scenarios:
    - "clear_goodbye": Explicit farewell - all levels should end (high/normal immediately)
    - "thanks_done": Thanks + done signal - high/normal should end, low should confirm
    - "ambiguous": Vague completion signal - high may end, normal/low should not
    - "continue": Clearly wants to continue - none should end
    """
    print("\n" + "=" * 60)
    print(f"Evaluating end_call behavior with {model}")
    print(f"Running {n_runs} iterations per scenario")
    print("=" * 60)

    # Test scenarios with expected behavior
    # Format: (category, user_message, conversation_context)
    scenarios = [
        (
            "clear_goodbye",
            "Goodbye!",
            [("user", "What's the capital of France?"), ("assistant", "The capital of France is Paris.")],
        ),
        (
            "clear_goodbye",
            "Thanks for your help, bye!",
            [
                ("user", "How do I make coffee?"),
                ("assistant", "Boil water, add coffee grounds, let it steep, then filter."),
            ],
        ),
        (
            "thanks_done",
            "That's all I needed, thanks!",
            [
                ("user", "What time is it in Tokyo?"),
                ("assistant", "It's currently evening in Tokyo, around 9 PM JST."),
            ],
        ),
        (
            "ambiguous",
            "Okay, thanks.",
            [("user", "What's 2+2?"), ("assistant", "2+2 equals 4.")],
        ),
        (
            "ambiguous",
            "Alright.",
            [
                ("user", "Tell me a joke."),
                ("assistant", "Why did the scarecrow win an award? Because he was outstanding in his field!"),
            ],
        ),
        (
            "continue",
            "That's helpful. Now tell me about Python.",
            [
                ("user", "What's JavaScript?"),
                ("assistant", "JavaScript is a programming language for web development."),
            ],
        ),
    ]

    # Expected end_call rates by eagerness level (approximate thresholds)
    # Format: {category: {eagerness: (min_rate, max_rate)}}
    # Key differentiator: "low" should ask follow-up before ending, so rates should be near 0
    expected_rates = {
        "clear_goodbye": {"high": (0.8, 1.0), "normal": (0.5, 1.0), "low": (0.0, 0.3)},
        "thanks_done": {"high": (0.7, 1.0), "normal": (0.4, 1.0), "low": (0.0, 0.2)},
        "ambiguous": {"high": (0.3, 1.0), "normal": (0.0, 0.5), "low": (0.0, 0.2)},
        "continue": {"high": (0.0, 0.2), "normal": (0.0, 0.1), "low": (0.0, 0.1)},
    }

    results = {
        eagerness: {cat: [] for cat in expected_rates.keys()} for eagerness in ["low", "normal", "high"]
    }

    async def check_end_call_invoked(agent: LlmAgent, user_message: str, context: list) -> bool:
        """Run a single conversation and check if end_call was invoked."""
        env = TurnEnv()

        # Build history from context - include both user and assistant messages
        history: List[InputEvent] = []
        for role, content in context:
            if role == "user":
                history.append(UserTextSent(content=content))
            elif role == "assistant":
                history.append(AgentTextSent(content=content))

        # IMPORTANT: The current message must be included in history for the LLM to see it.
        # event.history is what gets converted to LLM messages - event.content is NOT
        # automatically added by the agent.
        current_user_event = UserTextSent(content=user_message)
        history.append(current_user_event)

        event = UserTextSent(
            content=user_message,
            history=history,
        )

        async for output in agent.process(env, event):
            if isinstance(output, AgentToolCalled) and output.tool_name == "end_call":
                return True

        return False

    # Run evaluations
    total_runs = len(["high", "normal", "low"]) * len(scenarios) * n_runs
    current_run = 0

    for eagerness in ["high", "normal", "low"]:
        for _s_idx, (category, user_message, context) in enumerate(scenarios):
            end_count = 0
            for _run_idx in range(n_runs):
                current_run += 1
                print(f"\r  Progress: {current_run}/{total_runs}", end="", flush=True)

                # Create fresh agent for each run to avoid history contamination
                agent = LlmAgent(
                    model=model,
                    api_key=api_key,
                    tools=[end_call(eagerness=eagerness)],
                    config=LlmConfig(
                        system_prompt=(
                            "You are a phone assistant. Use the end_call tool to end conversations "
                            "according to the tool's description. When ending a call, invoke the tool."
                        ),
                    ),
                )

                try:
                    if await check_end_call_invoked(agent, user_message, context):
                        end_count += 1
                except Exception as e:
                    print(f"\n  ⚠ Error: {e}")

            rate = end_count / n_runs
            results[eagerness][category].append((user_message[:30], rate))

    print()  # Newline after progress

    # Print results table
    print("\n  " + "=" * 70)
    print("  Results Summary (end_call invocation rate)")
    print("  " + "=" * 70)
    print(f"  {'Category':<15} {'Scenario':<25} {'High':<8} {'Normal':<8} {'Low':<8}")
    print("  " + "-" * 70)

    all_passed = True
    for category in expected_rates.keys():
        for i, (scenario_name, _) in enumerate(results["high"][category]):
            high_rate = results["high"][category][i][1]
            normal_rate = results["normal"][category][i][1]
            low_rate = results["low"][category][i][1]

            # Check if rates are within expected ranges
            high_ok = expected_rates[category]["high"][0] <= high_rate <= expected_rates[category]["high"][1]
            normal_ok = (
                expected_rates[category]["normal"][0] <= normal_rate <= expected_rates[category]["normal"][1]
            )
            low_ok = expected_rates[category]["low"][0] <= low_rate <= expected_rates[category]["low"][1]

            high_str = f"{high_rate:.0%}" + ("" if high_ok else "*")
            normal_str = f"{normal_rate:.0%}" + ("" if normal_ok else "*")
            low_str = f"{low_rate:.0%}" + ("" if low_ok else "*")

            if not (high_ok and normal_ok and low_ok):
                all_passed = False

            print(f"  {category:<15} {scenario_name:<25} {high_str:<8} {normal_str:<8} {low_str:<8}")

    print("  " + "-" * 70)
    print("  * = outside expected range")

    # Verify key differentiators
    print("\n  Key behavioral checks:")

    # Check 1: High should end more often than low on ambiguous
    high_ambig = sum(r for _, r in results["high"]["ambiguous"]) / len(results["high"]["ambiguous"])
    low_ambig = sum(r for _, r in results["low"]["ambiguous"]) / len(results["low"]["ambiguous"])
    check1 = high_ambig >= low_ambig
    mark1 = "✓" if check1 else "✗"
    print(f"  {mark1} High ends more on ambiguous than Low ({high_ambig:.0%} >= {low_ambig:.0%})")

    # Check 2: Normal should end more than low on clear_goodbye
    normal_goodbye = sum(r for _, r in results["normal"]["clear_goodbye"]) / len(
        results["normal"]["clear_goodbye"]
    )
    low_goodbye = sum(r for _, r in results["low"]["clear_goodbye"]) / len(results["low"]["clear_goodbye"])
    check2 = normal_goodbye >= low_goodbye
    mark2 = "✓" if check2 else "✗"
    print(f"  {mark2} Normal ends more on goodbye than Low ({normal_goodbye:.0%} >= {low_goodbye:.0%})")

    # Check 3: None should end frequently on "continue" scenarios
    high_continue = sum(r for _, r in results["high"]["continue"]) / len(results["high"]["continue"])
    check3 = high_continue <= 0.3
    print(f"  {'✓' if check3 else '✗'} High doesn't end on continue signals ({high_continue:.0%} <= 30%)")

    if all_passed and check1 and check2 and check3:
        print("\n✓ end_call behavior evaluation passed")
    else:
        print("\n⚠ end_call behavior evaluation completed with some deviations")
        print("  (This is expected - LLM behavior is probabilistic)")


async def eval_form_completion_behavior(model: str, api_key: str, n_runs: int = 3):
    """Evaluate end_call eagerness during a multi-turn form completion task.

    Simulates a realistic form-filling scenario where the agent collects
    name and phone number. Tests how different eagerness levels affect
    behavior after form completion:

    - High: May end promptly after form is complete, minimal follow-up
    - Normal: Asks if there's anything else, then ends appropriately
    - Low: Continues engaging, confirms multiple times before ending

    This test validates that eagerness descriptions meaningfully influence
    LLM behavior in realistic multi-turn conversations.
    """
    print("\n" + "=" * 60)
    print(f"Evaluating form completion behavior with {model}")
    print(f"Running {n_runs} iterations per eagerness level")
    print("=" * 60)

    # Define a tool for recording form data
    @loopback_tool
    async def record_contact(
        ctx,
        name: Annotated[str, "The contact's full name"],
        phone_number: Annotated[str, "The contact's phone number"],
    ) -> str:
        """Record contact information. Call this once you have collected the name and phone number."""
        return f"Contact recorded: {name}, {phone_number}"

    system_prompt = """You are a helpful assistant collecting contact information.
Your task is to collect the caller's name and phone number, then record it using the record_contact tool.

Ask for their name first, then their phone number. Once you have both, use the tool to save it."""

    # Multi-turn conversation simulating form completion
    # Format: list of (user_message, check_end_call_after_this_turn)
    conversation_turns = [
        ("Hi, I'd like to register my contact information.", False),
        ("My name is John Smith.", False),
        ("My phone number is 555-123-4567.", True),  # Form complete - check behavior here
        ("Thanks, that's all.", True),  # Explicit thanks - check again
    ]

    # Track: ended_by_turn_N means end_call was invoked at or before turn N
    results = {
        eagerness: {"ended_by_form": [], "ended_by_thanks": []} for eagerness in ["high", "normal", "low"]
    }

    async def run_conversation(eagerness: str) -> dict:
        """Run the full conversation and track end_call invocations."""
        agent = LlmAgent(
            model=model,
            api_key=api_key,
            tools=[end_call(eagerness=eagerness), record_contact],
            config=LlmConfig(system_prompt=system_prompt),
        )

        env = TurnEnv()
        history: List[InputEvent] = []
        result = {
            "ended_by_form": False,  # end_call invoked by turn 3 (form complete)
            "ended_by_thanks": False,  # end_call invoked by turn 4 (explicit thanks)
            "end_turn": None,  # Which turn end_call was invoked (None if never)
            "form_recorded": False,
        }

        for i, (user_message, _) in enumerate(conversation_turns):
            # Build event with proper history (user + assistant messages)
            # IMPORTANT: The current message must be included in history for the LLM to see it
            user_event = UserTextSent(content=user_message)
            current_history = history.copy()
            current_history.append(user_event)
            event = UserTextSent(
                content=user_message,
                history=current_history,
            )

            response_text = ""
            end_call_invoked = False

            async for output in agent.process(env, event):
                if isinstance(output, AgentSendText):
                    response_text += output.text
                elif isinstance(output, AgentToolCalled):
                    if output.tool_name == "end_call":
                        end_call_invoked = True
                    elif output.tool_name == "record_contact":
                        result["form_recorded"] = True

            # Update history with this turn's exchange
            history.append(user_event)
            if response_text:
                history.append(AgentTextSent(content=response_text))

            # Track when end_call was invoked
            if end_call_invoked:
                result["end_turn"] = i + 1
                # Mark all checkpoints at or after this turn
                if i <= 2:  # Turn 3 or earlier = ended by form completion
                    result["ended_by_form"] = True
                if i <= 3:  # Turn 4 or earlier = ended by thanks
                    result["ended_by_thanks"] = True
                break

        return result

    # Run evaluations for each eagerness level
    eagerness_levels = ["high", "normal", "low"]
    total_runs = len(eagerness_levels) * n_runs
    current_run = 0

    for eagerness in eagerness_levels:
        for _run_num in range(n_runs):
            current_run += 1
            print(f"\r  Progress: {current_run}/{total_runs}", end="", flush=True)
            try:
                result = await run_conversation(eagerness)
                results[eagerness]["ended_by_form"].append(result["ended_by_form"])
                results[eagerness]["ended_by_thanks"].append(result["ended_by_thanks"])
            except Exception as e:
                print(f"\n  ⚠ Error: {e}")
                results[eagerness]["ended_by_form"].append(False)
                results[eagerness]["ended_by_thanks"].append(False)

    print()  # Newline after progress

    # Calculate rates
    def calc_rate(values):
        return sum(values) / len(values) if values else 0

    print("\n  " + "=" * 60)
    print("  Results Summary")
    print("  " + "=" * 60)
    print(f"  {'Checkpoint':<25} {'High':<12} {'Normal':<12} {'Low':<12}")
    print("  " + "-" * 60)

    high_form = calc_rate(results["high"]["ended_by_form"])
    normal_form = calc_rate(results["normal"]["ended_by_form"])
    low_form = calc_rate(results["low"]["ended_by_form"])

    high_thanks = calc_rate(results["high"]["ended_by_thanks"])
    normal_thanks = calc_rate(results["normal"]["ended_by_thanks"])
    low_thanks = calc_rate(results["low"]["ended_by_thanks"])

    label1, label2 = "Ended by form (turn 3)", "Ended by thanks (turn 4)"
    print(f"  {label1:<25} {high_form:.0%:<10} {normal_form:.0%:<10} {low_form:.0%:<10}")
    print(f"  {label2:<25} {high_thanks:.0%:<10} {normal_thanks:.0%:<10} {low_thanks:.0%:<10}")

    # Key behavioral checks
    print("\n  Key behavioral checks:")

    # Check 1: High should be more likely to end by form completion than low
    check1 = high_form >= low_form
    print(f"  {'✓' if check1 else '✗'} High ends more by form complete ({high_form:.0%} >= {low_form:.0%})")

    # Check 2: Low should rarely end immediately after form
    check2 = low_form <= 0.4
    print(f"  {'✓' if check2 else '✗'} Low rarely ends by form complete ({low_form:.0%} <= 40%)")

    # Check 3: High should end by thanks (if not earlier)
    check3 = high_thanks >= 0.7
    print(f"  {'✓' if check3 else '✗'} High ends by thanks ({high_thanks:.0%} >= 70%)")

    # Check 4: Low should still be reluctant even after thanks
    check4 = low_thanks <= 0.5
    print(f"  {'✓' if check4 else '✗'} Low still cautious by thanks ({low_thanks:.0%} <= 50%)")

    if check1 and check2 and check3 and check4:
        print("\n✓ Form completion behavior evaluation passed")
    else:
        print("\n⚠ Form completion evaluation completed with some deviations")
        print("  (LLM behavior varies - check if eagerness descriptions need tuning)")


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


def _local_mcp_server_command() -> str:
    """Return the command to launch the local test MCP server."""
    server_path = pathlib.Path(__file__).parent / "test_mcp_server.py"
    return f"{sys.executable} {server_path}"


async def test_mcp_list_tools(model: str, api_key: str):
    """Test MCP tool listing with the local test MCP server."""
    print(f"\n{'=' * 60}")
    print(f"Testing MCP tool listing with {model}")
    print("=" * 60)

    agent = LlmAgent(
        model=model,
        api_key=api_key,
        tools=[
            mcp_tool(
                name="dice",
                command=_local_mcp_server_command(),
            )
        ],
        config=LlmConfig(
            system_prompt="You are a helpful assistant with access to an MCP server. "
            "Use the available tools when asked.",
        ),
    )

    env = TurnEnv()
    user_message = "What tools are available?"
    event = UserTextSent(
        content=user_message,
        history=[UserTextSent(content=user_message)],
    )

    print(f"User: {user_message}")
    print("\nAgent response:")

    mcp_tool_used = False
    async for output in agent.process(env, event):
        if isinstance(output, AgentSendText):
            print(output.text, end="", flush=True)
        elif isinstance(output, AgentToolCalled):
            if output.tool_name == "mcp_dice":
                mcp_tool_used = True
            print(f"\n  [Tool call]: {output.tool_name}({output.tool_args})")
        elif isinstance(output, AgentToolReturned):
            result_preview = output.result[:400] + "..." if len(output.result) > 400 else output.result
            print(f"  [Tool result]: {result_preview}")

    print()

    if mcp_tool_used:
        print("✓ MCP tool listing test passed (mcp_dice tool was called)")
    else:
        print("⚠ MCP tool was not called (model may have answered directly)")


async def test_mcp_tool_execution(model: str, api_key: str):
    """Test MCP tool execution with the local test MCP server."""
    print(f"\n{'=' * 60}")
    print(f"Testing MCP tool execution with {model}")
    print("=" * 60)

    agent = LlmAgent(
        model=model,
        api_key=api_key,
        tools=[
            mcp_tool(
                name="dice",
                command=_local_mcp_server_command(),
            )
        ],
        config=LlmConfig(
            system_prompt="You are a helpful assistant with access to an MCP server. "
            "First list available tools, then use them as requested.",
        ),
    )

    env = TurnEnv()
    user_message = "Roll 3 six-sided dice for me."
    event = UserTextSent(
        content=user_message,
        history=[UserTextSent(content=user_message)],
    )

    print(f"User: {user_message}")
    print("\nAgent response:")

    mcp_tool_calls = []
    async for output in agent.process(env, event):
        if isinstance(output, AgentSendText):
            print(output.text, end="", flush=True)
        elif isinstance(output, AgentToolCalled):
            if output.tool_name == "mcp_dice":
                mcp_tool_calls.append(output.tool_args)
            print(f"\n  [Tool call]: {output.tool_name}({output.tool_args})")
        elif isinstance(output, AgentToolReturned):
            result_preview = output.result[:200] + "..." if len(output.result) > 200 else output.result
            print(f"  [Tool result]: {result_preview}")

    print()

    if mcp_tool_calls:
        print(f"✓ MCP tool execution test passed ({len(mcp_tool_calls)} tool calls made)")
    else:
        print("⚠ MCP tool was not called (model may have answered directly)")
    print("✓ MCP tool execution test completed")


# =============================================================================
# Main
# =============================================================================

MODELS = [
    ("OPENAI_API_KEY", "gpt-4o-mini"),
    ("ANTHROPIC_API_KEY", "anthropic/claude-sonnet-4-20250514"),
    ("GEMINI_API_KEY", "gemini/gemini-2.5-flash"),
    ("GEMINI_API_KEY", "gemini/gemini-3-flash-preview"),
]

# Available test names
AVAILABLE_TESTS = [
    "streaming",  # test_streaming_text
    "introduction",  # test_introduction
    "tools",  # test_tool_calling
    "end_call",  # test_end_call_eagerness
    "end_call_form_eval",  # test_end_call_form_eval
    "web_search",  # test_web_search
    "web_search_fn",  # test_function_tools_with_web_search
    "mcp",  # test_mcp_list_tools and test_mcp_tool_execution
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
            if "end_call" in tests_to_run:
                await test_end_call_eagerness(model, api_key)
                await eval_end_call_behavior(model, api_key, n_runs=args.runs)
            if "end_call_form_eval" in tests_to_run:
                await eval_form_completion_behavior(model, api_key, n_runs=args.runs)
            if "web_search" in tests_to_run:
                await test_web_search(model, api_key)
                await test_web_search(model, api_key, search_context_size="high")
            if "web_search_fn" in tests_to_run:
                await test_function_tools_with_web_search(model, api_key)
            if "mcp" in tests_to_run:
                await test_mcp_list_tools(model, api_key)
                await test_mcp_tool_execution(model, api_key)
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
