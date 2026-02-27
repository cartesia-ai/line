#!/usr/bin/env python3
"""
Test script to explore Anthropic API message configurations and error conditions.

Usage:
    ANTHROPIC_API_KEY=your-key uv run python line/llm_agent/scripts/test_anthropic_messages.py

This script tests various message configurations to understand API behavior:
1. Only system message (no user/assistant messages)
2. Empty messages list
3. System + empty user message
4. Messages ending in assistant message
5. Messages ending in system message
"""

import asyncio
import os

import anthropic


async def test_message_config(
    client: anthropic.AsyncAnthropic,
    name: str,
    messages: list,
    system: str | None = None,
):
    """Test a specific message configuration and report the result."""
    print(f"\n{'=' * 60}")
    print(f"Test: {name}")
    print(f"{'=' * 60}")
    print(f"System: {system!r}")
    print(f"Messages: {messages}")

    try:
        kwargs = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        response = await client.messages.create(**kwargs)
        print(f"✓ SUCCESS")
        print(f"Response: {response.content[0].text[:100]}...")
    except anthropic.BadRequestError as e:
        print(f"✗ BadRequestError: {e.message}")
    except anthropic.APIError as e:
        print(f"✗ APIError: {e}")
    except Exception as e:
        print(f"✗ Error ({type(e).__name__}): {e}")


async def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Usage: ANTHROPIC_API_KEY=your-key uv run python line/llm_agent/scripts/test_anthropic_messages.py")
        return 1

    client = anthropic.AsyncAnthropic(api_key=api_key)

    print("Anthropic API Message Configuration Tests")
    print("Model: claude-opus-4-6")

    # Test 1: Only system message (no user/assistant messages)
    await test_message_config(
        client,
        "1. Only system message (empty messages list)",
        messages=[],
        system="You are a helpful assistant.",
    )

    # Test 2: Empty messages list (no system either)
    await test_message_config(
        client,
        "2. Empty messages list (no system)",
        messages=[],
        system=None,
    )

    # Test 3: System + empty user message content
    await test_message_config(
        client,
        "3. System + empty user message content",
        messages=[{"role": "user", "content": ""}],
        system="You are a helpful assistant.",
    )

    # Test 4: System + whitespace-only user message
    await test_message_config(
        client,
        "4. System + whitespace-only user message",
        messages=[{"role": "user", "content": "   \n\t  "}],
        system="You are a helpful assistant.",
    )

    # Test 5: Messages ending in assistant message
    await test_message_config(
        client,
        "5. Messages ending in assistant message",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        system="You are a helpful assistant.",
    )

    # Test 6: Messages ending in system message (inline system)
    # Note: Anthropic uses top-level system param, but let's test inline
    await test_message_config(
        client,
        "6. Attempting system role in messages array",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Remember to be brief."},
        ],
        system=None,
    )

    # Test 7: Valid conversation for comparison
    await test_message_config(
        client,
        "7. Valid: user message only",
        messages=[{"role": "user", "content": "Say hi"}],
        system="You are a helpful assistant.",
    )

    # Test 8: Valid multi-turn conversation
    await test_message_config(
        client,
        "8. Valid: multi-turn conversation ending in user",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "How are you?"},
        ],
        system="You are a helpful assistant.",
    )

    # Test 9: Consecutive user messages
    await test_message_config(
        client,
        "9. Consecutive user messages",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Are you there?"},
        ],
        system="You are a helpful assistant.",
    )

    # Test 10: Consecutive assistant messages
    await test_message_config(
        client,
        "10. Consecutive assistant messages",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "assistant", "content": "How can I help?"},
        ],
        system="You are a helpful assistant.",
    )

    # Test 11: Starting with assistant message
    await test_message_config(
        client,
        "11. Starting with assistant message",
        messages=[
            {"role": "assistant", "content": "Hello! How can I help?"},
        ],
        system="You are a helpful assistant.",
    )

    print(f"\n{'=' * 60}")
    print("All tests completed!")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
