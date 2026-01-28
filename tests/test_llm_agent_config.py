"""
Tests for LlmConfig.

uv run pytest tests/test_config.py -v
"""

from typing import Optional

from line.llm_agent.config import FALLBACK_INTRODUCTION, FALLBACK_SYSTEM_PROMPT, LlmConfig
from line.voice_agent_app import AgentConfig, CallRequest


def make_call_request(
    system_prompt: Optional[str] = None,
    introduction: Optional[str] = None,
) -> CallRequest:
    """Helper to create a CallRequest with given agent config."""
    return CallRequest(
        call_id="test-call",
        from_="user",
        to="agent",
        agent_call_id="test-agent-call",
        agent=AgentConfig(
            system_prompt=system_prompt,
            introduction=introduction,
        ),
    )


def test_from_call_request_uses_defaults_when_none():
    """Test that None values use defaults."""
    call_request = make_call_request(system_prompt=None, introduction=None)

    config = LlmConfig.from_call_request(call_request)

    assert config.system_prompt == FALLBACK_SYSTEM_PROMPT
    assert config.introduction == FALLBACK_INTRODUCTION


def test_from_call_request_uses_provided_values():
    """Test that provided values are used."""
    call_request = make_call_request(
        system_prompt="Custom prompt",
        introduction="Custom intro",
    )

    config = LlmConfig.from_call_request(call_request)

    assert config.system_prompt == "Custom prompt"
    assert config.introduction == "Custom intro"


def test_from_call_request_empty_string_skips_intro():
    """Test that empty string introduction is preserved (skips intro)."""
    call_request = make_call_request(
        system_prompt="Custom prompt",
        introduction="",  # Empty string = skip intro
    )

    config = LlmConfig.from_call_request(call_request)

    assert config.system_prompt == "Custom prompt"
    assert config.introduction == ""  # Empty string preserved


def test_from_call_request_empty_system_prompt_uses_default():
    """Test that empty string system_prompt falls back to SDK default."""
    call_request = make_call_request(
        system_prompt="",  # Empty string = fall back to default
        introduction="Custom intro",
    )

    config = LlmConfig.from_call_request(call_request)

    assert config.system_prompt == FALLBACK_SYSTEM_PROMPT  # Falls back to SDK default
    assert config.introduction == "Custom intro"


def test_from_call_request_with_extra_kwargs():
    """Test that extra kwargs are passed through."""
    call_request = make_call_request()

    config = LlmConfig.from_call_request(
        call_request,
        temperature=0.7,
        max_tokens=300,
    )

    assert config.temperature == 0.7
    assert config.max_tokens == 300
    # Defaults still applied
    assert config.system_prompt == FALLBACK_SYSTEM_PROMPT
    assert config.introduction == FALLBACK_INTRODUCTION


def test_from_call_request_mixed_none_and_provided():
    """Test mixing None and provided values."""
    call_request = make_call_request(
        system_prompt="Custom prompt",
        introduction=None,  # Should use default
    )

    config = LlmConfig.from_call_request(call_request)

    assert config.system_prompt == "Custom prompt"
    assert config.introduction == FALLBACK_INTRODUCTION


# =============================================================================
# Tests: User-provided defaults
# =============================================================================


def test_user_default_used_when_call_request_is_none():
    """Test that user-provided defaults are used when CallRequest has None."""
    call_request = make_call_request(system_prompt=None, introduction=None)

    config = LlmConfig.from_call_request(
        call_request,
        fallback_system_prompt="My app's default prompt",
        fallback_introduction="My app's default intro",
    )

    assert config.system_prompt == "My app's default prompt"
    assert config.introduction == "My app's default intro"


def test_call_request_overrides_user_default():
    """Test that CallRequest values take priority over user defaults."""
    call_request = make_call_request(
        system_prompt="From API",
        introduction="From API intro",
    )

    config = LlmConfig.from_call_request(
        call_request,
        fallback_system_prompt="My app's default prompt",
        fallback_introduction="My app's default intro",
    )

    # CallRequest values should win
    assert config.system_prompt == "From API"
    assert config.introduction == "From API intro"


def test_empty_string_from_call_request_overrides_user_default():
    """Test that empty string from CallRequest overrides user default (skips intro)."""
    call_request = make_call_request(
        system_prompt=None,
        introduction="",  # Explicitly skip intro
    )

    config = LlmConfig.from_call_request(
        call_request,
        fallback_system_prompt="My default prompt",
        fallback_introduction="My default intro",
    )

    assert config.system_prompt == "My default prompt"  # Uses user default
    assert config.introduction == ""  # Empty string preserved (skips intro)


def test_empty_system_prompt_from_call_request_uses_user_default():
    """Test that empty string system_prompt from CallRequest falls back to user default."""
    call_request = make_call_request(
        system_prompt="",  # Empty string = fall back
        introduction="Custom intro",
    )

    config = LlmConfig.from_call_request(
        call_request,
        fallback_system_prompt="My default prompt",
    )

    assert config.system_prompt == "My default prompt"  # Falls back to user default
    assert config.introduction == "Custom intro"


def test_empty_system_prompt_everywhere_uses_sdk_default():
    """Test that empty strings everywhere for system_prompt fall back to SDK default."""
    call_request = make_call_request(
        system_prompt="",  # Empty string
        introduction="Custom intro",
    )

    config = LlmConfig.from_call_request(
        call_request,
        fallback_system_prompt="",  # Empty string default
    )

    assert config.system_prompt == FALLBACK_SYSTEM_PROMPT  # Falls back to SDK default
    assert config.introduction == "Custom intro"


def test_partial_user_defaults():
    """Test providing only some user defaults."""
    call_request = make_call_request(system_prompt=None, introduction=None)

    # Only provide fallback_system_prompt
    config = LlmConfig.from_call_request(
        call_request,
        fallback_system_prompt="My custom prompt",
        # No fallback_introduction provided
    )

    assert config.system_prompt == "My custom prompt"
    assert config.introduction == FALLBACK_INTRODUCTION  # Falls back to SDK default


def test_priority_call_request_then_user_then_sdk():
    """Test full priority chain: CallRequest > user default > SDK default."""
    # Case 1: CallRequest set -> use CallRequest
    call_request = make_call_request(system_prompt="From CallRequest", introduction=None)
    config = LlmConfig.from_call_request(
        call_request,
        fallback_system_prompt="User default",
    )
    assert config.system_prompt == "From CallRequest"

    # Case 2: CallRequest None, user default set -> use user default
    call_request = make_call_request(system_prompt=None, introduction=None)
    config = LlmConfig.from_call_request(
        call_request,
        fallback_system_prompt="User default",
    )
    assert config.system_prompt == "User default"

    # Case 3: CallRequest None, user default None -> use SDK default
    call_request = make_call_request(system_prompt=None, introduction=None)
    config = LlmConfig.from_call_request(call_request)
    assert config.system_prompt == FALLBACK_SYSTEM_PROMPT


def test_user_defaults_with_extra_kwargs():
    """Test that user defaults and extra kwargs work together."""
    call_request = make_call_request(system_prompt=None, introduction=None)

    config = LlmConfig.from_call_request(
        call_request,
        fallback_system_prompt="Sales assistant prompt",
        fallback_introduction="Welcome to our store!",
        temperature=0.8,
        max_tokens=500,
    )

    assert config.system_prompt == "Sales assistant prompt"
    assert config.introduction == "Welcome to our store!"
    assert config.temperature == 0.8
    assert config.max_tokens == 500
