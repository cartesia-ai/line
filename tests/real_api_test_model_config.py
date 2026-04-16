"""Live API integration tests for model configuration and parameter acceptance.

Usage:
    uv run python -m pytest tests/real_api_test_model_config.py -v

Coverage:
    - Live vendor probes via ``litellm.acompletion`` verifying that providers
      still accept the parameters we rely on (temperature, streaming, tools,
      max_completion_tokens).
    - Reasoning-model specific checks (temperature rejection, completion
      without temperature, Anthropic extended thinking).

Static registry tests (get_supported_openai_params, _get_model_config) live in
``test_llm_agent_provider.py`` — they need no API keys.

Requires API keys in the environment (OPENAI_API_KEY, ANTHROPIC_API_KEY).
"""

import os

import litellm
from litellm.exceptions import BadRequestError, UnsupportedParamsError
import pytest

litellm.suppress_debug_info = True

# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------

_has_openai_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
_has_anthropic_key = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_MIN_USER_MESSAGE = [{"role": "user", "content": "Reply with one word: ok."}]
_MIN_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "identity",
            "description": "Return a fixed marker.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]

# All standard-chat models (HTTP, accept temperature).
_STANDARD_MODELS_OPENAI = ["openai/gpt-4o-mini"]
_STANDARD_MODELS_ANTHROPIC = [
    "anthropic/claude-haiku-4-5",
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-opus-4-20250514",
]
_STANDARD_MODELS_OPENAI_IDS = ["gpt-4o-mini"]
_STANDARD_MODELS_ANTHROPIC_IDS = ["claude-haiku-4-5", "claude-sonnet-4", "claude-opus-4"]

# Reasoning HTTP models (reject arbitrary temperature).
_REASONING_HTTP_MODELS = ["openai/o3-mini"]
_REASONING_HTTP_IDS = ["o3-mini"]


# ---------------------------------------------------------------------------
# Live vendor tests — standard models (temperature, streaming, tools)
# ---------------------------------------------------------------------------


class TestLiveStandardCompletion:
    """Verify standard models accept a basic completion with temperature."""

    @_has_openai_key
    @pytest.mark.parametrize("model", _STANDARD_MODELS_OPENAI, ids=_STANDARD_MODELS_OPENAI_IDS)
    async def test_openai(self, model: str):
        r = await litellm.acompletion(
            model=model,
            messages=_MIN_USER_MESSAGE,
            max_tokens=1024,
            temperature=0.5,
        )
        assert r.choices[0].message is not None

    @_has_anthropic_key
    @pytest.mark.parametrize("model", _STANDARD_MODELS_ANTHROPIC, ids=_STANDARD_MODELS_ANTHROPIC_IDS)
    async def test_anthropic(self, model: str):
        r = await litellm.acompletion(
            model=model,
            messages=_MIN_USER_MESSAGE,
            max_tokens=1024,
            temperature=0.5,
        )
        assert r.choices[0].message is not None


class TestLiveStandardStreaming:
    """Verify standard models produce streaming output."""

    @_has_openai_key
    @pytest.mark.parametrize("model", _STANDARD_MODELS_OPENAI, ids=_STANDARD_MODELS_OPENAI_IDS)
    async def test_openai(self, model: str):
        stream = await litellm.acompletion(
            model=model,
            messages=_MIN_USER_MESSAGE,
            max_tokens=1024,
            temperature=0.5,
            stream=True,
        )
        saw_text = False
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                saw_text = True
                break
        assert saw_text, f"{model}: streaming returned no assistant text"

    @_has_anthropic_key
    @pytest.mark.parametrize("model", _STANDARD_MODELS_ANTHROPIC, ids=_STANDARD_MODELS_ANTHROPIC_IDS)
    async def test_anthropic(self, model: str):
        stream = await litellm.acompletion(
            model=model,
            messages=_MIN_USER_MESSAGE,
            max_tokens=1024,
            temperature=0.5,
            stream=True,
        )
        saw_text = False
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                saw_text = True
                break
        assert saw_text, f"{model}: streaming returned no assistant text"


class TestLiveStandardToolAcceptance:
    """Verify standard models accept tool definitions without erroring.

    Uses ``tool_choice`` to force a tool call so the assertion is deterministic
    (``auto`` may legitimately respond with text instead).
    """

    @_has_openai_key
    @pytest.mark.parametrize("model", _STANDARD_MODELS_OPENAI, ids=_STANDARD_MODELS_OPENAI_IDS)
    async def test_openai(self, model: str):
        r = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "Call the identity tool."}],
            max_tokens=1024,
            temperature=0.3,
            tools=_MIN_TOOL,
            tool_choice={"type": "function", "function": {"name": "identity"}},
        )
        tc = getattr(r.choices[0].message, "tool_calls", None)
        assert tc, f"{model}: expected a forced tool call, got {r.choices[0].message!r}"

    @_has_anthropic_key
    @pytest.mark.parametrize("model", _STANDARD_MODELS_ANTHROPIC, ids=_STANDARD_MODELS_ANTHROPIC_IDS)
    async def test_anthropic(self, model: str):
        r = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "Call the identity tool."}],
            max_tokens=1024,
            temperature=0.3,
            tools=_MIN_TOOL,
            tool_choice={"type": "function", "function": {"name": "identity"}},
        )
        tc = getattr(r.choices[0].message, "tool_calls", None)
        assert tc, f"{model}: expected a forced tool call, got {r.choices[0].message!r}"


# ---------------------------------------------------------------------------
# Live vendor tests — reasoning HTTP models
# ---------------------------------------------------------------------------


class TestLiveReasoningHttp:
    """Reasoning HTTP models (e.g. o3-mini) that reject arbitrary temperature."""

    @_has_openai_key
    @pytest.mark.parametrize("model", _REASONING_HTTP_MODELS, ids=_REASONING_HTTP_IDS)
    async def test_rejects_arbitrary_temperature(self, model: str):
        """Non-default temperature should be rejected by litellm or the vendor."""
        try:
            await litellm.acompletion(
                model=model,
                messages=_MIN_USER_MESSAGE,
                max_tokens=1024,
                temperature=0.5,
            )
        except (UnsupportedParamsError, BadRequestError):
            return
        pytest.fail(
            f"{model}: expected temperature=0.5 to be rejected; "
            "if the API now accepts arbitrary temperature, update this test."
        )

    @_has_openai_key
    @pytest.mark.parametrize("model", _REASONING_HTTP_MODELS, ids=_REASONING_HTTP_IDS)
    async def test_completion_without_temperature(self, model: str):
        """Minimal completion without temperature — validates the model works.

        Uses max_completion_tokens (not max_tokens) because reasoning models
        need token budget for internal chain-of-thought.
        """
        r = await litellm.acompletion(
            model=model,
            messages=_MIN_USER_MESSAGE,
            max_completion_tokens=1024,
        )
        assert (r.choices[0].message.content or "").strip(), f"{model}: empty completion body"


# ---------------------------------------------------------------------------
# Live vendor tests — Anthropic extended thinking
# ---------------------------------------------------------------------------


class TestLiveAnthropicExtendedThinking:
    """Verify Anthropic extended-thinking models work with and without thinking."""

    @_has_anthropic_key
    @pytest.mark.parametrize(
        "model",
        ["anthropic/claude-sonnet-4-20250514"],
        ids=["claude-sonnet-4"],
    )
    async def test_completion_without_thinking(self, model: str):
        """Model should produce a normal completion when thinking is not requested."""
        r = await litellm.acompletion(
            model=model,
            messages=_MIN_USER_MESSAGE,
            max_tokens=1024,
            temperature=0.5,
        )
        assert (r.choices[0].message.content or "").strip(), f"{model}: empty completion body"

    @_has_anthropic_key
    @pytest.mark.parametrize(
        "model",
        ["anthropic/claude-sonnet-4-20250514"],
        ids=["claude-sonnet-4"],
    )
    async def test_completion_with_thinking(self, model: str):
        """Model should produce a completion when extended thinking is enabled."""
        r = await litellm.acompletion(
            model=model,
            messages=_MIN_USER_MESSAGE,
            max_tokens=2048,
            thinking={"type": "enabled", "budget_tokens": 1024},
        )
        assert (r.choices[0].message.content or "").strip(), f"{model}: empty completion body"
