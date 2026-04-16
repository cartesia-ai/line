"""Live API integration tests for model configuration and parameter acceptance.

Usage:
    uv run python -m pytest tests/real_api_test_model_config.py -v

Coverage:
    - Live vendor probes via ``litellm.acompletion`` verifying that providers
      still accept the parameters we rely on (temperature, tools,
      max_completion_tokens).
    - Reasoning-model specific checks (temperature rejection, completion
      without temperature, Anthropic extended thinking).

Static registry tests (get_supported_openai_params, _get_model_config) live in
``test_llm_agent_provider.py`` — they need no API keys.

Requires both OPENAI_API_KEY and ANTHROPIC_API_KEY; the module raises at import
if either is missing so pytest errors during collection instead of skipping tests.
"""

import os

import litellm
from litellm.exceptions import BadRequestError, UnsupportedParamsError
import pytest

litellm.suppress_debug_info = True

# ---------------------------------------------------------------------------
# API keys — required at import so collection fails before any test runs
# ---------------------------------------------------------------------------

_missing_api_keys = [name for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if not os.getenv(name)]
if _missing_api_keys:
    raise RuntimeError(
        "tests/real_api_test_model_config.py requires OPENAI_API_KEY and ANTHROPIC_API_KEY "
        f"in the environment; missing: {', '.join(_missing_api_keys)}"
    )

# ---------------------------------------------------------------------------
# Model lists
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

# Standard-chat models (accept temperature). Tuples of (model, id).
_STANDARD_MODELS = [
    ("openai/gpt-4o", "gpt-4o"),
    ("gpt-5.2", "gpt-5.2"),
    ("gpt-5.4", "gpt-5.4"),
    ("anthropic/claude-haiku-4-5", "claude-haiku-4-5"),
    ("anthropic/claude-sonnet-4-20250514", "claude-sonnet-4"),
    ("anthropic/claude-opus-4-20250514", "claude-opus-4"),
]

# Reasoning HTTP models (reject arbitrary temperature).
_REASONING_HTTP_MODELS = [
    ("openai/o3-mini", "o3-mini"),
]

# Anthropic models that support extended thinking.
_ANTHROPIC_THINKING_MODELS = [
    ("anthropic/claude-haiku-4-5", "claude-haiku-4-5"),
    ("anthropic/claude-sonnet-4-20250514", "claude-sonnet-4"),
    ("anthropic/claude-opus-4-20250514", "claude-opus-4"),
]


def _models(pairs: list[tuple[str, str]]) -> list[str]:
    return [m for m, _ in pairs]


def _ids(pairs: list[tuple[str, str]]) -> list[str]:
    return [i for _, i in pairs]


# ---------------------------------------------------------------------------
# Live vendor tests — standard models (temperature, streaming, tools)
# ---------------------------------------------------------------------------


class TestLiveStandardCompletion:
    """Verify standard models accept a basic completion with temperature."""

    @pytest.mark.parametrize("model", _models(_STANDARD_MODELS), ids=_ids(_STANDARD_MODELS))
    async def test_completion(self, model: str):
        r = await litellm.acompletion(
            model=model,
            messages=_MIN_USER_MESSAGE,
            max_tokens=1024,
            temperature=0.5,
        )
        assert r.choices[0].message is not None


class TestLiveStandardToolAcceptance:
    """Verify standard models accept tool definitions without erroring.

    Uses ``tool_choice`` to force a tool call so the assertion is deterministic
    (``auto`` may legitimately respond with text instead).
    """

    @pytest.mark.parametrize("model", _models(_STANDARD_MODELS), ids=_ids(_STANDARD_MODELS))
    async def test_forced_tool_call(self, model: str):
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

    @pytest.mark.parametrize("model", _models(_REASONING_HTTP_MODELS), ids=_ids(_REASONING_HTTP_MODELS))
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

    @pytest.mark.parametrize("model", _models(_REASONING_HTTP_MODELS), ids=_ids(_REASONING_HTTP_MODELS))
    async def test_completion_without_temperature(self, model: str):
        """Minimal completion without temperature.

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
    """Verify Anthropic extended-thinking models work with thinking enabled."""

    @pytest.mark.parametrize(
        "model", _models(_ANTHROPIC_THINKING_MODELS), ids=_ids(_ANTHROPIC_THINKING_MODELS)
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
