"""Integration tests for _get_model_config against real litellm model metadata.

Usage:
    uv run python -m pytest tests/real_api_test_model_config.py -v

Coverage:
    - _ModelConfig (backend, supports_reasoning_effort, default_reasoning_effort)
      for OpenAI and Anthropic models across HTTP and WebSocket backends.
    - Core supported-param baseline (temperature, max_tokens, tools, tool_choice,
      stream) to catch litellm silently dropping params we depend on.
    - Unsupported-model rejection (ValueError).

These tests call litellm's get_supported_openai_params / get_llm_provider /
get_optional_params without mocking, so a litellm upgrade that changes model
metadata will surface here.

Requires API keys in the environment (OPENAI_API_KEY, ANTHROPIC_API_KEY) so
they run alongside other real-API tests rather than in the fast unit-test suite.
"""

import os

from litellm import get_supported_openai_params
import pytest

from line.llm_agent.provider import _get_model_config

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
# Parametrized real-model tests
# ---------------------------------------------------------------------------


class TestGetModelConfigRealModels:
    """Assert _ModelConfig values for real models via live litellm lookups."""

    @_has_openai_key
    @pytest.mark.parametrize(
        "model, expected_backend, expected_supports_reasoning, expected_default_reasoning",
        [
            # Non-reasoning OpenAI models
            ("openai/gpt-4o-mini", "http", False, None),
            ("openai/o1-mini", "http", False, None),
            # Reasoning OpenAI models — litellm accepts reasoning_effort="none"
            # so default_reasoning_effort is "none" (not "low").
            ("openai/o3-mini", "http", True, "none"),
            # WebSocket-capable reasoning models
            ("gpt-5.2", "websocket", True, "low"),
            ("gpt-5.4", "websocket", True, "low"),
        ],
        ids=["gpt-4o-mini", "o1-mini", "o3-mini", "gpt-5.2", "gpt-5.4"],
    )
    def test_openai_models(
        self,
        model,
        expected_backend,
        expected_supports_reasoning,
        expected_default_reasoning,
    ):
        cfg = _get_model_config(model)
        assert cfg.backend == expected_backend
        assert cfg.supports_reasoning_effort is expected_supports_reasoning
        assert cfg.default_reasoning_effort == expected_default_reasoning

    @_has_anthropic_key
    @pytest.mark.parametrize(
        "model, expected_backend, expected_supports_reasoning, expected_default_reasoning",
        [
            # Non-extended-thinking Anthropic model
            ("anthropic/claude-3-5-sonnet-20241022", "http", False, None),
            # Extended-thinking Anthropic models — reasoning supported but
            # default is None because litellm rejects "none" for Anthropic.
            ("anthropic/claude-haiku-4-5", "http", True, None),
            ("anthropic/claude-sonnet-4-20250514", "http", True, None),
            ("anthropic/claude-opus-4-20250514", "http", True, None),
        ],
        ids=["claude-3-5-sonnet", "claude-haiku-4-5", "claude-sonnet-4", "claude-opus-4"],
    )
    def test_anthropic_models(
        self,
        model,
        expected_backend,
        expected_supports_reasoning,
        expected_default_reasoning,
    ):
        cfg = _get_model_config(model)
        assert cfg.backend == expected_backend
        assert cfg.supports_reasoning_effort is expected_supports_reasoning
        assert cfg.default_reasoning_effort == expected_default_reasoning

    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError, match="is not supported"):
            _get_model_config("fake-provider/nonexistent-model-xyz")


class TestGetSupportedOpenaiParamsBaseline:
    """Spot-check that litellm still exposes the core params we rely on.

    Catches litellm silently dropping params (e.g. temperature, tools) that
    _get_model_config wouldn't surface because it only inspects reasoning_effort.
    """

    # Params that every HTTP-backend model we use must support.
    _CORE_PARAMS = {"temperature", "max_tokens", "tools", "tool_choice", "stream"}

    @_has_openai_key
    @pytest.mark.parametrize("model", ["openai/gpt-4o-mini", "openai/o3-mini"])
    def test_openai_core_params(self, model):
        supported = set(get_supported_openai_params(model=model) or [])
        missing = self._CORE_PARAMS - supported
        assert not missing, f"{model} lost core params: {missing}"

    @_has_anthropic_key
    @pytest.mark.parametrize("model", ["anthropic/claude-sonnet-4-20250514", "anthropic/claude-haiku-4-5"])
    def test_anthropic_core_params(self, model):
        supported = set(get_supported_openai_params(model=model) or [])
        missing = self._CORE_PARAMS - supported
        assert not missing, f"{model} lost core params: {missing}"
