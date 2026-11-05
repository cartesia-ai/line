"""Tests for the wire-format models in line._harness_types."""

from pydantic import ValidationError
import pytest

from line._harness_types import TTSConfig


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"speed": 0.6},
        {"speed": 1.5},
        {"volume": 0.5},
        {"volume": 2.0},
        {"emotion": "cheerful"},
        {"voice_id": "voice-123", "speed": 1.2, "volume": 0.8, "emotion": "calm"},
    ],
)
def test_tts_config_accepts_valid_generation_controls(kwargs):
    config = TTSConfig(**kwargs)
    for key, value in kwargs.items():
        assert getattr(config, key) == value


@pytest.mark.parametrize(
    "kwargs",
    [
        {"speed": 0.59},
        {"speed": 1.51},
        {"volume": 0.49},
        {"volume": 2.01},
        {"emotion": ""},
    ],
)
def test_tts_config_rejects_out_of_bounds_generation_controls(kwargs):
    with pytest.raises(ValidationError):
        TTSConfig(**kwargs)


def test_tts_config_generation_controls_default_to_none():
    config = TTSConfig(voice_id="voice-123")
    assert config.speed is None
    assert config.volume is None
    assert config.emotion is None
    # exclude_none keeps unset controls off the wire, so the harness treats
    # them as "not provided" rather than explicit overrides.
    assert config.model_dump(exclude_none=True) == {"voice_id": "voice-123"}


def test_config_output_omits_unset_generation_controls_on_the_wire():
    """Mirrors the websocket send path (model_dump(exclude_none=True)):
    an AgentUpdateCall-style ConfigOutput must not emit null speed/volume/
    emotion keys, which the harness would otherwise treat as explicit
    overrides once mid-call generation-control updates are supported."""
    from line._harness_types import ConfigOutput, STTConfig

    message = ConfigOutput(
        tts=TTSConfig(voice_id="voice-123", language="en"),
        stt=STTConfig(language="en"),
        language="en",
    )
    payload = message.model_dump(exclude_none=True)
    assert payload["tts"] == {"voice_id": "voice-123", "language": "en"}
    assert "speed" not in payload["tts"]
    assert "volume" not in payload["tts"]
    assert "emotion" not in payload["tts"]
