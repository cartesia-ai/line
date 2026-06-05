"""Cheap-LM voicemail detection sidecar for :class:`LlmAgent`.

This module provides an opt-in, *independent* voicemail/answering-machine
classifier that runs concurrently with the main LM. It uses a separate, cheap
LiteLLM-backed model (e.g. ``openai/gpt-4o-mini``) with a tiny deterministic
config and a strict-JSON prompt. The classifier never sees the main agent's
tools or system prompt — it only receives the latest user transcript.

The agent wiring (buffering the main LM's first user-visible output behind a
short gate and suppressing it on a ``voicemail`` verdict) lives in
``LlmAgent``; this module only owns the detector itself.

Detection is conservative and fail-open: invalid JSON, timeouts, and provider
errors all collapse to ``"unknown"`` so the call continues normally.
"""

from dataclasses import dataclass
import json
from typing import Literal, Optional

from loguru import logger

from line.llm_agent.config import LlmConfig
from line.llm_agent.provider import LlmProvider, Message, _get_model_config, parse_model_id

# Result of a single classification. "unknown" is the conservative fail-open
# verdict used whenever evidence is ambiguous or the detector errors.
VoicemailClassification = Literal["voicemail", "human", "unknown"]


# The detector is fully self-contained: this is the only system prompt it sees,
# and it is given no tools. Keeping it independent of the main agent means the
# main system prompt / tool list never leaks into (or biases) the classifier.
_DETECTOR_SYSTEM_PROMPT = """You are a strict voicemail / answering-machine detector for a live phone call.

You are given the most recent thing the other party said near the start of a call. \
Decide whether the call has reached a VOICEMAIL / answering machine, or a live HUMAN.

Respond with ONLY a JSON object with exactly these two keys and nothing else:
{"classification": "voicemail" | "human" | "unknown", "reason": "<short reason>"}

Be conservative — prefer "unknown" when the evidence is weak:
- "voicemail": obvious machine evidence, e.g. "please leave a message", "at the tone",
  "you've reached the voicemail of...", "is not available right now", an explicit beep, or a
  one-sided recorded greeting with no expectation of a reply.
- "human": clearly interactive speech that expects a reply, e.g. "Hello?", "Hi, who's this?",
  "How can I help you?".
- "unknown": ambiguous, noisy, partial, or empty evidence.

Do not include markdown, code fences, or any text outside the JSON object."""


@dataclass
class VoicemailDetectionConfig:
    """Opt-in configuration for :class:`LlmAgent` voicemail detection.

    Args:
        model: Cheap classifier model id (LiteLLM naming, e.g. ``"openai/gpt-4o-mini"``).
        api_key: API key for the classifier provider.
        message: Optional fixed message spoken (uninterruptible) when a voicemail is
            detected. If omitted, the call ends silently.
        initial_gate_ms: How long to buffer the main LM's first user-visible output while
            waiting for the detector. Advanced latency/accuracy tradeoff; default ``200``.
        min_transcript_words: Minimum number of words in the user turn before detection runs
            at all. Below this the agent skips detection and continues normally — so the
            sidecar never hangs up on a too-short greeting (e.g. "Hello?", "Yep?") and
            effectively waits to hear more content first. ``0`` (default) runs on any
            non-empty transcript.
        timeout: Hard timeout (seconds) for a single classifier request.
    """

    model: str
    api_key: str
    message: Optional[str] = None
    initial_gate_ms: int = 200
    min_transcript_words: int = 0
    timeout: float = 5.0


@dataclass
class VoicemailDetectionResult:
    """Internal detector result. No confidence score is exposed or used."""

    classification: VoicemailClassification
    reason: str = ""


def _parse_detector_output(text: Optional[str]) -> VoicemailDetectionResult:
    """Parse the classifier's raw text into a :class:`VoicemailDetectionResult`.

    Fail-open: any malformed / unexpected output becomes ``"unknown"``.
    """
    if not text or not text.strip():
        return VoicemailDetectionResult("unknown", "empty detector output")

    cleaned = text.strip()
    # Tolerate models that wrap JSON in ```...``` fences despite the instructions.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    # Extract the first complete-looking JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return VoicemailDetectionResult("unknown", "no JSON object in detector output")

    try:
        data = json.loads(cleaned[start : end + 1])
    except (ValueError, TypeError):
        return VoicemailDetectionResult("unknown", "invalid JSON from detector")

    if not isinstance(data, dict):
        return VoicemailDetectionResult("unknown", "detector JSON was not an object")

    classification = data.get("classification")
    reason = data.get("reason", "")
    if classification not in ("voicemail", "human", "unknown"):
        return VoicemailDetectionResult("unknown", f"unexpected classification: {classification!r}")

    return VoicemailDetectionResult(classification, reason if isinstance(reason, str) else "")


class _VoicemailDetector:
    """Cheap-LM classifier wrapper used by :class:`LlmAgent`.

    Owns its own :class:`LlmProvider` so it is fully independent of the main
    agent's provider, tools, and system prompt.
    """

    def __init__(self, config: VoicemailDetectionConfig):
        self._config = config

        # Reasoning models (e.g. gpt-5, o-series) only allow their default
        # temperature and reject temperature=0; non-reasoning models (e.g.
        # gpt-4o-mini) take temperature=0 for determinism. Use reasoning-effort
        # support as the proxy and omit temperature when it isn't honored.
        is_reasoning_model = _get_model_config(parse_model_id(config.model)).supports_reasoning_effort

        # Deterministic config. We only need a short JSON object back, but the
        # budget must also cover reasoning tokens on reasoning models, where
        # max_tokens caps completion + reasoning — too small and the model spends
        # it all thinking and returns nothing (→ fail-open "unknown").
        detector_config = LlmConfig(
            system_prompt=_DETECTOR_SYSTEM_PROMPT,
            temperature=None if is_reasoning_model else 0,
            max_tokens=512,
            reasoning_effort="none",
            timeout=config.timeout,
        )
        # No tools — the classifier must never call into the main agent's tools.
        self._provider = LlmProvider(
            model=config.model,
            api_key=config.api_key,
            config=detector_config,
            tools=[],
        )

    async def classify(self, transcript: str) -> VoicemailDetectionResult:
        """Classify a single user transcript. Fail-open to ``"unknown"`` on any error."""
        messages = [Message(role="user", content=transcript)]
        try:
            text = ""
            async for chunk in self._provider.chat(messages, None):
                if chunk.text:
                    text += chunk.text
            return _parse_detector_output(text)
        except Exception as e:  # noqa: BLE001 — detection is fail-open by design.
            logger.warning(f"Voicemail detector error: {e}")
            return VoicemailDetectionResult("unknown", f"detector error: {e}")

    async def aclose(self) -> None:
        """Close the underlying detector provider."""
        await self._provider.aclose()


__all__ = [
    "VoicemailClassification",
    "VoicemailDetectionConfig",
    "VoicemailDetectionResult",
]
