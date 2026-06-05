"""Compare two voicemail-detection approaches side by side.

Approach 1 — built-in ``voicemail`` tool: the main LM decides, given the call's
opening line, whether to call the ``voicemail`` tool (which ends the call with
``reason="voicemail_detected"``). We run a fresh LlmAgent for each sample and
check whether it emitted that end-call event.

Approach 2 — cheap-LM detection sidecar: a separate, cheap classifier model
labels the opening line ``voicemail`` / ``human`` / ``unknown`` independently of
the main LM.

Both are run over the labeled dataset in ``transcripts.py`` and reported with
accuracy and average latency so you can decide which approach to ship.

Usage:
    # Main agent (Approach 1) — defaults to Anthropic; detector (Approach 2) to OpenAI.
    export ANTHROPIC_API_KEY=...      # main LM for Approach 1
    export OPENAI_API_KEY=...         # cheap classifier for Approach 2
    uv run python examples/voicemail_detection/compare.py

    # Override models:
    MAIN_MODEL=openai/gpt-4o DETECTOR_MODEL=openai/gpt-4o-mini \
        uv run python examples/voicemail_detection/compare.py
"""

import asyncio
from dataclasses import dataclass
import os
import time
from typing import List, Optional

from transcripts import SAMPLES, Sample

from line.agent import AgentEnv, TurnEnv
from line.events import AgentEndCall, UserTextSent, UserTurnEnded
from line.llm_agent import LlmAgent, LlmConfig, voicemail
from line.llm_agent.tools.system import end_call
from line.llm_agent.voicemail_detection import VoicemailDetectionConfig, _VoicemailDetector

# Models (override via env). Approach 1 uses a full conversational model; Approach 2
# uses a cheap classifier.
MAIN_MODEL = os.getenv("MAIN_MODEL", "anthropic/claude-haiku-4-5-20251001")
MAIN_API_KEY = os.getenv("MAIN_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
DETECTOR_MODEL = os.getenv("DETECTOR_MODEL", "openai/gpt-4o-mini")
DETECTOR_API_KEY = os.getenv("DETECTOR_API_KEY") or os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = (
    "You are an outbound voice agent calling a customer about their recent order. "
    "Greet the person and confirm you're speaking to the right customer. If you reach "
    "a voicemail or answering machine instead of a live person, use the voicemail tool."
)
VOICEMAIL_MESSAGE = "Hi, this is a courtesy call about your recent order. Please call us back. Thanks!"


@dataclass
class Outcome:
    predicted_voicemail: bool
    detail: str
    latency_ms: float


async def run_approach1(sample: Sample) -> Outcome:
    """Run the main LM with the voicemail tool over the call's opening line."""
    agent = LlmAgent(
        model=MAIN_MODEL,
        api_key=MAIN_API_KEY,
        tools=[voicemail(message=VOICEMAIL_MESSAGE), end_call],
        # introduction="" → outbound call: the agent waits for the callee to speak first.
        config=LlmConfig(system_prompt=SYSTEM_PROMPT, introduction=""),
        # Keep the tool available for this single opening turn.
        voicemail_tool_active_turns=1,
    )
    env = TurnEnv(agent_env=AgentEnv())
    user_msg = UserTextSent(content=sample.transcript)
    event = UserTurnEnded(content=[UserTextSent(content=sample.transcript)], history=[user_msg])

    start = time.perf_counter()
    detected = False
    try:
        async for output in agent.process(env, event):
            if isinstance(output, AgentEndCall) and output.reason == "voicemail_detected":
                detected = True
    finally:
        await agent.cleanup()
    latency_ms = (time.perf_counter() - start) * 1000
    return Outcome(detected, "called voicemail tool" if detected else "no voicemail tool call", latency_ms)


async def run_approach2(detector: _VoicemailDetector, sample: Sample) -> Outcome:
    """Run the cheap-LM classifier over the call's opening line."""
    start = time.perf_counter()
    result = await detector.classify(sample.transcript)
    latency_ms = (time.perf_counter() - start) * 1000
    return Outcome(result.classification == "voicemail", result.classification, latency_ms)


def _summary(name: str, samples: List[Sample], outcomes: List[Outcome]) -> str:
    correct = sum(
        1
        for s, o in zip(samples, outcomes, strict=False)
        if o.predicted_voicemail == (s.label == "voicemail")
    )
    avg_latency = sum(o.latency_ms for o in outcomes) / len(outcomes) if outcomes else 0.0
    return f"{name}: accuracy {correct}/{len(samples)} ({100 * correct / len(samples):.0f}%), avg latency {avg_latency:.0f}ms"


def _missing_keys() -> Optional[str]:
    missing = []
    if not MAIN_API_KEY:
        missing.append("ANTHROPIC_API_KEY (or MAIN_API_KEY) for Approach 1")
    if not DETECTOR_API_KEY:
        missing.append("OPENAI_API_KEY (or DETECTOR_API_KEY) for Approach 2")
    return ", ".join(missing) if missing else None


async def main() -> None:
    missing = _missing_keys()
    if missing:
        raise SystemExit(f"Missing API key(s): {missing}")

    detector = _VoicemailDetector(VoicemailDetectionConfig(model=DETECTOR_MODEL, api_key=DETECTOR_API_KEY))

    a1: List[Outcome] = []
    a2: List[Outcome] = []
    try:
        header = f"{'truth':<10} {'A1 (tool)':<12} {'A2 (sidecar)':<14} transcript"
        print(header)
        print("-" * len(header))
        for sample in SAMPLES:
            o1 = await run_approach1(sample)
            o2 = await run_approach2(detector, sample)
            a1.append(o1)
            a2.append(o2)
            a1_mark = "voicemail" if o1.predicted_voicemail else "human"
            a2_mark = o2.detail
            snippet = sample.transcript[:48] + ("…" if len(sample.transcript) > 48 else "")
            print(f"{sample.label:<10} {a1_mark:<12} {a2_mark:<14} {snippet}")
    finally:
        await detector.aclose()

    print()
    print(_summary("Approach 1 (voicemail tool)", SAMPLES, a1))
    print(_summary("Approach 2 (detection sidecar)", SAMPLES, a2))


if __name__ == "__main__":
    asyncio.run(main())
