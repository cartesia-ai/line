"""Compare two voicemail-detection approaches over multi-turn conversations.

Approach 1 — built-in ``voicemail`` tool: the main LM decides whether to call the
``voicemail`` tool (which ends the call with ``reason="voicemail_detected"``).

Approach 2 — cheap-LM detection sidecar: a separate classifier runs concurrently
with the main LM and can end the call on a ``voicemail`` verdict.

Each scenario in ``transcripts.py`` is a short *conversation*, so the harness
measures detection quality (confusion matrix, accuracy / precision / recall) AND
**per-turn latency** — splitting the opening turn (where the tool/sidecar is
active) from later in-conversation turns, to show the overhead each approach adds.

By default both mechanisms are kept active for *every* turn
(``*_ACTIVE_TURNS=None``) so that overhead is visible. In production you'd set
them to ``1`` (only check the opening turn); pass ``TOOL_ACTIVE_TURNS=1`` /
``DETECTOR_ACTIVE_TURNS=1`` to see that.

Usage:
    export ANTHROPIC_API_KEY=...      # main LM for Approach 1 + 2
    export OPENAI_API_KEY=...         # cheap classifier for Approach 2
    uv run python examples/voicemail_detection/compare.py
"""

import asyncio
from dataclasses import dataclass, field
import os
import time
from typing import Dict, List, Optional

import litellm
from transcripts import SCENARIOS, Scenario

from line.agent import AgentEnv, TurnEnv
from line.events import AgentEndCall, AgentSendText, AgentTextSent, UserTextSent, UserTurnEnded
from line.llm_agent import LlmAgent, LlmConfig, VoicemailDetectionConfig, voicemail
from line.llm_agent.tools.system import end_call
from line.llm_agent.voicemail_detection import _VoicemailDetector

# Models (override via env). Approach 1 uses a full conversational model; Approach 2
# uses a cheap classifier.
MAIN_MODEL = os.getenv("MAIN_MODEL", "anthropic/claude-haiku-4-5-20251001")
MAIN_API_KEY = os.getenv("MAIN_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
DETECTOR_MODEL = os.getenv("DETECTOR_MODEL", "openai/gpt-5-nano")
DETECTOR_API_KEY = os.getenv("DETECTOR_API_KEY") or os.getenv("OPENAI_API_KEY")


def _opt_int(name: str) -> Optional[int]:
    """Parse an optional int env var; empty / 'none' → None (active for all turns)."""
    raw = os.getenv(name)
    if raw is None or raw.strip().lower() in ("", "none"):
        return None
    return int(raw)


# Keep both mechanisms active across all turns by default so per-turn overhead is
# measurable; set to 1 to mimic the recommended production config.
TOOL_ACTIVE_TURNS = _opt_int("TOOL_ACTIVE_TURNS")
DETECTOR_ACTIVE_TURNS = _opt_int("DETECTOR_ACTIVE_TURNS")
DETECTOR_MIN_WORDS = int(os.getenv("DETECTOR_MIN_WORDS", "0"))
# The sidecar only acts on a verdict that returns within this gate. A slow detector
# (e.g. a reasoning model like gpt-5-nano, ~1s) will miss a 200ms gate every time and
# never hang up — widen the gate or use a fast non-reasoning detector (gpt-4o-mini).
DETECTOR_GATE_MS = int(os.getenv("DETECTOR_GATE_MS", "200"))

SYSTEM_PROMPT = (
    "You are an outbound voice agent calling a customer about their recent order. "
    "Greet the person and confirm you're speaking to the right customer, then keep it brief. "
    "If you reach a voicemail or answering machine instead of a live person, use the voicemail tool."
)
VOICEMAIL_MESSAGE = "Hi, this is a courtesy call about your recent order. Please call us back. Thanks!"


@dataclass
class ScenarioOutcome:
    detected_voicemail: bool = False
    per_turn_ms: List[float] = field(default_factory=list)


def _approach1_agent() -> LlmAgent:
    return LlmAgent(
        model=MAIN_MODEL,
        api_key=MAIN_API_KEY,
        tools=[voicemail(message=VOICEMAIL_MESSAGE), end_call],
        config=LlmConfig(system_prompt=SYSTEM_PROMPT, introduction=""),
        voicemail_detection=VoicemailDetectionConfig(tool_active_turns=TOOL_ACTIVE_TURNS),
    )


def _approach2_agent() -> LlmAgent:
    return LlmAgent(
        model=MAIN_MODEL,
        api_key=MAIN_API_KEY,
        tools=[end_call],
        config=LlmConfig(system_prompt=SYSTEM_PROMPT, introduction=""),
        voicemail_detection=VoicemailDetectionConfig(
            model=DETECTOR_MODEL,
            api_key=DETECTOR_API_KEY,
            message=VOICEMAIL_MESSAGE,
            initial_gate_ms=DETECTOR_GATE_MS,
            active_turns=DETECTOR_ACTIVE_TURNS,
            min_transcript_words=DETECTOR_MIN_WORDS,
        ),
    )


async def _run_scenario(agent: LlmAgent, scenario: Scenario) -> ScenarioOutcome:
    """Drive a whole conversation through one agent, timing each turn."""
    env = TurnEnv(agent_env=AgentEnv())
    history: list = []
    outcome = ScenarioOutcome()
    try:
        for text in scenario.turns:
            history.append(UserTextSent(content=text))
            event = UserTurnEnded(content=[UserTextSent(content=text)], history=list(history))

            start = time.perf_counter()
            agent_text: List[str] = []
            async for output in agent.process(env, event):
                if isinstance(output, AgentEndCall) and output.reason == "voicemail_detected":
                    outcome.detected_voicemail = True
                elif isinstance(output, AgentSendText):
                    agent_text.append(output.text)
            outcome.per_turn_ms.append((time.perf_counter() - start) * 1000)

            if agent_text:
                history.append(AgentTextSent(content=" ".join(agent_text)))
            if outcome.detected_voicemail:
                break  # the call has ended
    finally:
        await agent.cleanup()
    return outcome


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


@dataclass
class ConfusionMatrix:
    """Binary confusion matrix with the positive class = "voicemail"."""

    tp: int = 0  # actual voicemail, predicted voicemail (correctly hung up)
    fn: int = 0  # actual voicemail, predicted human (missed → kept talking to a machine)
    fp: int = 0  # actual human, predicted voicemail (WORST: hung up on a real person)
    tn: int = 0  # actual human, predicted human (correctly continued)

    @property
    def total(self) -> int:
        return self.tp + self.fn + self.fp + self.tn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0


def _confusion_matrix(scenarios: List[Scenario], outcomes: List[ScenarioOutcome]) -> ConfusionMatrix:
    cm = ConfusionMatrix()
    for s, o in zip(scenarios, outcomes, strict=False):
        actual = s.label == "voicemail"
        if actual and o.detected_voicemail:
            cm.tp += 1
        elif actual and not o.detected_voicemail:
            cm.fn += 1
        elif not actual and o.detected_voicemail:
            cm.fp += 1
        else:
            cm.tn += 1
    return cm


def _latency_split(outcomes: List[ScenarioOutcome]) -> tuple:
    """Average opening-turn latency vs. later in-conversation-turn latency."""
    first = [o.per_turn_ms[0] for o in outcomes if o.per_turn_ms]
    later = [ms for o in outcomes for ms in o.per_turn_ms[1:]]
    first_avg = sum(first) / len(first) if first else 0.0
    later_avg = sum(later) / len(later) if later else 0.0
    return first_avg, later_avg, len(later)


def _report(name: str, scenarios: List[Scenario], outcomes: List[ScenarioOutcome]) -> None:
    cm = _confusion_matrix(scenarios, outcomes)
    first_avg, later_avg, n_later = _latency_split(outcomes)

    print(f"\n{name}")
    print(f"  accuracy {cm.accuracy:.0%}  precision {cm.precision:.0%}  recall {cm.recall:.0%}")
    print("  confusion matrix          predicted")
    print("                         voicemail   human")
    print(f"    actual voicemail   {cm.tp:>9}   {cm.fn:>5}")
    print(f"    actual human       {cm.fp:>9}   {cm.tn:>5}")
    if cm.fp:
        print(f"  ⚠  {cm.fp} false positive(s): hung up on a real human")
    if cm.fn:
        print(f"  ⚠  {cm.fn} false negative(s): missed a voicemail (kept talking to a machine)")
    print(f"  latency  opening turn: {first_avg:>5.0f}ms   later turns: {later_avg:>5.0f}ms (n={n_later})")


def _category_breakdown(
    scenarios: List[Scenario], a1: List[ScenarioOutcome], a2: List[ScenarioOutcome]
) -> None:
    cats: Dict[str, List[int]] = {}
    for i, s in enumerate(scenarios):
        cats.setdefault(s.category, []).append(i)

    print("\nPer-category accuracy (where the approaches differ)")
    print(f"  {'category':<20} {'n':>3}  {'A1 (tool)':>10}  {'A2 (sidecar)':>12}")
    print("  " + "-" * 50)
    for category in sorted(cats):
        idxs = cats[category]
        n = len(idxs)
        a1_ok = sum(1 for i in idxs if a1[i].detected_voicemail == (scenarios[i].label == "voicemail"))
        a2_ok = sum(1 for i in idxs if a2[i].detected_voicemail == (scenarios[i].label == "voicemail"))
        flag = "  ←" if a1_ok != a2_ok else ""
        print(f"  {category:<20} {n:>3}  {a1_ok:>4}/{n:<5} {a2_ok:>6}/{n:<5}{flag}")


def _missing_keys() -> Optional[str]:
    missing = []
    if not MAIN_API_KEY:
        missing.append("ANTHROPIC_API_KEY (or MAIN_API_KEY) for the main LM")
    if not DETECTOR_API_KEY:
        missing.append("OPENAI_API_KEY (or DETECTOR_API_KEY) for the sidecar")
    return ", ".join(missing) if missing else None


async def main() -> None:
    missing = _missing_keys()
    if missing:
        raise SystemExit(f"Missing API key(s): {missing}")

    print(
        f"main={MAIN_MODEL}  detector={DETECTOR_MODEL}  gate_ms={DETECTOR_GATE_MS}  "
        f"tool_active_turns={TOOL_ACTIVE_TURNS}  detector_active_turns={DETECTOR_ACTIVE_TURNS}  "
        f"min_words={DETECTOR_MIN_WORDS}"
    )
    a1: List[ScenarioOutcome] = []
    a2: List[ScenarioOutcome] = []
    a2_raw: List[ScenarioOutcome] = []  # detector verdict with a full await (no gate)
    # Standalone detector to measure the model's *raw* verdict + latency, so we can
    # tell model quality apart from the gate dropping late verdicts.
    raw_detector = _VoicemailDetector(
        VoicemailDetectionConfig(model=DETECTOR_MODEL, api_key=DETECTOR_API_KEY)
    )
    try:
        header = f"{'truth':<10} {'A1':<7} {'A2':<7} {'turns':>5}  scenario"
        print(header)
        print("-" * len(header))
        for scenario in SCENARIOS:
            o1 = await _run_scenario(_approach1_agent(), scenario)
            o2 = await _run_scenario(_approach2_agent(), scenario)
            # Raw detector verdict on the opening line, fully awaited (no gate).
            start = time.perf_counter()
            res = await raw_detector.classify(scenario.turns[0])
            raw_ms = (time.perf_counter() - start) * 1000
            a1.append(o1)
            a2.append(o2)
            a2_raw.append(ScenarioOutcome(res.classification == "voicemail", [raw_ms]))
            m1 = "vm" if o1.detected_voicemail else "human"
            m2 = "vm" if o2.detected_voicemail else "human"
            print(f"{scenario.label:<10} {m1:<7} {m2:<7} {len(scenario.turns):>5}  {scenario.name}")
    finally:
        await raw_detector.aclose()
        # litellm caches async HTTP clients globally; close them before the event
        # loop tears down or their SSL transports raise "Event loop is closed".
        await litellm.close_litellm_async_clients()

    _report("Approach 1 (voicemail tool)", SCENARIOS, a1)
    _report("Approach 2 (detection sidecar, as the agent applied it)", SCENARIOS, a2)
    _report("Approach 2 (raw detector verdict, full await — no gate)", SCENARIOS, a2_raw)

    # Diagnostic: separate model quality from the gate. If the detector's raw
    # latency exceeds the gate, its verdicts arrive too late and are dropped —
    # which shows up as the "applied" matrix being all-negative regardless of model.
    raw_lat = [o.per_turn_ms[0] for o in a2_raw if o.per_turn_ms]
    avg_lat = sum(raw_lat) / len(raw_lat) if raw_lat else 0.0
    fits = avg_lat <= DETECTOR_GATE_MS
    print(
        f"\ndetector raw latency avg {avg_lat:.0f}ms vs gate {DETECTOR_GATE_MS}ms — "
        + (
            "fits the gate."
            if fits
            else "EXCEEDS the gate, so verdicts are dropped (raise DETECTOR_GATE_MS or use a faster detector)."
        )
    )
    _category_breakdown(SCENARIOS, a1, a2)


if __name__ == "__main__":
    asyncio.run(main())
