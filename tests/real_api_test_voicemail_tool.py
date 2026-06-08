#!/usr/bin/env python3
"""Live-API evaluation of the built-in voicemail tool (Approach 1).

Runs a real LlmAgent (with the ``voicemail`` tool) over labeled multi-turn
scenarios and reports a confusion matrix. The priority metric is **human-safety**:
the agent must never hang up on a real person (zero false positives).

This is a live test — it makes real LLM calls and is intentionally NOT collected
by a bare ``pytest`` run (filename starts with ``real_api_test_``). Run it either
way:

    # as a script (prints the matrix, exits non-zero on a false positive)
    ANTHROPIC_API_KEY=... uv run python tests/real_api_test_voicemail_tool.py

    # under pytest (skips when no key is set)
    ANTHROPIC_API_KEY=... uv run pytest tests/real_api_test_voicemail_tool.py -s

Environment:
    VOICEMAIL_EVAL_MODEL  main model (default anthropic/claude-haiku-4-5-20251001)
    ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY  matched to the model
"""

import asyncio
from dataclasses import dataclass, field
import os
import sys
from typing import List, Literal, Optional

from line.agent import AgentEnv, TurnEnv
from line.events import AgentEndCall, AgentSendText, AgentTextSent, UserTextSent, UserTurnEnded
from line.llm_agent import LlmAgent, LlmConfig, end_call, voicemail

MODEL = os.getenv("VOICEMAIL_EVAL_MODEL", "anthropic/claude-haiku-4-5-20251001")
SYSTEM_PROMPT = (
    "You are an outbound voice agent making a brief courtesy call about a customer's recent order. "
    "Wait for the other party to speak first. If you've reached a live person, greet them and keep it short. "
    "If you've reached a voicemail or answering machine, use the voicemail tool."
)
VOICEMAIL_MESSAGE = "Hi, this is a courtesy call about your recent order. Please call us back. Thanks!"


@dataclass
class Scenario:
    label: Literal["voicemail", "human"]
    category: str
    turns: List[str]


# Human-weighted, adversarial. The point is to measure false positives with
# enough samples to trust the result, since hanging up on a person is the
# catastrophic error. "voicemail_classic" greetings are unambiguous and used for
# the recall floor assertion.
SCENARIOS: List[Scenario] = [
    # --- voicemail ---
    Scenario(
        "voicemail",
        "voicemail_classic",
        ["Hi, you've reached the voicemail of Alex Carter. Please leave a message after the tone."],
    ),
    Scenario(
        "voicemail",
        "voicemail_classic",
        ["The person you are trying to reach is not available. At the tone, please record your message."],
    ),
    Scenario(
        "voicemail",
        "voicemail_classic",
        ["You've reached the Google subscriber. Please record your message after the tone."],
    ),
    Scenario(
        "voicemail",
        "voicemail_business",
        ["Thank you for calling Brightwave Solutions. Our office is closed. Please leave a message."],
    ),
    Scenario("voicemail", "voicemail_subtle", ["Hey, it's Dana. You know what to do."]),
    Scenario(
        "voicemail",
        "voicemail_subtle",
        ["Hi, this is Mike. I can't get to my phone right now. Catch you later."],
    ),
    Scenario("voicemail", "voicemail_terse", ["Leave a message."]),
    # --- human (multi-turn, adversarial) ---
    Scenario("human", "human_short", ["Hello?", "Yeah, this is me. What's up?"]),
    Scenario("human", "human_short", ["Yeah, hello?", "Who's this?"]),
    Scenario(
        "human", "human_name", ["Hello, this is Sarah speaking.", "Yes, that's me — what's this about?"]
    ),
    Scenario("human", "human_name", ["Good afternoon, Daniel here.", "Who's calling?"]),
    Scenario("human", "human_name", ["Hi, you've got Marcus.", "Yeah? What's up?"]),
    Scenario(
        "human",
        "human_message_word",
        ["Oh hey, I got your voicemail earlier — what's up?", "Right, the order. Go on."],
    ),
    Scenario(
        "human",
        "human_message_word",
        ["Sorry, my voicemail's full, good thing you caught me.", "What did you need?"],
    ),
    Scenario(
        "human",
        "human_message_word",
        ["Hey, did you leave me a message? I saw a missed call.", "Okay, what's it about?"],
    ),
    Scenario("human", "human_screening", ["Who's calling please?", "And what's this regarding?"]),
    Scenario("human", "human_screening", ["Is this a sales call?", "Fine, what is it?"]),
    Scenario("human", "human_busy", ["Hang on— okay, sorry, hi. You there?", "Go ahead."]),
    Scenario("human", "human_busy", ["Sorry, I'm driving, you're on speaker. Who's this?", "Okay, quickly?"]),
    Scenario(
        "human",
        "human_business",
        ["Good morning, Brightwave Solutions, this is Jordan, how can I help?", "Sure, one moment."],
    ),
    Scenario("human", "human_business", ["Front desk, this is Lena.", "Who did you need?"]),
    Scenario(
        "human", "human_callback", ["Hi, I got a missed call from this number?", "Oh, okay, what's it about?"]
    ),
    Scenario(
        "human",
        "human_confused",
        ["Hello? Who is this now?", "Speak up, I can't hear you.", "Oh. Okay, go on."],
    ),
    Scenario(
        "human", "human_skeptical", ["Yeah?... who's this exactly?", "Uh huh. And why are you calling?"]
    ),
]


@dataclass
class Matrix:
    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0
    fp_examples: List[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 1.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def accuracy(self) -> float:
        t = self.tp + self.fn + self.fp + self.tn
        return (self.tp + self.tn) / t if t else 0.0


def _resolve_api_key(model: str) -> Optional[str]:
    if model.startswith("anthropic"):
        return os.getenv("ANTHROPIC_API_KEY")
    if model.startswith("gemini"):
        return os.getenv("GEMINI_API_KEY")
    return os.getenv("OPENAI_API_KEY")


async def _detected_voicemail(model: str, api_key: str, scenario: Scenario) -> bool:
    """Run a scenario through a fresh agent; True if it hung up as voicemail."""
    agent = LlmAgent(
        model=model,
        api_key=api_key,
        tools=[voicemail(message=VOICEMAIL_MESSAGE), end_call],
        config=LlmConfig(system_prompt=SYSTEM_PROMPT, introduction=""),
        voicemail_tool_active_turns=1,
    )
    env = TurnEnv(agent_env=AgentEnv())
    history: list = []
    detected = False
    try:
        for text in scenario.turns:
            history.append(UserTextSent(content=text))
            event = UserTurnEnded(content=[UserTextSent(content=text)], history=list(history))
            agent_text: List[str] = []
            async for output in agent.process(env, event):
                if isinstance(output, AgentEndCall) and output.reason == "voicemail_detected":
                    detected = True
                elif isinstance(output, AgentSendText):
                    agent_text.append(output.text)
            if agent_text:
                history.append(AgentTextSent(content=" ".join(agent_text)))
            if detected:
                break
    finally:
        await agent.cleanup()
    return detected


async def run_eval(model: str, api_key: str) -> Matrix:
    m = Matrix()
    print(f"\nVoicemail-tool eval — model={model}, {len(SCENARIOS)} scenarios\n")
    print(f"{'truth':<10} {'predicted':<10} category")
    print("-" * 44)
    for s in SCENARIOS:
        detected = await _detected_voicemail(model, api_key, s)
        actual_vm = s.label == "voicemail"
        if actual_vm and detected:
            m.tp += 1
        elif actual_vm and not detected:
            m.fn += 1
        elif not actual_vm and detected:
            m.fp += 1
            m.fp_examples.append(f"{s.category}: {s.turns[0]!r}")
        else:
            m.tn += 1
        pred = "voicemail" if detected else "human"
        flag = "  ⚠ FALSE POSITIVE" if (not actual_vm and detected) else ""
        print(f"{s.label:<10} {pred:<10} {s.category}{flag}")

    humans = m.tn + m.fp
    print("\nconfusion matrix          predicted")
    print("                       voicemail   human")
    print(f"  actual voicemail   {m.tp:>9}   {m.fn:>5}")
    print(f"  actual human       {m.fp:>9}   {m.tn:>5}")
    print(f"\naccuracy {m.accuracy:.0%}  precision {m.precision:.0%}  recall {m.recall:.0%}")
    print(f"human-safety: {m.tn}/{humans} humans preserved" + (f"  ⚠ {m.fp} WRONGLY HUNG UP" if m.fp else ""))
    for ex in m.fp_examples:
        print(f"  FP → {ex}")
    return m


async def _main_async() -> int:
    api_key = _resolve_api_key(MODEL)
    if not api_key:
        print(f"No API key for model {MODEL!r}. Set ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY.")
        return 2
    m = await run_eval(MODEL, api_key)
    # Hard guarantee: never hang up on a human.
    if m.fp:
        print(f"\nFAIL: {m.fp} false positive(s) — the agent hung up on a real person.")
        return 1
    print("\nPASS: zero false positives (no human was hung up on).")
    return 0


def test_voicemail_tool_confusion_matrix():
    """Pytest entry: skips without an API key; asserts zero false positives."""
    import pytest

    api_key = _resolve_api_key(MODEL)
    if not api_key:
        pytest.skip(f"no API key for live voicemail eval (model={MODEL})")
    matrix = asyncio.run(run_eval(MODEL, api_key))
    # The non-negotiable: never hang up on a human.
    assert matrix.fp == 0, f"hung up on {matrix.fp} human(s): {matrix.fp_examples}"
    # Sanity floor: unambiguous classic greetings should be caught.
    assert matrix.recall > 0.0, "detected no voicemails at all — check the tool/prompt"


if __name__ == "__main__":
    sys.exit(asyncio.run(_main_async()))
