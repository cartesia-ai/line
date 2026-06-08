#!/usr/bin/env python3
"""Live-API evaluation of the built-in voicemail tool (Approach 1).

Two phases (both make real LLM calls):

1. **Detection** — runs labeled scenarios through an agent with the voicemail
   tool and reports a confusion matrix. Priority metric: human-safety (never hang
   up on a real person → zero false positives).

2. **Latency sweep** — runs longer human conversations at several
   ``voicemail_tool_active_turns`` limits (1, 2, 5, None) and reports the average
   per-turn latency for each, to show how keeping the tool available on more
   turns adds per-turn overhead (an extra tool sits in the LLM's schema on every
   turn the tool is still present).

This is a live test (real LLM calls) and is intentionally NOT collected by a bare
``pytest`` run (filename starts with ``real_api_test_``). Run it either way:

    # full run (detection + latency sweep), prints everything
    ANTHROPIC_API_KEY=... uv run python tests/real_api_test_voicemail_tool.py

    # pytest: detection only (asserts zero false positives), skips without a key
    ANTHROPIC_API_KEY=... uv run pytest tests/real_api_test_voicemail_tool.py -s

Environment:
    VOICEMAIL_EVAL_MODEL  main model (default anthropic/claude-haiku-4-5-20251001)
    ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY  matched to the model
"""

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
import os
import sys
import time
from typing import List, Literal, Optional

import litellm

from line.agent import AgentEnv, TurnEnv
from line.events import AgentEndCall, AgentSendText, AgentTextSent, UserTextSent, UserTurnEnded
from line.llm_agent import LlmAgent, LlmConfig, end_call, voicemail

MODEL = os.getenv("VOICEMAIL_EVAL_MODEL", "anthropic/claude-haiku-4-5-20251001")
# Tool-active-turn limits compared in the latency sweep.
TURN_LIMITS: List[Optional[int]] = [1, 2, 5, None]

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


# Detection set — human-weighted and adversarial so the false-positive rate is
# trustworthy. "voicemail_classic" greetings are unambiguous.
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

# Latency set — longer (8-turn) live human conversations, used only for the
# sweep. They must exceed the largest finite limit (5) so the 5-vs-None
# difference is observable. These are the callee's (human) turns.
LATENCY_SCENARIOS: List[List[str]] = [
    [
        "Hello?",
        "Yeah, this is Sarah.",
        "Oh, about my order? Sure.",
        "Right, the one I placed last week.",
        "It hasn't arrived yet, actually.",
        "Okay, so when should I expect it?",
        "Got it, that works.",
        "Thanks for calling, bye.",
    ],
    [
        "Who's this?",
        "Uh huh, and what's this about?",
        "My order, okay. Make it quick.",
        "It was supposed to come Tuesday.",
        "So it's delayed again?",
        "Fine, just keep me posted then.",
        "Yeah, this number's fine.",
        "Okay. Bye.",
    ],
    [
        "Oh, hi there!",
        "Yes, this is the right number.",
        "The order, yes, I've been wondering about it.",
        "I ordered the blue one, not the red.",
        "Right, so can you fix that?",
        "Wonderful, thank you so much.",
        "And how long will that take?",
        "Perfect, appreciate it!",
    ],
    [
        "Hello? Who is this?",
        "Speak up please, I can't hear well.",
        "My... order? What order?",
        "Oh, the package. Yes, yes.",
        "It came already, I think.",
        "No, wait, maybe not.",
        "Can you check for me, dear?",
        "Alright, thank you.",
    ],
    [
        "Hang on— okay, hi.",
        "Sorry, who's calling?",
        "Right, the order, go ahead.",
        "Can you hold a sec? ... okay, back.",
        "So what about it?",
        "It's late? Ugh.",
        "Okay, just let me know.",
        "Thanks, bye.",
    ],
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


def _avg(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _tool_present_on_turn(turn_index_1based: int, active_turns: Optional[int]) -> bool:
    """Whether the voicemail tool is still in the schema on a given (1-based) turn."""
    return active_turns is None or turn_index_1based <= active_turns


async def _run_conversation(
    model: str, api_key: str, turns: List[str], active_turns: Optional[int]
) -> tuple[bool, List[float]]:
    """Drive a conversation through a fresh agent.

    Returns (detected_voicemail, per_turn_latency_ms).
    """
    agent = LlmAgent(
        model=model,
        api_key=api_key,
        tools=[voicemail(message=VOICEMAIL_MESSAGE), end_call],
        config=LlmConfig(system_prompt=SYSTEM_PROMPT, introduction=""),
        voicemail_tool_active_turns=active_turns,
    )
    env = TurnEnv(agent_env=AgentEnv())
    history: list = []
    detected = False
    latencies: List[float] = []
    try:
        for text in turns:
            history.append(UserTextSent(content=text))
            event = UserTurnEnded(content=[UserTextSent(content=text)], history=list(history))
            start = time.perf_counter()
            agent_text: List[str] = []
            async for output in agent.process(env, event):
                if isinstance(output, AgentEndCall) and output.reason == "voicemail_detected":
                    detected = True
                elif isinstance(output, AgentSendText):
                    agent_text.append(output.text)
            latencies.append((time.perf_counter() - start) * 1000)
            if agent_text:
                history.append(AgentTextSent(content=" ".join(agent_text)))
            if detected:
                break
    finally:
        await agent.cleanup()
    return detected, latencies


async def run_detection_eval(model: str, api_key: str) -> Matrix:
    """Phase 1: confusion matrix over the labeled detection scenarios (limit=1)."""
    m = Matrix()
    print(f"\n[1] Detection — model={model}, {len(SCENARIOS)} scenarios (voicemail_tool_active_turns=1)\n")
    print(f"{'truth':<10} {'predicted':<10} category")
    print("-" * 44)
    for s in SCENARIOS:
        detected, _ = await _run_conversation(model, api_key, s.turns, active_turns=1)
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


async def run_latency_sweep(model: str, api_key: str) -> None:
    """Phase 2: average per-turn latency at each tool-active-turn limit."""
    n_turns = sum(len(t) for t in LATENCY_SCENARIOS)
    print(
        f"\n[2] Latency sweep — {len(LATENCY_SCENARIOS)} human conversations, "
        f"{n_turns} turns each limit, model={model}\n"
    )
    print(f"  {'active_turns':<13} {'avg ms/turn':>12} {'tool-present':>14} {'tool-absent':>13}")
    print("  " + "-" * 54)

    present_pool: List[float] = []
    absent_pool: List[float] = []
    for limit in TURN_LIMITS:
        per_turn: List[tuple[int, float]] = []
        for turns in LATENCY_SCENARIOS:
            _, lats = await _run_conversation(model, api_key, turns, limit)
            per_turn.extend((i, ms) for i, ms in enumerate(lats, start=1))
        present = [ms for i, ms in per_turn if _tool_present_on_turn(i, limit)]
        absent = [ms for i, ms in per_turn if not _tool_present_on_turn(i, limit)]
        present_pool += present
        absent_pool += absent
        all_ms = [ms for _, ms in per_turn]
        label = "None" if limit is None else str(limit)
        present_str = f"{_avg(present):.0f}ms" if present else "—"
        absent_str = f"{_avg(absent):.0f}ms" if absent else "—"
        print(f"  {label:<13} {_avg(all_ms):>10.0f}ms {present_str:>14} {absent_str:>13}")

    dp, da = _avg(present_pool), _avg(absent_pool)
    print(
        f"\n  Pooled across limits — tool present: {dp:.0f}ms/turn   tool absent: {da:.0f}ms/turn   "
        f"per-turn cost of keeping the tool: {dp - da:+.0f}ms"
    )


async def _run_all() -> int:
    api_key = _resolve_api_key(MODEL)
    if not api_key:
        print(f"No API key for model {MODEL!r}. Set ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY.")
        return 2
    try:
        m = await run_detection_eval(MODEL, api_key)
        await run_latency_sweep(MODEL, api_key)
    finally:
        # Close litellm's cached async HTTP clients inside the loop, else their
        # SSL transports raise "Event loop is closed" after the run.
        with suppress(Exception):
            await litellm.close_litellm_async_clients()

    if m.fp:
        print(f"\nFAIL: {m.fp} false positive(s) — the agent hung up on a real person.")
        return 1
    print("\nPASS: zero false positives (no human was hung up on).")
    return 0


def test_voicemail_tool_confusion_matrix():
    """Pytest entry: detection only. Skips without an API key; asserts zero false positives."""
    import pytest

    api_key = _resolve_api_key(MODEL)
    if not api_key:
        pytest.skip(f"no API key for live voicemail eval (model={MODEL})")

    async def _go() -> Matrix:
        try:
            return await run_detection_eval(MODEL, api_key)
        finally:
            with suppress(Exception):
                await litellm.close_litellm_async_clients()

    matrix = asyncio.run(_go())
    # The non-negotiable: never hang up on a human.
    assert matrix.fp == 0, f"hung up on {matrix.fp} human(s): {matrix.fp_examples}"
    # Sanity floor: unambiguous classic greetings should be caught.
    assert matrix.recall > 0.0, "detected no voicemails at all — check the tool/prompt"


if __name__ == "__main__":
    sys.exit(asyncio.run(_run_all()))
