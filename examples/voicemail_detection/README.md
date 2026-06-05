# Voicemail Detection — Approach Comparison

Two ways to detect that an outbound call reached a voicemail / answering machine,
plus a small harness to evaluate them side by side.

## The two approaches

**Approach 1 — built-in `voicemail` tool.** The main LM is given the `voicemail`
tool and decides, from the call's opening line, whether to call it. The tool
speaks an optional fixed message and ends the call with
`reason="voicemail_detected"`. The agent automatically **removes the tool after
the first user turn** (`voicemail_tool_active_turns=1`) so the conversation is
"deemed started" and the LM can't hang up mid-conversation.

```python
from line.llm_agent import LlmAgent, LlmConfig, end_call, voicemail

agent = LlmAgent(
    model="anthropic/claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    tools=[voicemail(message="Please call us back."), end_call],
    config=LlmConfig(system_prompt=SYSTEM_PROMPT, introduction=""),
    voicemail_tool_active_turns=1,  # drop the tool once the conversation starts
)
```

**Approach 2 — cheap-LM detection sidecar.** A separate, cheap classifier model
runs concurrently with the main LM on each completed user turn. The main LM's
first user-visible output is buffered for `initial_gate_ms` (default `200`); if
the detector returns `voicemail` within that gate, the main output is suppressed
and the agent emits the configured message plus
`AgentEndCall(reason="voicemail_detected")`. Otherwise the buffered output is
released and the call continues. Detection is conservative and fail-open
(invalid JSON, timeouts, and errors all become `unknown`).

```python
from line.llm_agent import LlmAgent, LlmConfig, VoicemailDetectionConfig, end_call

agent = LlmAgent(
    model="anthropic/claude-sonnet-4-20250514",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    tools=[end_call],
    config=LlmConfig(system_prompt=SYSTEM_PROMPT, introduction=""),
    voicemail_detection=VoicemailDetectionConfig(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        message="Hi, please call us back when you can.",
        initial_gate_ms=200,
    ),
)
```

## Outbound calls and `introduction=""`

For outbound calls the agent should hear the callee's opening line before
speaking, so set `introduction=""`. The agent stays silent on `CallStarted` and
only responds once it receives the first `UserTurnEnded`. Both approaches then
act on that opening line: Approach 1 lets the LM call the `voicemail` tool;
Approach 2 classifies it with the sidecar.

## Running the comparison harness

`compare.py` runs both approaches over the labeled dataset in `transcripts.py`
and reports, per approach, a confusion matrix (positive class = voicemail),
accuracy / precision / recall, and average latency. False positives (hanging up
on a real human) and false negatives (missing a voicemail) are flagged
explicitly, since hanging up on a person is the costlier mistake.

It also prints a **per-category accuracy breakdown** so you can see *where* the
approaches diverge. The dataset is intentionally diverse and adversarial — it
includes subtle voicemails with no "leave a message" keyword, live people who
answer with their name (`Hi, this is Sarah`) or say the word "message", noisy
partial ASR, and long monologues vs. interactive greetings. Categories where the
two approaches disagree are marked with `←`.

The classifier (Approach 2) defaults to `openai/gpt-5`; the main agent
(Approach 1) defaults to `anthropic/claude-haiku-4-5-20251001`.

```bash
export ANTHROPIC_API_KEY=...   # main LM for Approach 1
export OPENAI_API_KEY=...      # cheap classifier for Approach 2
uv run python examples/voicemail_detection/compare.py

# Override models:
MAIN_MODEL=openai/gpt-4o DETECTOR_MODEL=openai/gpt-5-mini \
    uv run python examples/voicemail_detection/compare.py
```

## Running the agent

```bash
ANTHROPIC_API_KEY=... VOICEMAIL_APPROACH=tool    uv run python main.py   # Approach 1
ANTHROPIC_API_KEY=... OPENAI_API_KEY=... VOICEMAIL_APPROACH=sidecar uv run python main.py   # Approach 2
```
