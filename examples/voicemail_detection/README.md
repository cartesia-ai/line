# Voicemail Detection — Approach Comparison

Two ways to detect that an outbound call reached a voicemail / answering machine,
plus a small harness to evaluate them side by side.

## The two approaches

Both approaches are configured through a single
`voicemail_detection=VoicemailDetectionConfig(...)` parameter, and both only act
during the **opening turns** of the call.

**Approach 1 — built-in `voicemail` tool.** The main LM is given the `voicemail`
tool and decides, from the call's opening line, whether to call it. The tool
speaks an optional fixed message and ends the call with
`reason="voicemail_detected"`. The agent **drops the tool after `tool_active_turns`
turns** (default `1`) so once the conversation is "deemed started" the LM can't
hang up mid-conversation. (No `model`/`api_key` on the config ⇒ tool only, no
sidecar.)

```python
from line.llm_agent import LlmAgent, LlmConfig, VoicemailDetectionConfig, end_call, voicemail

agent = LlmAgent(
    model="anthropic/claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    tools=[voicemail(message="Please call us back."), end_call],
    config=LlmConfig(system_prompt=SYSTEM_PROMPT, introduction=""),
    voicemail_detection=VoicemailDetectionConfig(tool_active_turns=1),
)
```

**Approach 2 — cheap-LM detection sidecar.** Setting `model`/`api_key` enables a
separate, cheap classifier that runs concurrently with the main LM for the first
`active_turns` turns (default `1`). The main LM's first user-visible output is
buffered for `initial_gate_ms` (default `200`); if the detector returns
`voicemail` within that gate, the main output is suppressed and the agent emits
the configured message plus `AgentEndCall(reason="voicemail_detected")`.
Otherwise the buffered output is released and the call continues. Detection is
conservative and fail-open (invalid JSON, timeouts, and errors all become
`unknown`).

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
        active_turns=1,
    ),
)
```

> Both `tool_active_turns` and `active_turns` default to `1`: voicemail is only
> worth checking at the very start, so neither approach keeps running once you're
> in conversation. Set them higher (or `None`) only to measure the per-turn cost.

## Outbound calls and `introduction=""`

For outbound calls the agent should hear the callee's opening line before
speaking, so set `introduction=""`. The agent stays silent on `CallStarted` and
only responds once it receives the first `UserTurnEnded`. Both approaches then
act on that opening line: Approach 1 lets the LM call the `voicemail` tool;
Approach 2 classifies it with the sidecar.

## Running the comparison harness

`compare.py` runs both approaches over the **multi-turn conversation scenarios**
in `transcripts.py`. Voicemail scenarios are a single greeting turn; human
scenarios run several back-and-forth turns. For each approach it reports a
confusion matrix (positive class = voicemail), accuracy / precision / recall, and
a **per-turn latency split** — the opening turn (where the tool/sidecar is active)
vs. later in-conversation turns. False positives (hanging up on a real human) and
false negatives (missing a voicemail) are flagged explicitly.

A **per-category accuracy breakdown** shows *where* the approaches diverge
(marked `←`). Scenarios are adversarial — subtle keyword-free voicemails, terse
machine greetings, business/carrier mailboxes, people who answer with their name
(`Hi, this is Sarah`) or say "message", and call screeners.

To make each approach's overhead visible, the harness keeps **both mechanisms
active on every turn by default** (`TOOL_ACTIVE_TURNS`/`DETECTOR_ACTIVE_TURNS`
unset ⇒ `None`). That's the opposite of the production default (`1`) — set them to
`1` to confirm the later-turn cost disappears once detection stops.

The classifier defaults to the small/cheap `openai/gpt-5-nano`; the main agent to
`anthropic/claude-haiku-4-5-20251001`.

```bash
export ANTHROPIC_API_KEY=...   # main LM (both approaches)
export OPENAI_API_KEY=...      # cheap classifier (Approach 2)
uv run python examples/voicemail_detection/compare.py

# Override models / windows:
MAIN_MODEL=openai/gpt-4o DETECTOR_MODEL=openai/gpt-5-mini \
TOOL_ACTIVE_TURNS=1 DETECTOR_ACTIVE_TURNS=1 \
    uv run python examples/voicemail_detection/compare.py
```

> **The detector must answer within the gate.** The sidecar only acts on a
> `voicemail` verdict that returns within `initial_gate_ms` (`DETECTOR_GATE_MS`,
> default 200). A *reasoning* model — including the small `gpt-5-nano` — takes
> ~1s per classification, so it misses a 200ms gate every time and **never hangs
> up** (you'll see Approach 2 predict voicemail 0 times). Either use a fast,
> non-reasoning detector or widen the gate:
>
> ```bash
> # Fast non-reasoning classifier (recommended) — fits the default gate:
> DETECTOR_MODEL=openai/gpt-4o-mini TOOL_ACTIVE_TURNS=1 DETECTOR_ACTIVE_TURNS=1 \
>     uv run python examples/voicemail_detection/compare.py
>
> # Or keep a reasoning detector but give it time (adds first-reply latency):
> DETECTOR_GATE_MS=1500 TOOL_ACTIVE_TURNS=1 DETECTOR_ACTIVE_TURNS=1 \
>     uv run python examples/voicemail_detection/compare.py
> ```

### Tuning the sidecar: hear more before deciding

A cheap classifier can jump the gun on a short opening line and hang up on a
real person. Two `VoicemailDetectionConfig` knobs control this:

- **`min_transcript_words`** — skip detection until the turn has at least this
  many words. Below the threshold the agent never hangs up and waits to hear more
  on a later turn. This is the lever for "listen to more content first": raising
  it cuts false positives on short greetings, at the cost of possibly deferring
  on *terse* voicemails ("Leave a message."). In the harness, set it with
  `DETECTOR_MIN_WORDS`:

  ```bash
  DETECTOR_MIN_WORDS=5 uv run python examples/voicemail_detection/compare.py
  ```

- **`initial_gate_ms`** — how long the main reply is buffered while waiting for
  the verdict. This is a *timing* knob (latency vs. catching the verdict in
  time); it does **not** give the detector more transcript to read.

## Running the agent

```bash
ANTHROPIC_API_KEY=... VOICEMAIL_APPROACH=tool    uv run python main.py   # Approach 1
ANTHROPIC_API_KEY=... OPENAI_API_KEY=... VOICEMAIL_APPROACH=sidecar uv run python main.py   # Approach 2
```
