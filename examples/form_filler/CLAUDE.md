# Line SDK - Form Filler Example

## About This Example

This example demonstrates a **YAML-driven form pattern** with a loopback tool for structured data collection. The `FormFiller` class loads questions from a YAML file and provides a dynamically generated system prompt and tool for recording answers.

> Line is Cartesia's open-source SDK for building real-time voice AI agents that connect any LLM to Cartesia's low-latency text-to-speech, enabling natural conversational experiences over phone calls and other voice interfaces.

## FormFiller Class Pattern

The `FormFiller` class encapsulates form logic and provides:

1. **`get_system_prompt()`** - Dynamically generates a system prompt including form structure and current state
2. **`record_answer_tool`** property - Returns a configured loopback tool for recording answers
3. **`get_current_question_text()`** - Gets the current question for the introduction

```python
from form_filler import FormFiller

form = FormFiller(str(FORM_PATH), system_prompt=USER_PROMPT)

agent = LlmAgent(
    model="anthropic/claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    tools=[form.record_answer_tool, end_call],
    config=LlmConfig(
        system_prompt=form.get_system_prompt(),
        introduction=f"Hi! {form.get_current_question_text()}",
    ),
)
```

**Tool returns structured status dict:**

```python
{
    "success": True,
    "completed": {"patient_name": "John Doe", ...},
    "remaining": ["date_of_birth", "visit_type", ...],
    "next_question": "And what is your date of birth?",
    "is_complete": False,
}
```

## YAML Form Definition

Forms are defined in YAML with questions, types, and conditional logic.

**Question types:**

| Type | Description | Example |
|------|-------------|---------|
| `string` | Free-form text | Name, date, notes |
| `number` | Numeric with optional min/max | Age, quantity |
| `boolean` | Yes/no questions | Confirmation |
| `select` | Multiple choice with options | Visit type, time preference |
| `date` | Date input | Appointment date |

**Conditional questions with `dependsOn`:**

```yaml
- id: "symptoms"
  text: "Can you describe your symptoms?"
  type: "string"
  dependsOn:
    questionId: "visit_type"
    operator: "in"
    value: ["sick_visit", "new_concern"]
```

**Operators:**

| Operator | Description |
|----------|-------------|
| `equals` | Exact match (default) |
| `not_equals` | Not equal to value |
| `in` | Value is in list |
| `not_in` | Value is not in list |

**Select question example:**

```yaml
- id: "preferred_time"
  text: "Do you prefer morning or afternoon?"
  type: "select"
  options:
    - value: "morning"
      text: "Morning - before noon"
    - value: "afternoon"
      text: "Afternoon - after noon"
```

## LlmAgent Configuration

```python
import os
from line.llm_agent import LlmAgent, LlmConfig

agent = LlmAgent(
    model="anthropic/claude-haiku-4-5-20251001",  # LiteLLM format
    api_key=os.getenv("ANTHROPIC_API_KEY"),  # Must be explicitly provided
    tools=[...],
    config=LlmConfig(...),
    max_tool_iterations=10,
)
```

**LlmConfig options:**

- `system_prompt`, `introduction` - Agent behavior
- `temperature`, `max_tokens`, `top_p`, `stop`, `seed` - Sampling
- `presence_penalty`, `frequency_penalty` - Penalties
- `num_retries`, `fallbacks`, `timeout` - Resilience
- `extra` - Provider-specific pass-through (dict)

**Dynamic configuration via Calls API:**

The [Calls API](https://docs.cartesia.ai/line/integrations/calls-api) connects client-side audio (web/mobile apps or telephony) to your agent via WebSocket. When initiating a call, clients can pass agent configuration that your agent receives in `CallRequest`.

Use `LlmConfig.from_call_request()` to allow callers to customize agent behavior at runtime:

```python
async def get_agent(env: AgentEnv, call_request: CallRequest):
    return LlmAgent(
        model="anthropic/claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        tools=[end_call, web_search],
        config=LlmConfig.from_call_request(
            call_request,
            fallback_system_prompt=SYSTEM_PROMPT,
            fallback_introduction=INTRODUCTION,
        ),
    )
```

**How it works:**

- Callers can pass `system_prompt` and `introduction` when initiating a call
- Priority: Caller's value > your fallback > SDK default
- For `system_prompt`: empty string is treated as unset (uses fallback)
- For `introduction`: empty string IS preserved (agent waits for user to speak first)

**Use cases:**

- Multi-tenant apps: Different system prompts per customer
- A/B testing: Test different agent personalities
- Contextual customization: Pass user-specific context at call time

**Model ID formats (LiteLLM):**

| Provider | Format |
|----------|--------|
| OpenAI | `gpt-5-nano-2025-08-07`, `gpt-4o` |
| Anthropic | `anthropic/claude-sonnet-4-20250514`, `anthropic/claude-haiku-4-5-20251001` |
| Gemini | `gemini/gemini-3-flash-preview` |

Full list of supported models: <https://models.litellm.ai/>

**Model selection strategy:** Use fast models (gpt-5-nano, claude-haiku, gemini-flash) for the main agent loop to minimize latency. Use powerful models (gpt-4o, claude-sonnet) inside background tools where latency is hidden.

## Tool Decorators

**Decision tree:**

```text
@loopback_tool           → Result goes to LLM
@loopback_tool(is_background=True) → Long-running, yields interim values
@passthrough_tool        → Yields OutputEvents directly
@handoff_tool            → Transfer to another handler
```

**Signatures:**

```python
from line.llm_agent import loopback_tool, passthrough_tool, handoff_tool, ToolEnv
from line import AgentSendText
from typing import Annotated

# Loopback - result sent to LLM
@loopback_tool
async def my_tool(ctx: ToolEnv, param: Annotated[str, "desc"]) -> str:
    return "result"

# Background - yields interim + final
@loopback_tool(is_background=True)
async def slow_tool(ctx: ToolEnv, query: Annotated[str, "desc"]):
    yield "Working..."
    yield await slow_work()

# Passthrough - yields OutputEvents
@passthrough_tool
async def direct_tool(ctx: ToolEnv, msg: Annotated[str, "desc"]):
    yield AgentSendText(text=msg)

# Handoff - requires event param
@handoff_tool
async def transfer(ctx: ToolEnv, reason: Annotated[str, "desc"], event):
    """Transfer to another agent."""
    async for output in other_agent.process(ctx.turn_env, event):
        yield output
```

**ToolEnv:** `ctx.turn_env` provides turn context (TurnEnv instance).

## Built-in Tools

```python
# Built-in tools
from line.llm_agent import end_call, send_dtmf, transfer_call, web_search, agent_as_handoff

agent = LlmAgent(
    tools=[
        end_call,
        agent_as_handoff(other_agent, name="transfer", description="Transfer to specialist"),
    ]
)
```

| Tool | Type | Purpose |
|------|------|---------|
| `end_call` | passthrough | End call gracefully |
| `send_dtmf` | passthrough | Send DTMF tone (0-9, *, #) |
| `transfer_call` | passthrough | Transfer to E.164 number |
| `web_search` | WebSearchTool | Real-time search (native or DuckDuckGo fallback) |
| `agent_as_handoff` | helper | Create handoff tool from an Agent (pass to tools list) |

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Missing `end_call` | Include it so agent can end calls (otherwise waits for user hangup) |
| Raising exceptions | Return error string. This will cause the agent to hang up the call. |
| Missing `ctx: ToolEnv` | First param always |
| No `Annotated` descriptions | Add for all params. This is used to describe the parameters of the tool to the LLM. |
| Slow model for main agent | Use fast model, offload to background |
| Missing `event` in handoff | Required final param |
| Calling record_answer without valid answer | Only call when user clearly provides an answer matching the question |
| Not using `get_system_prompt()` | Include form structure in system prompt for LLM context |
| Blocking nested agent call | Use `is_background=True` |
| Forgetting conversation history | Pass `history` in `UserTextSent` |
| Not cleaning up nested agents | Call cleanup on all agents in `_cleanup()` |

## Documentation

Full SDK documentation: <https://docs.cartesia.ai/line/sdk/overview>
