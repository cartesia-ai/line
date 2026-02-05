# Line SDK - Chat/Supervisor Example

## About This Example

This example demonstrates a two-tier agent architecture: a fast "chat" model (Haiku) handles routine conversations while escalating complex questions to a powerful "supervisor" (Opus). The supervisor receives full conversation history for context.

> Line is Cartesia's open-source SDK for building real-time voice AI agents that connect any LLM to Cartesia's low-latency text-to-speech, enabling natural conversational experiences over phone calls and other voice interfaces.

## Architecture Pattern

```text
User → Chat (Haiku) → Direct response
              ↓
         Complex question?
              ↓
        Supervisor (Opus) → Synthesized response
```

**When to use this pattern:**

- Complex reasoning tasks (math proofs, multi-step logic)
- Questions requiring deep domain expertise
- Cost optimization (cheap fast model + expensive powerful model on demand)
- When you need thoughtful analysis without blocking the conversation

## AgentClass Implementation

Use `AgentClass` to wrap multiple `LlmAgent` instances with shared state:

```python
from line.agent import AgentClass, TurnEnv
from line.llm_agent import LlmAgent, LlmConfig

class ChatSupervisorAgent(AgentClass):
    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._current_event: Optional[InputEvent] = None

        # Powerful model for deep reasoning
        self._supervisor = LlmAgent(
            model="anthropic/claude-opus-4-5",
            api_key=self._api_key,
            config=LlmConfig(system_prompt=SUPERVISOR_SYSTEM_PROMPT),
        )

        # Fast model for conversation
        self._chatter = LlmAgent(
            model="anthropic/claude-haiku-4-5",
            api_key=self._api_key,
            tools=[self.ask_supervisor, end_call],
            config=LlmConfig(
                system_prompt=CHAT_SYSTEM_PROMPT,
                introduction=CHAT_INTRODUCTION,
            ),
        )

    async def process(self, env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        self._input_event = event

        if isinstance(event, CallEnded):
            await self._cleanup()
            return

        async for output in self._chatter.process(env, event):
            yield output

    async def _cleanup(self):
        await self._chatter.cleanup()
        await self._supervisor.cleanup()
```

**Key points:**

- Tools can be class methods (e.g., `self.ask_supervisor`) - they close over instance state
- Store the current event to access conversation history later
- Always implement cleanup to release resources

## Background Tools for Nested Agents

Use `@loopback_tool(is_background=True)` when calling a nested agent to avoid blocking:

```python
@loopback_tool(is_background=True)
async def ask_supervisor(
    self,
    ctx: ToolEnv,
    question: Annotated[str, "The complex question requiring deep reasoning"],
) -> AsyncIterable[str]:
    """Consult with a more powerful reasoning model for complex questions."""
    history = self._input_event.history if self._input_event else []
    yield "Pondering your question deeply, will get back to you shortly"

    # Create event with full conversation history
    supervisor_event = UserTextSent(
        content=question,
        history=history + [UserTextSent(content=question)]
    )

    # Get response from supervisor
    full_response = ""
    async for output in self._supervisor.process(ctx.turn_env, supervisor_event):
        if isinstance(output, AgentSendText):
            full_response += output.text
    yield full_response
```

**Key points:**

- `is_background=True` allows yielding interim messages while processing
- First `yield` sends immediate feedback to the user
- Use `ctx.turn_env` when calling nested agent's `process()`
- Pass conversation history via `UserTextSent(content=..., history=[...])`
- Final `yield` returns the complete result

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

**Model selection strategy:** Use fast models (gpt-5-nano, claude-haiku, gemini-flash) for the main agent loop to minimize latency. Use powerful models (gpt-4o, claude-sonnet, claude-opus) inside background tools where latency is hidden.

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
| Missing `end_call` | Always include it |
| Raising exceptions | Return error string |
| Missing `ctx: ToolEnv` | First param always |
| No `Annotated` descriptions | Add for all params |
| Slow model for main agent | Use fast model, offload to background |
| Missing `event` in handoff | Required final param |
| Blocking nested agent call | Use `is_background=True` |
| Forgetting conversation history | Pass `history` in `UserTextSent` |
| Not cleaning up nested agents | Call cleanup on all agents in `_cleanup()` |

## Documentation

Full SDK documentation: <https://docs.cartesia.ai/line/sdk/overview>
