# Line SDK - Transfer Agent Example

## About This Example

This example demonstrates multi-agent handoff pattern using `agent_as_handoff()` for language-based routing between agents. The main agent can transfer to a Spanish-speaking agent when the user requests it.

> Line is Cartesia's open-source SDK for building real-time voice AI agents that connect any LLM to Cartesia's low-latency text-to-speech, enabling natural conversational experiences over phone calls and other voice interfaces.

## agent_as_handoff() Pattern

```python
from line.llm_agent import LlmAgent, LlmConfig, agent_as_handoff, end_call

# Create the target agent with its own system prompt
spanish_agent = LlmAgent(
    model="anthropic/claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    tools=[end_call],
    config=LlmConfig(
        system_prompt=(
            "Eres un asistente amable y servicial. "
            "Tenga una conversación natural con el usuario. "
            "Habla sólo en español."
        ),
        introduction="¡Hola! Soy tu asistente de IA. ¿Cómo puedo ayudarte hoy?",
    ),
)

# Main agent with handoff tool
main_agent = LlmAgent(
    model="anthropic/claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    tools=[
        end_call,
        agent_as_handoff(
            spanish_agent,
            handoff_message="Transferring you to our Spanish-speaking agent now...",
            name="transfer_to_spanish",
            description="Transfer the call to a Spanish-speaking agent. Use this when the user requests to speak in Spanish.",
        ),
    ],
    config=LlmConfig(
        system_prompt=(
            "You are a friendly and helpful assistant. "
            "Have a natural conversation with the user. "
            "If the user asks to speak in Spanish or requests a Spanish speaker, "
            "use the transfer_to_spanish tool."
        ),
        introduction="Hello! I'm your AI assistant. How can I help you today?",
    ),
)
```

## Multi-Agent Configuration

**Key concepts:**

1. **Define target agents separately** - Each has its own system prompt, tools, and introduction
2. **Wrap as tools using `agent_as_handoff()`** - Creates a handoff tool the LLM can invoke
3. **LLM decides when to invoke** - Based on system prompt guidance and conversation context

**agent_as_handoff parameters:**

| Parameter | Description |
|-----------|-------------|
| `agent` | The target LlmAgent to hand off to |
| `handoff_message` | Message spoken to user during transfer |
| `name` | Tool name the LLM will use |
| `description` | Tool description for LLM to understand when to use |

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
| Gemini | `gemini/gemini-2.5-flash-preview-09-2025` |

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
| Blocking nested agent call | Use `is_background=True` |
| Forgetting conversation history | Pass `history` in `UserTextSent` |
| Not cleaning up nested agents | Call cleanup on all agents in `_cleanup()` |
| Missing handoff_message | Include message for smooth user transition |
| Not limiting target agent tools | Target agent should have appropriate tools only |

## Documentation

Full SDK documentation: <https://docs.cartesia.ai/line/sdk/overview>
