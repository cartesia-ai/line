# LLM Agent Module for Line SDK

A unified interface for building LLM-powered voice agents with 100+ provider support via LiteLLM.

## Quick Start

```python
import os

from line.llm_agent import LlmAgent, LlmConfig, end_call
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

async def get_agent(env: AgentEnv, call_request: CallRequest):
    return LlmAgent(
        model="gemini/gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        tools=[end_call],
        config=LlmConfig.from_call_request(call_request),
    )

app = VoiceAgentApp(get_agent=get_agent)
app.run()
```

## Models (via LiteLLM)

| Provider | Model format |
|----------|--------------|
| **OpenAI** | `gpt-5-nano`, `gpt-5.2` |
| **Anthropic** | `anthropic/claude-haiku-4-5-20251001`, `anthropic/claude-sonnet-4-5` |
| **Google** | `gemini/gemini-2.5-flash-preview-09-2025`, `gemini/gemini-3.0-preview` |

See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for 100+ more providers.

## Configuration

```python
config = LlmConfig(
    # Behavior
    system_prompt="You are a helpful assistant.",
    introduction="Hello!",  # Sent on CallStarted

    # Sampling
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
    stop=["\n\n"],
    seed=42,

    # Penalties
    presence_penalty=0.1,
    frequency_penalty=0.1,

    # Resilience
    num_retries=3,
    fallbacks=["anthropic/claude-3-5-sonnet-20241022"],
    timeout=30.0,

    # Provider-specific options
    extra={"logprobs": True},
)
```

### Creating Config from CallRequest

Use `LlmConfig.from_call_request()` to automatically extract configuration from incoming call requests with sensible defaults:

```python
from line.llm_agent import LlmAgent, LlmConfig

async def get_agent(env, call_request):
    return LlmAgent(
        model="gemini/gemini-2.0-flash",
        tools=[...],
        config=LlmConfig.from_call_request(call_request),
    )
```

**Priority chain** (highest to lowest):
1. **CallRequest value** - If the API provides `system_prompt` or `introduction`
2. **User fallback** - Your app's custom fallbacks via `fallback_system_prompt` / `fallback_introduction`
3. **SDK default** - Built-in defaults (`FALLBACK_SYSTEM_PROMPT`, `FALLBACK_INTRODUCTION`)

```python
# Provide custom fallbacks for your app (used when CallRequest doesn't specify)
config = LlmConfig.from_call_request(
    call_request,
    fallback_system_prompt="You are a sales assistant for Acme Corp.",
    fallback_introduction="Hi! How can I help with your purchase today?",
    temperature=0.7,  # Additional LlmConfig options
)
```

**Empty string handling**:
- `system_prompt=""` is treated as None and falls back to defaults (a valid system prompt is always required)
- `introduction=""` is preserved (agent waits for user to speak first rather than using a default)

## Built-in Tools

The SDK provides commonly-used tools out of the box:

```python
from line.llm_agent import end_call, send_dtmf, transfer_call, web_search, agent_as_handoff
```

| Tool | Type | Description |
|------|------|-------------|
| `end_call` | passthrough | End the call |
| `send_dtmf` | passthrough | Send DTMF tones (0-9, *, #) |
| `transfer_call` | passthrough | Transfer to a phone number (E.164 format) |
| `web_search` | loopback | Web search with native LLM support or DuckDuckGo fallback |
| `agent_as_handoff` | handoff | Create a handoff tool from another agent |

### Web Search

Uses native LLM web search when available, falls back to DuckDuckGo:

```python
from line.llm_agent import LlmAgent, web_search

# Default settings
agent = LlmAgent(model="gpt-4o", tools=[web_search])

# Custom settings
agent = LlmAgent(model="gpt-4o", tools=[web_search(search_context_size="high")])
```

### Agent Handoffs

Use `agent_as_handoff` to create a tool that transfers control to another agent:

```python
from line.llm_agent import LlmAgent, LlmConfig, agent_as_handoff, end_call

spanish_agent = LlmAgent(
    model="gpt-4o",
    tools=[end_call],
    config=LlmConfig(
        system_prompt="You are a helpful assistant. Speak only in Spanish.",
        introduction="¡Hola! ¿Cómo puedo ayudarte hoy?",
    ),
)

main_agent = LlmAgent(
    model="gemini/gemini-2.0-flash",
    tools=[
        end_call,
        agent_as_handoff(
            spanish_agent,
            handoff_message="Transferring you to our Spanish-speaking agent...",
            name="transfer_to_spanish",
            description="Transfer to a Spanish-speaking agent when requested.",
        ),
    ],
    config=LlmConfig(
        system_prompt="You are a helpful assistant. If the user asks to speak in Spanish, use transfer_to_spanish.",
        introduction="Hello! How can I help you today?",
    ),
)
```

## Creating Custom Tools

Define custom tools using decorators. The SDK uses the function docstring and `Annotated` parameters to generate the tool schema for the LLM.

### Loopback Tools (Default)

Result is sent back to the LLM for continued generation. Use for information retrieval.

```python
from typing import Annotated
from line.llm_agent import LlmAgent, LlmConfig, end_call, loopback_tool

@loopback_tool
async def get_weather(ctx, city: Annotated[str, "City name"]) -> str:
    """Get weather for a city."""
    return f"72°F in {city}"

# Create agent with custom tool + built-in end_call
agent = LlmAgent(
    model="gpt-4o",
    tools=[get_weather, end_call],
    config=LlmConfig(
        system_prompt="You are a helpful assistant.",
        introduction="Hello! How can I help?",
    ),
)
```

### Passthrough Tools

Response bypasses the LLM and goes directly to the user. Use for deterministic actions.

```python
from typing import Annotated
from line.events import AgentSendText, AgentEndCall
from line.llm_agent import passthrough_tool

@passthrough_tool
async def goodbye_and_hang_up(ctx, message: Annotated[str, "Goodbye message"]):
    """Say a custom goodbye message and end the call."""
    yield AgentSendText(text=message)
    yield AgentEndCall()
```

### Handoff Tools

Transfers control to another agent. After handoff, all future input events are processed by the handoff tool, which routes them to the target agent. Use for multi-agent workflows.

```python
from typing import Annotated
from line.events import AgentHandedOff, AgentSendText
from line.llm_agent import handoff_tool

@handoff_tool
async def transfer_to_billing(
    ctx,
    reason: Annotated[str, "Reason for transfer"],
    event,  # Required: receives AgentHandedOff on first call, then subsequent InputEvents
):
    """Transfer to billing department."""
    # On initial handoff, send a message to the user
    if isinstance(event, AgentHandedOff):
        yield AgentSendText(text=f"Transferring you to billing for: {reason}")

    # Route all events (including initial) to the target agent
    async for output in billing_agent.process(ctx.turn_env, event):
        yield output
```

The `event` parameter is required for handoff tools:
- **First call**: `event` is `AgentHandedOff` - use this to send initial transfer messages
- **Subsequent calls**: `event` is the actual input event (e.g., `UserTextSent`) - route to target agent

### Tool Parameters

The SDK uses the function docstring and `Annotated` parameters to tell the LLM how to use the tool and its parameters.

> **Note:** The first argument to every tool must be `ctx`. This is currently unused but is a placeholder for forward compatibility.

```python
from typing import Annotated, Literal, Optional
from line.llm_agent import loopback_tool

@loopback_tool
async def search_products(
    ctx,
    query: Annotated[str, "Search query"],                                    # Required
    category: Annotated[Literal["electronics", "clothing", "home"], "Category"],  # Required enum
    limit: Annotated[int, "Max results"] = 10,                                # Optional with default
    note: Annotated[Optional[str], "Optional note"] = None,                   # Optional
) -> str:
    """Search products in the catalog."""
    ...
```

### Tool Context

Tools receive a `ToolEnv` object as their first argument (`ctx`):

```python
from line.llm_agent import ToolEnv, loopback_tool

@loopback_tool
async def my_tool(ctx: ToolEnv, ...) -> str:
    # Access turn environment (session metadata)
    print(ctx.turn_env)
```

### Long-Running Tools

By default, tool calls are terminated when the agent is interrupted. If you have a tool that takes a long time to complete, use `is_background=True` to keep it running:

```python
from typing import Annotated
from line.llm_agent import loopback_tool

@loopback_tool(is_background=True)
async def search_database(ctx, query: Annotated[str, "Search query"]) -> str:
    """Search the database - may take several seconds."""
    results = await slow_database_search(query)
    return format_results(results)
```

Background tools continue running even if the user interrupts. The result will be included in the next generation once complete.

## Events

**Input Events** (agent receives):
- `CallStarted` - Call initiated
- `UserTextSent` - User speech transcribed
- `UserTurnEnded` - User finished speaking
- `UserDtmfSent` - DTMF tone pressed
- `AgentHandedOff` - Passed to handoff tools on initial handoff

**Output Events** (agent yields):
- `AgentSendText` - Send text to user
- `AgentEndCall` - End the call
- `AgentTransferCall` - Transfer call
- `AgentSendDtmf` - Send DTMF tones
- `AgentToolCalled` - Tool was called
- `AgentToolReturned` - Tool returned result

## Customizing Your Agent's Implementation

While `LlmAgent` handles most use cases, you can create custom agents by wrapping it or implementing the agent protocol directly. This is useful for adding preprocessing, postprocessing, or custom routing logic.

### Wrapping LlmAgent

Create a wrapper class that delegates to `LlmAgent` while adding custom behavior:

```python
from typing import AsyncIterable

from line.agent import TurnEnv
from line.events import AgentSendText, InputEvent, OutputEvent, UserTurnEnded
from line.llm_agent import LlmAgent, LlmConfig, end_call


class BirdAgent:
    """Custom agent that adds special handling for bird-related messages."""

    def __init__(self):
        self._llm_agent = LlmAgent(
            model="gemini/gemini-2.0-flash",
            tools=[end_call],
            config=LlmConfig(
                system_prompt="You are a helpful assistant.",
                introduction="Hello! How can I help you today?",
            ),
        )

    async def process(self, env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        # Add custom preprocessing
        if isinstance(event, UserTurnEnded):
            user_text = " ".join(
                item.content for item in event.content if hasattr(item, "content")
            )
            if "bird" in user_text.lower():
                yield AgentSendText(text="Did you mention birds? I love birds!")

        # Delegate to the inner LLM agent
        async for output in self._llm_agent.process(env, event):
            yield output

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self._llm_agent.cleanup()
```

See the [guardrails_wrapper example](https://github.com/cartesia-ai/line/tree/main/v02/examples/guardrails_wrapper) for a complete implementation with content filtering.
