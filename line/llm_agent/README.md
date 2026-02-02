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

### With Custom Tools

```python
from typing import Annotated
from line.llm_agent import LlmAgent, LlmConfig, end_call, loopback_tool

# Define a custom tool
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

## Models (using LiteLLM)

| Provider | Format | Examples |
|----------|--------|----------|
| OpenAI | `model-name` | `gpt-5-nano`, `gpt-5.2` |
| Anthropic | `anthropic/model-name` | `anthropic/claude-haiku-4-5-20251001`, `anthropic/claude-sonnet-4-5` |
| Google | `gemini/model-name` | `gemini/gemini-2.5-flash-preview-09-2025`, `gemini/gemini-3.0-preview` |

See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for 100+ more providers.

## Tool Types

### Loopback Tools (Default)

Result is sent back to the LLM for continued generation. Use for information retrieval.

```python
from typing import Annotated
from line.llm_agent import loopback_tool

@loopback_tool
async def lookup_order(ctx, order_id: Annotated[str, "Order ID"]) -> str:
    """Look up order status."""
    order = await db.get_order(order_id)
    return f"Order {order_id}: {order.status}"
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

Built-in passthrough tools: `end_call`, `send_dtmf`, `transfer_call`

### Handoff Tools

Transfers control to another agent. After handoff, all future input events are processed by the
handoff tool, which routes them to the target agent. Use for multi-agent workflows.

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

## Tool Parameters

The SDK leverages the function docstring, and the `Annotated` parameters to tell the LLM how to use the tool and its params. *Note: The first argument to every tool call must be `env`*

```python
from typing import Annotated, Literal, Optional
from line.llm_agent import loopback_tool

@loopback_tool
async def search_products(
    ctx,
    query: Annotated[str, "Search query"],                                    # Required
    category: Annotated[Literal["electronics", "clothing", "home"], "Category"],  # Required with enum
    limit: Annotated[int, "Max results"] = 10,                                # Optional with default
    note: Annotated[Optional[str], "Optional note"],                          # Optional (None if not provided)
) -> str:
    """Search products."""
    ...
```

## Tool Context

Tools receive a `ToolEnv` object:

```python
from line.llm_agent import ToolEnv, loopback_tool

@loopback_tool
async def my_tool(ctx: ToolEnv, ...) -> str:
    # Access turn environment (session metadata)
    print(ctx.turn_env)
```

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

## Built-in Tools

The SDK provides ready-to-use tools:

```python
from line.v02.llm import end_call, send_dtmf, transfer_call, web_search

agent = LlmAgent(
    model="gpt-4o",
    tools=[end_call, send_dtmf, transfer_call, web_search],
    ...
)
```

- **`end_call`** - Ends the call. Note: this just ends the call without saying anything. If you want a goodbye message, instruct your agent to say goodbye before calling this tool.
- **`send_dtmf`** - Sends DTMF tones (for IVR navigation)
- **`transfer_call`** - Transfers to another phone number (E.164 format)
- **`web_search`** - Search the web (uses native LLM search or DuckDuckGo fallback)

## Testing

Run the integration test script to verify LLM provider connectivity:

```bash
# Set your API keys
export OPENAI_API_KEY=your-key
export ANTHROPIC_API_KEY=your-key
export GEMINI_API_KEY=your-key

# Run tests (will test whichever providers have keys set)
uv run python line/llm_agent/scripts/test_provider.py
```
