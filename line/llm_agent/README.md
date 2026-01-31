# LLM Agent Module for Line SDK

A unified interface for building LLM-powered voice agents with 100+ provider support via LiteLLM.

## Quick Start

```python
from typing import Annotated, Literal, Optional
from line.events import AgentSendText, AgentEndCall, AgentHandedOff
from line.llm_agent import (
    LlmAgent, LlmConfig, loopback_tool, passthrough_tool, handoff_tool,
)

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

## Model Naming (LiteLLM Format)

| Provider | Format | Examples |
|----------|--------|----------|
| OpenAI | `model-name` | `gpt-4o`, `gpt-4o-mini`, `o1` |
| Anthropic | `anthropic/model-name` | `anthropic/claude-3-5-sonnet-20241022` |
| Google | `gemini/model-name` | `gemini/gemini-2.0-flash` |

See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for 100+ more providers.

## Tool Types

### Loopback Tools (Default)

Result is sent back to the LLM for continued generation. Use for information retrieval.

```python
@loopback_tool
async def lookup_order(ctx, order_id: Annotated[str, "Order ID"]) -> str:
    """Look up order status."""
    order = await db.get_order(order_id)
    return f"Order {order_id}: {order.status}"
```

### Passthrough Tools

Response bypasses the LLM and goes directly to the user. Use for deterministic actions.

```python
@passthrough_tool
async def transfer_call(ctx, department: Annotated[str, "Department name"]):
    """Transfer to another department."""
    yield AgentSendText(text=f"Transferring to {department}...")
    yield AgentTransferCall(target_phone_number=DEPT_NUMBERS[department])
```

### Handoff Tools

Transfers control to another agent. After handoff, all future input events are processed by the
handoff tool, which routes them to the target agent. Use for multi-agent workflows.

```python
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

## Tool Parameters

Use `Annotated` with a string description to define parameters:

```python
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

## Streaming

Responses stream automatically. Tool calls arrive incrementally:

```python
async for output in agent.process(env, event):
    if isinstance(output, AgentSendText):
        print(output.text, end="", flush=True)
    elif isinstance(output, AgentToolCalled):
        print(f"\nCalling {output.tool_name}...")
    elif isinstance(output, AgentToolReturned):
        print(f"Result: {output.result}")
```

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
