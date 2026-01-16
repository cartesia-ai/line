# LLM Module for Line SDK

A unified interface for building LLM-powered voice agents with 100+ provider support via LiteLLM.

## Quick Start

```python
from typing import Annotated
from line.v02.llm import (
    LlmAgent, LlmConfig, loopback_tool, passthrough_tool, handoff_tool,
    Field, ToolContext, AgentSendText, AgentEndCall,
)

# Define tools
@loopback_tool
async def get_weather(
    ctx: ToolContext,
    city: Annotated[str, Field(description="City name")]
) -> str:
    """Get weather for a city."""
    return f"72°F in {city}"

@passthrough_tool
async def end_call(
    ctx: ToolContext,
    message: Annotated[str, Field(description="Goodbye message")]
):
    """End the call."""
    yield AgentSendText(text=message)
    yield AgentEndCall()

# Create agent
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
async def lookup_order(
    ctx: ToolContext,
    order_id: Annotated[str, Field(description="Order ID")]
) -> str:
    """Look up order status."""
    order = await db.get_order(order_id)
    return f"Order {order_id}: {order.status}"
```

### Passthrough Tools

Response bypasses the LLM and goes directly to the user. Use for deterministic actions.

```python
@passthrough_tool
async def transfer_call(
    ctx: ToolContext,
    department: Annotated[str, Field(description="Department name")]
):
    """Transfer to another department."""
    yield AgentSendText(text=f"Transferring to {department}...")
    yield AgentTransferCall(target_phone_number=DEPT_NUMBERS[department])
```

### Handoff Tools

Transfers control to another agent. Use for multi-agent workflows.

```python
@handoff_tool
async def transfer_to_billing(ctx: ToolContext):
    """Transfer to billing agent."""
    yield AgentSendText(text="Connecting you to billing...")
    yield BillingAgent()
```

## Tool Parameters

Use `Annotated` with `Field` to define parameters:

```python
@loopback_tool
async def search_products(
    ctx: ToolContext,
    query: Annotated[str, Field(description="Search query")],
    category: Annotated[str, Field(description="Category", enum=["electronics", "clothing", "home"])],
    limit: Annotated[int, Field(description="Max results", default=10)]
) -> str:
    """Search products."""
    # query is required, category is required with enum constraint, limit defaults to 10
    ...
```

## ToolContext

Tools receive a `ToolContext` with access to:

```python
@loopback_tool
async def my_tool(ctx: ToolContext, ...) -> str:
    # Conversation history
    for event in ctx.conversation_history:
        print(event)

    # Current tool call info
    print(ctx.tool_call_id, ctx.tool_name)

    # Turn environment (session metadata)
    print(ctx.turn_env)

    # LLM config (system prompt, etc.)
    print(ctx.config.system_prompt)
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

## Streaming

Responses stream automatically. Tool calls arrive incrementally:

```python
async for output in agent.process(env, event):
    if isinstance(output, AgentSendText):
        print(output.text, end="", flush=True)
    elif isinstance(output, ToolCallEvent):
        print(f"\nCalling {output.tool_name}...")
    elif isinstance(output, ToolResultEvent):
        print(f"Result: {output.result}")
```

## Events

**Input Events** (agent receives):
- `CallStarted` - Call initiated
- `UserTextSent` - User speech transcribed
- `UserTurnEnded` - User finished speaking
- `UserDtmfSent` - DTMF tone pressed

**Output Events** (agent yields):
- `AgentSendText` - Send text to user
- `AgentEndCall` - End the call
- `AgentTransferCall` - Transfer call
- `AgentSendDTMF` - Send DTMF tones

**Observability Events**:
- `ToolCallEvent` - Tool was called
- `ToolResultEvent` - Tool returned result
- `AgentHandoff` - Control transferred to another agent
