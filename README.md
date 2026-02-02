# Line SDK

Build production-ready, low-latency voice AI agents with real-time speech and tool calling.

## Quick Start

**1. Clone and run an example:**

```bash
git clone https://github.com/cartesia-ai/line.git
cd line/line/v02/examples/basic_chat
GEMINI_API_KEY=your-key uv run python main.py
```

**2. Or create from scratch:**

```bash
mkdir my-agent && cd my-agent
```

Create `pyproject.toml`:

```toml
[project]
name = "my-agent"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["cartesia-line"]
```

Create `main.py`:

```python
import os
from line.llm_agent import LlmAgent, LlmConfig, end_call
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

async def get_agent(env: AgentEnv, call_request: CallRequest):
    return LlmAgent(
        model="gemini/gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        tools=[end_call],
        config=LlmConfig(
            system_prompt="You are a helpful voice assistant.",
            introduction="Hello! How can I help you today?",
        ),
    )

app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    app.run()
```

Run it:

```bash
uv sync && uv run python main.py
```

**3. (Optional) Install the CLI to test locally:**

```bash
curl -fsSL https://cartesia.sh | sh
```

Then chat with your agent:

```bash
PORT=8000 uv run python main.py
cartesia chat 8000
```

See the [CLI documentation](https://docs.cartesia.ai/line/cli) for deployment and management commands.

---

## Customize Your Agent's Prompt

### System Prompt & Introduction

Configure your agent's personality and behavior via `LlmConfig`:

```python
config = LlmConfig(
    system_prompt="You are a customer service agent for Acme Corp. Be friendly and concise.",
    introduction="Hi! Thanks for calling Acme. How can I help?",
)
```

- **`system_prompt`** — Defines the agent's personality, rules, and context
- **`introduction`** — First message spoken when the call starts (set to `""` to wait for user)

### Dynamic Prompts from API

Use `LlmConfig.from_call_request()` to configure prompts dynamically from your API:

```python
async def get_agent(env: AgentEnv, call_request: CallRequest):
    # Prompts come from call_request.agent.system_prompt and call_request.agent.introduction
    # Falls back to your defaults if not provided
    return LlmAgent(
        model="gemini/gemini-2.0-flash",
        tools=[end_call],
        config=LlmConfig.from_call_request(
            call_request,
            fallback_system_prompt="You are a helpful assistant.",
            fallback_introduction="Hello! How can I help?",
        ),
    )
```

---

## Add Tools to Your Agent

### Built-in Tools

Ready-to-use tools for common actions:

```python
from line.llm_agent import LlmAgent, LlmConfig, end_call, send_dtmf, transfer_call, web_search

agent = LlmAgent(
    model="gemini/gemini-2.0-flash",
    tools=[end_call, send_dtmf, transfer_call, web_search],
    config=LlmConfig(...),
)
```

| Tool | What it does |
|------|--------------|
| `end_call` | Ends the call |
| `send_dtmf` | Presses phone buttons (0-9, *, #) |
| `transfer_call` | Transfers to a phone number (E.164 format) |
| `web_search` | Searches the web (native LLM search or DuckDuckGo fallback) |

### Loopback Tools — Fetch Data & Call APIs

Results go back to the LLM for a natural language response:

```python
from typing import Annotated
from line.llm_agent import loopback_tool

@loopback_tool
async def get_order_status(ctx, order_id: Annotated[str, "The order ID"]) -> str:
    """Look up order status."""
    order = await db.get_order(order_id)
    return f"Order {order_id}: {order.status}"

agent = LlmAgent(tools=[get_order_status, end_call], ...)
```

**User:** "What's the status of order 12345?"
**Agent:** Calls tool → LLM responds: "Your order was delivered on January 5th!"

### Passthrough Tools — Deterministic Actions

Output goes directly to the user, bypassing the LLM:

```python
from typing import Annotated
from line.events import AgentSendText, AgentTransferCall
from line.llm_agent import passthrough_tool

@passthrough_tool
async def transfer_to_support(ctx, reason: Annotated[str, "Why they need support"]):
    """Transfer to support team."""
    yield AgentSendText(text="Transferring you to support now.")
    yield AgentTransferCall(target_phone_number="+18005551234")

agent = LlmAgent(tools=[transfer_to_support, end_call], ...)
```

### Handoff Tools — Multi-Agent Workflows

Transfer control to a specialized agent:

```python
from line.llm_agent import LlmAgent, LlmConfig, agent_as_handoff, end_call

spanish_agent = LlmAgent(
    model="gpt-4o",
    tools=[end_call],
    config=LlmConfig(
        system_prompt="You speak only in Spanish.",
        introduction="¡Hola! ¿Cómo puedo ayudarte?",
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
            description="Transfer when user wants to speak Spanish.",
        ),
    ],
    config=LlmConfig(system_prompt="Transfer to Spanish if requested."),
)
```

### Tool Types Summary

| Type | Decorator | Result goes to | Use for |
|------|-----------|----------------|---------|
| **Loopback** | `@loopback_tool` | Back to LLM | API calls, data lookup |
| **Passthrough** | `@passthrough_tool` | Directly to user | Deterministic actions |
| **Handoff** | `@handoff_tool` | Another agent | Multi-agent workflows |

### Long-Running Tools

By default, tool calls are terminated when the agent is interrupted (though any reasoning and tool call response values already produced are preserved for use in the next generation).

For tools that take a long time to complete, set `is_background=True`. The tool will continue running in the background until completion regardless of interruptions, then loop back to the LLM:

```python
@loopback_tool(is_background=True)
async def search_database(ctx, query: Annotated[str, "Search query"]) -> str:
    """Search that may take a while."""
    results = await slow_database_search(query)
    return results
```

---

## Customize Your Agent's Implementation

### Wrap with Custom Logic

Implement the `Agent` protocol to add guardrails, logging, or preprocessing:

```python
from line.agent import TurnEnv
from line.events import InputEvent, UserTextSent, AgentSendText

class GuardedAgent:
    def __init__(self, inner_agent):
        self.inner = inner_agent
        self.blocked_words = ["competitor", "confidential"]

    async def process(self, env: TurnEnv, event: InputEvent):
        # Pre-process: check input
        if isinstance(event, UserTextSent):
            if any(word in event.content.lower() for word in self.blocked_words):
                yield AgentSendText(text="I can't discuss that topic.")
                return

        # Delegate to inner agent
        async for output in self.inner.process(env, event):
            # Post-process: modify or log outputs here
            yield output

async def get_agent(env, call_request):
    inner = LlmAgent(model="gemini/gemini-2.0-flash", tools=[end_call], ...)
    return GuardedAgent(inner)
```


## LLM Provider Support

100+ providers via [LiteLLM](https://docs.litellm.ai/docs/providers):

| Provider | Model format |
|----------|--------------|
| **OpenAI** | `gpt-5-nano`, `gpt-5.2` |
| **Anthropic** | `anthropic/claude-haiku-4-5-20251001`, `anthropic/claude-sonnet-4-5` |
| **Google** | `gemini/gemini-2.5-flash-preview-09-2025`, `gemini/gemini-3.0-preview` |
| **Meta (via Together)** | `together_ai/meta-llama/Llama-3.1-70B` |


## Agent Examples

| Example | Description |
|---------|-------------|
| [Basic Chat](./line/v02/examples/basic_chat) | Simple conversational agent |
| [Form Filler](./line/v02/examples/form_filler) | Collect structured data |
| [Phone Transfer](./line/v02/examples/transfer_phone_call) | IVR navigation & transfers |
| [Multi-Agent](./line/v02/examples/transfer_agent) | Hand off between agents |
| [Echo Tool](./line/v02/examples/echo) | Custom handoff tool |

### Integrations

| Integration | Description |
|-------------|-------------|
| [Exa Web Research](./example_integrations/exa) | Real-time web search |
| [Browserbase](./example_integrations/browserbase) | Fill web forms via voice |


## Documentation

- **[SDK Overview](https://docs.cartesia.ai/line/sdk/overview)** — Architecture and installation
- **[Tools Guide](https://docs.cartesia.ai/line/sdk/tools)** — Tool types in depth
- **[Agents Guide](https://docs.cartesia.ai/line/sdk/agents)** — LlmAgent, custom agents, conversation loop
- **[Events Reference](https://docs.cartesia.ai/line/sdk/events)** — Input/output events

## Getting Help

- [Full Documentation](https://docs.cartesia.ai/line/introduction)
- [Discord Community](https://discord.gg/GExXcjM7)
- [Email Support](mailto:support@cartesia.ai)
