# Cartesia Line SDK

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/cartesia-ai/line)

Build intelligent, low-latency voice agents with Line.

Line brings voice to your text agents with Cartesia's state-of-the-art speech models. We handle audio orchestration, deployment, and observability so you can focus on your agent's reasoning.

## Features

- **Real-time interruption support** — Handles audio interruptions and turn-taking out-of-the-box
- **Tool calling** — Connect to databases, APIs, and external services
- **Multi-agent handoffs** — Route conversations between specialized agents
- **Web search** — Built-in tool for real-time information lookup
- **100+ LLM providers** — Works with any LLM via LiteLLM
- **Instant deployment** — Build, deploy, and start talking in minutes

## Quick Start

**1. Clone and run an example:**

```bash
git clone https://github.com/cartesia-ai/line.git
cd line/examples/basic_chat
GEMINI_API_KEY=your-key uv run python main.py
```

**2. Or create from scratch:**

```bash
mkdir my-agent && cd my-agent
uv init && uv add cartesia-line
```

Create `main.py`:

```python
import os
from line.llm_agent import LlmAgent, LlmConfig, end_call
from line.voice_agent_app import VoiceAgentApp

async def get_agent(env, call_request):
    return LlmAgent(
        model="gemini/gemini-2.5-flash-preview-09-2025",
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
GEMINI_API_KEY=your-key uv run python main.py
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
        model="gemini/gemini-2.5-flash-preview-09-2025",
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
from line.llm_agent import LlmAgent, LlmConfig, end_call, knowledge_base, send_dtmf, transfer_call, voicemail, web_search

agent = LlmAgent(
    model="gemini/gemini-2.5-flash-preview-09-2025",
    tools=[end_call, send_dtmf, transfer_call, voicemail, web_search, knowledge_base],
    config=LlmConfig(...),
)
```

| Tool | What it does |
|------|--------------|
| `end_call` | Ends the call |
| `send_dtmf` | Presses phone buttons (0-9, *, #) |
| `transfer_call` | Transfers to a phone number (E.164 format) |
| `voicemail` | Call when you reach a voicemail: optionally leaves a message, then hangs up the call. Configure with `voicemail(message="…")` |
| `web_search` | Searches the web (native LLM search or DuckDuckGo fallback) |
| `knowledge_base` | Looks up information from the agent's knowledge base via natural-language query. Call `knowledge_base(filters={...}, top_k=10)` to pre-filter retrievals or override `top_k` |
| `http_server_tool` | Creates an HTTP tool from JSON schemas (see below) |

### HTTP Tools — Connect to HTTP APIs Without Code

`http_server_tool` creates a tool that makes HTTP requests when the LLM calls it. Define the request shape with JSON schemas — no custom tool function needed:

```python
from line.llm_agent import http_server_tool

create_ticket = http_server_tool(
    name="create_ticket",
    description="Creates a support ticket for the caller.",
    url="https://api.example.com/v1/{tenant_id}/tickets",
    method="POST",
    request_body_schema={
        "type": "object",
        "required": ["subject", "priority"],
        "properties": {
            "subject": {"type": "string", "description": "Short summary of the issue."},
            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
            # constant_value: hidden from the LLM, baked into every request
            "source": {"type": "string", "constant_value": "voice_agent"},
        },
    },
    # ${ENV_VAR} resolved from os.environ at build time
    auth={"Authorization": "Bearer ${SUPPORT_API_KEY}"},
)

agent = LlmAgent(tools=[create_ticket, end_call], ...)
```

The LLM sees `subject`, `priority`, and `tenant_id` (from the URL template). It never sees `source` — that's injected automatically. The `${SUPPORT_API_KEY}` is resolved from your environment when the tool is created.

**Query parameter tools** work the same way for GET requests:

```python
search_orders = http_server_tool(
    name="search_orders",
    description="Search orders by status.",
    url="https://api.example.com/orders",
    method="GET",
    query_params_schema={
        "type": "object",
        "required": ["status"],
        "properties": {
            "status": {"type": "string", "enum": ["pending", "shipped", "delivered"]},
            "api_key": {"type": "string", "constant_value": "pk_live_abc123"},
        },
    },
)
```

**Response format** — the LLM always receives structured JSON:

```json
{"ok": true,  "status": 201, "body": "{\"ticket_id\": \"TKT-001\"}"}
{"ok": false, "status": 500, "error": "Internal server error"}
{"ok": false, "status": null, "error": "Request timed out after 5.0s."}
```

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
    model="anthropic/claude-sonnet-4-5",
    tools=[end_call],
    config=LlmConfig(
        system_prompt="You speak only in Spanish.",
        introduction="¡Hola! ¿Cómo puedo ayudarte?",
    ),
)

main_agent = LlmAgent(
    model="gemini/gemini-2.5-flash-preview-09-2025",
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

| Type | How to create | Result goes to | Use for |
|------|---------------|----------------|---------|
| **Loopback** | `@loopback_tool` | Back to LLM | API calls, data lookup |
| **Passthrough** | `@passthrough_tool` | Directly to user | Deterministic actions |
| **Handoff** | `agent_as_handoff()` or `@handoff_tool` | Another agent | Multi-agent workflows |

### Long-Running Tools

By default, tool calls are terminated when the agent is interrupted (though any reasoning and tool call response values already produced are preserved for use in the next generation).

For tools that take a long time to complete, set `is_background=True`. The tool will continue running in the background until completion regardless of interruptions, then loop back to the LLM:

```python
from typing import Annotated
from line.llm_agent import loopback_tool

@loopback_tool(is_background=True)
async def search_database(ctx, query: Annotated[str, "Search query"]) -> str:
    """Search that may take a while."""
    results = await slow_database_search(query)
    return results
```

---

## Context Management

Control what the LLM sees in its conversation history using `agent.history.add_entry` and `agent.history.update`.

### Inject Context with `agent.history.add_entry`

Insert text into the LLM's conversation history. This is useful for injecting context for controlling exactly what the LLM sees from tool calls, or integrating information from external APIs.

```python
from line.llm_agent import LlmAgent, LlmConfig, loopback_tool

agent = LlmAgent(
    model="gemini/gemini-2.5-flash-preview-09-2025",
    api_key=os.getenv("GEMINI_API_KEY"),
    config=LlmConfig(system_prompt="You are a helpful assistant."),
)

# Inject context before the conversation starts
agent.history.add_entry("The customer's name is Alice and she has a premium account.")

# Or inject context from within a tool call
@loopback_tool
async def lookup_customer(ctx, customer_id: str) -> str:
    """Look up customer details."""
    customer = await db.get_customer(customer_id)
    # Inject rich context that persists across turns
    agent.history.add_entry(f"Customer profile: {customer.summary}")
    return f"Found customer {customer.name}"
```

Each entry defaults to a user message (`role="user"`). Pass `role="system"` to inject a system note instead. By default entries are appended at the end of history; pass the `before=` or `after=` anchor keyword (a `HistoryEvent` already in history) to insert relative to a specific event.

### Rewrite History with `agent.history.update`

`agent.history.update(events, *, start=None, end=None)` replaces a segment of history with a new list of `HistoryEvent` items. The optional `start` and `end` anchors (events already present in history) determine which segment is replaced:

- **Neither anchor** — `events` are prefixed before the existing history.
- **`start` only** — replaces from `start` through the end of history.
- **`end` only** — replaces from the beginning of history through `end` (inclusive).
- **Both anchors** — replaces the segment `[start..end]` inclusive.

```python
from line import CustomHistoryEntry

# Prefix the history with a reminder (neither anchor)
agent.history.update([CustomHistoryEntry(content="Remember: be concise and friendly.")])

# Replace everything from `marker` onward (start only)
agent.history.update(
    [CustomHistoryEntry(content="Conversation summarized.")],
    start=marker,
)

# Replace the inclusive segment between two known events (both anchors)
agent.history.update(
    [CustomHistoryEntry(content="(redacted)")],
    start=first_event,
    end=last_event,
)
```

`update` raises `ValueError` if an anchor is not found in the current history, or if `end` appears before `start`.

---

## Customize Your Agent's Implementation

### Wrap with Custom Logic

Implement the `Agent` protocol to add guardrails, logging, or preprocessing:

```python
from line.agent import TurnEnv
from line.events import InputEvent, OutputEvent, UserTurnEnded, AgentSendText
from line.llm_agent import LlmAgent, LlmConfig, end_call

class GuardedAgent:
    def __init__(self, inner_agent):
        self.inner = inner_agent
        self.blocked_words = ["competitor", "confidential"]

    async def process(self, env: TurnEnv, event: InputEvent):
        # Pre-process: check user input for blocked words
        if isinstance(event, UserTurnEnded):
            user_text = " ".join(
                item.content for item in event.content if hasattr(item, "content")
            )
            if any(word in user_text.lower() for word in self.blocked_words):
                yield AgentSendText(text="I can't discuss that topic.")
                return

        # Delegate to inner agent
        async for output in self.inner.process(env, event):
            yield output

async def get_agent(env, call_request):
    inner = LlmAgent(
        model="gemini/gemini-2.5-flash-preview-09-2025",
        tools=[end_call],
        config=LlmConfig(system_prompt="You are a helpful assistant."),
    )
    return GuardedAgent(inner)
```

## LLM Provider Support

Line leverages [LiteLLM](https://github.com/BerriAI/litellm) to support 100+ LLM providers. Pass any LiteLLM-compatible model string to `LlmAgent`:

| Provider | Model format |
|----------|--------------|
| **OpenAI** | `gpt-5-nano`, `gpt-5.2` |
| **Anthropic** | `anthropic/claude-haiku-4-5-20251001`, `anthropic/claude-sonnet-4-5` |
| **Google** | `gemini/gemini-2.5-flash-preview-09-2025` |

## Agent Examples

| Example | Description |
|---------|-------------|
| [Basic Chat](./examples/basic_chat) | Simple conversational agent |
| [Form Filler](./examples/form_filler) | Collect structured data |
| [Phone Transfer](./examples/transfer_phone_call) | IVR navigation & transfers |
| [Multi-Agent](./examples/transfer_agent) | Hand off between agents |
| [Echo Tool](./examples/echo) | Custom handoff tool |

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

## Acknowledgments

Line leverages the fantastic work by the maintainers of [LiteLLM](https://github.com/BerriAI/litellm). Their open-source library provides the unified LLM interface that makes it possible to support 100+ providers out of the box.

LiteLLM is licensed under the [MIT License](https://github.com/BerriAI/litellm/blob/main/LICENSE).
