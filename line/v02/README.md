# Line SDK

Build production-ready voice AI agents with real-time speech capabilities.

Line SDK provides a unified interface for creating conversational voice agents powered by LLMs, with support for 100+ model providers via [LiteLLM](https://github.com/BerriAI/litellm), tool calling, multi-agent handoffs, and streaming responses.

## Get Started

**Quick start:** Clone the repo and run an example directly:

```bash
git clone https://github.com/cartesia-ai/line.git
cd line/line/v02/examples/basic_chat
ANTHROPIC_API_KEY=your-api-key uv run python main.py
```

**Or follow the steps below to start from scratch:**

### 1. Install uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that handles dependencies automatically.

```bash
pip install uv
```

### 2. Create a new project

```bash
mkdir my-voice-agent && cd my-voice-agent
```

Create a `pyproject.toml`:

```toml
[project]
name = "my-voice-agent"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "cartesia-line-v02",
    "cartesia-line",
]
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Install the CLI (for testing)

```bash
curl -fsSL https://cartesia.sh | sh
```

### 5. Set your API key

```bash
export ANTHROPIC_API_KEY=your-api-key
# Or use OPENAI_API_KEY, GEMINI_API_KEY, etc.
```

## Hello World

Create a `main.py` file:

```python
import os
from line.call_request import CallRequest
from line.v02.llm import LlmAgent, LlmConfig, end_call
from line.v02.voice_agent_app import AgentEnv, VoiceAgentApp

async def get_agent(env: AgentEnv, call_request: CallRequest):
    return LlmAgent(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
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
uv run python main.py
```

## Adding Tools

Tools let your agent perform actions and retrieve information. Use `@loopback_tool` when the result should go back to the LLM for processing:

```python
import os
from typing import Annotated
from line.call_request import CallRequest
from line.v02.llm import LlmAgent, LlmConfig, end_call, loopback_tool
from line.v02.voice_agent_app import AgentEnv, VoiceAgentApp

@loopback_tool
async def get_weather(
    ctx,
    city: Annotated[str, "The city to check weather for"]
):
    """Get the current weather for a city."""
    # In production, call a real weather API
    return f"72°F and sunny in {city}"

async def get_agent(env: AgentEnv, call_request: CallRequest):
    return LlmAgent(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        tools=[end_call, get_weather],
        config=LlmConfig(
            system_prompt="You are a helpful voice assistant that can check the weather.",
            introduction="Hello! I can help you check the weather. What city are you interested in?",
        ),
    )

app = VoiceAgentApp(get_agent=get_agent)
```

The SDK supports three tool types:

| Type | Decorator | Use Case |
|------|-----------|----------|
| **Loopback** | `@loopback_tool` | Information retrieval, API calls—results go back to the LLM |
| **Passthrough** | `@passthrough_tool` | Deterministic actions like `end_call`—results go directly to user |
| **Handoff** | `@handoff_tool` | Transfer control to another agent or handler |

## Passthrough Tools

Use `@passthrough_tool` when you want deterministic actions that bypass the LLM. The output goes directly to the user:

```python
import os
from typing import Annotated
from line.call_request import CallRequest
from line.v02.llm import LlmAgent, LlmConfig, end_call, passthrough_tool
from line.v02.events import AgentSendText, AgentTransferCall
from line.v02.voice_agent_app import AgentEnv, VoiceAgentApp

@passthrough_tool
async def transfer_to_support(
    ctx,
    reason: Annotated[str, "Why the customer needs support"]
):
    """Transfer the customer to the support line."""
    yield AgentSendText(text="Let me transfer you to our support team right away.")
    yield AgentTransferCall(phone_number="+18000000000")

async def get_agent(env: AgentEnv, call_request: CallRequest):
    return LlmAgent(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        tools=[end_call, transfer_to_support],
        config=LlmConfig(
            system_prompt="You are a helpful assistant. Transfer to support if the user has technical issues.",
            introduction="Hello! How can I help you today?",
        ),
    )

app = VoiceAgentApp(get_agent=get_agent)
```

When the LLM calls `transfer_to_support`, the message and transfer happen immediately without additional LLM processing.

## Agent Handoffs

Transfer control between specialized agents using `agent_as_handoff`:

```python
import os
from line.call_request import CallRequest
from line.v02.llm import LlmAgent, LlmConfig, agent_as_handoff, end_call
from line.v02.voice_agent_app import AgentEnv, VoiceAgentApp

async def get_agent(env: AgentEnv, call_request: CallRequest):
    # Create a Spanish-speaking agent
    spanish_agent = LlmAgent(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        tools=[end_call],
        config=LlmConfig(
            system_prompt="You are a helpful assistant. You speak only in Spanish.",
            introduction="¡Hola! ¿Cómo puedo ayudarte hoy?",
        ),
    )

    # Main agent with handoff capability
    return LlmAgent(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        tools=[
            end_call,
            agent_as_handoff(
                spanish_agent,
                handoff_message="Transferring you to our Spanish-speaking agent...",
                name="transfer_to_spanish",
                description="Transfer to a Spanish-speaking agent when the user wants to speak in Spanish.",
            ),
        ],
        config=LlmConfig(
            system_prompt="You are a helpful assistant. If the user wants to speak in Spanish, use the transfer_to_spanish tool.",
            introduction="Hello! How can I help you today?",
        ),
    )

app = VoiceAgentApp(get_agent=get_agent)
```

## Putting It All Together

Here's a complete customer service agent with multiple tool types:

```python
import os
from typing import Annotated
from line.call_request import CallRequest
from line.v02.llm import (
    LlmAgent,
    LlmConfig,
    agent_as_handoff,
    end_call,
    loopback_tool,
    passthrough_tool,
)
from line.v02.events import AgentSendText, AgentTransferCall
from line.v02.voice_agent_app import AgentEnv, VoiceAgentApp


@loopback_tool
async def get_order_status(ctx, order_id: Annotated[str, "The order ID to look up"]):
    """Look up the current status of a customer's order."""
    # In production, fetch from your database
    return f"Order {order_id} was delivered on January 5th"


@passthrough_tool
async def transfer_to_billing(ctx, reason: Annotated[str, "Why the customer needs billing support"]):
    """Transfer the customer to the billing department."""
    yield AgentSendText(text="Let me transfer you to our billing department.")
    yield AgentTransferCall(phone_number="+18005551234")


async def get_agent(env: AgentEnv, call_request: CallRequest):
    spanish_agent = LlmAgent(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        tools=[end_call],
        config=LlmConfig(
            system_prompt="You are a helpful customer service agent. You speak only in Spanish.",
            introduction="¡Hola! ¿Cómo puedo ayudarte hoy?",
        ),
    )

    return LlmAgent(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        tools=[
            end_call,
            get_order_status,
            transfer_to_billing,
            agent_as_handoff(
                spanish_agent,
                handoff_message="Un momento por favor...",
                name="transfer_to_spanish",
                description="Transfer to a Spanish-speaking agent when requested.",
            ),
        ],
        config=LlmConfig(
            system_prompt=(
                "You are a helpful customer service agent. "
                "You can look up order statuses, transfer to billing, "
                "and transfer to a Spanish-speaking agent if requested."
            ),
            introduction="Hello! How can I help you today?",
        ),
    )


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    app.run()
```

**How it works:**

- **User:** "What's the status of order 12345?" → Calls `get_order_status`, LLM responds naturally
- **User:** "I need help with my bill" → Calls `transfer_to_billing`, immediately transfers
- **User:** "Can I speak in Spanish?" → Hands off to `spanish_agent`

## LLM Provider Support

Line SDK supports 100+ LLM providers through [LiteLLM](https://github.com/BerriAI/litellm):

| Provider | Model Examples |
|----------|----------------|
| **Anthropic** | `claude-sonnet-4-20250514`, `claude-opus-4-20250514` |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `o1` |
| **Google** | `gemini/gemini-2.0-flash`, `gemini/gemini-1.5-pro` |
| **Meta** | `together_ai/meta-llama/Llama-3.1-70B` |
| **Mistral** | `mistral/mistral-large-latest` |

Plus AWS Bedrock, Azure OpenAI, Vertex AI, Groq, Together AI, Replicate, Ollama (local), and [many more](https://docs.litellm.ai/docs/providers).

## Examples

| Example | Description |
|---------|-------------|
| [Basic Chat](./examples/basic_chat) | Simple conversational agent |
| [Form Filler](./examples/form_filler) | Collect structured information |
| [Phone Transfer](./examples/transfer_phone_call) | IVR navigation & call transfers |
| [Guardrails Wrapper](./examples/guardrails_wrapper) | Content filtering & safety |

### Example Integrations

| Integration | Description |
|-------------|-------------|
| [Exa Web Research](./example_integrations/exa) | Voice agent with real-time web search |
| [Browserbase](./example_integrations/browserbase) | Fill web forms via voice conversation |

## Documentation

- **[SDK Overview](https://docs.cartesia.ai/line/sdk/overview)** — Installation and quick start
- **[Build Your Voice Agent](https://docs.cartesia.ai/line/sdk/build-your-voice-agent)** — Prompting, tools, and handoffs
- **[Agents](https://docs.cartesia.ai/line/sdk/agents)** — LlmAgent, custom agents, CallRequest
- **[Tools](https://docs.cartesia.ai/line/sdk/tools)** — Loopback, passthrough, and handoff tools
- **[Events](https://docs.cartesia.ai/line/sdk/events)** — Input/output events and history

## Acknowledgements

Line SDK builds on these open-source projects:

- **[LiteLLM](https://github.com/BerriAI/litellm)** — Unified interface for 100+ LLM providers
- **[Pydantic](https://github.com/pydantic/pydantic)** — Data validation powering our configuration system

## Getting Help

- [Documentation](https://docs.cartesia.ai/line/introduction)
- [Discord Community](https://discord.gg/GExXcjM7)
- [Email Support](mailto:support@cartesia.ai)
