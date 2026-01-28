# Line SDK v0.2

Build production-ready voice AI agents with real-time speech capabilities.

Line SDK provides a unified interface for creating conversational voice agents powered by LLMs, with support for 100+ model providers via LiteLLM, tool calling, multi-agent handoffs, and streaming responses.

## Quick Start

Get up and running in minutes:

- [5-Minute Quick Start](https://example.com/quick-start) - Build your first voice agent
- [Installation Guide](https://example.com/installation) - Setup and dependencies
- [API Keys & Configuration](https://example.com/configuration) - Configure LLM providers

### Minimal Example

```python
import os
from line.v02.llm import LlmAgent, LlmConfig, end_call
from line.v02.voice_agent_app import VoiceAgentApp

async def get_agent(env, call_request):
    return LlmAgent(
        model="anthropic/claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        tools=[end_call],
        config=LlmConfig(
            system_prompt="You are a helpful voice assistant.",
            introduction="Hello! How can I help you today?",
        ),
    )

app = VoiceAgentApp(get_agent=get_agent)
app.run()
```

## Templates & Examples

Ready-to-use examples to jumpstart your project:

| Template | Description |
|----------|-------------|
| [Basic Chat](./examples/basic_chat) | Simple conversational agent |
| [Form Filler](./examples/form_filler) | Collect structured information |
| [Transfer Agent](./examples/transfer_agent) | Multi-agent handoffs |
| [Phone Transfer](./examples/transfer_phone_call) | IVR navigation & call transfers |
| [Guardrails Wrapper](./examples/guardrails_wrapper) | Content filtering & safety |

Browse more templates:
- [All Examples](https://example.com/examples)
- [Community Templates](https://example.com/community-templates)

## Example Integrations

Production-ready integrations with third-party services:

| Integration | Description | Services |
|-------------|-------------|----------|
| [Exa Web Research](./example_integrations/exa) | Voice agent with real-time web search | Exa API, OpenAI |
| [Browserbase Form Filler](./example_integrations/browserbase) | Fill web forms via voice conversation | Browserbase, Stagehand, Gemini |

These integrations demonstrate:
- Custom loopback tools for external APIs
- Passthrough tools for deterministic conversation flow
- Async processing patterns for non-blocking operations
- Agent wrapper patterns for extended functionality

## Documentation

### Core Concepts

| Topic | Description |
|-------|-------------|
| [Agents](https://example.com/docs/agents) | LlmAgent, custom agents, agent lifecycle |
| [Tools](https://example.com/docs/tools) | Loopback, passthrough, and handoff tools |
| [Events](https://example.com/docs/events) | Input/output events, history management |
| [Configuration](https://example.com/docs/configuration) | LlmConfig, system prompts, sampling |

### Guides

- [Tool Calling Deep Dive](https://example.com/guides/tool-calling) - Build custom tools
- [Multi-Agent Systems](https://example.com/guides/multi-agent) - Handoffs and orchestration
- [Streaming & Real-Time](https://example.com/guides/streaming) - Low-latency responses
- [Error Handling](https://example.com/guides/error-handling) - Resilience patterns
- [Testing Voice Agents](https://example.com/guides/testing) - Test strategies

### API Reference

- [LlmAgent](https://example.com/api/llm-agent)
- [LlmConfig](https://example.com/api/llm-config)
- [Tool Decorators](https://example.com/api/tool-decorators)
- [Events Reference](https://example.com/api/events)
- [VoiceAgentApp](https://example.com/api/voice-agent-app)

## Tips & Best Practices

### Voice-Optimized Prompts

Voice agents need different prompting than chat agents:

```python
LlmConfig(
    system_prompt="""You are a voice assistant. Keep responses brief (1-2 sentences).
    Speak naturally - no bullet points, URLs, or formatted text.
    Say goodbye before calling end_call.""",
)
```

See [Prompting for Voice](https://example.com/tips/voice-prompts) for more.

### Tool Design

- **Loopback tools** - For information retrieval (results go back to LLM)
- **Passthrough tools** - For deterministic actions (end_call, transfer)
- **Handoff tools** - For multi-agent workflows

See [Choosing Tool Types](https://example.com/tips/tool-types) for guidance.

### Performance

- Use fast models (e.g., `gemini/gemini-2.0-flash`) for low latency
- Keep system prompts concise
- Minimize tool call chains

See [Latency Optimization](https://example.com/tips/latency) for benchmarks.

### Common Patterns

- [Wrapper Pattern](https://example.com/tips/wrapper-pattern) - Pre/post processing
- [Graceful Degradation](https://example.com/tips/fallbacks) - Fallback models
- [Session State](https://example.com/tips/session-state) - Managing context

## LLM Provider Support

Line SDK supports 100+ LLM providers through [LiteLLM](https://github.com/BerriAI/litellm), providing a unified interface across all major model providers.

### Popular Models

| Provider | Models | Format |
|----------|--------|--------|
| **OpenAI** | GPT-4o, GPT-4o-mini, o1, o3-mini | `gpt-4o`, `o1` |
| **Anthropic** | Claude Opus 4, Claude Sonnet 4 | `anthropic/claude-sonnet-4-20250514` |
| **Google** | Gemini 2.0 Flash, Gemini Pro | `gemini/gemini-2.0-flash` |
| **Meta** | Llama 3.1, Llama 3.2 | `together_ai/meta-llama/Llama-3.1-70B` |
| **Mistral** | Mistral Large, Mixtral | `mistral/mistral-large-latest` |
| **Cohere** | Command R+, Command R | `cohere/command-r-plus` |

### Additional Providers

AWS Bedrock, Azure OpenAI, Vertex AI, Groq, Together AI, Replicate, Hugging Face, Ollama (local), and [many more](https://docs.litellm.ai/docs/providers).


## Project Structure

```
line/v02/
├── llm/                    # LLM integration layer
│   ├── llm_agent.py        # Main LlmAgent class
│   ├── tools.py            # Built-in tools (end_call, web_search, etc.)
│   ├── tool_types.py       # @loopback_tool, @passthrough_tool, @handoff_tool
│   └── config.py           # LlmConfig
├── events.py               # Event types (InputEvent, OutputEvent)
├── agent.py                # Base agent types
├── voice_agent_app.py      # VoiceAgentApp server
├── examples/               # Example implementations
└── example_integrations/   # Third-party service integrations
```

## Getting Help

- [Line Docs](https://docs.cartesia.ai/line/introduction)
- [Support Email](mailto:support@cartesia.ai)
- [Discord Community](https://discord.gg/GExXcjM7)
