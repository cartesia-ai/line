# Web Research Agent with Exa and Cartesia (v0.2 SDK)

A real-time web research voice agent that combines Cartesia's voice capabilities with Exa's powerful web search API to provide accurate, up-to-date information through natural conversation.

This example uses the Line v0.2 SDK with the simplified `LlmAgent` and `@loopback_tool` pattern.

## Getting Started

### Prerequisites

- [Cartesia account](https://play.cartesia.ai/agents) and API key
- [OpenAI API key](https://platform.openai.com/api-keys) for conversation synthesis
- [Exa API key](https://dashboard.exa.ai/api-keys) for web search

### Environment Variables

Create a `.env` file in this directory with your API keys:

```bash
OPENAI_API_KEY=your-openai-key-here
EXA_API_KEY=your-exa-key-here
```

### Installation

```bash
# Install dependencies
pip install exa-py openai cartesia-line python-dotenv loguru

# Or using uv (recommended)
uv add exa-py openai cartesia-line python-dotenv loguru
```

## Running the Agent

```bash
# Run the agent
python main.py

# Or using uv
uv run python main.py
```

Then connect with the Cartesia chat client:

```bash
cartesia chat 8000
```

## Architecture

This example demonstrates the v0.2 SDK patterns:

### LlmAgent

The `LlmAgent` class provides a unified interface to 100+ LLM providers via LiteLLM. It handles:
- Conversation history management
- Tool calling with automatic result processing
- Streaming responses

### Loopback Tools

The Exa web search is implemented as a `@loopback_tool`. Loopback tools:
- Return results back to the LLM for further processing
- Are ideal for information retrieval where the LLM needs to synthesize the response
- Support async operations via `asyncio.to_thread`

### Files

| File | Description |
|------|-------------|
| `main.py` | Entry point - creates `VoiceAgentApp` with `LlmAgent` |
| `exa_utils.py` | Exa web search tool using `@loopback_tool` decorator |
| `config.py` | System prompts and configuration constants |

## Key Differences from v0.1 SDK

| v0.1 SDK | v0.2 SDK |
|----------|----------|
| Custom `ReasoningNode` subclass | `LlmAgent` with tools |
| Manual OpenAI client management | Automatic via LiteLLM |
| Manual tool schema definitions | `@loopback_tool` decorator with type hints |
| Bridge and event routing | Automatic event handling |

## Configuration

### Exa Search Parameters

The integration uses optimized search settings:

```python
results = exa.search_and_contents(
    query,
    num_results=10,
    type="fast",
    livecrawl="never",
    text={"max_characters": 1000}
)
```

### LLM Configuration

Configured via `LlmConfig`:

```python
LlmConfig(
    system_prompt=SYSTEM_PROMPT,
    introduction=INTRODUCTION,
    max_tokens=300,
    temperature=0.7,
)
```

## Deployment

Deploy to Cartesia's platform:

1. Add API keys to your Cartesia dashboard
2. Upload the integration files
3. Deploy and start talking to your research agent

Perfect for building voice-powered research tools, fact-checking assistants, and information discovery applications.
