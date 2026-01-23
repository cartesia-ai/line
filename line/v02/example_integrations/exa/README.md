# Web Research Agent with Exa and Cartesia (v0.2 SDK)

A real-time web research voice agent that combines Cartesia's voice capabilities with Exa's powerful web search API to provide accurate, up-to-date information through natural conversation.

This example uses the Line v0.2 SDK with the simplified `LlmAgent` and `@loopback_tool` pattern.

## Getting Started

### Prerequisites

- [Cartesia account](https://play.cartesia.ai/agents) and API key
- [OpenAI API key](https://platform.openai.com/api-keys) for conversation synthesis
- [Exa API key](https://dashboard.exa.ai/api-keys) for web search

### Environment Variables

Make sure to add these API keys to your `.env` file or Cartesia dashboard:

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for conversation | Yes |
| `EXA_API_KEY` | Exa API key for web search | Yes |

### Installation

```bash
# Using uv (recommended)
uv sync

# Or install dependencies manually
pip install exa-py python-dotenv
```

## Architecture

### Core Components

1. **LlmAgent** (`main.py`)
   - Unified interface to LLM providers via LiteLLM
   - Automatic conversation history management
   - Tool calling with automatic result processing
   - Streaming responses

2. **Web Search Tool** (`exa_tools.py`)
   - `@loopback_tool` decorator for automatic schema generation
   - Async Exa API calls via `asyncio.to_thread`
   - Formats search results for LLM synthesis

### Key Differences from v0.1 SDK

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

## Local Development

1. **Set up environment variables**:
   ```bash
   # Create .env with your API keys
   OPENAI_API_KEY=your-openai-key-here
   EXA_API_KEY=your-exa-key-here
   ```

2. **Run locally**:
   ```bash
   python main.py
   ```

3. **Test with voice**:
   ```bash
   cartesia chat 8000
   ```

## Deployment

Deploy to Cartesia's platform:

1. **Add API keys** to your Cartesia dashboard
2. **Upload the integration** files
3. **Deploy** and start talking to your research agent

Perfect for building voice-powered research tools, fact-checking assistants, and information discovery applications.
