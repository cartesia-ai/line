# Web Research Agent with Exa (v0.2 SDK)

A voice agent that searches the web using Exa API and synthesizes results into conversational responses.

## Setup

### Prerequisites

- [OpenAI API key](https://platform.openai.com/api-keys)
- [Exa API key](https://dashboard.exa.ai/api-keys)

### Environment Variables

Create a `.env` file:

```bash
OPENAI_API_KEY=your-openai-key
EXA_API_KEY=your-exa-key
```

### Installation

```bash
uv sync
```

## Running

```bash
python main.py
```

Then connect:

```bash
cartesia chat 8000
```

## How It Works

Everything is in `main.py`:

1. **`web_search`** - A `@loopback_tool` that calls Exa API and returns formatted results to the LLM
2. **`get_agent`** - Creates an `LlmAgent` with the web search tool and system prompt
3. **`VoiceAgentApp`** - Handles the voice connection

The tool uses `asyncio.to_thread` to run the synchronous Exa API without blocking.

## Configuration

### Exa Search Parameters

```python
client.search_and_contents(
    query,
    num_results=10,
    type="fast",
    livecrawl="never",
    text={"max_characters": 1000}
)
```

### LLM Configuration

```python
LlmConfig(
    system_prompt=SYSTEM_PROMPT,
    introduction=INTRODUCTION,
    max_tokens=300,
    temperature=0.7,
)
```
