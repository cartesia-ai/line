# Web Research Agent with SpecStream

A voice agent that searches the live web and extracts page content using the [SpecStream API](https://specstream.ai/docs/search-api), then synthesizes results into spoken responses. It uses SpecStream's `fast` search tier by default for low-latency voice calls, with optional `default` answer synthesis and `/v1/extract` for reading specific pages.

## Setup

### Prerequisites

- [OpenAI API key](https://platform.openai.com/api-keys)
- [SpecStream API key](https://specstream.ai/workspace/agents?tab=keys)

### Environment Variables

Create a `.env` file:

```bash
OPENAI_API_KEY=your-openai-key
SPECSTREAM_API_KEY=your-specstream-key
```

### Installation

```bash
cd example_integrations/specstream
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

## File Overview

| File | Purpose |
|------|---------|
| `main.py` | `SpecStreamTools` (`web_search`, `web_extract`), `get_agent`, and `VoiceAgentApp` setup |
| `pyproject.toml` | Package metadata and dependencies |
| `cartesia.toml` | Cartesia Line deployment marker |
| `README.md` | Setup, configuration, and API reference |

## How It Works

1. **`web_search`** (`@loopback_tool`): Calls `GET /v1/web-search` with `speed="fast"` and `num=5` (or `speed="default"` when answer synthesis is requested, and `news=true` for recent news) and returns formatted sources and any synthesized answer to the LLM.
2. **`web_extract`** (`@loopback_tool`): Calls `POST /v1/extract` with `format="markdown"` to extract the main text of a specific URL when snippets are not enough.
3. **`get_agent`**: Builds an `LlmAgent` with both tools, `end_call`, and a voice-formatted system prompt.
4. **`VoiceAgentApp`**: Serves the agent over the Cartesia Line runtime.

Both tools share a single `aiohttp.ClientSession` on `SpecStreamTools` for connection reuse.

## Configuration

### SpecStream Search Parameters

By default, `web_search` sends:

```text
GET https://api.specstream.ai/v1/web-search?q=<query>&speed=fast&num=5
```

- **`speed`**: `"fast"` returns search snippets in roughly one second (`$1` per 1,000 searches); `"default"` adds answer synthesis and structured rows (`$5` per 1,000 searches).
- **`news`**: Set to `true` to query recent news items.
- **`num`**: Number of sources returned (set to `5` in `main.py` via `SEARCH_NUM_RESULTS`).

See the [SpecStream Search API docs](https://specstream.ai/docs/search-api) for additional parameters such as `country`, `language`, `passages`, and `compact`.

### SpecStream Extract Parameters

`web_extract` sends:

```json
{"urls": ["<url>"], "format": "markdown"}
```

Extracted text is truncated to 3,000 characters (`EXTRACT_MAX_CHARS` in `main.py`) to keep voice context concise. See the [SpecStream Extract API docs](https://specstream.ai/docs/extract-api) for details on supported document types and output formats.

### LLM Configuration

```python
LlmConfig(
    system_prompt=SYSTEM_PROMPT_TEMPLATE.format(today=today),
    introduction=INTRODUCTION,
    max_tokens=600,
    temperature=0.7,
)
```
