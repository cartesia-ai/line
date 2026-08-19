# Web Research Agent with Keenable

A voice agent that searches the web and fetches page content using the [Keenable](https://keenable.ai) API, then synthesizes results into conversational responses. Keenable is a search API built for AI agents, **keyless by default**: the agent works with no API key against the public endpoints (rate-limited); set `KEENABLE_API_KEY` to lift the cap.

## Setup

### Prerequisites

- [OpenAI API key](https://platform.openai.com/api-keys)
- A Keenable API key is **optional** (keyless works out of the box); get one at [keenable.ai/console](https://keenable.ai/console) to lift the rate limit.

### Environment Variables

Create a `.env` file:

```bash
OPENAI_API_KEY=your-openai-key
# Optional — keyless by default:
# KEENABLE_API_KEY=keen_...
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

1. **`web_search`** - A `@loopback_tool` that calls Keenable Search and returns formatted results to the LLM
2. **`web_fetch`** - A `@loopback_tool` that fetches the full content of a webpage by URL as clean text, useful for deep-diving into a promising search result
3. **`get_agent`** - Creates an `LlmAgent` with both tools and a voice-optimized system prompt
4. **`VoiceAgentApp`** - Handles the voice connection

Both tools share a single `httpx.AsyncClient`. No provider SDK is required — the agent calls the Keenable HTTP API directly, selecting the public (keyless) or authenticated endpoint based on whether `KEENABLE_API_KEY` is set.

## Configuration

### Search

`web_search` posts to `POST /v1/search/public` (keyless) or `POST /v1/search` (with an `X-API-Key` header when `KEENABLE_API_KEY` is set):

```python
payload = {"query": query, "mode": "pro"}
# optional: payload["published_after"] = "2026-01-01"
```

Results are trimmed to the first 5 for voice-friendly latency. You can extend the tool with Keenable filters such as `site`, `published_after` / `published_before`, and `acquired_after` / `acquired_before`.

### Fetch

`web_fetch` calls `GET /v1/fetch/public?url=...` (keyless) or `GET /v1/fetch?url=...` (keyed) and returns the page's main content as markdown, truncated to 3000 characters to keep LLM context manageable (adjust `FETCH_MAX_CHARS` in `main.py`).

### LLM Configuration

```python
LlmConfig(
    system_prompt=SYSTEM_PROMPT,
    introduction=INTRODUCTION,
    max_tokens=600,
    temperature=0.7,
)
```
