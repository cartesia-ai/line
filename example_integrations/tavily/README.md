# Web Research Agent with Tavily

A voice agent that searches the web and extracts page content using the Tavily API, then synthesizes results into conversational responses. Uses Tavily's `fast` search depth for low-latency voice interactions and `extract` for deep-diving into specific pages.

## Setup

### Prerequisites

- [OpenAI API key](https://platform.openai.com/api-keys)
- [Tavily API key](https://app.tavily.com/home)

### Environment Variables

Create a `.env` file:

```bash
OPENAI_API_KEY=your-openai-key
TAVILY_API_KEY=your-tavily-key
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

1. **`web_search`** - A `@loopback_tool` that calls Tavily Search and returns formatted results to the LLM
2. **`web_extract`** - A `@loopback_tool` that extracts the full content of a webpage by URL, useful for deep-diving into a promising search result
3. **`get_agent`** - Creates an `LlmAgent` with both tools and a voice-optimized system prompt
4. **`VoiceAgentApp`** - Handles the voice connection

Both tools use Tavily's `AsyncTavilyClient`, which provides native async support and automatically reads `TAVILY_API_KEY` from the environment.

## Configuration

### Tavily Search Parameters

The `web_search` tool calls `AsyncTavilyClient.search()` with these defaults:

```python
response = await client.search(
    query=query,
    search_depth="fast",
    max_results=5,
)
```

#### Search Depth

| Depth | Latency | Content Type | Cost | Best For |
|-------|---------|--------------|------|----------|
| `ultra-fast` | Lowest | NLP summary per URL | 1 credit | Voice agents, real-time chat |
| `fast` | Low | Reranked chunks per URL | 1 credit | Chunk-based results with low latency |
| `basic` | Medium | NLP summary per URL | 1 credit | General-purpose search |
| `advanced` | Higher | Reranked chunks per URL | 2 credits | Precision-critical queries |

#### Additional Parameters

You can extend the `web_search` tool with Tavily features like:

- **`topic`** - `"general"`, `"news"`, or `"finance"` to focus results
- **`time_range`** - `"day"`, `"week"`, `"month"`, or `"year"` for recency filtering
- **`include_domains`** / **`exclude_domains`** - restrict or block specific sources
- **`include_answer`** - `"basic"` or `"advanced"` to get an LLM-generated answer alongside results
- **`country`** - boost results from a specific country (available for `"general"` topic)

See the [Tavily Search API docs](https://docs.tavily.com/documentation/api-reference/endpoint/search) and the [Python SDK reference](https://docs.tavily.com/sdk/python/reference) for the full parameter list.

### Tavily Extract Parameters

The `web_extract` tool calls `AsyncTavilyClient.extract()` with minimal defaults:

```python
response = await client.extract(urls=[url])
```

Extracted content is truncated to 3000 characters to keep LLM context manageable. You can adjust this in `main.py` or add parameters like:

- **`extract_depth`** - `"basic"` (default) or `"advanced"` for tables and embedded content
- **`format`** - `"markdown"` (default) or `"text"` for plain text

See the [Tavily Extract API docs](https://docs.tavily.com/documentation/api-reference/endpoint/extract) for more options.

### LLM Configuration

```python
LlmConfig(
    system_prompt=SYSTEM_PROMPT,
    introduction=INTRODUCTION,
    max_tokens=600,
    temperature=0.7,
)
```
