# Voice Research Agent with Parallel Free Search

A deployable Cartesia Line voice agent that searches the live web through the hosted [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp) endpoint.

## Prerequisites

- Python 3.10 or later
- An [OpenAI API key](https://platform.openai.com/api-keys) for the voice agent model
- A Cartesia Line project

Parallel Free Search does not require an API key.

## Environment Variables

Set only the model credential:

```bash
OPENAI_API_KEY=your-openai-key
```

Search objectives and search queries are sent to `https://search.parallel.ai/mcp` when the agent invokes `web_search`. Do not include private or sensitive information in requests. No URLs are fetched by this search-only integration.

## Installation and Running

```bash
uv sync
uv run python main.py
```

In another terminal, connect to the running Line agent:

```bash
cartesia chat 8000
```

The same directory can be connected to the Line platform through the [Cartesia GitHub integration](https://docs.cartesia.ai/line/integrations/github).

## File Overview

- `main.py` registers a `@loopback_tool` on `LlmAgent`, calls the hosted MCP `web_search` tool with its supported `objective` and `search_queries` fields, and returns at most five structured results to the model. Text content is used only when structured content is absent.
- `cartesia.toml` marks the directory as a Line deployment.
- `pyproject.toml` declares the Line SDK and official MCP client dependencies.

## Configuration

The agent keeps the standard OpenAI model used by the existing search examples. Change its model settings in `get_agent` if needed. The hosted search endpoint needs no Parallel credential and the integration adds no custom request headers.
