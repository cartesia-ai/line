# Basic Chat (OpenAI Realtime)

Single system prompt text agent using OpenAI Realtime API.

## Template Information

### Prerequisites

- [Cartesia account](https://play.cartesia.ai)
- [OpenAI API key](https://platform.openai.com/api-keys) with access to Realtime API

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key with Realtime API access | - |
| `PORT` | Server port | 8000 |

### Use Cases

Real-time streaming text conversations, low-latency chat support, live customer service, interactive educational platforms.

### File Overview

```
├── main.py              # Entry point, server setup
├── chat.py              # OpenAI Realtime chat logic
├── config.py            # Agent prompt and settings
├── prompts.py           # Message templates
├── cartesia.toml        # Cartesia deployment config
├── pyproject.toml       # Python project dependencies
└── uv.lock              # Dependency lock file
```

## Local Setup

Install the Cartesia CLI.
```zsh
curl -fsSL https://cartesia.sh | sh
cartesia auth login
cartesia auth status
```

### Run the Example

1. Set up your environment variables.
   ```zsh
   export OPENAI_API_KEY=your_api_key_here
   ```

2. Install dependencies and run.

   **uv (recommended)**
   ```zsh
   PORT=8000 uv run python main.py
   ```

   **pip**
   ```zsh
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   PORT=8000 python main.py
   ```

   **conda**
   ```zsh
   conda create -n basic-chat-openai-realtime python=3.11 -y
   conda activate basic-chat-openai-realtime
   pip install -e .
   PORT=8000 python main.py
   ```

3. Access the agent at `http://localhost:8000`. Use the `/chats` endpoint to obtain a websocket URL.

## Remote Deployment

Read the [Cartesia docs](https://docs.cartesia.ai/line/) to learn how to deploy templates to the Cartesia Line platform.

## Customization

Update constants in `config.py`:
- **AGENT_PROMPT**: Define personality, role, and conversation style
- **LOCATION**: Set location context
- **INITIAL_MESSAGE**: First message sent to user (set to None for outbound agents)

## Features

Uses OpenAI's Realtime API in text-only mode with `gpt-4o-mini-realtime-preview-2024-12-17` model. Provides real-time streaming responses via WebSocket connection with automatic call ending detection.
