# Web Search (Gemini Live)

Single text agent using Gemini Live API with web search capabilities. Agent searches the web first and responds with search results.

## Template Information

### Prerequisites

- [Cartesia account](https://play.cartesia.ai)
- [Google Gemini API key](https://aistudio.google.com/app/apikey)

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `PORT` | Server port | 8000 |

### Use Cases

Real-time research assistants, live fact-checking agents, current events chatbots, streaming information retrieval services.

### File Overview

```
├── main.py              # Entry point, server setup
├── chat.py              # Gemini Live search logic
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
   export GEMINI_API_KEY=your_api_key_here
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
   conda create -n web-search-gemini-live python=3.11 -y
   conda activate web-search-gemini-live
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
