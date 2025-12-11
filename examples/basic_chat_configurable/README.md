# Basic Chat

Single system prompt voice agent that uses Gemini.

## Template Information

### Prerequisites

- [Cartesia account](https://play.cartesia.ai)
- [Google Gemini API key](https://aistudio.google.com/app/apikey)

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `MODEL_ID` | Gemini model to use | gemini-2.5-flash |

### Use Cases

Customer service, personal assistants, educational tutoring, business receptionists.

### File Overview

```
├── main.py              # Entry point, event routing
├── chat_node.py         # Gemini chat node
├── config.py            # System prompt and model settings
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
   conda create -n basic-chat python=3.11 -y
   conda activate basic-chat
   pip install -e .
   PORT=8000 python main.py
   ```

4. Chat locally by running in a different terminal.
   ```zsh
   cartesia chat 8000
   ```

## Remote Deployment
Read the [Cartesia docs](https://docs.cartesia.ai/line/) to learn how to deploy templates to the Cartesia Line platform.

## Ending Calls
Our basic chat template implements ending the call using LLM Tool Calls. This means that the accuracy of when the agent will attempt to end the call will vary by your chosen LLM's ability to perform tool calls robustly. We accommodate for this by prompting the agent to ask for confirmation before ending the call, but note that the agent's performance will change if you sub out different LLM(s).
