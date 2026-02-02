# Basic Chat Example

A simple conversational voice agent that demonstrates the core Line SDK features.

## Overview

This example creates a basic voice agent that:
- Has a natural conversation with the user
- Accepts configurable system prompts and introductions via the call request
- Uses the `end_call` tool to gracefully end conversations

## Running the Example

```bash
cd v02/examples/basic_chat
GEMINI_API_KEY=your-key uv run python main.py
```

## How It Works

The agent is configured with:
- **Model**: Gemini 2.0 Flash (via LiteLLM)
- **Tools**: `end_call` - allows the agent to end the call
- **Config**: System prompt and introduction from the call request, with defaults

```python
async def get_agent(env: AgentEnv, call_request: CallRequest):
    return LlmAgent(
        model="gemini/gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        tools=[end_call],
        config=LlmConfig.from_call_request(call_request),
    )
```

The `LlmConfig.from_call_request()` helper automatically uses prompts from the call request with sensible defaults.

## Key Concepts

- **`VoiceAgentApp`**: The main application wrapper that handles WebSocket connections
- **`get_agent`**: Factory function called for each new call to create an agent instance
- **`LlmConfig`**: Configuration for the agent's behavior (system prompt, introduction)
- **`end_call`**: A passthrough tool that sends a goodbye message and ends the call
