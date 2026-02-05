# Basic Chat Example

A simple conversational voice agent that demonstrates the core Line SDK features.

## Overview

This example creates a basic voice agent that:
- Has a natural conversation with the user
- Accepts configurable system prompts and introductions via the call request
- Uses the `end_call` tool to gracefully end conversations

## Running the Example

```bash
cd examples/basic_chat
ANTHROPIC_API_KEY=your-key uv run python main.py
```

## How It Works

The agent is configured with:
- **Model**: Anthropic Claude Haiku 4.5
- **Tools**: `end_call` - allows the agent to end the call
- **Config**: System prompt and introduction from the call request, with defaults

```python
async def get_agent(env: AgentEnv, call_request: CallRequest):
    return LlmAgent(
        model="anthropic/claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        tools=[end_call],
        config=LlmConfig(
            system_prompt=call_request.agent.system_prompt or "You are a friendly...",
            introduction=call_request.agent.introduction or "Hello! I'm your AI assistant...",
        ),
    )
```

## Key Concepts

- **`VoiceAgentApp`**: The main application wrapper that handles WebSocket connections
- **`get_agent`**: Factory function called for each new call to create an agent instance
- **`LlmConfig`**: Configuration for the agent's behavior (system prompt, introduction)
- **`end_call`**: A passthrough tool that sends a goodbye message and ends the call
