# Transfer Agent Example

A voice agent that demonstrates agent handoffs using the Line SDK.

## Overview

This example creates a voice agent that:
- Has a natural conversation with the user in English
- Transfers to a Spanish-speaking agent when requested
- Uses the `agent_as_handoff` utility to define transfer behavior

## Running the Example

```bash
cd examples/transfer_agent
ANTHROPIC_API_KEY=your-key uv run python main.py
```

## How It Works

The example creates two agents:
1. **Main Agent**: English-speaking, with a tool to transfer to Spanish
2. **Spanish Agent**: Handles conversations in Spanish

```python
spanish_agent = LlmAgent(
    model="anthropic/claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    tools=[end_call],
    config=LlmConfig(
        system_prompt="Eres un asistente amable y servicial...",
        introduction="¡Hola! Soy tu asistente de IA...",
    ),
)

return LlmAgent(
    model="anthropic/claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    tools=[
        end_call,
        agent_as_handoff(
            spanish_agent,
            handoff_message="Transferring you to our Spanish-speaking agent now...",
            name="transfer_to_spanish",
            description="Transfer the call to a Spanish-speaking agent.",
        ),
    ],
    ...
)
```

## Key Concepts

- **`agent_as_handoff`**: Wraps an agent as a tool that the main agent can invoke to transfer control
- **`handoff_message`**: Message spoken to the user during the transfer
- **`name`**: Tool name the LLM uses to invoke the transfer
- **`description`**: Helps the LLM understand when to use the transfer tool
