# Chat/Supervisor Example

A voice agent demonstrating a two-tier model architecture: a fast "chat" model (Claude Haiku) for conversational interactions, with access to a powerful "supervisor" (Claude Opus) for complex reasoning tasks.

## Overview

This example creates a voice agent that:
- Uses **Claude 4.5 Haiku** as the primary conversational model (fast, efficient)
- Can escalate to **Claude 4.5 Opus** via a tool call when facing difficult questions
- Passes the **full conversation history** to the supervisor for context
- Synthesizes the supervisor's deep analysis into natural spoken responses

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────┐
│      ChatSupervisorAgent            │
│  ┌─────────────┐                    │
│  │    Chat     │  (Claude Haiku)    │
│  │  LlmAgent   │                    │
│  └─────────────┘                    │
│        │                            │
│        │ Simple question? → Respond │
│        │                            │
│        │ Complex question?          │
│        │ ↓ ask_supervisor tool      │
│        │ (has access to full        │
│        │  conversation history)     │
│        ▼                            │
│  ┌─────────────┐                    │
│  │  Supervisor │  (Claude Opus)     │
│  │  LlmAgent   │                    │
│  └─────────────┘                    │
│        │                            │
│        │ Detailed analysis          │
│        ▼                            │
│  Chat synthesizes response          │
└─────────────────────────────────────┘
     │
     ▼
Spoken Response
```

## Key Implementation Details

The `ChatSupervisorAgent` class wraps both LlmAgents:
- Stores the current event to give the `ask_supervisor` tool access to conversation history
- The tool is a method on the class, allowing it to close over the agent's state
- Both chat and supervisor use `LlmAgent` for consistent behavior

## When the Supervisor is Consulted

The chat agent is instructed to use `ask_supervisor` for:
- Complex mathematical problems or proofs
- Multi-step logical reasoning puzzles
- Questions requiring deep domain expertise
- Philosophical or ethical dilemmas
- Any question the chat model is uncertain about

## Running the Example

```bash
cd line/v02/examples/chat_supervisor
ANTHROPIC_API_KEY=your-key uv run python main.py
```

## Example Conversations

**Simple question (Haiku responds directly):**
> User: "What's the capital of France?"
> Agent: "The capital of France is Paris!"

**Complex question (escalates to Opus with full context):**
> User: "Can you explain the proof of the Pythagorean theorem using similar triangles?"
> Agent: *calls ask_supervisor with full conversation history*
> Agent: "Great question! The proof using similar triangles works like this..."

## Cost Considerations

- Haiku is significantly cheaper and faster than Opus
- Opus is only invoked when truly needed for complex reasoning
- The full conversation history provides Opus with necessary context
- This architecture balances cost, latency, and quality
