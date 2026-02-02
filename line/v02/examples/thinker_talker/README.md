# Thinker/Talker Example

A voice agent demonstrating a two-tier model architecture: a fast "talker" (Claude Haiku) for conversational interactions, with access to a powerful "thinker" (Claude Opus) for complex reasoning tasks.

## Overview

This example creates a voice agent that:
- Uses **Claude 4.5 Haiku** as the primary conversational model (fast, efficient)
- Can escalate to **Claude 4.5 Opus** via a tool call when facing difficult questions
- Passes the **full conversation history** to the thinker for context
- Synthesizes the thinker's deep analysis into natural spoken responses

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────┐
│      ThinkerTalkerAgent             │
│  ┌─────────────┐                    │
│  │   Talker    │  (Claude Haiku)    │
│  │  LlmAgent   │                    │
│  └─────────────┘                    │
│        │                            │
│        │ Simple question? → Respond │
│        │                            │
│        │ Complex question?          │
│        │ ↓ ask_thinker tool         │
│        │ (has access to full        │
│        │  conversation history)     │
│        ▼                            │
│  ┌─────────────┐                    │
│  │   Thinker   │  (Claude Opus)     │
│  │  LlmAgent   │                    │
│  └─────────────┘                    │
│        │                            │
│        │ Detailed analysis          │
│        ▼                            │
│  Talker synthesizes response        │
└─────────────────────────────────────┘
     │
     ▼
Spoken Response
```

## Key Implementation Details

The `ThinkerTalkerAgent` class wraps both LlmAgents:
- Stores the current event to give the `ask_thinker` tool access to conversation history
- The tool is a method on the class, allowing it to close over the agent's state
- Both talker and thinker use `LlmAgent` for consistent behavior

## When the Thinker is Consulted

The talker agent is instructed to use `ask_thinker` for:
- Complex mathematical problems or proofs
- Multi-step logical reasoning puzzles
- Questions requiring deep domain expertise
- Philosophical or ethical dilemmas
- Any question the talker is uncertain about

## Running the Example

```bash
cd line/v02/examples/thinker_talker
ANTHROPIC_API_KEY=your-key uv run python main.py
```

## Example Conversations

**Simple question (Haiku responds directly):**
> User: "What's the capital of France?"
> Agent: "The capital of France is Paris!"

**Complex question (escalates to Opus with full context):**
> User: "Can you explain the proof of the Pythagorean theorem using similar triangles?"
> Agent: *calls ask_thinker with full conversation history*
> Agent: "Great question! The proof using similar triangles works like this..."

## Cost Considerations

- Haiku is significantly cheaper and faster than Opus
- Opus is only invoked when truly needed for complex reasoning
- The full conversation history provides Opus with necessary context
- This architecture balances cost, latency, and quality
