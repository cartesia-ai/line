# DEXA Scan Intake Agent

A voice agent that answers questions about DEXA scans using an embedded knowledge base and Exa web search.

## Overview

This example creates a DEXA scan customer support agent that:
- Answers common questions about DEXA scans from an embedded knowledge base
- Searches the web via Exa for information beyond its knowledge base
- Uses natural, voice-friendly responses without formatting
- Gracefully ends calls when the user is done

## Setup

### Prerequisites

- [Anthropic API key](https://console.anthropic.com/)
- [Exa API key](https://dashboard.exa.ai/api-keys)

### Environment Variables

```bash
export ANTHROPIC_API_KEY=your-anthropic-key
export EXA_API_KEY=your-exa-key
```

### Installation

```bash
cd examples/dexa_scan_intake
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

The agent has two main components:

1. **Knowledge Base** - A comprehensive DEXA FAQ embedded in the system prompt covering:
   - What DEXA is and how it works
   - What measurements DEXA provides
   - Accuracy and safety information
   - Preparation and what to expect
   - How often to scan
   - Who should get a DEXA scan

2. **`search_dexa_info`** - A `@loopback_tool` that searches the web via Exa when questions go beyond the knowledge base

## Configuration

### Exa Search Parameters

```python
client.search_and_contents(
    f"DEXA scan {query}",
    num_results=5,
    type="auto",
    text={"max_characters": 800}
)
```

### LLM Configuration

```python
LlmConfig(
    system_prompt=SYSTEM_PROMPT,
    introduction=INTRODUCTION,
    max_tokens=300,
    temperature=0.7,
)
```

## Example Conversations

**User**: "What exactly does a DEXA scan measure?"

**Agent**: "A DEXA scan measures your total body fat percentage and how it's distributed, your lean muscle mass broken down by body region, bone mineral density, and visceral fat around your organs. It also shows symmetry between your left and right sides."

**User**: "How much radiation is there?"

**Agent**: "DEXA uses very low radiation, about one tenth of a standard chest X-ray. A single scan is roughly 0.001 millisieverts, which is actually less than the natural background radiation you'd get in a typical day. It's considered very safe."

## Roadmap

Future features planned for this agent:
- Appointment scheduling assistance
- Intake form completion
