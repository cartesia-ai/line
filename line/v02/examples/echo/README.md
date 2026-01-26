# Echo Example

Demonstrates the **handoff tool** pattern - where control is transferred from the LLM to a custom function.

## Overview

This example creates a voice agent that:
- Starts as a normal conversational assistant
- When the user says "I'm ready to talk to myself", hands off to an echo function
- The echo function then repeats everything the user says with a configurable prefix

## Running the Example

```bash
cd v02/examples/echo
GEMINI_API_KEY=your-key uv run python main.py
```

## How It Works

### The Handoff Tool

The `echo` tool is decorated with `@handoff_tool`, which means:
1. When called, it takes over processing of all future events
2. The LLM is no longer involved after the handoff
3. The tool receives an `event` parameter for each subsequent user input

```python
@handoff_tool
async def echo(ctx: ToolEnv, prefix: Annotated[str, "A prefix to add..."], event):
    """Echo the user's message back to them with a prefix."""
    if isinstance(event, AgentHandedOff):
        # Called once when handoff occurs
        yield AgentSendText(text=f"Echo mode activated! I'll prefix everything with '{prefix}'")
        return

    if isinstance(event, UserTurnEnded):
        # Called for each user message after handoff
        for item in event.content:
            if isinstance(item, SpecificUserTextSent):
                yield AgentSendText(text=f"{prefix}: {item.content}")
```

### Tool Arguments

- **`ctx`**: Injected by the system - provides access to the turn environment
- **`prefix`**: Provided by the LLM when calling the tool - captured at handoff time
- **`event`**: Injected by the system - the current input event being processed

## Key Concepts

- **Handoff tools**: Transfer control from the LLM to custom logic
- **`AgentHandedOff`**: Event received when the handoff first occurs
- **`UserTurnEnded`**: Event received for subsequent user messages
- **Argument capture**: LLM-provided arguments (like `prefix`) are captured at handoff time and reused for all subsequent calls
