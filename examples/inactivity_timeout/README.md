# Inactivity Timeout Example

This example demonstrates the **inactivity timeout** feature, which allows agents to detect when users haven't responded and re-prompt them.

## How It Works

1. Agent finishes speaking
2. A timer starts (configured via `inactivity_timeout_ms`)
3. If the user doesn't start speaking before the timer expires, an `InactivityTimeout` event fires
4. The agent receives this event and can respond appropriately (e.g., "Are you still there?")

## Configuration

Set the timeout in `AgentConfig`:

```python
# Via Calls API request
{
    "agent": {
        "inactivity_timeout_ms": 5000  // 5 seconds
    }
}

# Or in get_agent
async def get_agent(env, call_request):
    call_request.agent.inactivity_timeout_ms = 5000
    return MyAgent(...)
```

## Handling InactivityTimeout Events

The `InactivityTimeout` event is passed to your agent's `process()` method:

```python
from line.events import InactivityTimeout

async def process(self, env, event):
    if isinstance(event, InactivityTimeout):
        # User hasn't responded - re-prompt them
        yield AgentSendText(text="Are you still there?")
    else:
        # Normal event handling
        ...
```

With `LlmAgent`, you can handle it via the system prompt or wrap the agent:

```python
class InactivityAwareAgent:
    def __init__(self, llm_agent):
        self.llm_agent = llm_agent
        self.inactivity_count = 0

    async def process(self, env, event):
        if isinstance(event, InactivityTimeout):
            self.inactivity_count += 1
            # Track repeated timeouts, maybe end call after too many
        else:
            self.inactivity_count = 0  # Reset on user activity

        async for output in self.llm_agent.process(env, event):
            yield output
```

## Running the Example

```bash
ANTHROPIC_API_KEY=your-key uv run python main.py
```

The agent will:
- Greet the user
- Wait for responses with a 5-second timeout
- Re-prompt if the user is silent
- Gracefully handle repeated silence
