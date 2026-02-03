# Guardrails Wrapper Example

This example demonstrates how to create a wrapper around `LlmAgent` that preprocesses user inputs. It showcases the flexibility of the Line SDK by implementing content filtering and conversation guardrails.

## Use Case

A Cartesia AI assistant that:
- Answers questions about Cartesia, voice AI, and the competitive landscape
- Uses web search for up-to-date information
- Protects against toxic content, prompt injection, and off-topic conversations
- Ends the call after repeated policy violations

## Architecture

```
User Input (speech → text)
         ↓
┌────────────────────────────────────┐
│       GuardrailsWrapper            │
│                                    │
│  ┌─────────────────────────────┐   │
│  │ PREPROCESS                  │   │
│  │                             │   │
│  │ Guardrail LLM (batched)     │◄───── Anthropic Haiku
│  │    → toxicity check         │   │
│  │    → prompt injection check │   │
│  │    → off-topic check        │   │
│  └─────────────────────────────┘   │
│              ↓                     │
│     ┌────────────────────┐         │
│     │ Block?             │         │
│     │ • Toxic → respond  │         │
│     │ • Injection → warn │         │
│     │ • Off-topic → warn │         │
│     │ • 3 strikes → end  │         │
│     └────────────────────┘         │
│              ↓                     │
│  ┌─────────────────────────────┐   │
│  │     Inner LlmAgent          │◄───── Anthropic Haiku
│  │     (Cartesia assistant     │   │
│  │      with web search)       │   │
│  └─────────────────────────────┘   │
│                                    │
└────────────────────────────────────┘
         ↓
    Output (text → speech)
```

## Guardrail Checks

| Check | Method | Action on Violation |
|-------|--------|---------------------|
| **Toxicity** | LLM classification | Block, send warning message |
| **Prompt Injection** | LLM classification | Block, send warning message |
| **Off-Topic** | LLM classification | Block, redirect to allowed topics |

### Allowed Topics

The assistant will discuss:
- Cartesia AI products and technology
- Voice AI, TTS, speech synthesis
- AI/ML and software engineering
- Competitors (ElevenLabs, PlayHT, Amazon Polly, etc.)
- Voice AI market and landscape

Off-topic requests (recipes, sports, dating advice, etc.) are politely declined.

### Escalation

After 3 violations of any type, the agent ends the call:

> "It seems like you might have other things on your mind right now. Feel free to call back when you're ready to chat about Cartesia or voice AI. Have a great day!"

## Model Architecture

This example uses two different LLMs optimized for their roles:

| Role | Model | Why |
|------|-------|-----|
| **Inner Agent** | Anthropic Claude | Supports web search + function calling together |
| **Guardrail LLM** | Gemini Flash | Fast & cheap for classification (no tools needed) |

> **Note**: Gemini's standard API doesn't support combining Google Search with function calling in the same request. Anthropic Claude doesn't have this limitation.

## Running the Example

```bash
# Set your API keys
export ANTHROPIC_API_KEY=your-anthropic-key

# Run the example
uv run python main.py
```

## Configuration

The `GuardrailConfig` class provides full control over the wrapper behavior:

```python
GuardrailConfig(
    # What topics are allowed
    allowed_topics="Cartesia AI, voice AI, TTS, ...",

    # Model for guardrail classification (use a fast model)
    guardrail_model="anthropic/claude-haiku-4-5",
    guardrail_api_key=None,  # Uses env var if not set

    # Toggle individual checks
    block_toxicity=True,
    block_prompt_injection=True,
    enforce_topic=True,

    # Escalation threshold
    max_violations_before_end_call=3,

    # Custom response messages
    toxic_response="...",
    injection_response="...",
    off_topic_warning="...",
    end_call_message="...",
)
```

## How It Works

### 1. Preprocessing (Input)

When a `UserTurnEnded` event arrives:

1. **LLM Classification** (single batched call):
   - Sends user text to guardrail LLM
   - Returns JSON: `{toxic, prompt_injection, off_topic, reasoning}`
   - Uses temperature=0 for deterministic classification

2. **Violation Handling**:
   - Toxic/Injection: Block completely, return warning
   - Off-topic: Return redirect message
   - All violations increment counter toward escalation

### 2. Passthrough

Other events (`CallStarted`, `CallEnded`, tool events) pass through to the inner agent.

## Extending the Wrapper

This pattern can be adapted for other use cases:

```python
class MyCustomWrapper:
    def __init__(self, inner_agent: LlmAgent):
        self.inner_agent = inner_agent

    async def process(self, env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        # Preprocess
        event = self.preprocess(event)

        # Delegate to inner agent
        async for output in self.inner_agent.process(env, event):
            # Postprocess
            output = self.postprocess(output)
            yield output
```

### Other Wrapper Ideas

- **Translation**: Detect language, translate to English, process, translate back
- **RAG**: Inject retrieved context into the conversation
- **Analytics**: Log metrics, track conversation quality
- **A/B Testing**: Route to different models based on conditions
- **Rate Limiting**: Throttle requests per user/session

## Files

```
guardrails_wrapper/
├── README.md           # This file
├── main.py             # Entry point, Cartesia AI assistant
├── guardrails.py       # GuardrailConfig + GuardrailsWrapper
└── pyproject.toml      # Dependencies
```
