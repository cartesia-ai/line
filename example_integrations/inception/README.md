# Quick-Service Ordering Agent with Inception Mercury 2

A low-latency order-taking voice agent built on [Mercury 2](https://www.inceptionlabs.ai/blog/introducing-mercury-2), Inception Labs' diffusion large language model, and the Cartesia Line SDK.

Mercury 2 generates over 1,000 tokens per second by refining all tokens in parallel instead of decoding them one at a time. On a live voice call that cuts both time-to-first-token and end-to-end response latency: a ~300-token reasoning trace completes in under 300 ms, so the model can actually deliberate — tool selection, policy compliance, multi-step workflows — while staying inside the ~500 ms end-to-end LLM budget that natural conversation demands. This example leans into that with a use case where snappy turnaround matters: taking a coffee order with rapid back-and-forth and tool calls on nearly every turn.

The [Inception API](https://docs.inceptionlabs.ai/) is OpenAI-compatible, so Mercury 2 plugs into `LlmAgent` through the SDK's HTTP/LiteLLM backend — no extra dependencies, just a custom `api_base`:

```python
LlmAgent(
    model="openai/mercury-2",
    api_key=os.environ["INCEPTION_API_KEY"],
    config=LlmConfig(
        ...,
        extra={
            "api_base": "https://api.inceptionlabs.ai/v1",
            "extra_body": {"reasoning_effort": "medium", "realtime": True},
        },
    ),
)
```

## Setup

### Prerequisites

- An [Inception API key](https://platform.inceptionlabs.ai/) (new accounts include free tokens)
- A [Cartesia](https://play.cartesia.ai/agents) account and the Cartesia CLI:

```bash
curl -fsSL https://cartesia.sh | sh
```

### Environment Variables

Create a `.env` file:

```bash
INCEPTION_API_KEY=your-inception-key
```

### Installation

```bash
# uv (recommended)
uv sync
# or with pip
pip install -e .
```

## Running Locally

1) Start the agent:

```bash
uv run python main.py
```

2) In another terminal, chat with the agent in text-only mode:

```bash
cartesia chat 8000
```

Try: "Can I get a medium latte with oat milk and a croissant?" then "Actually make that a large" and "That's everything."

## File Overview

| File | Description |
|------|-------------|
| `main.py` | Agent, order tools, prompts, and Mercury 2 configuration |
| `pyproject.toml` | Project metadata and dependencies |
| `cartesia.toml` | Line platform deployment configuration |

## How It Works

Everything is in `main.py`:

1. **`OrderTools`** — a per-call class holding the in-progress order, exposing three `@loopback_tool` methods: `add_item`, `remove_item`, and `confirm_order`
2. **`get_agent`** — creates an `LlmAgent` pointed at Mercury 2 with the order tools plus the built-in `end_call`
3. **`VoiceAgentApp`** — handles the voice connection

## Mercury 2 Configuration

### Model Selection

The Inception API serves several models — see [Models, Endpoints, and Pricing](https://docs.inceptionlabs.ai/get-started/models) for the current list. This example uses `mercury-2`. Because the API is OpenAI-compatible, the model is addressed as `openai/mercury-2` with `api_base` pointing at `https://api.inceptionlabs.ai/v1`.

### Reasoning Effort

Mercury 2 is a reasoning model with a `reasoning_effort` control: `"instant"`, `"low"`, `"medium"`, or `"high"`. This example defaults to `"medium"`, [Inception's recommended setting for production voice agents](https://docs.inceptionlabs.ai/usecases/voice/quickstart). How to choose:

- **`instant`** trades intelligence for reflex speed — suitable for acknowledgments, backchannels, and turns that don't require tool calls.
- **`low`** already beats GPT 4.1 on instruction following at a fraction of the latency; pair it with `realtime` for the most TTFT-sensitive workflows.
- **`medium`** is the headline setting: it beats GPT 4.1 by 27 points on IFBench and 24 points on Tau3Bench Telecom while still decoding faster than GPT 4.1's non-reasoning baseline. Since this agent makes a tool call on nearly every turn, medium is the right default here.
- **`high`** spends the most time reasoning for highest intelligence; rarely needed on a live call.

The setting is passed through `extra_body` so LiteLLM forwards it verbatim, along with `realtime`, Inception's recommended low-latency serving mode for voice:

```python
extra={
    "api_base": "https://api.inceptionlabs.ai/v1",
    "extra_body": {"reasoning_effort": "medium", "realtime": True},
}
```

Note: use `extra_body` rather than the `LlmConfig.reasoning_effort` field — that field is validated against LiteLLM's model registry, which does not know Mercury's effort levels.

### Sampling

```python
LlmConfig(
    system_prompt=SYSTEM_PROMPT,
    introduction=INTRODUCTION,
    temperature=0.75,
    max_tokens=4096,
)
```

These follow the [voice quickstart](https://docs.inceptionlabs.ai/usecases/voice/quickstart) recommendations: `temperature=0.75` (the API default) suits most voice use cases — consider lowering toward `0.6` if tool routing needs to be extra consistent. Because Mercury is a reasoning model, its reasoning tokens count against `max_tokens`, so Inception recommends a budget of at least 3,000 at `reasoning_effort="medium"`.

### Prompting

The system prompt in `main.py` follows [Inception's prompt guide](https://docs.inceptionlabs.ai/resources/prompt-guide): persona and menu up top, few-shot tool-routing examples (including a negative example), and critical voice rules — markdown suppression, one question at a time — placed last, where Mercury weights context most heavily.

## Deploying to Cartesia

1) Initialize the agent:

```bash
cartesia init
```

2) Deploy:

```bash
cartesia deploy
```

3) Upload your API key:

```bash
cartesia env set INCEPTION_API_KEY=your-inception-key
```
