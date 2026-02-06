# Line SDK - Text to Agent Examples

## About This Example

Template examples for text-to-agent feature showing provider-specific implementations (Gemini, Gemini Live, OpenAI, OpenAI Realtime) with basic chat and web search variants. Use these as starting points for building custom ReasoningNode-based agents.

> Line is Cartesia's open-source SDK for building real-time voice AI agents that connect any LLM to Cartesia's low-latency text-to-speech, enabling natural conversational experiences over phone calls and other voice interfaces.

## Directory Structure

```
text_to_agent/
├── basic_chat/
│   ├── gemini/          # Standard Gemini API
│   ├── gemini_live/     # Gemini Live API (persistent WebSocket)
│   ├── openai/          # OpenAI Responses API
│   └── openai_realtime/ # OpenAI Realtime WebSocket API
└── web_search/
    ├── gemini_live/     # Single agent with Google Search
    └── gemini_background/ # Multi-agent with background search
```

## ReasoningNode Pattern

All chat implementations extend `ReasoningNode`:

```python
from line import ConversationContext, ReasoningNode
from line.events import AgentResponse, EndCall

class ChatNode(ReasoningNode):
    def __init__(self, max_context_length: int = 100):
        self.system_prompt = get_chat_system_prompt()
        super().__init__(self.system_prompt, max_context_length)
        # Initialize provider-specific client

    async def process_context(
        self, context: ConversationContext
    ) -> AsyncGenerator[AgentResponse | EndCall, None]:
        """Process conversation and yield streaming responses."""
        messages = convert_messages_to_gemini(context.events)
        # ... stream from provider
        yield AgentResponse(content=chunk)
```

**Key features:**
- Conversation context management
- Tool handling
- Message history with configurable `max_context_length`

## Event Bridge Architecture

```python
from line import Bridge, VoiceAgentSystem
from line.events import UserStartedSpeaking, UserStoppedSpeaking, UserTranscriptionReceived

async def handle_new_call(system: VoiceAgentSystem, call_request: CallRequest):
    chat_node = ChatNode()
    chat_bridge = Bridge(chat_node)
    system.with_speaking_node(chat_node, chat_bridge)

    # Map transcription events to node
    chat_bridge.on(UserTranscriptionReceived).map(chat_node.add_event)

    # Stream generation with interruption handling
    (
        chat_bridge.on(UserStoppedSpeaking)
        .interrupt_on(UserStartedSpeaking, handler=chat_node.on_interrupt_generate)
        .stream(chat_node.generate)
        .broadcast()
    )

    await system.start()
    await system.send_initial_message("Hello! How can I help you today?")
    await system.wait_for_shutdown()
```

## Voice Optimization

```python
VOICE_RESTRICTION_PROMPT = """
### IMPORTANT: Voice/Phone Context
Your responses will be said out loud over the phone. Therefore:
- Do NOT use emojis or any special characters
- Do NOT use formatting like asterisks, newlines, bold, italics, bullet points, em-dash, etc.
- You are ONLY allowed to use alphanumeric characters, spaces, punctuation, and commas.
- Spell out all units, dates, years, and abbreviations
- Use as few words as possible to get your point across.
- Speak naturally as if you're having a phone conversation
"""
```

## Provider-Specific Patterns

### Gemini (Standard API)

```python
from google.genai import Client
from google.genai.types import GenerateContentConfig, ThinkingConfig

class ChatNode(ReasoningNode):
    def __init__(self):
        self.client = Client()
        self.generation_config = GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=0.7,
            thinking_config=ThinkingConfig(thinking_budget=0),
            tools=[EndCallTool.to_gemini_tool()],
        )

    async def process_context(self, context):
        messages = convert_messages_to_gemini(context.events)
        stream = await self.client.aio.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=messages,
            config=self.generation_config,
        )
        async for msg in stream:
            if msg.text:
                yield AgentResponse(content=msg.text)
```

### Gemini Live (Persistent WebSocket)

```python
import google.genai as genai
from google.genai.types import LiveConnectConfig

class ChatNode(ReasoningNode):
    def __init__(self):
        self.client = genai.Client(http_options={"api_version": "v1alpha"})
        self.live_config = LiveConnectConfig(
            system_instruction=self.system_prompt,
            temperature=0.7,
            response_modalities=["TEXT"],
            tools=[EndCallTool.to_gemini_tool()],
        )

    async def process_context(self, context):
        messages = convert_messages_to_gemini(context.events, text_events_only=True)
        async with self.client.aio.live.connect(
            model="gemini-live-2.5-flash-preview", config=self.live_config
        ) as stream:
            await stream.send_client_content(turns=messages, turn_complete=True)
            async for msg in stream.receive():
                if msg.text:
                    yield AgentResponse(content=msg.text)
```

### OpenAI Realtime (Manual WebSocket)

```python
import websockets

class ChatNode(ReasoningNode):
    async def _connect_websocket(self):
        url = f"wss://api.openai.com/v1/realtime?model={CHAT_MODEL_ID}"
        headers = [
            ("Authorization", f"Bearer {self.api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
        ]
        self.websocket = await websockets.connect(url, additional_headers=headers)
        await self._configure_session()

    async def _configure_session(self):
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "instructions": self.system_prompt,
                "tools": [EndCallTool.to_openai_tool()],
                "tool_choice": "auto",
                "temperature": 0.7,
            },
        }
        await self.websocket.send(json.dumps(session_config))

    async def _send_text_message(self, content: str):
        message = {
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": content}]},
        }
        await self.websocket.send(json.dumps(message))
        await self.websocket.send(json.dumps({"type": "response.create", "response": {"modalities": ["text"]}}))
```

## Web Search with Gemini Live

```python
self.live_config = LiveConnectConfig(
    system_instruction=self.system_prompt,
    tools=[{"google_search": {}}],  # Enable native Google Search
    response_modalities=["TEXT"],
)

# Parse search metadata from responses
def _parse_search_queries(self, msg: LiveServerMessage) -> set[str]:
    queries = set()
    if msg.server_content and msg.server_content.grounding_metadata:
        if msg.server_content.grounding_metadata.web_search_queries:
            queries.update(msg.server_content.grounding_metadata.web_search_queries)
    return queries
```

## LlmAgent Configuration

```python
import os
from line.llm_agent import LlmAgent, LlmConfig

agent = LlmAgent(
    model="anthropic/claude-haiku-4-5-20251001",  # LiteLLM format
    api_key=os.getenv("ANTHROPIC_API_KEY"),  # Must be explicitly provided
    tools=[...],
    config=LlmConfig(...),
    max_tool_iterations=10,
)
```

**LlmConfig options:**

- `system_prompt`, `introduction` - Agent behavior
- `temperature`, `max_tokens`, `top_p`, `stop`, `seed` - Sampling
- `presence_penalty`, `frequency_penalty` - Penalties
- `num_retries`, `fallbacks`, `timeout` - Resilience
- `extra` - Provider-specific pass-through (dict)

**Dynamic configuration via Calls API:**

The [Calls API](https://docs.cartesia.ai/line/integrations/calls-api) connects client-side audio (web/mobile apps or telephony) to your agent via WebSocket. When initiating a call, clients can pass agent configuration that your agent receives in `CallRequest`.

Use `LlmConfig.from_call_request()` to allow callers to customize agent behavior at runtime:

```python
async def get_agent(env: AgentEnv, call_request: CallRequest):
    return LlmAgent(
        model="anthropic/claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        tools=[end_call, web_search],
        config=LlmConfig.from_call_request(
            call_request,
            fallback_system_prompt=SYSTEM_PROMPT,
            fallback_introduction=INTRODUCTION,
        ),
    )
```

**How it works:**

- Callers can pass `system_prompt` and `introduction` when initiating a call
- Priority: Caller's value > your fallback > SDK default
- For `system_prompt`: empty string is treated as unset (uses fallback)
- For `introduction`: empty string IS preserved (agent waits for user to speak first)

**Use cases:**

- Multi-tenant apps: Different system prompts per customer
- A/B testing: Test different agent personalities
- Contextual customization: Pass user-specific context at call time

**Model ID formats (LiteLLM):**

| Provider | Format |
|----------|--------|
| OpenAI | `gpt-5-nano-2025-08-07`, `gpt-4o` |
| Anthropic | `anthropic/claude-sonnet-4-20250514`, `anthropic/claude-haiku-4-5-20251001` |
| Gemini | `gemini/gemini-2.5-flash-preview-09-2025` |

Full list of supported models: <https://models.litellm.ai/>

**Model selection strategy:** Use fast models (gpt-5-nano, claude-haiku, gemini-flash) for the main agent loop to minimize latency. Use powerful models (gpt-4o, claude-sonnet) inside background tools where latency is hidden.

## Tool Decorators

**Decision tree:**

```text
@loopback_tool           → Result goes to LLM
@loopback_tool(is_background=True) → Long-running, yields interim values
@passthrough_tool        → Yields OutputEvents directly
@handoff_tool            → Transfer to another handler
```

**Signatures:**

```python
from line.llm_agent import loopback_tool, passthrough_tool, handoff_tool, ToolEnv
from line import AgentSendText
from typing import Annotated

# Loopback - result sent to LLM
@loopback_tool
async def my_tool(ctx: ToolEnv, param: Annotated[str, "desc"]) -> str:
    return "result"

# Background - yields interim + final
@loopback_tool(is_background=True)
async def slow_tool(ctx: ToolEnv, query: Annotated[str, "desc"]):
    yield "Working..."
    yield await slow_work()

# Passthrough - yields OutputEvents
@passthrough_tool
async def direct_tool(ctx: ToolEnv, msg: Annotated[str, "desc"]):
    yield AgentSendText(text=msg)

# Handoff - requires event param
@handoff_tool
async def transfer(ctx: ToolEnv, reason: Annotated[str, "desc"], event):
    """Transfer to another agent."""
    async for output in other_agent.process(ctx.turn_env, event):
        yield output
```

**ToolEnv:** `ctx.turn_env` provides turn context (TurnEnv instance).

## Built-in Tools

```python
# Built-in tools
from line.llm_agent import end_call, send_dtmf, transfer_call, web_search, agent_as_handoff

agent = LlmAgent(
    tools=[
        end_call,
        agent_as_handoff(other_agent, name="transfer", description="Transfer to specialist"),
    ]
)
```

| Tool | Type | Purpose |
|------|------|---------|
| `end_call` | passthrough | End call gracefully |
| `send_dtmf` | passthrough | Send DTMF tone (0-9, *, #) |
| `transfer_call` | passthrough | Transfer to E.164 number |
| `web_search` | WebSearchTool | Real-time search (native or DuckDuckGo fallback) |
| `agent_as_handoff` | helper | Create handoff tool from an Agent (pass to tools list) |

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Missing `end_call` | Include it so agent can end calls (otherwise waits for user hangup) |
| Raising exceptions | Return error string. This will cause the agent to hang up the call. |
| Missing `ctx: ToolEnv` | First param always |
| No `Annotated` descriptions | Add for all params. This is used to describe the parameters of the tool to the LLM. |
| Slow model for main agent | Use fast model, offload to background |
| Missing `event` in handoff | Required final param |
| Blocking nested agent call | Use `is_background=True` |
| Forgetting conversation history | Pass `history` in `UserTextSent` |
| Not cleaning up nested agents | Call cleanup on all agents in `_cleanup()` |
| Not handling stream interruption | Use `.interrupt_on()` for user interruption |
| Formatting in voice output | Use VOICE_RESTRICTION_PROMPT to prevent special characters |

## Documentation

Full SDK documentation: <https://docs.cartesia.ai/line/sdk/overview>
