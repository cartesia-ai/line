# Line SDK - Sales with Leads Example

## About This Example

This example demonstrates a multi-node pipeline orchestrating parallel analysis (chat, leads extraction, research) with custom events for real-time sales intelligence. Three nodes run independently, each with its own Bridge, communicating via custom events.

> Line is Cartesia's open-source SDK for building real-time voice AI agents that connect any LLM to Cartesia's low-latency text-to-speech, enabling natural conversational experiences over phone calls and other voice interfaces.

## Multi-Node Pipeline Pattern

Three parallel nodes with separate Bridges for independent event handling:

```python
from line import Bridge, VoiceAgentSystem
from config import LeadsAnalysis, ResearchAnalysis

# Main conversation node
chat_node = ChatNode(gemini_client=chat_client)
conversation_bridge = Bridge(chat_node)
system.with_speaking_node(chat_node, conversation_bridge)

# Leads extraction node (parallel)
leads_node = LeadsExtractionNode(gemini_client=leads_client)
leads_bridge = Bridge(leads_node)
system.with_node(leads_node, leads_bridge)

# Research node (parallel)
research_node = ResearchNode(gemini_client=research_client)
research_bridge = Bridge(research_node)
system.with_node(research_node, research_bridge)
```

**Key insight:** Each node gets its own Gemini client to avoid interference.

## Event-Driven Orchestration

```python
# Chat node listens to user transcription and agent speech
conversation_bridge.on(UserTranscriptionReceived).map(chat_node.add_event)
conversation_bridge.on(AgentSpeechSent).map(chat_node.add_event)
(
    conversation_bridge.on(UserStoppedSpeaking)
    .interrupt_on(UserStartedSpeaking, handler=chat_node.on_interrupt_generate)
    .stream(chat_node.generate)
    .broadcast()
)

# Leads node triggers on user speech
leads_bridge.on(UserTranscriptionReceived).map(leads_node.add_event)
leads_bridge.on(UserStoppedSpeaking).stream(leads_node.generate).broadcast()

# Chat node receives leads analysis
conversation_bridge.on(LeadsAnalysis).map(chat_node.add_event)

# Research node triggers on leads analysis
research_bridge.on(LeadsAnalysis).map(research_node.add_event).stream(research_node.generate).broadcast()
conversation_bridge.on(ResearchAnalysis).map(chat_node.add_event)
```

## Custom Events

Define custom events using Pydantic BaseModel:

```python
from pydantic import BaseModel

class LeadsAnalysis(BaseModel):
    """Leads analysis results from conversation."""
    leads_info: dict
    confidence: str = "medium"
    timestamp: str

class ResearchAnalysis(BaseModel):
    """Research analysis results about a company/lead."""
    company_info: dict
    research_summary: str
    confidence: str = "medium"
    timestamp: str
```

## ReasoningNode Extension

All nodes extend `ReasoningNode` base class:

```python
from line.nodes.reasoning import ReasoningNode
from line.nodes.conversation_context import ConversationContext

class ChatNode(ReasoningNode):
    def __init__(self, gemini_client, model_id: str, max_context_length: int = 100):
        super().__init__(system_prompt=SYSTEM_PROMPT, max_context_length=max_context_length)
        self.client = gemini_client

    async def process_context(self, context: ConversationContext) -> AsyncGenerator[AgentResponse, None]:
        messages = convert_messages_to_gemini(context.get_committed_events(), handlers=EVENT_HANDLERS)
        # ... process with Gemini
```

## Event Handlers for Gemini Conversion

Register custom event handlers to convert custom events to Gemini format:

```python
from google.genai import types

def leads_analysis_handler(event: LeadsAnalysis) -> types.UserContent:
    leads_json = json.dumps(event.leads_info, indent=2)
    leads_message = f"[LEADS_ANALYSIS] {leads_json} [/LEADS_ANALYSIS]"
    return types.UserContent(parts=[types.Part.from_text(text=leads_message)])

def research_analysis_handler(event: ResearchAnalysis) -> types.UserContent:
    research_json = json.dumps(event.company_info, indent=2)
    research_message = f"[RESEARCH_ANALYSIS] {research_json} [/RESEARCH_ANALYSIS]"
    return types.UserContent(parts=[types.Part.from_text(text=research_message)])

EVENT_HANDLERS = {
    LeadsAnalysis: leads_analysis_handler,
    ResearchAnalysis: research_analysis_handler,
}

# Use in node
messages = convert_messages_to_gemini(context.events, handlers=EVENT_HANDLERS)
```

## Context Injection via Tagged Blocks

System prompt instructs agent how to use injected context:

```
The conversation context may contain special annotated blocks:
- `[LEADS_ANALYSIS] {...} [/LEADS_ANALYSIS]` - Extracted lead information
- `[RESEARCH_ANALYSIS] {...} [/RESEARCH_ANALYSIS]` - Researched company information

These are SYSTEM-GENERATED context. Do NOT acknowledge these blocks to the user.
Simply use the information naturally in your responses.
```

## Gemini Live API with Google Search

ResearchNode uses Live API for real-time Google Search:

```python
from google.genai import Client, types
from google.genai.types import LiveConnectConfig

class ResearchNode(ReasoningNode):
    def __init__(self, gemini_client, model_id: str = "gemini-live-2.5-flash-preview"):
        self.live_client = Client(http_options={"api_version": "v1alpha"})
        self.live_config = LiveConnectConfig(
            system_instruction=self.system_prompt,
            tools=[{"google_search": {}}],
            response_modalities=["TEXT"],
        )
        # Cache to prevent duplicate research
        self.previous_leads: dict[str, dict] = {}

    async def _perform_research(self, leads_info: dict):
        async with self.live_client.aio.live.connect(
            model=self.model_id, config=self.live_config
        ) as stream:
            await stream.send_client_content(turns=[search_content], turn_complete=True)
            async for msg in stream.receive():
                if msg.text:
                    research_summary += msg.text
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
| Gemini | `gemini/gemini-3-flash-preview` |

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
| Missing EVENT_HANDLERS | Register custom event handlers for Gemini conversion |
| Not caching research | Implement deduplication to avoid redundant searches |

## Documentation

Full SDK documentation: <https://docs.cartesia.ai/line/sdk/overview>
