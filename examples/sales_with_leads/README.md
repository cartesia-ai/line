# Sales with Leads Extraction and Research

A sales representative voice agent with stateful leads extraction and company research. Uses a three-agent architecture similar to `chat_supervisor`.

## Quick Start

```bash
ANTHROPIC_API_KEY=your-key uv run python main.py
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          SalesWithLeadsAgent                                 │
│                                                                              │
│  ┌─────────────────┐      ┌──────────────────────────────────────┐          │
│  │   LeadsState    │      │         Chat Agent                   │          │
│  │   (dataclass)   │      │      (claude-haiku-4-5-20251001)              │          │
│  │                 │      │                                      │          │
│  │  - name         │      │  Tools:                              │          │
│  │  - company      │      │  ├── extract_leads ──────────────────┼─────┐    │
│  │  - phone        │      │  ├── research_company ───────────────┼────┐│    │
│  │  - email        │      │  └── end_call                        │    ││    │
│  │  - interest     │      └──────────────────────────────────────┘    ││    │
│  │  - pain_points  │                                                  ││    │
│  │  - budget       │◀─────────────────────────────────────────────────┘│    │
│  │  - next_steps   │      ┌──────────────────────────────────────┐     │    │
│  │  - notes        │      │    Leads Extractor Agent             │     │    │
│  └─────────────────┘      │      (claude-haiku-4-5-20251001)              │◀────┘    │
│                           │                                      │          │
│  ┌─────────────────┐      │  Extracts structured lead info       │          │
│  │  Company Cache  │      │  and returns JSON                    │          │
│  │                 │      └──────────────────────────────────────┘          │
│  │ {company: info} │                                                        │
│  └─────────────────┘      ┌──────────────────────────────────────┐          │
│           ▲               │       Researcher Agent               │◀─────────┘
│           │               │      (claude-haiku-4-5-20251001)              │
│           └───────────────│                                      │
│                           │  Tools: [web_search]                 │
│                           └──────────────────────────────────────┘
└──────────────────────────────────────────────────────────────────────────────┘
```

## How It Works

This agent has three LLM agents:

1. **Chat Agent** (`_chatter`) - Handles the sales conversation (Haiku)
2. **Leads Extractor** (`_leads_extractor`) - Extracts structured lead info (Haiku)
3. **Researcher Agent** (`_researcher`) - Researches companies via web_search (Opus)

### Tools

- **extract_leads**: Delegates to `_leads_extractor` agent
  - Extracts lead info from conversation summary
  - Merges into `LeadsState` (accumulates, doesn't overwrite)
  - Returns current state + missing required fields

- **research_company**: Delegates to `_researcher` agent
  - Uses web_search to find company info, pain points, key people
  - Caches results to avoid duplicate research
  - Returns structured JSON + research summary

## Key Pattern

```python
class SalesWithLeadsAgent(AgentClass):
    def __init__(self):
        # Leads extraction agent
        self._leads_extractor = LlmAgent(
            model="anthropic/claude-haiku-4-5-20251001",
            config=LlmConfig(system_prompt=LEADS_EXTRACTION_PROMPT),
        )

        # Research agent (like _supervisor in chat_supervisor)
        self._researcher = LlmAgent(
            model="anthropic/cclaude-haiku-4-5-20251001",
            tools=[web_search],
            config=LlmConfig(system_prompt=RESEARCH_PROMPT),
        )

        # Main chat agent
        self._chatter = LlmAgent(
            model="anthropic/claude-haiku-4-5-20251001",
            tools=[self.extract_leads, self.research_company, end_call],
            config=LlmConfig(system_prompt=SALES_SYSTEM_PROMPT),
        )

    @loopback_tool
    async def extract_leads(self, ctx, conversation_summary):
        """Delegates to _leads_extractor agent."""
        async for output in self._leads_extractor.process(ctx.turn_env, request):
            ...

    @loopback_tool(is_background=True)
    async def research_company(self, ctx, company_name, contact_name=None):
        """Delegates to _researcher agent."""
        async for output in self._researcher.process(ctx.turn_env, request):
            ...
```

## Stateful Leads Accumulation

```python
@dataclass
class LeadsState:
    name: str = ""
    company: str = ""
    phone: str = ""
    email: str = ""
    interest_level: str = "unknown"
    pain_points: list[str] = field(default_factory=list)
    budget_mentioned: bool = False
    next_steps: str = ""
    notes: str = ""

    def merge(self, extracted: dict) -> list[str]:
        """Merge new data, accumulate pain_points and notes."""
```

## Research Output

The `research_company` tool returns structured JSON:

```json
{
  "company_overview": "1-2 sentence company description",
  "pain_points": ["Challenge 1", "Challenge 2"],
  "key_people": ["CEO Name", "CTO Name"],
  "sales_opportunities": ["Voice AI opportunity 1", "Opportunity 2"]
}
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key |
