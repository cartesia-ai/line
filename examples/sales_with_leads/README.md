# Sales with Leads Extraction and Research

A sales representative voice agent with stateful leads extraction and company research. Uses a two-tier architecture similar to `chat_supervisor`.

## Quick Start

```bash
ANTHROPIC_API_KEY=your-key uv run python main.py
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       SalesWithLeadsAgent                            │
│                                                                      │
│  ┌──────────────────┐                                                │
│  │   LeadsState     │                                                │
│  │   (dataclass)    │                                                │
│  │                  │                                                │
│  │  - name          │     ┌─────────────────────────────────────┐   │
│  │  - company       │     │           Chat Agent                │   │
│  │  - phone         │◀────│       (claude-haiku-4-5)            │   │
│  │  - email         │     │                                     │   │
│  │  - interest_level│     │  Tools:                             │   │
│  │  - pain_points   │     │  ├── extract_leads (background)     │   │
│  │  - budget        │     │  ├── research_company (background)  │   │
│  │  - next_steps    │     │  └── end_call                       │   │
│  │  - notes         │     └─────────────────────────────────────┘   │
│  └──────────────────┘                     │                          │
│                                           │ research_company         │
│  ┌──────────────────┐                     ▼                          │
│  │ Company Research │     ┌─────────────────────────────────────┐   │
│  │     Cache        │◀────│        Researcher Agent             │   │
│  │                  │     │       (claude-haiku-4-5)            │   │
│  │  {company: info} │     │                                     │   │
│  └──────────────────┘     │  Tools: [web_search]                │   │
│                           └─────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

## How It Works

Similar to `chat_supervisor`, this agent has two LLM agents:

1. **Chat Agent** (`_chatter`) - Handles the sales conversation
2. **Researcher Agent** (`_researcher`) - Handles company research via web_search

### Tools

- **extract_leads**: Called after each user response
  - Uses litellm to extract lead info from conversation
  - Merges into `LeadsState` (accumulates, doesn't overwrite)
  - Returns current state + missing required fields

- **research_company**: Triggers when company is identified
  - Delegates to `_researcher` agent with web_search
  - Finds company info, pain points, key people, opportunities
  - Caches results to avoid duplicate research
  - Returns structured JSON + research summary

## Key Pattern

```python
class SalesWithLeadsAgent(AgentClass):
    def __init__(self):
        # Researcher agent for company research (like _supervisor in chat_supervisor)
        self._researcher = LlmAgent(
            model="anthropic/claude-haiku-4-5",
            tools=[web_search],
            config=LlmConfig(system_prompt=RESEARCH_PROMPT),
        )

        # Main chat agent
        self._chatter = LlmAgent(
            model="anthropic/claude-haiku-4-5",
            tools=[self.extract_leads, self.research_company, end_call],
            config=LlmConfig(system_prompt=SALES_SYSTEM_PROMPT),
        )

    @loopback_tool(is_background=True)
    async def research_company(self, ctx, company_name, contact_name=None):
        """Delegates to _researcher agent."""
        async for output in self._researcher.process(ctx.turn_env, request):
            ...
        yield result
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
