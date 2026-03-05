# Sales Development Rep Agent

A fun, engaging SDR (Sales Development Representative) voice agent for Cartesia that qualifies leads and captures contact information.

## Features

- **Hybrid Knowledge Base**: Core Cartesia info in the prompt, detailed product/feature info via tool
- **Lead Capture**: Tracks name, company, email, phone, role, interest, pain points, and timeline
- **Mock CRM Integration**: Leads are stored as JSON and submitted to a mock API (easily replaceable)
- **Guardrails**: In-prompt soft guardrails + wrapper that ends calls after 3 violations
- **Personality**: Fun, witty, and genuinely enthusiastic about voice AI

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SDRGuardrailsWrapper                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              SalesDevRepAgent                        │    │
│  │  ┌──────────────┐  ┌──────────────┐                 │    │
│  │  │  Chat Agent  │  │Lead Extractor│                 │    │
│  │  │(Gemini Flash)│  │(Gemini Flash)│                 │    │
│  │  └──────────────┘  └──────────────┘                 │    │
│  │         │                                            │    │
│  │         ▼                                            │    │
│  │  ┌──────────────────────────────────────────┐       │    │
│  │  │ Tools:                                    │       │    │
│  │  │ - extract_lead_info (after every turn)   │       │    │
│  │  │ - get_cartesia_info (detailed KB lookup) │       │    │
│  │  │ - submit_lead (CRM submission)           │       │    │
│  │  │ - end_call (graceful termination)        │       │    │
│  │  └──────────────────────────────────────────┘       │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│              Guardrail LLM (Gemini Flash)                   │
│              - Toxicity detection                            │
│              - Prompt injection detection                    │
│              - Off-topic detection                           │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Set your API key
export GEMINI_API_KEY=your-key

# Run the agent
uv run python main.py
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | Agent class, tools, mock CRM, app entry point |
| `prompts.py` | System prompt, knowledge bases, guardrail config |
| `state.py` | LeadState dataclass for tracking lead info |
| `guardrails.py` | SDRGuardrailsWrapper for content filtering |

## Knowledge Base

### Inline (always in context)
- Company overview
- Core products (Sonic, Ink, Line)
- Key differentiators
- Pricing
- Target industries

### Tool-based (on demand)
- Detailed product specs
- Technical specifications
- Enterprise capabilities
- Competitor comparisons
- Industry use cases

## Lead Capture

The agent tracks the following information:

```python
@dataclass
class LeadState:
    name: str           # Contact's full name
    company: str        # Company/organization
    email: str          # Email address
    phone: str          # Phone number
    role: str           # Job title/role
    interest_area: str  # What they're interested in
    interest_level: str # high/medium/low
    pain_points: list   # Challenges mentioned
    timeline: str       # Implementation timeline
    notes: str          # Additional observations
```

### Required fields for submission:
- Name
- Company
- Phone OR Email

## CRM Integration

The mock CRM client (`MockCRMClient`) shows the expected API format:

```python
{
    "lead": {
        "contact": {
            "name": "Alex Chen",
            "company": "TechStartup Inc",
            "email": "alex@techstartup.com",
            "phone": "",
            "role": "CTO"
        },
        "qualification": {
            "interest_area": "customer service voice agents",
            "interest_level": "high",
            "pain_points": ["high latency", "robotic voices"],
            "timeline": "1-3 months"
        },
        "metadata": {
            "notes": "Building customer support chatbot",
            "call_id": "call-123",
            "call_start_time": "2024-01-15T10:30:00Z",
            "submitted_at": "2024-01-15T10:45:00Z",
            "source": "sdr_voice_agent"
        }
    }
}
```

To integrate with a real CRM, replace `MockCRMClient` with your API client.

## Guardrails

### In-prompt (soft)
The system prompt instructs the agent to:
- Stay on topic (Cartesia, voice AI)
- Not bash competitors
- Redirect off-topic questions politely
- Respect user boundaries

### Wrapper (hard)
The `SDRGuardrailsWrapper` intercepts every user message and checks for:
- **Toxicity**: Profanity, harassment, threats
- **Prompt injection**: Attempts to manipulate the AI
- **Off-topic**: Content unrelated to allowed topics

After 3 violations, the call ends gracefully.

## Customization

### Change the personality
Edit `SYSTEM_PROMPT` in `prompts.py` to adjust tone, conversation flow, or discovery questions.

### Update the knowledge base
- Inline: Edit `CARTESIA_CORE_KNOWLEDGE` in `prompts.py`
- Detailed: Edit `DETAILED_KNOWLEDGE` dict in `prompts.py`

### Adjust guardrails
Edit `GuardrailConfig` defaults in `guardrails.py` or pass custom config in `main.py`.

### Change the model
Update `MODEL` constant in `main.py`. Supports any model via LiteLLM format:
- `gemini/gemini-2.5-flash-preview-09-2025`
- `anthropic/claude-haiku-4-5-20251001`
- `gpt-4o`

## Example Conversation

```
Sam: Hey there! This is Sam from Cartesia - we make voice AI that's so fast and
     natural, people forget they're talking to an AI. Who do I have the pleasure
     of speaking with?

User: Hey Sam! I'm Alex, I work at a startup called TechBot.

Sam: Nice to meet you, Alex! TechBot sounds interesting. What kind of stuff are
     you building over there?

User: We're building a customer support chatbot and looking into adding voice.

Sam: Oh nice! Voice is huge for support - customers hate waiting on hold. Are you
     handling calls in-house or thinking about automating with voice agents?

User: We want to automate. Our current solution sounds too robotic though.

Sam: Totally get that - nothing worse than a robot voice when you're frustrated!
     That's actually Cartesia's sweet spot. Our Sonic model sounds so natural it
     can laugh and express emotion. Plus we're 4x faster than alternatives - under
     90ms latency. What volume of calls are you looking at?

...
```

## Development

```bash
# Install dependencies
uv sync

# Run with debug logging
LOGURU_LEVEL=DEBUG GEMINI_API_KEY=your-key uv run python main.py
```
