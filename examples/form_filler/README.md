# Doctor Appointment Scheduling Agent

A voice agent that helps patients schedule doctor appointments by completing an intake form and booking available time slots.

## Overview

This example creates a doctor appointment scheduling agent that:
- Guides callers through an intake form (first name, last name, reason for visit, date of birth, phone)
- Checks availability and books appointment slots
- Uses natural, voice-friendly responses
- Gracefully ends calls when the user is done

## Setup

### Prerequisites

- [Gemini API key](https://aistudio.google.com/apikey)

### Environment Variables

```bash
export GEMINI_API_KEY=your-gemini-key
```

### Installation

```bash
cd examples/form_filler
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

The agent:
- **Intake form** – Asks one question at a time (first name, last name, reason for visit, date of birth, phone), confirms each answer, and supports editing previous answers.
- **Appointment scheduling** – After the form is complete, checks availability, lets the user pick a slot, and books using contact info from the form.
- **Call flow** – Starts with a greeting, collects intake data, offers to schedule, then summarizes and ends the call when the user is done.

## File Structure

```
form_filler/
├── main.py                  # Agent factory, system prompt, LlmAgent setup
├── intake_form.py           # IntakeForm class and form-filling tools
├── appointment_scheduler.py # AppointmentScheduler class and booking tools
├── cartesia.toml            # Server configuration
└── pyproject.toml           # Dependencies
```

## Key Concepts

### Per-Call State with Tool Factories

Each call gets its own `IntakeForm` and `AppointmentScheduler` instances. Tools are created bound to these instances using factory functions:

```python
async def get_agent(env: AgentEnv, call_request: CallRequest):
    form = IntakeForm()
    scheduler = AppointmentScheduler()

    intake_tools = create_intake_tools(form)
    scheduler_tools = create_scheduler_tools(scheduler, form)

    return LlmAgent(tools=[*intake_tools, *scheduler_tools], ...)
```

### Tool Decorators

This example uses two tool patterns:

- **`@loopback_tool`** – Result is sent back to the LLM for the next response. Used for `record_answer`, `list_answer`, `check_availability`, `select_appointment_slot`.

- **`@loopback_tool(is_background=True)`** – For longer operations. Yields interim status, then final result. Used for `book_appointment_and_submit_form`.

### Form-Driven Conversation

The `IntakeForm` tracks which question to ask next. Each `record_answer` call returns instructions for the agent (e.g., "spell back the name", "ask the next question"), guiding natural conversation flow.

## Example Flow

**Agent**: "Hi, this is Jane from UCSF Medical. I'm here to help you schedule an appointment. Who am I speaking with?"

**User**: "This is John Smith."

**Agent**: "Hi John! I have that as J-O-H-N S-M-I-T-H. What's the main reason for your visit today?"

**User**: "Annual checkup."

**Agent**: "Got it, annual checkup. And what's your date of birth?"

… (intake continues) …

**Agent**: "I have openings Tuesday morning and Thursday afternoon. Which works better?"

**User**: "Tuesday morning."

**Agent**: "Got it! I've selected 9:00 AM on Tuesday, February 12. Ready to confirm the booking using the contact info from the intake form."

**Agent**: "Your appointment is confirmed! You're scheduled for 9:00 AM on Tuesday, February 12. A confirmation will be sent to your phone. Thanks for calling—take care!"
