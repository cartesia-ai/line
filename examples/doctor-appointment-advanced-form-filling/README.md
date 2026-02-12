# Doctor Appointment Scheduling Agent

A voice agent that helps patients schedule doctor appointments by completing an intake form and booking available time slots.

## Overview

This example creates a doctor appointment scheduling agent that:
- Guides callers through an intake form (reason for visit, name, DOB, preferences, contact info)
- Checks availability and books appointment slots
- Uses natural, voice-friendly responses
- Gracefully ends calls when the user is done

## Setup

### Prerequisites

- [Anthropic API key](https://console.anthropic.com/)

### Environment Variables

```bash
export ANTHROPIC_API_KEY=your-anthropic-key
```

### Installation

```bash
cd examples/doctor-appointment-advanced-form-filling
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
- **Intake form** – Asks one question at a time (reason for visit, full name, date of birth, time preferences, email, phone), confirms each answer, and supports editing previous answers.
- **Appointment scheduling** – After the form is complete, checks availability, lets the user pick a slot, and books using contact info from the form.
- **Call flow** – Starts with a greeting, collects intake data, offers to schedule, then summarizes and ends the call when the user is done.

## Configuration

### LLM Configuration

```python
LlmConfig(
    system_prompt=SYSTEM_PROMPT,
    introduction=introduction,
    max_tokens=MAX_OUTPUT_TOKENS,
    temperature=TEMPERATURE,
)
```

## Example Flow

**Agent**: "Hi, this is Jane speaking from UCSF medical. I'd be happy to help schedule an appointment for you. I have a few questions for you. First, what's the main reason for your visit?"

**User**: "Annual checkup."

**Agent**: "Got it, annual checkup. And your full name?"

… (intake continues) …

**Agent**: "I have openings Tuesday morning and Thursday afternoon. Which works better?"

**User**: "Tuesday morning."

**Agent**: "Got it! I've selected 9:00 AM on Tuesday, February 12. Ready to confirm the booking using the contact info from the intake form."

**Agent**: "Your appointment is confirmed! You're scheduled for 9:00 AM on Tuesday, February 12. An email is on its way to you. Thanks for calling—take care!"
