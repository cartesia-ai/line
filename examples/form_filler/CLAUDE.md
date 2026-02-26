# Line SDK - Form Filler Example

## About This Example

A doctor appointment scheduling agent that demonstrates multi-step form filling and appointment booking. Shows how to manage per-call state, create tool factories, and guide conversation flow with tool return values.

> Line is Cartesia's open-source SDK for building real-time voice AI agents that connect any LLM to Cartesia's low-latency text-to-speech, enabling natural conversational experiences over phone calls and other voice interfaces.

## Architecture

```
get_agent()
    ├── IntakeForm (per-call state)
    │   └── create_intake_tools() → [record_answer, list_answer]
    ├── AppointmentScheduler (per-call state)
    │   └── create_scheduler_tools() → [check_availability, select_appointment_slot, book_appointment_and_submit_form]
    └── LlmAgent (combines all tools)
```

## Key Patterns Demonstrated

### 1. Per-Call State with Tool Factories

Each call creates fresh instances of `IntakeForm` and `AppointmentScheduler`. Tools are bound to these instances via factory functions:

```python
async def get_agent(env: AgentEnv, call_request: CallRequest):
    form = IntakeForm()
    form.start_form()
    scheduler = AppointmentScheduler()

    intake_tools = create_intake_tools(form)
    scheduler_tools = create_scheduler_tools(scheduler, form)

    return LlmAgent(
        model="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        tools=[*intake_tools, *scheduler_tools],
        config=LlmConfig(...),
    )
```

### 2. Loopback Tools

Results go back to the LLM for the next response:

```python
@loopback_tool
async def record_answer(
    ctx: ToolEnv,
    field_id: Annotated[str, "The ID of the field to record"],
    answer: Annotated[str, "The user's answer"],
) -> str:
    result = await form.record_answer(field_id, answer)
    return f"[{result.recorded_field}: {result.recorded_value}] Next: {result.next_question}"
```

### 3. Background Tools

For longer operations, use `is_background=True` to yield interim status:

```python
@loopback_tool(is_background=True)
async def book_appointment_and_submit_form(ctx: ToolEnv) -> str:
    contact = form.get_contact_info()
    if not contact:
        yield "Contact info missing..."
        return

    await form.submit_form()
    result = await scheduler.book_appointment(...)
    yield f"Appointment confirmed for {result['appointment']['time']}..."
```

### 4. Tool Return Values Guide Conversation

The `record_answer` tool returns instructions that guide agent behavior:

```python
# In FORM_FIELDS definition:
{
    "id": "first_name",
    "post_record_instructions": "Spell the user's answer back with dashes..."
}

# Tool returns these instructions:
return f"[{field}: {value}] {field.post_record_instructions} Next: {next_question}"
```

## Form Fields

Defined in `intake_form.py`:

| Field | Type | Post-Record Behavior |
|-------|------|---------------------|
| `first_name` | string | Spell back with dashes |
| `last_name` | string | Spell back with dashes |
| `reason_for_visit` | string | Repeat answer |
| `date_of_birth` | date | Repeat answer |
| `phone` | string | Repeat answer |

## Tools Reference

### Intake Tools (`intake_form.py`)

| Tool | Type | Purpose |
|------|------|---------|
| `record_answer` | loopback | Save form field, return next question |
| `list_answer` | loopback | Show all recorded answers |

### Scheduler Tools (`appointment_scheduler.py`)

| Tool | Type | Purpose |
|------|------|---------|
| `check_availability` | loopback | Get available slots (mocked) |
| `select_appointment_slot` | loopback | Select a slot by date/time |
| `book_appointment_and_submit_form` | background | Book slot and submit form |

## LlmAgent Configuration

```python
LlmAgent(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    tools=[*intake_tools, *scheduler_tools],
    config=LlmConfig(
        system_prompt=system_prompt,
        introduction="Hi, this is Jane from UCSF Medical...",
        max_tokens=250,
        temperature=0.7,
    ),
)
```

## Common Patterns

### Dynamic System Prompts

Inject current date/time into the system prompt:

```python
now = datetime.now(ZoneInfo("America/Los_Angeles"))
system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
    current_date=now.strftime("%A, %B %d, %Y"),
    current_time=now.strftime("%I:%M %p %Z"),
)
```

### Cross-Tool State Sharing

The scheduler needs access to form data for booking:

```python
def create_scheduler_tools(scheduler: AppointmentScheduler, form: IntakeForm):
    @loopback_tool(is_background=True)
    async def book_appointment_and_submit_form(ctx: ToolEnv):
        contact = form.get_contact_info()  # Access form state
        await scheduler.book_appointment(contact["first_name"], ...)
```

## Documentation

Full SDK documentation: https://docs.cartesia.ai/line/sdk/overview
