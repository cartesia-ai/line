# Form Filler Example

Demonstrates a **loopback tool** pattern for collecting structured data via a YAML-defined form, with **DTMF input support** for phone numbers and dates.

## Overview

This example creates a voice agent that:
- Loads form questions from a YAML file
- Guides the user through each question in sequence
- Validates answers based on question type
- Supports conditional questions (only shown based on previous answers)
- **Accepts DTMF (touch-tone) input** for phone numbers and dates
- Summarizes collected data and asks for confirmation

## Running the Example

```bash
cd v02/examples/form_filler
GEMINI_API_KEY=your-key uv run python main.py
```

## How It Works

### FormFiller Class

The `FormFiller` class loads a YAML form definition and provides:
- A `record_answer` loopback tool for the LLM to call
- A `get_system_prompt()` method that includes form structure and the first question

```python
form = FormFiller("form.yaml", system_prompt="Your tone guidelines...")

agent = LlmAgent(
    tools=[form.tool, end_call],
    config=LlmConfig(
        system_prompt=form.get_system_prompt(),  # Includes form structure + current question
        introduction=f"Hi! I'm here to collect info. {form.get_current_question_text()}",
    ),
)
```

### System Prompt Generation

`get_system_prompt()` generates a comprehensive prompt that includes:
- Your custom system prompt (tone, style guidelines)
- The form title and all questions with types/options
- Clear instructions on how to conduct the form
- Current state (questions answered, current question to ask)

This ensures the LLM always knows:
- The full form structure
- What question to ask next
- How to use the `record_answer` tool

### The Loopback Tool

The `record_answer` tool returns comprehensive status after each answer:

```python
{
    "success": True,
    "completed": {"name": "John", "email": "john@example.com"},
    "remaining": ["phone", "contact_reason"],
    "next_question": "What is your phone number?",
    "is_complete": False
}
```

### YAML Form Definition

Forms support multiple question types and conditional logic:

```yaml
questionnaire:
  text: "Contact Information Form"
  questions:
    - id: "name"
      text: "What is your full name?"
      type: "string"
      required: true

    - id: "contact_reason"
      type: "select"
      options:
        - value: "sales"
          text: "Sales inquiry"
        - value: "support"
          text: "Technical support"

    - id: "budget"
      text: "What is your approximate budget?"
      type: "number"
      dependsOn:
        questionId: "contact_reason"
        value: "sales"
        operator: "equals"
```

### Supported Question Types

- `string`: Free text input
- `number`: Numeric input with optional min/max validation
- `boolean`: Yes/no questions
- `select`: Multiple choice from predefined options
- `date`: Date input

### Conditional Questions

Questions can depend on previous answers:

```yaml
dependsOn:
  questionId: "contact_reason"
  value: "sales"
  operator: "equals"  # equals, not_equals, in, not_in
```

## Key Concepts

- **Loopback tools**: Results are sent back to the LLM for continued processing
- **`get_system_prompt()`**: Generates a prompt that includes form structure and first question
- **Answer validation**: Each question type has appropriate validation
- **Conditional logic**: Questions can be shown/hidden based on previous answers
- **DTMF input wrapper**: Enables touch-tone input for specific field types

## DTMF Input Support

The `DtmfInputWrapper` enables users to enter phone numbers and dates using their phone's keypad instead of speaking them. This is useful when:
- Speech recognition has difficulty with phone numbers
- Users prefer touch-tone input for sensitive data
- Background noise makes speech unreliable

### How It Works

```
Voice Call → VoiceAgentApp → DtmfInputWrapper → LlmAgent
                                    ↓
                            Detects DTMF-eligible questions
                                    ↓
                            Prompts user about keypad option
                                    ↓
                            Collects DTMF digits (0-9)
                                    ↓
                            # pressed → converts to text
                                    ↓
                            Forwards to inner agent
```

### Using the Wrapper

```python
from dtmf_input_wrapper import DtmfInputWrapper
from line.v02.events import CallEnded, CallStarted, UserDtmfSent, UserTurnEnded, UserTurnStarted

# Event filters can be tuples of event types
# IMPORTANT: Include UserDtmfSent in run filter so the wrapper receives DTMF events
RUN_ON = (CallStarted, UserTurnEnded, CallEnded, UserDtmfSent)
CANCEL_ON = (UserTurnStarted,)

# Wrap your LlmAgent and return with filters
llm_agent = LlmAgent(...)
return (DtmfInputWrapper(llm_agent), RUN_ON, CANCEL_ON)
```

### DTMF Input Formats

**Phone Numbers:**
- User enters: `4155551234#`
- Converted to: "4 1 5 5 5 5 1 2 3 4" (spoken digits)

**Date of Birth (MMDDYYYY):**
- User enters: `01151990#`
- Converted to: "January 15, 1990"

### Configuration

The wrapper uses keyword detection to identify DTMF-eligible questions:

```python
from dtmf_input_wrapper import DtmfInputConfig, DtmfFieldConfig

config = DtmfInputConfig(
    dtmf_fields=[
        DtmfFieldConfig(
            field_id="phone",
            question_keywords=["phone number", "callback"],
            format_type="phone",
            min_digits=10,
            max_digits=15,
        ),
        DtmfFieldConfig(
            field_id="dob",
            question_keywords=["date of birth", "birthday"],
            format_type="date",
            min_digits=8,
            max_digits=8,
        ),
    ],
    dtmf_prompt="You can also enter this using your phone's keypad, then press pound when done.",
    termination_button="#",
)

agent = DtmfInputWrapper(llm_agent, config=config)
```

### YAML Field Annotation

You can annotate form fields with DTMF hints:

```yaml
- id: "callback_number"
  text: "What's the best phone number to reach you?"
  type: "string"
  dtmf_input:
    enabled: true
    format: "phone"
    hint: "Enter the digits of your phone number"

- id: "date_of_birth"
  text: "What is your date of birth?"
  type: "string"
  dtmf_input:
    enabled: true
    format: "MMDDYYYY"
    hint: "Enter as month, day, year"
```

### Input Behavior

- **# (pound)**: Submits the collected digits
- **Speech detected**: Clears DTMF buffer, uses speech instead
- **Turn ended with digits**: Submits if minimum digits met
