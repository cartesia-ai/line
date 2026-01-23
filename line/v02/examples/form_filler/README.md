# Form Filler Example

Demonstrates a **loopback tool** pattern for collecting structured data via a YAML-defined form.

## Overview

This example creates a voice agent that:
- Loads form questions from a YAML file
- Guides the user through each question in sequence
- Validates answers based on question type
- Supports conditional questions (only shown based on previous answers)
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
