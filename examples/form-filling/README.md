# Form Filling

Voice agent that conducts structured questionnaires using YAML configuration with conditional logic and response validation.

## Template Information

### Prerequisites

- [Cartesia account](https://play.cartesia.ai)
- [Google Gemini API key](https://aistudio.google.com/app/apikey)

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | - |

### Use Cases

Customer surveys, medical intake forms, registration processes, interview questionnaires, research data collection, onboarding workflows.

### File Overview

```
├── main.py                 # Entry point and system configuration
├── form_filling_node.py    # Core form-filling reasoning node
├── form_manager.py         # Form state and logic management
├── form_tools.py           # Gemini tools for answer recording
├── config.py               # System prompts and configuration
├── form.yaml               # Form question definitions
├── cartesia.toml           # Cartesia deployment config
├── requirements.txt        # Python dependencies (legacy)
└── pyproject.toml          # Python project dependencies (if present)
```

## Local Setup

Install the Cartesia CLI.
```zsh
curl -fsSL https://cartesia.sh | sh
cartesia auth login
cartesia auth status
```

### Run the Example

1. Set up your environment variables.
   ```zsh
   export GEMINI_API_KEY=your_api_key_here
   ```

2. Install dependencies and run.

   **uv (recommended)**
   ```zsh
   PORT=8000 uv run python main.py
   ```

   **pip**
   ```zsh
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   PORT=8000 python main.py
   ```

   **conda**
   ```zsh
   conda create -n form-filling python=3.11 -y
   conda activate form-filling
   pip install -r requirements.txt
   PORT=8000 python main.py
   ```

3. Chat locally by running in a different terminal.
   ```zsh
   cartesia chat 8000
   ```

## Remote Deployment

Read the [Cartesia docs](https://docs.cartesia.ai/line/) to learn how to deploy templates to the Cartesia Line platform.

## Architecture

### Core Components

1. **FormManager** (`form_manager.py`)
   - Loads and parses YAML form configuration
   - Manages question progression and conditional logic
   - Validates and stores user responses
   - Handles nested question groups

2. **FormFillingNode** (`form_filling_node.py`)
   - Extends ReasoningNode for form-specific logic
   - Orchestrates conversation flow
   - Manages context clearing between questions

3. **RecordAnswerTool** (`form_tools.py`)
   - Tool for capturing user responses
   - Triggers form state management
   - Enables structured data collection

4. **Form Configuration** (`form.yaml`)
   - YAML-based question definitions
   - Supports multiple question types and conditional logic

### Question Flow Logic

1. **Initial Setup**: Form loads from `form.yaml` and presents first question
2. **User Response**: User provides answer via voice
3. **LLM Processing**: LLM processes response and calls `record_answer` tool if satisfactory
4. **Answer Recording**: FormManager validates and stores the response
5. **Context Clearing**: Conversation history is cleared to maintain focus
6. **Next Question**: If more questions exist, present the next one
7. **Completion**: When all questions are answered, end the call gracefully

## Customization Guide

### Modifying the Form (`form.yaml`)

The form configuration supports various question types and features:

```yaml
questionnaire:
  id: "your_form_id"
  text: "Your Form Title"
  type: "group"
  questions:
    - id: "question_id"
      text: "Your question text?"
      type: "string|number|boolean|select|date"
      required: true|false

      # For select questions
      options:
        - value: "option1"
          text: "Option 1 Display Text"
        - value: "option2"
          text: "Option 2 Display Text"

      # For conditional questions
      dependsOn:
        questionId: "previous_question_id"
        value: expected_value
        operator: "equals|not_equals|in|not_in"

      # For number questions
      min: min_value
      max: max_value
```

#### Supported Question Types:
- **string**: Free-text responses
- **number**: Numeric values with optional min/max validation
- **boolean**: Yes/No questions
- **select**: Multiple choice with predefined options
- **date**: Date responses (stored as strings)

#### Conditional Logic:
Questions can be shown/hidden based on previous answers using the `dependsOn` field with operators:
- `equals`: Show if previous answer equals specified value
- `not_equals`: Show if previous answer doesn't equal specified value
- `in`: Show if previous answer is in specified list
- `not_in`: Show if previous answer is not in specified list

### Customizing the System Prompt (`config.py`)

Modify `SYSTEM_PROMPT` to change the agent's behavior. We've outlined a prompt that performs well, but feel free to substite in your own.

```python
SYSTEM_PROMPT = """
### You and your role
[Customize the agent's role and personality]

IMPORTANT: When you receive a clear answer from the user, use the record_answer tool to record their response.

### Your tone
[Define how the agent should communicate]
"""
```

### Adding Custom Question Types

1. **Extend FormManager**: Add validation logic in `_process_answer()` method
2. **Update Form Schema**: Define new question type in your YAML
3. **Modify System Prompt**: Instruct the LLM on how to handle the new type

### Customizing Response Validation

Edit the `_process_answer()` method in `FormManager` to add custom validation:

```python
def _process_answer(self, answer: str, question_type: str, question: Dict[str, Any]) -> Any:
    # Add your custom validation logic here
    if question_type == "custom_type":
        # Custom validation logic
        return processed_answer
    # ... existing logic
```

### Modifying Conversation Flow

To customize how questions are presented or how the conversation flows:

1. **Question Formatting**: Modify `format_question_for_llm()` in FormManager
2. **Response Messages**: Update response templates in FormFillingNode
3. **Context Management**: Adjust when and how context is cleared
4. **Completion Behavior**: Customize the end-of-form experience

### Adding Post-Processing

To process completed forms (save to database, send emails, etc.):

1. **Form Completion Hook**: Add logic in FormFillingNode when `form_manager.is_form_complete()` returns True
2. **Data Export**: Use `form_manager.get_form_summary()` to access all collected data
3. **Integration Points**: Add API calls or database saves as needed
