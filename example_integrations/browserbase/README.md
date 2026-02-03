# Voice Agent with Real-time Web Form Filling

This example demonstrates a voice agent that conducts phone questionnaires while automatically filling out web forms in real-time using Stagehand browser automation powered by Browserbase.

## Features

- **Voice Conversations**: Natural voice interactions using Line SDK
- **Real-time Form Filling**: Automatically fills web forms as answers are collected
- **Browser Automation**: Uses Stagehand AI to interact with any web form
- **Deterministic Flow**: Passthrough tools provide predictable conversation flow without LLM generation
- **Async Processing**: Non-blocking form filling maintains conversation flow - form fields are filled in background tasks
- **Auto-submission**: Submits forms automatically when complete or when call ends

## Architecture

```
Voice Call → VoiceAgentApp → FormFillingAgent (wraps LlmAgent)
                                    ↓
                            Passthrough Tools
                                    ↓
                    start_questionnaire / record_form_field
                                    ↓
                         StagehandFormFiller
                                    ↓
                         Browserbase Session
                                    ↓
                           Web Form Filled
```

The implementation uses:
- **`VoiceAgentApp`**: The main application entry point
- **`LlmAgent`**: Handles LLM interactions with tool calling
- **Passthrough tools**: Directly emit `OutputEvent`s (like `AgentSendText`) without requiring LLM generation, providing deterministic conversation flow

## Getting Started

You will need:
- A [Cartesia](https://play.cartesia.ai/agents) account and API key
- A [Gemini API Key](https://aistudio.google.com/apikey)
- A [Browserbase API Key and Project ID](https://www.browserbase.com/overview)

### Dependencies

```
cartesia-line
stagehand>=3.0.0
google-genai>=1.26.0
loguru>=0.7.0
pydantic>=2.0.0
```

## Setup

1. Install dependencies:
```bash
pip install -e .
```

2. Set up environment variables - create a `.env` file:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
BROWSERBASE_API_KEY=your_browserbase_api_key_here
BROWSERBASE_PROJECT_ID=your_browserbase_project_id_here
```

3. Run the agent:
```bash
python main.py
```

## Project Structure

### `main.py`
Entry point for the voice agent. Creates a `VoiceAgentApp` with a `FormFillingAgent` that wraps an `LlmAgent`. The agent is configured with:
- A system prompt guiding the LLM to use the form-filling tools
- An introduction message to greet users
- Two passthrough tools for deterministic conversation flow
- Configuration:
  - `FORM_URL`: Target web form to fill
  - `MODEL_ID`: LLM model to use (e.g., Gemini Pro)


### `stagehand_form_filler.py`
Browser automation manager that handles all web interactions. Key components:
- **`StagehandFormFiller`**: Main class that manages the browser session and provides tool methods
- **Passthrough tools**: `start_questionnaire` and `record_form_field` emit `OutputEvent`s directly
- **Eager initialization**: Browser session starts immediately on construction
- **Background filling**: Form fields are filled asynchronously without blocking conversation

### `pyproject.toml`
Package configuration with dependencies for the Line SDK.

## Example Flow

1. User calls the voice agent
2. Agent greets: "Hello! I'm here to help you fill out an application form today..."
3. User says "yes" or "ready"
4. Agent calls `start_questionnaire` tool → asks first question
5. User answers each question
6. Agent calls `record_form_field` tool:
   - Records the answer
   - Fills the form field in the browser (background task)
   - Asks the next question (deterministically, via passthrough)
7. After all questions, form is submitted automatically
8. Call ends and browser resources are cleaned up

## Advanced Features

- **Eager Browser Initialization**: Browser session starts immediately on agent construction, reducing latency when filling the first field
- **Background Processing**: Form filling happens asynchronously - conversation remains smooth
- **Auto-submission on Disconnect**: If the user hangs up, collected data is still submitted
- **Error Recovery**: Continues conversation even if form filling fails
- **Progress Tracking**: Monitor form completion status
- **Screenshot Debugging**: Captures screenshots after each field

### Passthrough Tools
The `@passthrough_tool` decorator marks tools that directly yield `OutputEvent`s:

```python
@passthrough_tool
async def start_questionnaire(self, ctx: ToolEnv) -> AsyncGenerator[OutputEvent, None]:
    yield AgentSendText(text="Great! Let's begin. What is your full name?")
```

This bypasses LLM generation for responses, providing:
- Deterministic, predictable conversation flow
- Lower latency (no LLM call needed for tool responses)
- Consistent question phrasing

## Testing

Test with different scenarios:
- Complete questionnaire flow
- Interruptions and corrections
- Various answer formats
- Multi-page forms
- Form validation errors


## Production Considerations

- Configure proper error logging
- Add retry logic for form submission
- Implement form validation checks
- Consider rate limiting for API calls
- Handle browser session timeouts gracefully
