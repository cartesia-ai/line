# "Evals as unit tests" framework

### Motivation

Vibe checking an agent is rather painful, especially if the scenarios are rather complex. If we leverage the fact that a "code-first" voice agent can natively integrate with the python testing ecosystem, we can allow developers to build and evaluate agent behavior with native python tools.

### Goal

"Evals as unit tests" is a testing framework for validating that your agent has the correct text-to-text behavior and models this as unit tests. You can specify a conversation and assert certain qualities of your agent -  mentioned something or has made the correct tool calls.


## Overview

You can simply write tests and run them with

```
# Installs dev pytest dependencies
uv sync --extra dev

# Run all pytests
uv run pytest
```

## Leveraging pytest

One way to improve your robustness is to run with multiple counts automatically load balanced across all available workers (usually the number of cores you have on your machine):
```
uv run pytest tests/test_similarity_utils.py   --count 8 -n auto
```

Or you can run certain tests
```
uv run pytest line/tests/test_*
```


## Usage Example

```python
from eval.conversation_runner import ConversationRunner
from eval.turn import AgentTurn, ToolCall, UserTurn

reasoning_node = ExampleWeatherNode()

# Define expected conversation flow
expected_conversation = [
    UserTurn(text="What's the weather like in San Francisco?"),
    AgentTurn(
        text="<mentions current weather conditions>",
        tool_calls=[ToolCall(
            name="get_weather",
            arguments={"location": "San Francisco", "units": "fahrenheit"},
            result={"temperature": None, "condition": "*"}
        )]
    ),
    UserTurn(text="What about tomorrow?"),
    AgentTurn(
        text="*",  # Accept any response about tomorrow's weather
        tool_calls=[ToolCall(
            name="get_weather",
            arguments={"location": "San Francisco", "forecast": "tomorrow"},
            result={"temperature": None, "condition": "*"}
        )]
    )
]

# Run the test
test_conv = ConversationRunner(reasoning_node, expected_conversation)
await test_conv.run()
```

## Dependencies
Ensure the following are added as dependencies (the following example uses `uv`)

```
[tool.uv]
dev-dependencies = [
    "pytest==8.4.2",
    "pytest-asyncio==1.2.0",
    "pytest-cov==7.0.0",
    "pytest-xdist==3.8.0",
    "pytest-repeat==0.9.4"
]
```

# Comparison logic

Due to the non-deterministic nature of LLMs, we support the following syntax for ensuring that turns match

**Wildcard Support:**
- `"*"` - Matches any string content
- Example: `AgentTurn(text="*")` accepts any agent response

**Semantic Matching:**
- Uses Gemini AI to compare meaning beyond exact text matching
- Handles paraphrasing, synonyms, and filler words
- Example: "What's your name?" ≈ "Can you tell me your name?"

**Statement Patterns:**
- `"<statement about content>"` - Matches text that satisfies the statement
- Example: `"<mentions SOC-2 compliance>"` matches "We are SOC-2 compliant"
- Example: `"<asks for user name>"` matches "What's your name?"

**Match different responses
- `AgentTurn(text=["I can say hello", "<mentions how are you>"])

### Dictionary Comparison

**Recursive Checking:**
- Nested dictionaries are compared recursively
- String values use semantic similarity matching
- Other types require exact matches

**None Value Handling:**
- `None` values in expected dict skip that key's validation
- Allows partial dictionary matching for flexible validation

## Testing Patterns

### Environment Requirements

- Requires `GEMINI_API_KEY` environment variable for semantic similarity checking
- Tests are automatically skipped if API key is not available
