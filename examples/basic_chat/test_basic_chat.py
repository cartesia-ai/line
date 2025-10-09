"""
Run this via

```
uv sync --extra dev
uv run pytest test_basic_chat.py
```

To run with parallelism:
```
uv run pytest test_basic_chat.py --count 4 -n auto
```
"""

import os

from chat_node import ChatNode
from config import SYSTEM_PROMPT
from google import genai
import pytest

from line.evals.conversation_runner import ConversationRunner
from line.evals.turn import AgentTurn, UserTurn
from line.events import EndCall


@pytest.mark.asyncio
async def test_basic_chat():
    """
    Test a simple conversation
    """
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    conversation_node = ChatNode(
        system_prompt=SYSTEM_PROMPT,
        gemini_client=gemini_client,
    )

    expected_conversation = [
        UserTurn(
            text="Hello",
        ),
        AgentTurn(
            text="Hello! Have you had any experience with voice agents before?"
        ),  # Opener is same every time
        UserTurn(text="What do you like better: apples or oranges?"),
        AgentTurn(text=["<mentions apples>", "<mentions oranges>"]),
    ]

    test_conv = ConversationRunner(conversation_node, expected_conversation)
    await test_conv.run()


@pytest.mark.asyncio
async def test_basic_chat_can_end_call():
    """
    Test a simple conversation
    """
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    conversation_node = ChatNode(
        system_prompt=SYSTEM_PROMPT,
        gemini_client=gemini_client,
    )

    expected_conversation = [
        UserTurn(
            text="Goodbye",
        ),
        AgentTurn(
            text="*",
            telephony_events=[EndCall()],
        ),
    ]

    test_conv = ConversationRunner(conversation_node, expected_conversation)
    await test_conv.run()
