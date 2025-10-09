"""
Run this via

```
uv sync --extra dev
uv run pytest tests/test_cold_transfers.py
```
"""

import os

from chat_node import ChatNode
from google import genai
import pytest

from line.evals.conversation_runner import ConversationRunner
from line.evals.turn import AgentTurn, UserTurn
from line.events import TransferCall


@pytest.mark.asyncio
async def test_basic_chat():
    """
    Test a simple conversation
    """
    conversation_node = ChatNode()

    expected_conversation = [
        UserTurn(
            text="Speak with a representative",
        ),
        AgentTurn(text="*", telephony_events=[TransferCall(target_phone_number="+18005551234")]),
    ]

    test_conv = ConversationRunner(conversation_node, expected_conversation)
    await test_conv.run()
