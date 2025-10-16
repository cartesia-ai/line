import os

from chat_node import ChatNode
from config import get_system_prompt
from config_service import BusinessDetails
from google import genai

from line.evals.conversation_runner import ConversationRunner
from line.evals.turn import AgentTurn, ToolCall, UserTurn
from line.events import EndCall


async def test_human_conversation():
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    details = BusinessDetails(
        name="Tahoe snow resorts",
        address="123 Main St",
        phone_number="+1234567890",
    )
    reasoning_node = ChatNode(
        system_prompt=get_system_prompt(details),
        gemini_client=gemini_client,
    )

    expected_conversation = [
        UserTurn(text="Hello?"),
        AgentTurn(text="Hi, this is Caroline and from Acme Inc to see if can confirm your address"),
        UserTurn(text="Yes"),
        AgentTurn(text="Great - is 123 Main St the correct address?"),
        UserTurn(text="Yes"),
        AgentTurn(text="Thank you. And can I have the status on my rental gear?"),
        UserTurn(text="Yes, it seems to be in good condition"),
        AgentTurn(
            text="Thank you. And if you wouldn't mind looking outside, can you tell me if I should rent powder skis or all mountain skis?"
        ),
        UserTurn(text="Yes, it is currently snowing outside"),
        AgentTurn(text="That's all I need, would you like to end the call?"),
        UserTurn(text="Yes"),
        AgentTurn(text="*", telephony_events=[EndCall()]),
    ]

    # This should pass due to fuzzy matching
    test_conv = ConversationRunner(
        reasoning_node,
        expected_conversation,
    )

    await test_conv.run()


async def test_human_conversation_negative_response_should_not_trigger_end_call():
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    details = BusinessDetails(
        name="Jane Doe",
        address="123 Main St",
        phone_number="+1234567890",
    )
    reasoning_node = ChatNode(
        system_prompt=get_system_prompt(details),
        gemini_client=gemini_client,
    )

    expected_conversation = [
        UserTurn(text="Hello?"),
        AgentTurn(text="*"),
        UserTurn(text="No"),
        AgentTurn(text="*"),
    ]

    # This should pass due to fuzzy matching
    test_conv = ConversationRunner(
        reasoning_node,
        expected_conversation,
    )

    await test_conv.run()
