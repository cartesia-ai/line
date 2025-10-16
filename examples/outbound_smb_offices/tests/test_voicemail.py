import os

from chat_node import ChatNode
from config import get_system_prompt
from config_service import BusinessDetails
from google import genai

from line.evals.conversation_runner import ConversationRunner
from line.evals.turn import AgentTurn, ToolCall, UserTurn
from line.events import EndCall


async def test_dtmf_ivr_agent_voicemail_hangs_up():
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    details = BusinessDetails(
        name="Tahoe snow resorts",
        address="123 Main St, Tahoe, USA",
        phone_number="+1234567890",
    )
    reasoning_node = ChatNode(
        system_prompt=get_system_prompt(details),
        gemini_client=gemini_client,
    )

    expected_conversation = [
        UserTurn(
            text="Thank you for calling Tahoe snow resorts. Unfortunately, no one is available to answer your call"
            + " right now. Our office hours are Monday through Friday from 9 a .m. to 5 p .m. Please leave your name, number,"
            + "as well as the nature of your call, and someone will gladly follow up with you at our earliest convenience. weekdays "
            + "between the hours of 9 a .m. and 5 p .m. Thank you and have a nice day. Record your message at the tone when you are finished."
            + "Hang up or press pound for more options."
        ),
        AgentTurn(text="*", telephony_events=[EndCall()]),
    ]

    # This should pass due to fuzzy matching
    test_conv = ConversationRunner(
        reasoning_node,
        expected_conversation,
    )

    await test_conv.run()
