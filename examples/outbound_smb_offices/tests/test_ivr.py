import os

from chat_node import ChatNode
from config import get_system_prompt
from config_service import BusinessDetails
from google import genai

from line.evals.conversation_runner import ConversationRunner
from line.evals.turn import AgentTurn, UserTurn
from line.events import DTMFOutputEvent


async def test_dtmf_ivr_multi_step_expects_correct_ivr_response_and_then_correct_greeting():
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
            text="""
        Thank you for calling Tahoe snow resorts. If this is an emergency, please hang up and dial 911
        or contact the closest ski patrol to the closest emergency room. Calls may be recorded for quality purposes. Para espanol oprima numero ocho.

        Please select from the following. If you are calling to schedule, or reschedule an appointment, say or press one.
        For a weather briefing, say or press two. For ski rentals, say or press three.
        For all other inquiries, say or press zero. To repeat this menu, press pound.
        """
        ),
        AgentTurn(telephony_events=[DTMFOutputEvent(button="0")]),
        UserTurn(
            text="Good afternoon, thank you for calling Tahoe snow resorts. My name is Serena. How may I help you?"
        ),
        AgentTurn(text="<mentions Caroline>"),
    ]

    test_conv = ConversationRunner(
        reasoning_node,
        expected_conversation,
    )

    await test_conv.run()
