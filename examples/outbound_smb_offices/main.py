import os

from chat_node import ChatNode
from config import CONFIG_URL, get_system_prompt
from loguru import logger
from config_service import BusinessDetails
from google import genai
import requests

from line import Bridge, CallRequest, PreCallResult, VoiceAgentApp, VoiceAgentSystem
from line.events import (
    DTMFOutputEvent,
    UserStartedSpeaking,
    UserStoppedSpeaking,
    UserTranscriptionReceived,
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
else:
    gemini_client = None


async def pre_call_handler(_call_request: CallRequest) -> PreCallResult:
    """Configure voice settings before starting the call."""
    # Find more voices here: https://play.cartesia.ai/voices
    tts_config = {"tts": {"model": "sonic-3-preview"}}

    return PreCallResult(
        metadata={},
        config=tts_config,
    )


async def get_details(to_number: str) -> BusinessDetails:
    response = requests.post(f"{CONFIG_URL}/details", json={"to": to_number})
    if response.status_code != 200:
        raise Exception(
            f"Failed to get provider for {to_number}, response: {response.text}, response code: {response.status_code}"
        )

    details = BusinessDetails(**response.json())
    return details


async def handle_new_call(system: VoiceAgentSystem, call_request: CallRequest):
    # call_request.to will be None if the call originated from a non-phone service (e.g. webdialier in the cartesia playground)
    to_number = call_request.to if call_request.to != "unknown" else "+15551234567"

    details = await get_details(to_number)

    # Main conversation node
    conversation_node = ChatNode(
        system_prompt=get_system_prompt(details),
        gemini_client=gemini_client,
    )
    conversation_bridge = Bridge(conversation_node)
    system.with_speaking_node(conversation_node, bridge=conversation_bridge)

    # Setup events
    conversation_bridge.on(UserTranscriptionReceived).map(conversation_node.add_event)
    conversation_bridge.on(DTMFOutputEvent).map(conversation_node.on_dtmf_output)

    (
        conversation_bridge.on(UserStoppedSpeaking)
        .interrupt_on(UserStartedSpeaking, handler=conversation_node.on_interrupt_generate)
        .stream(conversation_node.generate)
        .broadcast()
    )
    async for item in conversation_node.warmup():
        logger.info(f"Received item from gemini client: {item}")

    await system.start()
    await system.wait_for_shutdown()


app = VoiceAgentApp(handle_new_call, pre_call_handler)

if __name__ == "__main__":
    app.run()
