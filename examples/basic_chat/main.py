import os

import httpx
import openai

from chat_node import ChatNode
from config import CONFIG_URL, get_system_prompt
from config_service import BusinessDetails
from google import genai
import httpx

from line import Bridge, CallRequest, PreCallResult, VoiceAgentApp, VoiceAgentSystem
from line.events import (
    UserStartedSpeaking,
    UserStoppedSpeaking,
    UserTranscriptionReceived, ToolResult,
)

from loguru import logger

print(

        os.getenv("OPENAI_API_KEY"),
        os.getenv("OPENAI_ORG_ID"))

openai_client = openai.AsyncClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORG_ID")
        )

httpx_client = httpx.AsyncClient()

cached_details = {}

async def pre_call_handler(_call_request: CallRequest) -> PreCallResult:
    """Configure voice settings before starting the call."""
    # Find more voices here: https://play.cartesia.ai/voices
    tts_config = {"tts": {"model": "sonic-3-preview"}}

    details = await get_details(_call_request.to, _call_request.from_)
    cached_details[(_call_request.to, _call_request.from_)] = details
    return PreCallResult(
        metadata={},
        config=tts_config,
    )


async def get_details(to_number: str, from_number: str) -> BusinessDetails:
    response = await httpx_client.post(f"{CONFIG_URL}/details", json={"to": to_number, "from": from_number})
    if response.status_code != 200:
        raise Exception(
            f"Failed to get provider for to:{to_number} from:{from_number}, response: {response.text}, response code: {response.status_code}"
        )

    details = BusinessDetails(**response.json())
    return details


async def handle_new_call(system: VoiceAgentSystem, call_request: CallRequest):
    # call_request.to will be None if the call originated from a non-phone service (e.g. webdialier in the cartesia playground)
    # to_number = call_request.to if call_request.to != "unknown" else "+15551234567"

    if not (details := cached_details.get((call_request.to, call_request.from_))):
        details = await get_details(call_request.to, call_request.from_)

    system_prompt = get_system_prompt(details)

    # Main conversation node
    conversation_node = ChatNode(
        system_prompt=system_prompt,
        openai_client=openai_client,
        max_context_length=300,
    )
    conversation_bridge = Bridge(conversation_node)
    system.with_speaking_node(conversation_node, bridge=conversation_bridge)

    # Setup events
    conversation_bridge.on(UserTranscriptionReceived).map(conversation_node.add_event)

    (
        conversation_bridge.on(UserStoppedSpeaking)
        .interrupt_on(UserStartedSpeaking, handler=conversation_node.on_interrupt_generate)
        .stream(conversation_node.generate)
        .broadcast()
    )

    # start generating again when tool result is received
    (
        conversation_bridge.on(ToolResult, filter_fn=lambda msg: msg.event.tool_name == 'find_agent_availability')
        .interrupt_on(UserStartedSpeaking, handler=conversation_node.on_interrupt_generate)
        .stream(conversation_node.generate)
        .broadcast()
    )

    async for item in conversation_node.warmup():
        logger.info(f"Received item from gemini client: {item}")

    await system.start()
    await system.send_initial_message("Hi, can you hear me?")
    await system.wait_for_shutdown()


app = VoiceAgentApp(handle_new_call, pre_call_handler)

if __name__ == "__main__":
    app.run()
