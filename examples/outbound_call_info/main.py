import os

from chat_node import ChatNode
from config import SYSTEM_PROMPT
import google.genai as genai
from loguru import logger

from line import Bridge, CallRequest, PreCallResult, VoiceAgentApp, VoiceAgentSystem
from line.events import AgentSpeechSent, UserStartedSpeaking, UserStoppedSpeaking, UserTranscriptionReceived

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


async def handle_call_request(call_request: CallRequest):
    logger.info(f"Handling call request: {call_request}")
    phone_number = call_request.to

    # Return None for rejecting outbounds to a some numbers
    if phone_number == "911":
        return None

    # Configure the TTS to return different voices based on the caller.
    if phone_number == "+15555555555":
        return PreCallResult(
            metadata={"extra_prompt": "This is a VIP caller, so treat them with extra care."},
            config={
                "tts": {
                    "voice": "146485fd-8736-41c7-88a8-7cdd0da34d84",
                    "language": "en",
                }
            },
        )
    return PreCallResult(
        metadata={"extra_prompt": "This is a normal caller, so treat them with normal care."},
        config={
            "tts": {
                "voice": "4322a30e-e1fb-4b06-bc79-06b04f079b07",
                "language": "es",
            }
        },
    )


async def handle_new_call(system: VoiceAgentSystem, call_request: CallRequest):
    extra_prompt = call_request.metadata.get("extra_prompt", "")

    # Main conversation node
    conversation_node = ChatNode(
        gemini_client=gemini_client,
        system_prompt=SYSTEM_PROMPT + f"\n\n {extra_prompt}",
    )
    conversation_bridge = Bridge(conversation_node)

    system.with_speaking_node(conversation_node, conversation_bridge)

    conversation_bridge.on(UserTranscriptionReceived).map(conversation_node.add_event)
    conversation_bridge.on(AgentSpeechSent).map(conversation_node.add_event)
    (
        conversation_bridge.on(UserStoppedSpeaking)
        .interrupt_on(UserStartedSpeaking, handler=conversation_node.on_interrupt_generate)
        .stream(conversation_node.generate)
        .broadcast()
    )

    await system.start()
    await system.send_initial_message("Hi there! How are you?")
    await system.wait_for_shutdown()


app = VoiceAgentApp(call_handler=handle_new_call, pre_call_handler=handle_call_request)

if __name__ == "__main__":
    app.run()
