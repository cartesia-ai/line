import os

from chat_node import ChatNode
from config import SYSTEM_PROMPT
from google import genai
from loguru import logger

from line import Bridge, CallRequest, VoiceAgentApp, VoiceAgentSystem
from line.events import AgentSpeechSent, UserStartedSpeaking, UserStoppedSpeaking, UserTranscriptionReceived

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
else:
    gemini_client = None


async def handle_new_call(system: VoiceAgentSystem, call_request: CallRequest):
    if call_request.metadata:
        logger.info(f"Received metadata in start call request: {call_request.metadata}")

    # Main conversation node
    conversation_node = ChatNode(
        system_prompt=SYSTEM_PROMPT,
        gemini_client=gemini_client,
    )
    conversation_bridge = Bridge(conversation_node)
    system.with_speaking_node(conversation_node, bridge=conversation_bridge)

    conversation_bridge.on(UserTranscriptionReceived).map(conversation_node.add_event)
    conversation_bridge.on(AgentSpeechSent).map(conversation_node.add_event)

    (
        conversation_bridge.on(UserStoppedSpeaking)
        .interrupt_on(UserStartedSpeaking, handler=conversation_node.on_interrupt_generate)
        .stream(conversation_node.generate)
        .broadcast()
    )

    await system.start()
    await system.send_initial_message(
        "Hello! I am your voice agent powered by Cartesia. What do you want to build?"
    )
    await system.wait_for_shutdown()


app = VoiceAgentApp(handle_new_call)

if __name__ == "__main__":
    app.run()
