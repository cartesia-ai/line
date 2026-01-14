import logging

from chat_node import ChatNode

from line import Bridge, CallRequest, VoiceAgentApp, VoiceAgentSystem
from line.events import AgentSpeechSent, UserStartedSpeaking, UserStoppedSpeaking, UserTranscriptionReceived

logger = logging.getLogger(__name__)


async def handle_new_call(system: VoiceAgentSystem, _call_request: CallRequest):
    print("Starting new call", _call_request)
    chat_node = ChatNode()
    chat_bridge = Bridge(chat_node)
    system.with_speaking_node(chat_node, chat_bridge)

    chat_bridge.on(UserTranscriptionReceived).map(chat_node.add_event)
    chat_bridge.on(AgentSpeechSent).map(chat_node.add_event)
    (
        chat_bridge.on(UserStoppedSpeaking)
        .interrupt_on(UserStartedSpeaking, handler=chat_node.on_interrupt_generate)
        .stream(chat_node.generate)
        .broadcast()
    )

    await system.start()
    await system.send_initial_message(chat_node.initial_message())
    await system.wait_for_shutdown()


app = VoiceAgentApp(handle_new_call)

if __name__ == "__main__":
    app.run()
