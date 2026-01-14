import os
import sys

from chat_node import ChatNode
from config import LeadsAnalysis, ResearchAnalysis
import google.genai as genai
from leads_extraction_node import LeadsExtractionNode
from loguru import logger
from research_node import ResearchNode

from line import Bridge, CallRequest, VoiceAgentApp, VoiceAgentSystem
from line.events import AgentSpeechSent, UserStartedSpeaking, UserStoppedSpeaking, UserTranscriptionReceived

# Configure logging to filter out DEBUG messages from cartesia modules
logger.remove()
logger.add(
    sys.stderr,
    filter=lambda record: not (record["name"].startswith("cartesia") and record["level"].name == "DEBUG"),
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


async def handle_new_call(system: VoiceAgentSystem, call_request: CallRequest):
    logger.info(
        f"Starting sales with leads call for {call_request.call_id} "
        f"from {call_request.from_} to {call_request.to}"
    )

    # Create separate Gemini clients for each node to avoid interference
    chat_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    leads_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    research_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Main conversation node
    chat_node = ChatNode(gemini_client=chat_client)
    conversation_bridge = Bridge(chat_node)
    system.with_speaking_node(chat_node, conversation_bridge)
    conversation_bridge.on(UserTranscriptionReceived).map(chat_node.add_event)
    conversation_bridge.on(AgentSpeechSent).map(chat_node.add_event)
    (
        conversation_bridge.on(UserStoppedSpeaking)
        .interrupt_on(UserStartedSpeaking, handler=chat_node.on_interrupt_generate)
        .stream(chat_node.generate)
        .broadcast()
    )

    # Add leads extraction node that triggers on UserStoppedSpeaking
    leads_node = LeadsExtractionNode(gemini_client=leads_client)
    leads_bridge = Bridge(leads_node)
    leads_bridge.on(UserTranscriptionReceived).map(leads_node.add_event)
    leads_bridge.on(UserStoppedSpeaking).stream(leads_node.generate).broadcast()

    # Add the leads events to the chat node
    conversation_bridge.on(LeadsAnalysis).map(chat_node.add_event)

    # Add research node that triggers on LeadsAnalysis
    research_node = ResearchNode(gemini_client=research_client)
    research_bridge = Bridge(research_node)
    research_bridge.on(LeadsAnalysis).map(research_node.add_event).stream(research_node.generate).broadcast()
    conversation_bridge.on(ResearchAnalysis).map(chat_node.add_event)

    # Add both nodes to system
    system.with_node(leads_node, leads_bridge)
    system.with_node(research_node, research_bridge)

    await system.start()
    await system.send_initial_message(
        "Hi! I'm Savannah, a Cartesia voice agent. Who do I have the pleasure of speaking with today?"
    )
    await system.wait_for_shutdown()


app = VoiceAgentApp(handle_new_call)

if __name__ == "__main__":
    app.run()
