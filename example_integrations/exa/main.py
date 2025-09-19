import os
import sys
sys.path.append('../../')  # Add path to line SDK

from dotenv import load_dotenv
import openai
from loguru import logger

from config import SYSTEM_PROMPT
from exa_utils import ExaSearchClient
from research_node import ResearchNode

from line import Bridge, CallRequest, VoiceAgentApp, VoiceAgentSystem
from line.events import UserStartedSpeaking, UserStoppedSpeaking, UserTranscriptionReceived

# Load environment variables
load_dotenv()

# Initialize clients
openai_client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
exa_client = ExaSearchClient(api_key=os.environ.get("EXA_API_KEY"))


async def handle_new_call(system: VoiceAgentSystem, call_request: CallRequest):
    """
    Handle new voice call with web research capabilities.
    
    Creates a single research node that can engage in conversation and
    perform web searches to provide accurate, up-to-date information.
    """
    logger.info(f"Starting web research call for {call_request.call_id}")
    
    # Create main research node
    research_node = ResearchNode(
        system_prompt=SYSTEM_PROMPT,
        openai_client=openai_client,
        exa_client=exa_client
    )
    
    # Set up bridge for the research node
    research_bridge = Bridge(research_node)
    system.with_speaking_node(research_node, research_bridge)
    
    # Configure event routing
    research_bridge.on(UserTranscriptionReceived).map(research_node.add_event)
    
    # Set up conversation flow with interruption handling
    (
        research_bridge.on(UserStoppedSpeaking)
        .interrupt_on(UserStartedSpeaking, handler=research_node.on_interrupt_generate)
        .stream(research_node.generate)
        .broadcast()
    )
    
    # Start the system
    await system.start()
    
    # Send initial greeting
    await system.send_initial_message(
        "Hello! I'm your web research assistant powered by Exa and Cartesia. "
        "I can search the web in real-time to answer your questions with up-to-date information. "
        "What would you like to know about?"
    )
    
    # Wait for call to end
    await system.wait_for_shutdown()


# Create the voice agent app
app = VoiceAgentApp(handle_new_call)

if __name__ == "__main__":
    app.run()
