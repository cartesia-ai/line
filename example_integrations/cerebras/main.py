import logging
import os

from cerebras.cloud.sdk import AsyncCerebras
from config import PerfAnalysis, prompt_agent1, prompt_agent2, prompt_agent3, prompt_main, schema_background
from dotenv import load_dotenv
from interviewer import TalkingNode
from judges import JudgeNode

from line import Bridge, CallRequest, VoiceAgentApp, VoiceAgentSystem
from line.events import AgentResponse, UserStartedSpeaking, UserStoppedSpeaking, UserTranscriptionReceived

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


cs_client = AsyncCerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))


async def handle_new_call(system: VoiceAgentSystem, chat_request: CallRequest):
    # Main conversation agent (authorized to speak)
    conversation_node = TalkingNode(system_prompt=prompt_main, client=cs_client)

    conversation_bridge = Bridge(conversation_node)
    # ... configure conversation routing
    system.with_speaking_node(conversation_node, conversation_bridge)
    conversation_bridge.on(UserTranscriptionReceived).map(conversation_node.add_event)

    (
        conversation_bridge.on(UserStoppedSpeaking)
        .interrupt_on(UserStartedSpeaking, handler=conversation_node.on_interrupt_generate)
        .stream(conversation_node.generate)
        .broadcast()
    )

    # Background analysis agent
    agent1_node = JudgeNode(
        system_prompt=prompt_agent1,
        client=cs_client,
        node_schema=schema_background,
        node_name="Technical Report",
    )
    agent2_node = JudgeNode(
        system_prompt=prompt_agent2,
        client=cs_client,
        node_schema=schema_background,
        node_name="Communication Report",
    )
    agent3_node = JudgeNode(
        system_prompt=prompt_agent3,
        client=cs_client,
        node_schema=schema_background,
        node_name="Reasoning Report",
    )

    agent1_bridge = Bridge(agent1_node)
    # ... configure conversation routing
    agent1_bridge.on(UserTranscriptionReceived).map(agent1_node.add_event)
    agent1_bridge.on(AgentResponse).map(agent1_node.add_event)
    agent1_bridge.on(UserStoppedSpeaking).stream(agent1_node.generate).broadcast()

    # Add the judge events to the conversation node
    conversation_bridge.on(PerfAnalysis).map(conversation_node.add_event)

    agent2_bridge = Bridge(agent2_node)
    agent2_bridge.on(UserTranscriptionReceived).map(agent2_node.add_event)
    agent2_bridge.on(AgentResponse).map(agent2_node.add_event)
    agent2_bridge.on(UserStoppedSpeaking).stream(agent2_node.generate).broadcast()

    # Add the judge events to the conversation node
    conversation_bridge.on(PerfAnalysis).map(conversation_node.add_event)

    agent3_bridge = Bridge(agent3_node)
    agent3_bridge.on(UserTranscriptionReceived).map(agent3_node.add_event)
    agent3_bridge.on(AgentResponse).map(agent3_node.add_event)
    agent3_bridge.on(UserStoppedSpeaking).stream(agent3_node.generate).broadcast()

    # Add the judge events to the conversation node
    conversation_bridge.on(PerfAnalysis).map(conversation_node.add_event)

    # Register both agents
    (
        system.with_speaking_node(conversation_node, conversation_bridge)  # Can speak to user
        .with_node(agent1_node, agent1_bridge)  # Background only
        .with_node(agent2_node, agent2_bridge)  # Background only
        .with_node(agent3_node, agent3_bridge)  # Background only
    )

    await system.start()
    await system.send_initial_message(
        "Welcome to the Interview Practice Platform! "
        "Please let me know what role you are applying for and when you are ready to begin the interview."
    )
    await system.wait_for_shutdown()


app = VoiceAgentApp(handle_new_call)
if __name__ == "__main__":
    app.run()
