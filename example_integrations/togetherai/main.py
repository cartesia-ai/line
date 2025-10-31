import logging
import os

from config import EscalationAlert, escalation_schema, prompt_escalation, prompt_main
from customer_service_node import CustomerServiceNode
from dotenv import load_dotenv
from escalation_node import EscalationNode
from openai import AsyncOpenAI

from line import Bridge, CallRequest, VoiceAgentApp, VoiceAgentSystem
from line.events import AgentResponse, UserStartedSpeaking, UserStoppedSpeaking, UserTranscriptionReceived

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


together_client = AsyncOpenAI(
    api_key=os.environ.get("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
)


async def handle_new_call(system: VoiceAgentSystem, chat_request: CallRequest):
    """
    Handle new customer service call with main agent and background escalation monitoring.

    Args:
        system: Voice agent system
        chat_request: Incoming call request
    """

    # Main customer service agent (authorized to speak)
    service_node = CustomerServiceNode(system_prompt=prompt_main, client=together_client)
    service_bridge = Bridge(service_node)

    # Configure main conversation routing
    system.with_speaking_node(service_node, service_bridge)
    service_bridge.on(UserTranscriptionReceived).map(service_node.add_event)

    # Handle user speech with interruption support
    (
        service_bridge.on(UserStoppedSpeaking)
        .interrupt_on(UserStartedSpeaking, handler=service_node.on_interrupt_generate)
        .stream(service_node.generate)
        .broadcast()
    )

    # Background escalation monitoring agent
    escalation_node = EscalationNode(
        system_prompt=prompt_escalation,
        client=together_client,
        node_schema=escalation_schema,
        node_name="Escalation Monitor",
    )

    escalation_bridge = Bridge(escalation_node)

    # Configure escalation monitoring
    escalation_bridge.on(UserTranscriptionReceived).map(escalation_node.add_event)
    escalation_bridge.on(AgentResponse).map(escalation_node.add_event)
    escalation_bridge.on(UserStoppedSpeaking).stream(escalation_node.generate).broadcast()

    # Route escalation alerts back to main service node
    service_bridge.on(EscalationAlert).map(service_node.add_event)

    # Register both nodes in the system
    (
        system.with_speaking_node(service_node, service_bridge).with_node(  # Can speak to customer
            escalation_node, escalation_bridge
        )  # Background monitoring only
    )

    # Start the system and send greeting
    await system.start()
    await system.send_initial_message(
        "Hello! Welcome to TechCorp customer support. "
        "I'm here to help you with any technical issues, billing questions, or general inquiries. "
        "How can I assist you today?"
    )
    await system.wait_for_shutdown()


app = VoiceAgentApp(handle_new_call)

if __name__ == "__main__":
    app.run()
