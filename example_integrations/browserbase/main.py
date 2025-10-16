"""Cartesia Line Voice Agent with real-time web form filling.

This module implements a voice agent that conducts phone questionnaires
while automatically filling out web forms in real-time using Stagehand
browser automation.
"""

import os

from config import SYSTEM_PROMPT
from form_filling_node import FormFillingNode
from google import genai

from line import Bridge, CallRequest, VoiceAgentApp, VoiceAgentSystem
from line.events import UserStartedSpeaking, UserStoppedSpeaking, UserTranscriptionReceived

# Target form URL - the actual web form to fill
FORM_URL = "https://forms.fillout.com/t/rff6XZTSApus"

# Initialize Gemini client
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


async def handle_new_call(
    system: VoiceAgentSystem,
    call_request: CallRequest
) -> None:
    """Handle incoming voice calls with real-time web form filling.
    
    This agent will:
    1. Conduct a voice conversation to gather information
    2. Open and fill an actual web form in the background
    3. Submit the form when the conversation is complete
    
    Args:
        system: The voice agent system instance.
        call_request: The incoming call request.
    """
    
    # Create form filling node with browser automation
    form_node = FormFillingNode(
        system_prompt=SYSTEM_PROMPT,
        gemini_client=gemini_client,
        form_url=FORM_URL,
        headless=False
    )
    
    # Set up bridge for event handling
    form_bridge = Bridge(form_node)
    system.with_speaking_node(form_node, bridge=form_bridge)
    
    # Connect transcription events
    form_bridge.on(UserTranscriptionReceived).map(form_node.add_event)
    
    # Handle interruptions and streaming
    (
        form_bridge.on(UserStoppedSpeaking)
        .interrupt_on(
            UserStartedSpeaking,
            handler=form_node.on_interrupt_generate
        )
        .stream(form_node.generate)
        .broadcast()
    )
    
    # Start the system
    await system.start()
    
    # Wait for call to end
    await system.wait_for_shutdown()
    
    # Ensure form is submitted when call ends
    await form_node.cleanup_and_submit()


# Create the voice agent application
app = VoiceAgentApp(handle_new_call)

if __name__ == "__main__":
    print("Starting Voice Agent with Web Form Automation")
    print(f"Will fill form at: {FORM_URL}")
    print("Ready to receive calls...")
    print("\nNote: The browser will run in background (headless mode).")
    print(
        "Form filling happens invisibly while processing voice calls.\n"
    )
    app.run()