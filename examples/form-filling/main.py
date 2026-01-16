import os
from pathlib import Path

from config import SYSTEM_PROMPT
from form_filling_node import FormFillingNode
from google import genai

from line import Bridge, CallRequest, VoiceAgentApp, VoiceAgentSystem
from line.events import AgentSpeechSent, UserStartedSpeaking, UserStoppedSpeaking, UserTranscriptionReceived

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


async def handle_new_call(system: VoiceAgentSystem, call_request: CallRequest):
    # Get the form path
    form_path = Path(__file__).parent / "form.yaml"

    # Main form filling node
    form_node = FormFillingNode(
        system_prompt=SYSTEM_PROMPT,
        gemini_client=gemini_client,
        form_path=str(form_path),
    )
    form_bridge = Bridge(form_node)
    system.with_speaking_node(form_node, bridge=form_bridge)

    form_bridge.on(UserTranscriptionReceived).map(form_node.add_event)
    form_bridge.on(AgentSpeechSent).map(form_node.add_event)
    (
        form_bridge.on(UserStoppedSpeaking)
        .interrupt_on(UserStartedSpeaking, handler=form_node.on_interrupt_generate)
        .stream(form_node.generate)
        .broadcast()
    )

    await system.start()

    # Get the first question to include in the initial message
    first_question = form_node.form_manager.get_current_question()
    if first_question:
        question_text = form_node.form_manager.format_question_for_llm(first_question)
        initial_message = f"Hello! I'll be conducting a brief questionnaire with you today. Let's get started. {question_text}"
    else:
        initial_message = "Hello! I'll be conducting a brief questionnaire with you today. Let's get started."

    await system.send_initial_message(initial_message)
    await system.wait_for_shutdown()


app = VoiceAgentApp(handle_new_call)

if __name__ == "__main__":
    app.run()
