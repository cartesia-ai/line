import os

from loguru import logger

from line.llm_agent import LlmAgent, LlmConfig, agent_as_handoff, end_call
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

#  OPENAI_API_KEY=your-key GEMINI_API_KEY=your-key uv run python main.py


async def get_agent(env: AgentEnv, call_request: CallRequest):
    logger.info(f"Starting new call for {call_request.call_id}")

    # Create the Spanish-speaking agent
    spanish_agent = LlmAgent(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        tools=[end_call],
        config=LlmConfig(
            system_prompt=(
                "Eres un asistente amable y servicial. "
                "Tenga una conversación natural con el usuario. "
                "Habla sólo en español."
            ),
            introduction=("¡Hola! Soy tu asistente de IA. ¿Cómo puedo ayudarte hoy?"),
        ),
    )

    return LlmAgent(
        model="gemini/gemini-2.5-flash-preview-09-2025",
        api_key=os.getenv("GEMINI_API_KEY"),
        tools=[
            end_call,
            agent_as_handoff(
                spanish_agent,
                handoff_message=("Transferring you to our Spanish-speaking agent now..."),
                name="transfer_to_spanish",
                description=(
                    "Transfer the call to a Spanish-speaking agent. "
                    "Use this when the user requests to speak in Spanish."
                ),
            ),
        ],
        config=LlmConfig(
            system_prompt=(
                "You are a friendly and helpful assistant. "
                "Have a natural conversation with the user. "
                "If the user asks to speak in Spanish or requests a Spanish "
                "speaker, use the transfer_to_spanish tool."
            ),
            introduction="Hello! I'm your AI assistant. How can I help you today?",
        ),
    )


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    print("Starting Transfer Agent app")
    app.run()
