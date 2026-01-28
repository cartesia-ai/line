import os

from loguru import logger

from line.call_request import CallRequest
from line.v02.llm import LlmAgent, LlmConfig, agent_as_handoff
from line.v02.llm.tools import end_call
from line.v02.voice_agent_app import AgentEnv, VoiceAgentApp

#  OPENAI_API_KEY=your-key GEMINI_API_KEY=your-key uv run python main.py


async def get_agent(env: AgentEnv, call_request: CallRequest):
    logger.info(
        f"Starting new call for {call_request.call_id}. "
        f"Agent system prompt: {call_request.agent.system_prompt}"
        f"Agent introduction: {call_request.agent.introduction}"
    )

    # Create the Spanish-speaking agent
    spanish_agent = LlmAgent(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        tools=[end_call],
        config=LlmConfig(
            system_prompt=(
                "You are a friendly and helpful assistant. "
                "Have a natural conversation with the user. "
                "You speak only in Spanish."
            ),
            introduction=("¡Hola! Soy tu asistente de IA. ¿Cómo puedo ayudarte hoy?"),
        ),
    )

    return LlmAgent(
        model="gemini/gemini-2.0-flash",
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
