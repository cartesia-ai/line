import os

from loguru import logger

from line.call_request import CallRequest
from line.v02.llm import LlmAgent, LlmConfig
from line.v02.tools import end_call
from line.v02.voice_agent_app import AgentEnv, VoiceAgentApp
from spanish_agent import SpanishAgentTransfer

#  GEMINI_API_KEY=your-key uv run python main.py


async def get_agent(env: AgentEnv, call_request: CallRequest):
    logger.info(
        f"Starting new call for {call_request.call_id}. "
        f"Agent system prompt: {call_request.agent.system_prompt}"
        f"Agent introduction: {call_request.agent.introduction}"
    )

    # Create the transfer tool instance (per-call to maintain separate state)
    spanish_transfer = SpanishAgentTransfer()

    return LlmAgent(
        model="gemini/gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        tools=[end_call, spanish_transfer.tool],
        config=LlmConfig(
            system_prompt="""You are a friendly and helpful assistant. Have a natural conversation with the user.
If the user asks to speak in Spanish or requests a Spanish speaker, use the transfer_to_spanish tool.""",
            introduction="Hello! I'm your AI assistant. How can I help you today?",
        ),
    )


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    print("Starting Transfer Agent app")
    app.run()
