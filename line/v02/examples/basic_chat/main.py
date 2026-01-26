import os

from loguru import logger

from line.call_request import CallRequest
from line.v02.llm import LlmAgent, LlmConfig, end_call
from line.v02.voice_agent_app import AgentEnv, VoiceAgentApp

#  GEMINI_API_KEY=your-key uv python main.py


async def get_agent(env: AgentEnv, call_request: CallRequest):
    logger.info(
        f"Starting new call for {call_request.call_id}. "
        f"Agent system prompt: {call_request.agent.system_prompt}"
        f"Agent introduction: {call_request.agent.introduction}"
    )

    return LlmAgent(
        model="gemini/gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        tools=[end_call],
        config=LlmConfig(
            system_prompt=call_request.agent.system_prompt
            or "You are a friendly and helpful assistant. Have a natural conversation with the user.",
            # Empty string = agent waits for user to speak first; non-empty = agent speaks first
            introduction=call_request.agent.introduction
            if call_request.agent.introduction is not None
            else "Hello! I'm your AI assistant. How can I help you today?",
        ),
    )


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    print("Starting app")
    app.run()
