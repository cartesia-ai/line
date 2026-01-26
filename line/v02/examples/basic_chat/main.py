import os

from loguru import logger

from line.call_request import CallRequest
from line.v02.llm import LlmAgent, LlmConfig
from line.v02.tools import end_call
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
        config=LlmConfig.from_call_request(call_request),
    )


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    print("Starting app")
    app.run()
