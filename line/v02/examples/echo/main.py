import os

from tools import echo

from line.call_request import CallRequest
from line.v02.llm import LlmAgent, LlmConfig, end_call
from line.v02.voice_agent_app import AgentEnv, VoiceAgentApp

#  GEMINI_API_KEY=your-key uv python main.py


async def get_agent(env: AgentEnv, call_request: CallRequest):
    return LlmAgent(
        model="gemini/gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        tools=[end_call, echo],
        config=LlmConfig(
            system_prompt="""
            You are a friendly and helpful assistant. Have a natural conversation with the user.
Once the user says `I'm ready to talk to myself`, call the echo tool to echo back what the user says.""",
            introduction="Hello! I'm your Echo Agent. How can I help you today?",
        ),
    )


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    print("Starting Echo app")
    app.run()
