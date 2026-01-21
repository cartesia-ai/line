import os

from line.call_request import CallRequest
from line.v02.llm import LlmAgent, LlmConfig
from line.v02.voice_agent_app import AgentEnv, VoiceAgentApp

#  GEMINI_API_KEY=your-key uv python main.py

async def get_agent(env: AgentEnv, call_request: CallRequest):
    print('Call request', call_request.agent)
    return LlmAgent(
        model="gemini/gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        config=LlmConfig(
            system_prompt="You are a friendly and helpful assistant. Have a natural conversation with the user.",
            introduction="Hello! I'm your AI assistant. How can I help you today?",
        ),
    )


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    print("Starting app")
    app.run()
