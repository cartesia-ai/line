"""
Interview Practice Agent with Cartesia Line SDK v0.2.0 and Cerebras.

This agent conducts mock interviews with real-time background analysis
using three judge agents that evaluate technical, communication, and reasoning skills.
"""

import logging

from dotenv import load_dotenv
from interviewer import InterviewAgent

from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


async def get_agent(env: AgentEnv, call_request: CallRequest):
    """Create an InterviewAgent for this call."""
    return InterviewAgent()


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    app.run()
