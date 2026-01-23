"""
Web Research Agent with Exa and Cartesia Line v0.2 SDK.

A real-time web research voice agent that combines Cartesia's voice capabilities
with Exa's powerful web search API to provide accurate, up-to-date information
through natural conversation.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from line.call_request import CallRequest
from line.v02.llm import LlmAgent, LlmConfig
from line.v02.tools import end_call
from line.v02.voice_agent_app import AgentEnv, VoiceAgentApp

from config import INTRODUCTION, MAX_OUTPUT_TOKENS, SYSTEM_PROMPT, TEMPERATURE
from exa_utlls import web_search

# Load environment variables
load_dotenv()


async def get_agent(env: AgentEnv, call_request: CallRequest):
    """
    Create a web research agent for handling voice calls.

    Uses LlmAgent with Exa web search capability to provide
    real-time information to users through natural conversation.
    """
    logger.info(f"Starting web research call for {call_request.call_id}")

    return LlmAgent(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        tools=[web_search, end_call],
        config=LlmConfig(
            system_prompt=SYSTEM_PROMPT,
            introduction=INTRODUCTION,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
        ),
    )


# Create the voice agent app
app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    app.run()
