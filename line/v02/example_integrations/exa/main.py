"""Web Research Agent with Exa and Cartesia Line v0.2 SDK."""

import os

from exa_tools import web_search

from line.call_request import CallRequest
from line.v02.llm import LlmAgent, LlmConfig
from line.v02.tools import end_call
from line.v02.voice_agent_app import AgentEnv, VoiceAgentApp

SYSTEM_PROMPT = """You are an intelligent web research assistant with access to real-time \
web search capabilities.

Your role is to help users find accurate, up-to-date information by searching the web and \
synthesizing the results into clear, helpful answers.

When a user asks a question, first determine if you need current information or specific facts. \
If so, use the web_search tool to find relevant information. Then analyze the search results \
carefully and provide a comprehensive answer based on what you found. Cite your sources when \
possible. If search results are insufficient, let the user know and suggest refining the question.

Always search for current information rather than relying on potentially outdated knowledge. Be \
concise but thorough in your responses. Distinguish between facts from search results and your \
own analysis. If you're unsure about information, say so. Use the end_call tool when the user \
wants to end the conversation.

CRITICAL: This is a voice interface. Never use any formatting or special characters in your \
responses. Do not use markdown bold, italics, numbered lists, bullet points, dashes, or \
asterisks. Speak naturally in plain text paragraphs as if you are having a conversation. \
Format everything as natural flowing speech."""

INTRODUCTION = (
    "Hello! I'm your web research assistant powered by Exa and Cartesia. "
    "I can search the web in real-time to answer your questions with up-to-date information. "
    "What would you like to know about?"
)

MAX_OUTPUT_TOKENS = 300
TEMPERATURE = 0.7


async def get_agent(env: AgentEnv, call_request: CallRequest):
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


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    app.run()
