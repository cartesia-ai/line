"""Web Research Agent with Exa and Cartesia Line v0.2 SDK."""

import asyncio
import os
from typing import Annotated

from exa_py import Exa
from loguru import logger

from line.call_request import CallRequest
from line.v02.llm import LlmAgent, LlmConfig, ToolEnv, end_call, loopback_tool
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


@loopback_tool
async def web_search(
    ctx: ToolEnv,
    query: Annotated[
        str,
        "The search query. Be specific and include key terms.",
    ],
) -> str:
    """Search the web for current information. Use when you need up-to-date facts or news."""
    logger.info(f"Performing Exa web search: '{query}'")

    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        return "Web search failed: EXA_API_KEY not set."

    try:
        client = Exa(api_key=api_key)
        results = await asyncio.to_thread(
            client.search_and_contents,
            query,
            num_results=10,
            type="fast",
            livecrawl="never",
            text={"max_characters": 1000},
        )

        if not results or not results.results:
            return "No relevant information found."

        # Format results for LLM
        content_parts = [f"Search Results for: '{query}'\n"]
        for i, result in enumerate(results.results[:10]):
            content_parts.append(f"\n--- Source {i + 1}: {result.title} ---\n")
            if result.text:
                content_parts.append(f"{result.text}\n")
            content_parts.append(f"URL: {result.url}\n")

        logger.info(f"Search completed: {len(results.results)} sources found")
        return "".join(content_parts)

    except Exception as e:
        logger.error(f"Exa search failed: {e}")
        return f"Web search failed: {e}"


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
