"""Voice research agent powered by Parallel Free Search MCP."""

import json
import os
from typing import Annotated, Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from line.llm_agent import LlmAgent, LlmConfig, ToolEnv, end_call, loopback_tool
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

PARALLEL_MCP_URL = "https://search.parallel.ai/mcp"
MAX_RESULTS = 5
SYSTEM_PROMPT = (
    "You are a concise voice research assistant. Use web_search whenever a user asks for "
    "current or factual web information. Base the answer on the returned sources and name "
    "sources naturally. If the search is inconclusive, say so. Speak in plain sentences "
    "without markdown. Use end_call when the user wants to finish."
)
INTRODUCTION = "Hello! I can search the web for current information. What would you like to know?"


def _limit_results(value: Any) -> Any:
    """Limit result collections before returning MCP data to the model."""
    if isinstance(value, dict):
        return {
            key: _limit_results(item[:MAX_RESULTS] if key == "results" and isinstance(item, list) else item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_limit_results(item) for item in value]
    return value


def _result_text(result: Any) -> str:
    """Render structured MCP output, falling back to text when it is absent."""
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return json.dumps(_limit_results(structured), ensure_ascii=False)

    text_parts = [item.text for item in getattr(result, "content", []) if hasattr(item, "text")]
    if not text_parts:
        return "No relevant information found."
    return "\n".join(text_parts)


@loopback_tool
async def web_search(
    ctx: ToolEnv,
    objective: Annotated[str, "A concise description of the information needed."],
    search_queries: Annotated[list[str], "One or more focused web search queries."],
) -> str:
    """Search the web for current information and return up to five results."""
    async with streamablehttp_client(PARALLEL_MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "web_search",
                arguments={"objective": objective, "search_queries": search_queries},
            )
    return _result_text(result)


async def get_agent(env: AgentEnv, call_request: CallRequest) -> LlmAgent:
    """Create the voice agent with the Parallel search loopback tool."""
    return LlmAgent(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        tools=[web_search, end_call],
        config=LlmConfig(
            system_prompt=SYSTEM_PROMPT,
            introduction=INTRODUCTION,
            max_tokens=400,
            temperature=0.5,
        ),
    )


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    app.run()
