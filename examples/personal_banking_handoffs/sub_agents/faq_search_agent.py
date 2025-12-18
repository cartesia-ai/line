"""FAQ Search Agent - Performs web searches to get real-time banking information using Gemini Live API."""

import re
from typing import AsyncGenerator, Union

from config import BANK_NAME
from google.genai import Client
from google.genai.types import Content, LiveConnectConfig, LiveServerMessage, Part
from loguru import logger
from prompts import make_voice_system_prompt

from line.events import AgentResponse, ToolResult

# FAQ search agent settings
FAQ_SEARCH_MODEL_ID = "gemini-2.5-flash"

# FAQ search agent prompt
FAQ_SEARCH_PROMPT = f"""
You are a {BANK_NAME} information specialist that searches for current, accurate information to help customers.

### Your Role:
- Search for real-time {BANK_NAME} information using web search
- Focus on official {BANK_NAME} sources when possible
- Provide complete, accurate responses using the most current data available
- Complete the chat agent's partial response with real-time information

### Search Focus:
- {BANK_NAME} official website content
- Current rates, fees, and promotional offers
- Branch locations, hours, and services
- Account types and features
- Recent policy updates or changes
- Online banking and mobile app features

### Response Style:
- Start with the exact chat message provided
- Continue naturally with current, accurate information
- Cite specific sources when helpful
- Be comprehensive but concise
"""


class FAQSearchAgent:
    """Search agent that performs web searches for banking information using Gemini Live API with Google Search grounding."""

    def __init__(self):
        """Initialize the FAQ search agent."""
        self.client = Client(http_options={"api_version": "v1alpha"})
        self.system_prompt = make_voice_system_prompt(FAQ_SEARCH_PROMPT)

        # Create Live API configuration with Google Search tool
        self.live_config = LiveConnectConfig(
            system_instruction=self.system_prompt,
            tools=[{"google_search": {}}],
            response_modalities=["TEXT"],
        )

    def _parse_search_queries(self, msg: LiveServerMessage) -> set[str]:
        """Parse search queries from message content."""
        queries = set()

        # Parse queries from grounding metadata
        if (
            msg.server_content
            and msg.server_content.grounding_metadata
            and msg.server_content.grounding_metadata.web_search_queries
        ):
            queries.update(msg.server_content.grounding_metadata.web_search_queries)

        # Parse queries from executable_code parts
        if msg.server_content and msg.server_content.model_turn:
            for part in msg.server_content.model_turn.parts:
                if hasattr(part, "executable_code") and part.executable_code:
                    code = part.executable_code.code
                    # Extract queries from google_search.search(queries=[...]) pattern
                    pattern = r"google_search\.search\(queries=\[(.*?)\]"
                    match = re.search(pattern, code)
                    if match:
                        queries_str = match.group(1)
                        # Extract individual quoted strings
                        query_pattern = r'"([^"]*)"'
                        extracted_queries = re.findall(query_pattern, queries_str)
                        queries.update(extracted_queries)

        return queries

    def _parse_search_pages(self, msg: LiveServerMessage) -> set[str]:
        """Parse search page titles from message grounding metadata."""
        pages = set()

        # Parse page titles from grounding metadata
        if (
            msg.server_content
            and msg.server_content.grounding_metadata
            and msg.server_content.grounding_metadata.grounding_chunks
        ):
            for chunk in msg.server_content.grounding_metadata.grounding_chunks:
                if chunk.web and chunk.web.title:
                    pages.add(chunk.web.uri)

        return pages

    async def get_search_response(
        self, user_msg: str, chat_msg: str, message_history: list[Content]
    ) -> AsyncGenerator[Union[AgentResponse, ToolResult], None]:
        """Get bank information to complete the chat agent's response using Gemini Live API with web search.

        Args:
            user_msg: The user's original question
            chat_msg: The chat agent's partial response to complete
            message_history: Previous conversation messages in Gemini format (excluding the current user message)

        Yields:
            AgentResponse: Streaming text chunks from Gemini (with chat_msg prefix stripped)
            ToolCall: Google Search tool call with queries and pages
        """
        # Add the search-specific user message focused on the bank name
        search_prompt = Content(
            role="user",
            parts=[
                Part(
                    text=f"""User asked: "{user_msg}"
Chat agent started responding with: "{chat_msg}"

Search for current {BANK_NAME} information and provide a complete response that:
1. Starts with the EXACT chat message: "{chat_msg}"
2. Then continues naturally with the real-time {BANK_NAME} data to complete the sentence
3. Focus on official {BANK_NAME} sources and current information"""
                )
            ],
        )

        search_queries = set()
        search_pages = set()
        chat_prefix_buffer = ""
        chat_msg_len = len(chat_msg)

        # Filter message history to only include messages with text content
        text_only_history = []
        for msg in message_history:
            # Only include messages that have text parts
            if any(hasattr(part, "text") and part.text for part in msg.parts):
                text_only_history.append(msg)

        async with self.client.aio.live.connect(model=FAQ_SEARCH_MODEL_ID, config=self.live_config) as stream:
            await stream.send_client_content(turns=text_only_history + [search_prompt], turn_complete=True)

            async for msg in stream.receive():
                # Handle text content
                if msg.text:
                    if len(chat_prefix_buffer) < chat_msg_len:
                        chat_prefix_buffer += msg.text

                        # Check if we have enough to determine prefix match
                        if len(chat_prefix_buffer) >= chat_msg_len:
                            if chat_prefix_buffer.startswith(chat_msg):
                                # Extract text after the chat_msg prefix
                                remaining = chat_prefix_buffer[chat_msg_len:].lstrip()
                                if remaining:
                                    yield AgentResponse(content=remaining)
                            else:
                                # No prefix match, yield all buffered content
                                yield AgentResponse(content=chat_prefix_buffer)
                    else:
                        # Already past prefix checking, just yield
                        yield AgentResponse(content=msg.text)

                # Extract search queries and pages from the message
                search_queries.update(self._parse_search_queries(msg))
                search_pages.update(self._parse_search_pages(msg))

            # After streaming completes, yield Google Search data if any was collected
            if search_queries or search_pages:
                logger.info(f"❓ FAQ search queries used: {list(search_queries)}")
                logger.info(f"📄 FAQ pages referenced: {list(search_pages)}")

                yield ToolResult(
                    tool_name="web_search",
                    tool_args={"queries": list(search_queries)},
                    result={"pages": list(search_pages)},
                )
