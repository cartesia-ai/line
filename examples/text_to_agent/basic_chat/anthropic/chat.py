"""ChatNode - Handles basic conversations using Anthropic."""

from typing import AsyncGenerator, Optional

from config import CHAT_MODEL_ID, CHAT_TEMPERATURE, MAX_TOKENS
from anthropic import AsyncAnthropic
from loguru import logger
from prompts import get_chat_system_prompt

from line import ConversationContext, ReasoningNode
from line.events import AgentResponse, EndCall
from line.tools.system_tools import EndCallArgs, EndCallTool, end_call
from line.utils.anthropic_utils import convert_messages_to_anthropic


class ChatNode(ReasoningNode):
    """Voice-optimized ReasoningNode for basic chat using Anthropic streaming.

    Provides simple conversation capabilities without external tools or search.
    """

    def __init__(self, api_key: Optional[str] = None, max_context_length: int = 100):
        """Initialize the Voice reasoning node with proven Anthropic configuration.

        Args:
            api_key: Anthropic API key. If None, will try to get from environment.
            max_context_length: Maximum number of conversation turns to keep.
        """
        self.system_prompt = get_chat_system_prompt()
        super().__init__(self.system_prompt, max_context_length)

        self.tools = []

        if not api_key:
            raise ValueError(
                "Anthropic API key is required. Please set the ANTHROPIC_API_KEY "
                "environment variable or pass it as a parameter."
            )

        self.client = AsyncAnthropic(api_key=api_key)
        self.tools.append(EndCallTool.to_anthropic_tool())

    async def process_context(
        self, context: ConversationContext
    ) -> AsyncGenerator[AgentResponse | EndCall, None]:
        """Basic chat processing using Anthropic streaming.

        Args:
            context: ConversationContext with messages, tools, and metadata

        Yields:
            AgentResponse: Streaming text chunks from Anthropic
            EndCall: End call event
        """
        user_message = context.get_latest_user_transcript_message()
        if user_message:
            logger.info(f'🧠 Processing user message: "{user_message}"')

        full_response = ""
        messages = convert_messages_to_anthropic(context.events)

        async with self.client.messages.stream(
            model=CHAT_MODEL_ID,
            max_tokens=MAX_TOKENS,
            temperature=CHAT_TEMPERATURE,
            system=self.system_prompt,
            messages=messages,
            tools=self.tools,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        full_response += event.delta.text
                        yield AgentResponse(content=event.delta.text)

                elif event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        logger.info(f"🤖 Tool use start: {event.content_block.name}")

                elif event.type == "content_block_stop":
                    if event.content_block.type == "tool_use":
                        tool_use = event.content_block
                        logger.info(
                            f"🤖 Tool use complete: {tool_use.name}, input={tool_use.input}"
                        )
                        if tool_use.name == EndCallTool.name():
                            goodbye_message = tool_use.input.get(
                                "goodbye_message", "Goodbye!"
                            )
                            args = EndCallArgs(goodbye_message=goodbye_message)
                            logger.info(
                                f"🤖 End call tool called. Ending conversation with goodbye message: "
                                f"{args.goodbye_message}"
                            )
                            async for item in end_call(args):
                                yield item

        if full_response:
            logger.info(
                f'🤖 Agent response: "{full_response}" ({len(full_response)} chars)'
            )
