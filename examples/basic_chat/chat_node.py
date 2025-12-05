"""
ChatNode - Voice-optimized ReasoningNode implementation using OpenAI
"""

import json
from typing import AsyncGenerator, Union

from line import Message
from line.utils.log_aiter import log_aiter_func

from context import TimeZoneInfo, find_availability
from config import CHAT_MODEL_ID
from loguru import logger
from openai import AsyncOpenAI, pydantic_function_tool
from openai._streaming import AsyncStream
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)

from line.events import AgentResponse, EndCall, ToolCall, ToolResult
from line.nodes.conversation_context import ConversationContext
from line.nodes.reasoning import ReasoningNode
from line.tools.system_tools import EndCallArgs, EndCallTool, end_call
from line.utils.openai_utils import convert_messages_to_openai

tool = pydantic_function_tool(TimeZoneInfo, name='find_agent_availability', description="Find the times when licensed agent is available to make a callback")

class ChatNode(ReasoningNode):
    """
    Voice-optimized ReasoningNode using template method pattern with OpenAI Responses API.
    - Uses ReasoningNode's template method generate() for consistent flow
    - Implements process_context() for OpenAI streaming
    - Integrates with end_call tool
    """

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        system_prompt: str,
        max_context_length: int = 100,
    ):
        """
        Initialize the Voice reasoning node with OpenAI configuration

        Args:
            system_prompt: System prompt for the LLM
            max_context_length: Maximum number of conversation turns to keep
        """
        super().__init__(system_prompt=system_prompt, max_context_length=max_context_length)

        # Initialize OpenAI client
        self.client = openai_client

        logger.info(f"ChatNode initialized with OpenAI model: {CHAT_MODEL_ID}")


    async def warmup(self) -> AsyncGenerator[Union[AgentResponse, EndCall], None]:
        resp = self.client.responses.create(
            model='gpt-4.1',
            instructions="say hello",
            input="",
            temperature=0.3,
            service_tier="priority",
        )
        yield resp

    def clean_tools_context(self, context: ConversationContext) -> None:
        # only check tool calls with raw_response, since only those are passed to OpenAI
        to_check = [ev for ev in context.events if isinstance(ev, ToolCall) and ev.raw_response]
        if not to_check:
            return

        result_call_ids = [ev.tool_call_id for ev in context.events if isinstance(ev, ToolResult)]
        for ev in to_check:
            if ev.tool_call_id not in result_call_ids:
                # nuke the raw response, which will cause the tool call to not be passed to OpenAI
                # this is important, because otherwise OpenAI will fail the call due to unresolved tool call
                ev.raw_response.clear()

    async def process_context(
        self, context: ConversationContext
    ) -> AsyncGenerator[Union[AgentResponse, EndCall, ToolCall, ToolResult], None]:
        """
        Process the conversation context and yield responses from OpenAI.

        Yields:
            AgentResponse: Text chunks from OpenAI
            EndCall: end_call Event
        """
        self.clean_tools_context(context)
        # Convert context events to OpenAI format
        messages = convert_messages_to_openai(context.events)
        # logger.info(f'MESSAGES: {messages}')

        user_message = context.get_latest_user_transcript_message()
        if user_message:
            logger.info(f'🧠 Processing user message: "{user_message}"')

        # Ensure we have at least one user message for context
        if not any(msg.get("role") == "user" for msg in messages):
            logger.warning("No user message found in conversation")
            return

        # Make the streaming request using Responses API with optimizations
        stream: AsyncStream[ResponseStreamEvent]
        async with self.client.responses.stream(
            model='gpt-4.1',
            instructions=self.system_prompt,
            input=messages,
            tools=[EndCallTool.to_openai_tool(), tool],
            #reasoning={"effort": "low"},
            text={"verbosity": "medium"},
            temperature=0.3,
            service_tier="priority",
        ) as stream:

            full_response = ""
            ended = False
            output_index: None | int = None
            async for event in stream:
                if isinstance(event, ResponseTextDeltaEvent):
                    if output_index is None:
                        output_index = event.output_index
                    elif output_index != event.output_index:
                        # we only take text deltas from the first output item of text type
                        continue
                    full_response += event.delta
                    yield AgentResponse(content=event.delta)

                if isinstance(event, ResponseOutputItemDoneEvent) and isinstance(
                    event.item, ResponseFunctionToolCall
                ):
                    if event.item.name == EndCallTool.name():
                        args = json.loads(event.item.arguments)
                        if output_index is not None:
                            # LLM has already generated some text, so we will ignore the end call message
                            args["goodbye_message"] = "" if "goodbye" in full_response.lower() else "Goodbye!"
                        end_call_args = EndCallArgs(goodbye_message=args.get("goodbye_message", "Goodbye!"))
                        logger.info(
                            f"🤖 End call tool called. Ending conversation with goodbye message: "
                            f"{end_call_args.goodbye_message}"
                        )
                        async for item in end_call(end_call_args):
                            yield item
                        ended = True
                    elif event.item.name == tool['function']['name']:
                        args = json.loads(event.item.arguments)
                        if output_index is None:
                            # If LLM has already generated text response, it probably says something like "wait until I..."
                            # and therefore there is no need for us to say anything else.
                            # However, if LLM has not generated any text, then we say the canned phrase:
                            yield AgentResponse(content="Just a moment while I check the agent availability...")
                        yield ToolCall(tool_name=event.item.name, tool_args=args, tool_call_id=event.item.call_id, raw_response=event.item.model_dump())
                        availability = find_availability(TimeZoneInfo(**args))
                        yield ToolResult(tool_name=event.item.name, tool_args=args, result=availability, tool_call_id=event.item.call_id)


            if not ended and "goodbye" in full_response.lower():
                yield EndCall()

            if full_response:
                logger.info(f'🤖 Agent response: "{full_response}" ({len(full_response)} chars)')

    def on_interrupt_generate(self, message: Message) -> None:
        """Handle interrupt event."""
        logger.info(f'on interrupt generate was called: {message}, events: {self.conversation_events}')
        super().on_interrupt_generate(message)
