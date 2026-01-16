"""
GeminiReasoningNode - Voice-optimized ReasoningNode implementation using proven Gemini logic
"""

from typing import AsyncGenerator, Optional, Union

from config import DEFAULT_MODEL_ID, DEFAULT_TEMPERATURE
from google import genai
from google.genai import types as gemini_types
from loguru import logger

from line.bus import Message
from line.events import AgentResponse, DTMFOutputEvent, EndCall, LogMetric
from line.nodes.conversation_context import ConversationContext
from line.nodes.reasoning import ReasoningNode
from line.utils.dtmf_lookahead_buffer import DTMFLookAheadStringBuffer
from line.utils.gemini_utils import convert_messages_to_gemini
from line.utils.log_aiter import log_aiter_func


class ChatNode(ReasoningNode):
    """
    Voice-optimized ReasoningNode using template method pattern with Gemini streaming.
    - Uses ReasoningNode's template method generate() for consistent flow
    - Implements process_context() for Gemini streaming
    - Integrates with end_call tool
    """

    def __init__(
        self,
        system_prompt: str,
        gemini_client: Optional[genai.Client] = None,
        model_id: str = DEFAULT_MODEL_ID,
        temperature: float = DEFAULT_TEMPERATURE,
        max_context_length: int = 100,
        max_output_tokens: int = 1000,
    ):
        """
        Initialize the Voice reasoning node with proven Gemini configuration

        Args:
            system_prompt: System prompt for the LLM
            gemini_client: Google Gemini client instance.
                If not provided, a canned (dummy) response will be streamed.
            model_id: Gemini model ID to use
            temperature: Temperature for generation
            max_context_length: Maximum number of conversation turns to keep
        """
        super().__init__(system_prompt=system_prompt, max_context_length=max_context_length)

        self.client = gemini_client
        self.model_id = model_id
        self.temperature = temperature

        # Interruption support
        self.stop_generation_event = None

        # Create generation config using utility function
        self.generation_config = gemini_types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=0.2,
            tools=[],
            max_output_tokens=max_output_tokens,
            thinking_config=gemini_types.ThinkingConfig(thinking_budget=0),
        )

        logger.info(f"GeminiNode initialized with model: {model_id}")

        self.most_recent_dtmf_message = None
        self.end_call_timer = None

    @log_aiter_func(message="Processing context")
    async def process_context(
        self, context: ConversationContext
    ) -> AsyncGenerator[Union[AgentResponse, EndCall], None]:
        """
        Process the conversation context and yield responses from Gemini.

        Yields:
            AgentResponse: Text chunks from Gemini
            AgentEndCall: end_call Event
        """
        if not context.events:
            logger.info("No messages to process")
            return

        messages = convert_messages_to_gemini(
            context.events,
            {
                DTMFOutputEvent: serialize_dtmf_output_event,
            },
        )

        user_message = context.get_latest_user_transcript_message()
        if user_message:
            logger.info(f'🧠 Processing user message: "{user_message}"')

        full_response = ""
        buffer = DTMFLookAheadStringBuffer()

        stream = await self.client.aio.models.generate_content_stream(
            model=self.model_id,
            contents=messages,
            config=self.generation_config,
        )

        # Confirm user pressed buttons
        full_response = ""

        # Process LLM content
        async for msg in stream:
            # Yield buffered items if present
            if msg.text:
                full_response += msg.text
                logger.info(f"raw text: {msg.text}")
                items = list(buffer.feed(msg.text))
                for item in items:
                    yield item

                    if isinstance(item, DTMFOutputEvent):
                        yield LogMetric(name="DTMF pressed", value=item.button)
                        logger.info(f"📊 Logged metric: DTMF pressed={item.button}")
                # yield AgentResponse(content=msg.text)

        if full_response:
            logger.info(f'🤖 Agent response: "{full_response}" ({len(full_response)} chars)')

        if "goodbye" in full_response.lower():
            yield EndCall()

        items = list(buffer.flush())
        for item in items:
            yield item

            if isinstance(item, DTMFOutputEvent):
                yield LogMetric(name="DTMF pressed", value=item.button)
                logger.info(f"📊 Logged metric: DTMF pressed={item.button}")

    def on_dtmf_output(self, message: Message) -> None:
        if not isinstance(message.event, DTMFOutputEvent):
            raise RuntimeError(f"expected DTMF output event but got {message.event=}")

        logger.info(f"Sent DTMF output: button={message.event.button}")

    @log_aiter_func(message="Warming up gemini client")
    async def warmup(self) -> AsyncGenerator[Union[AgentResponse, EndCall], None]:
        async for item in await self.client.aio.models.generate_content_stream(
            model=self.model_id,
            contents="ok",
            config=self.generation_config,
        ):
            yield item


def serialize_dtmf_output_event(event: DTMFOutputEvent) -> gemini_types.ModelContent:
    """
    Serialize the DTMF event to a string for gemini to process
    """
    return gemini_types.ModelContent(parts=[gemini_types.Part.from_text(text="")])
