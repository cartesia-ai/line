"""
GeminiReasoningNode - Voice-optimized ReasoningNode implementation using proven Gemini logic
"""

import asyncio
import random
from typing import AsyncGenerator, List, Optional, Union

from config import DEFAULT_MODEL_ID, DEFAULT_TEMPERATURE
from google import genai
from google.genai import types as gemini_types
from loguru import logger

from line.bus import Message
from line.events import AgentResponse, DTMFInputEvent, DTMFOutputEvent, DTMFStoppedEvent, EndCall
from line.nodes.conversation_context import ConversationContext
from line.nodes.reasoning import ReasoningNode
from line.tools.system_tools import EndCallArgs, EndCallTool, end_call
from line.utils.gemini_utils import convert_messages_to_gemini


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
            temperature=self.temperature,
            tools=[EndCallTool.to_gemini_tool()],
            max_output_tokens=max_output_tokens,
            thinking_config=gemini_types.ThinkingConfig(thinking_budget=0),
        )

        logger.info(f"GeminiNode initialized with model: {model_id}")

        self.most_recent_dtmf_message = None

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
                DTMFInputEvent: serialize_dtmf_input_event,
                DTMFOutputEvent: serialize_dtmf_output_event,
            },
        )

        user_message = context.get_latest_user_transcript_message()
        if user_message:
            logger.info(f'🧠 Processing user message: "{user_message}"')

        full_response = ""
        if not self.client:
            stream = canned_gemini_response_stream()
        else:
            stream = await self.client.aio.models.generate_content_stream(
                model=self.model_id,
                contents=messages,
                config=self.generation_config,
            )

        # Confirm user pressed buttons
        dtmf_events = get_dtmf_button_presses(context)
        if len(dtmf_events) > 0:
            buttons = "".join([event.button for event in dtmf_events])
            for button in buttons:
                if button.strip():
                    yield DTMFOutputEvent(button=button)

        # Process LLM content
        async for msg in stream:
            if msg.text:
                full_response += msg.text
                yield AgentResponse(content=msg.text)

            if msg.function_calls:
                for function_call in msg.function_calls:
                    if function_call.name == EndCallTool.name():
                        goodbye_message = function_call.args.get("goodbye_message", "Goodbye!")
                        args = EndCallArgs(goodbye_message=goodbye_message)
                        logger.info(
                            f"🤖 End call tool called. Ending conversation with goodbye message: "
                            f"{args.goodbye_message}"
                        )
                        async for item in end_call(args):
                            yield item

        if full_response:
            logger.info(f'🤖 Agent response: "{full_response}" ({len(full_response)} chars)')

    async def on_dtmf_event(self, message: Message):
        event = message.event
        if not isinstance(event, DTMFInputEvent):
            raise ValueError(f"Expected DTMFEventInput, got {type(event)=}: {event=}")

        self.most_recent_dtmf_message = message
        await asyncio.sleep(1.0)
        if self.most_recent_dtmf_message.id == message.id:
            logger.info(
                f"Publishing DTMFStoppedEvent as there was no other DTMF event queued. button pressed: {message.event.button}"
            )
            return DTMFStoppedEvent()

        logger.info(
            f"Did not publish DTMFStoppedEvent as there was another DTMF event queued. button pressed: {message.event.button}"
        )


def get_dtmf_button_presses(context: ConversationContext) -> List[DTMFInputEvent]:
    """
    Gets the most recent DTMF event from the context
    """
    i = len(context.events) - 1

    dtmf_events: List[DTMFInputEvent] = []

    # Start from the end and then scan until you hit an Agent Response
    while i >= 0:
        event = context.events[i]
        if isinstance(event, DTMFInputEvent):
            dtmf_events.append(event)
        i -= 1

        if isinstance(event, AgentResponse):
            break

    return list(reversed(dtmf_events))


async def canned_gemini_response_stream() -> AsyncGenerator[gemini_types.GenerateContentResponse, None]:
    """
    Stream a canned response from Gemini.

    This is to support running this example without a Gemini API key.
    """
    # Random messages about missing API key
    api_key_messages = [
        "I am a silly AI assistant because you didn't provide a Gemini API key. Add it to your environment variables.",
        "My brain is offline because I am missing a Gemini API key! Add the key to your environment variables.",
        "I'm like a car without keys - can't go anywhere. Add your Gemini API key for intelligence.",
    ]

    # Select a random message
    message = random.choice(api_key_messages)

    # Create the response structure
    part = gemini_types.Part(text=message)
    content = gemini_types.Content(parts=[part], role="model")
    candidate = gemini_types.Candidate(content=content, finish_reason="STOP")
    response = gemini_types.GenerateContentResponse(candidates=[candidate])

    await asyncio.sleep(0.005)
    yield response


def serialize_dtmf_input_event(event: DTMFInputEvent) -> gemini_types.UserContent:
    """
    Serialize the DTMF event to a string for gemini to process
    """
    return gemini_types.UserContent(parts=[gemini_types.Part.from_text(text=event.button)])


def serialize_dtmf_output_event(event: DTMFOutputEvent) -> gemini_types.ModelContent:
    """
    Serialize the DTMF event to a string for gemini to process
    """
    return gemini_types.ModelContent(parts=[gemini_types.Part.from_text(text="")])
