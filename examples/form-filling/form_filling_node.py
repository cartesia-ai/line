"""
FormFillingNode - Voice-optimized ReasoningNode for conducting questionnaires
"""

from typing import AsyncGenerator, Union

from config import DEFAULT_MODEL_ID, DEFAULT_TEMPERATURE
from form_manager import FormManager
from form_tools import RecordAnswerArgs, RecordAnswerTool, record_answer
from google.genai import types as gemini_types
from loguru import logger

from line.events import AgentResponse, EndCall, LogMetric
from line.nodes.conversation_context import ConversationContext
from line.nodes.reasoning import ReasoningNode
from line.tools.system_tools import EndCallArgs, end_call
from line.utils.gemini_utils import convert_messages_to_gemini


class FormFillingNode(ReasoningNode):
    """
    Voice-optimized ReasoningNode for conducting questionnaires using Gemini streaming.
    - Uses ReasoningNode's template method generate() for consistent flow
    - Implements process_context() for Gemini streaming and form management
    - Integrates with record_answer and end_call tools
    - Manages form state and question progression
    """

    def __init__(
        self,
        system_prompt: str,
        gemini_client,
        form_path: str,
        model_id: str = DEFAULT_MODEL_ID,
        temperature: float = DEFAULT_TEMPERATURE,
        max_context_length: int = 10,  # Keep context short for form filling
        max_output_tokens: int = 1000,
    ):
        """
        Initialize the Form Filling reasoning node with Gemini configuration

        Args:
            system_prompt: System prompt for the LLM
            gemini_client: Google Gemini client instance
            form_path: Path to the form YAML configuration file
            model_id: Gemini model ID to use
            temperature: Temperature for generation
            max_context_length: Maximum number of conversation turns to keep (kept short)
        """
        super().__init__(system_prompt=system_prompt, max_context_length=max_context_length)

        self.client = gemini_client
        self.model_id = model_id
        self.temperature = temperature

        # Form management
        self.form_manager = FormManager(form_path)

        # Interruption support
        self.stop_generation_event = None

        # Create generation config with form tools
        self.generation_config = gemini_types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=self.temperature,
            tools=[RecordAnswerTool.to_gemini_tool()],
            max_output_tokens=max_output_tokens,
            thinking_config=gemini_types.ThinkingConfig(thinking_budget=0),
        )

        logger.info(f"FormFillingNode initialized with model: {model_id} and form: {form_path}")

    async def process_context(
        self, context: ConversationContext
    ) -> AsyncGenerator[Union[AgentResponse, EndCall], None]:
        """
        Process the conversation context for form filling and yield responses from Gemini.

        This method handles the form filling logic:
        1. Check if we need to ask the next question
        2. Process user responses through Gemini
        3. Handle tool calls for recording answers
        4. Manage form completion

        Yields:
            AgentResponse: Text chunks from Gemini
            EndCall: end_call Event when form is complete
        """
        # Check if we need to ask the next question
        current_question = self.form_manager.get_current_question()

        if not current_question:
            # Form is complete
            summary = self.form_manager.get_form_summary()
            logger.info(f"📋 Form completed! Summary: {summary}")

            # End the call with form completion message
            args = EndCallArgs(
                goodbye_message="Thank you for completing the questionnaire. Have a great day!"
            )
            async for item in end_call(args):
                yield item
            return

        # If we have no conversation events, ask the first question
        if not context.events:
            question_text = self.form_manager.format_question_for_llm(current_question)
            logger.info(f"📝 Asking first question: {current_question['id']}")
            yield AgentResponse(content=question_text)
            return

        # Note that we use text_events_only=True here to improve the accuracy of tool calling by filtering out prior ToolResults
        messages = convert_messages_to_gemini(context.get_committed_events(), text_events_only=True)

        # Add current question context to help the LLM
        question_text = self.form_manager.format_question_for_llm(current_question)
        question_context = f"\\n\\nCurrent question: {question_text}\\nRemember to use the record_answer tool when the user provides their answer."

        # Add context about the current question to the system instruction
        enhanced_config = gemini_types.GenerateContentConfig(
            system_instruction=self.system_prompt + question_context,
            temperature=self.temperature,
            tools=[RecordAnswerTool.to_gemini_tool()],
            max_output_tokens=self.generation_config.max_output_tokens,
            thinking_config=gemini_types.ThinkingConfig(thinking_budget=0),
        )

        user_message = context.get_latest_user_transcript_message()
        if user_message:
            logger.info(f'🧠 Processing user message for question {current_question["id"]}: "{user_message}"')

        full_response = ""
        stream: AsyncGenerator[
            gemini_types.GenerateContentResponse
        ] = await self.client.aio.models.generate_content_stream(
            model=self.model_id,
            contents=messages,
            config=enhanced_config,
        )

        async for msg in stream:
            if msg.text:
                full_response += msg.text
                yield AgentResponse(content=msg.text)

            if msg.function_calls:
                for function_call in msg.function_calls:
                    if function_call.name == RecordAnswerTool.name():
                        answer = function_call.args.get("answer", "")
                        logger.info(f"📝 Recording answer: {answer}")

                        # Get current question info for metric logging BEFORE recording the answer
                        current_question = self.form_manager.get_current_question()

                        # Record the answer in form manager
                        success = self.form_manager.record_answer(answer)

                        if success:
                            # Log metric for the answered question
                            if current_question:
                                metric_name = current_question["id"]
                                yield LogMetric(name=metric_name, value=answer)
                                logger.info(f"📊 Logged metric: {metric_name}={answer}")

                            # Clear context to keep conversation focused
                            self.clear_context()

                            # Check if form is complete
                            if self.form_manager.is_form_complete():
                                summary = self.form_manager.get_form_summary()
                                logger.info(f"🎉 Form completed! Final summary: {summary}")
                                yield AgentResponse(
                                    content="Perfect! I've recorded all your answers. Thank you for completing the questionnaire."
                                )
                            else:
                                # Ask next question
                                next_question = self.form_manager.get_current_question()
                                if next_question:
                                    next_question_text = self.form_manager.format_question_for_llm(
                                        next_question
                                    )
                                    yield AgentResponse(content=f"Thank you. {next_question_text}")
                        else:
                            yield AgentResponse(
                                content="I didn't quite understand your answer. Could you please clarify?"
                            )

                        # Yield tool result for observability
                        async for item in record_answer(RecordAnswerArgs(answer=answer)):
                            yield item

        if full_response:
            logger.info(f'🤖 Agent response: "{full_response}" ({len(full_response)} chars)')
