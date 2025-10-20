"""Voice agent that fills web forms in real-time using Stagehand.

This module implements a ReasoningNode subclass that conducts voice
conversations while automatically filling web forms in the background.
The agent uses Stagehand for browser automation and handles async form
filling during conversation without blocking the voice flow.
"""

import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional, Union

from config import DEFAULT_MODEL_ID, DEFAULT_TEMPERATURE
from google.genai import types as gemini_types
from loguru import logger
from pydantic import BaseModel, Field
from stagehand_form_filler import StagehandFormFiller

from line.events import AgentResponse, EndCall, ToolResult
from line.nodes.conversation_context import ConversationContext
from line.nodes.reasoning import ReasoningNode
from line.tools.system_tools import EndCallArgs, end_call
from line.utils.gemini_utils import convert_messages_to_gemini


class RecordFormFieldArgs(BaseModel):
    """Arguments for recording a form field"""

    field_name: str = Field(description="The form field being filled")
    value: str = Field(description="The value to enter in the field")


class RecordFormFieldTool:
    """Tool for recording form field values"""

    @staticmethod
    def name() -> str:
        return "record_form_field"

    @staticmethod
    def description() -> str:
        return "Record a value for a form field that needs to be filled"

    @staticmethod
    def parameters() -> Dict:
        return RecordFormFieldArgs.model_json_schema()

    @staticmethod
    def to_gemini_tool():
        """Convert to Gemini tool format.

        Returns:
            A Gemini Tool object with function declarations.
        """
        return gemini_types.Tool(
            function_declarations=[
                gemini_types.FunctionDeclaration(
                    name=RecordFormFieldTool.name(),
                    description=RecordFormFieldTool.description(),
                    parameters=RecordFormFieldTool.parameters(),
                )
            ]
        )


@dataclass
class FormQuestion:
    """Represents a question to ask the user"""

    field_name: str
    question: str
    field_type: str = "text"
    required: bool = True


class FormFillingNode(ReasoningNode):
    """Voice agent that fills web forms while conducting conversations.

    This class uses Stagehand to read and fill web forms dynamically,
    maintains conversation flow while automating browser actions, and
    intelligently extracts form structure and asks relevant questions.
    """

    def __init__(
        self,
        system_prompt: str,
        gemini_client,
        form_url: str,
        model_id: str = DEFAULT_MODEL_ID,
        temperature: float = DEFAULT_TEMPERATURE,
        max_context_length: int = 15,
        max_output_tokens: int = 1000,
    ):
        """Initialize the Form Filling node with Stagehand integration.

        Args:
            system_prompt: System prompt for the LLM.
            gemini_client: Google Gemini client instance.
            form_url: URL of the web form to fill.
            model_id: Gemini model ID.
            temperature: Temperature for generation.
            max_context_length: Maximum conversation context length.
            max_output_tokens: Maximum tokens for generation.
        """
        super().__init__(system_prompt=system_prompt, max_context_length=max_context_length)

        self.client = gemini_client
        self.model_id = model_id
        self.temperature = temperature

        # Browser automation
        self.form_url = form_url
        self.stagehand_filler: Optional[StagehandFormFiller] = None

        # Form state
        self.collected_data: Dict[str, str] = {}
        # Pre-initialize questions so conversation can start immediately
        self.questions: List[FormQuestion] = self._create_questions()
        self.current_question_index = 0

        # Browser initialization
        self.browser_init_task = None
        self.browser_initializing = False

        # Enhanced prompt for form filling
        enhanced_prompt = (
            system_prompt
            + """
        
        You are conducting a voice conversation to help fill out a web
        form. As you collect information, it's being entered into an
        actual online form in real-time. Ask natural questions to gather
        the required information. Use the record_form_field tool to save
        each piece of information. Keep the conversation friendly and
        natural.
        """
        )

        # Generation config
        self.generation_config = gemini_types.GenerateContentConfig(
            system_instruction=enhanced_prompt,
            temperature=self.temperature,
            tools=[RecordFormFieldTool.to_gemini_tool()],
            max_output_tokens=max_output_tokens,
            thinking_config=gemini_types.ThinkingConfig(thinking_budget=0),
        )

        logger.info(f"FormFillingNode initialized for form: {form_url}")

        # Track if form was submitted
        self.form_submitted = False

    async def cleanup_and_submit(self) -> None:
        """Ensure form is submitted and cleanup when call ends.

        Returns:
            None.
        """
        # Submit form if we have any data and haven't submitted yet
        if not self.form_submitted and self.collected_data and self.stagehand_filler:
            logger.info("Call ending - auto-submitting form with collected data")
            try:
                await self._submit_form()
            except Exception as e:
                logger.error(f"Error during cleanup submission: {e}")

        # Clean up browser
        if self.stagehand_filler:
            await self.stagehand_filler.cleanup()

    async def _initialize_browser(self) -> None:
        """Initialize browser and extract form fields.

        Returns:
            None.
        """
        # Prevent multiple initializations
        if self.browser_initializing or self.stagehand_filler:
            logger.info("Browser already initializing or initialized, skipping")
            return

        self.browser_initializing = True
        try:
            logger.info("Initializing browser and analyzing form")
            self.stagehand_filler = StagehandFormFiller(form_url=self.form_url)
            await self.stagehand_filler.initialize()

            logger.info("Browser ready, form can now be filled")

        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            self.browser_initializing = False
            raise
        finally:
            self.browser_initializing = False

    def _create_questions(self) -> List[FormQuestion]:
        """Create questions for the form.

        Returns:
            A list of FormQuestion objects to ask the user.
        """
        # Define questions for the form fields we know about
        # This matches form at https://forms.fillout.com/t/34ccsqafUFus
        form_questions = [
            FormQuestion(
                field_name="full_name", question="What is your full name?", field_type="text", required=True
            ),
            FormQuestion(
                field_name="email", question="What is your email address?", field_type="email", required=True
            ),
            FormQuestion(
                field_name="phone", question="What is your phone number?", field_type="phone", required=False
            ),
            FormQuestion(
                field_name="work_eligibility",
                question="Are you legally eligible to work in this country?",
                field_type="radio",
                required=True,
            ),
            FormQuestion(
                field_name="availability_type",
                question=("What's your availability - temporary, part-time, or full-time?"),
                field_type="radio",
                required=True,
            ),
            FormQuestion(
                field_name="role_selection",
                question=(
                    "Which role are you applying for? We have openings "
                    "for Sales Manager, IT Support, Recruiting, "
                    "Software Engineer, or Marketing Specialist."
                ),
                field_type="checkbox",
                required=True,
            ),
            FormQuestion(
                field_name="previous_experience",
                question="Have you worked in a similar role before?",
                field_type="radio",
                required=True,
            ),
            FormQuestion(
                field_name="skills_experience",
                question=(
                    "What relevant skills and experience do you have "
                    "that make you a strong candidate for this position?"
                ),
                field_type="textarea",
                required=True,
            ),
            FormQuestion(
                field_name="additional_info",
                question=("Is there anything else you'd like to tell us about yourself?"),
                field_type="textarea",
                required=False,
            ),
        ]

        return form_questions

    async def _fill_form_field_async(self, field_name: str, value: str) -> None:
        """Fill a form field asynchronously in background (non-blocking).

        Args:
            field_name: The name of the form field to fill.
            value: The value to enter in the field.
        """
        try:
            # Wait for browser initialization if needed
            if self.browser_init_task:
                logger.info(f"Waiting for browser to initialize before filling {field_name}")
                await self.browser_init_task

            logger.info(f"Filling field '{field_name}' with: {value} in background")
            # Use StagehandFormFiller's fill_field method which
            # handles the mapping
            success = await self.stagehand_filler.fill_field(field_name, value)

            if success:
                logger.info(f"Successfully filled field: {field_name} in browser")
            else:
                logger.warning(f"Failed to fill field: {field_name}")

        except Exception as e:
            logger.error(f"Error filling field {field_name}: {e}")
            raise  # Re-raise so background task can catch it

    async def _submit_form(self) -> bool:
        """Submit the completed form.

        Returns:
            True if submission succeeded, False otherwise.
        """
        # Wait for browser initialization if needed
        if self.browser_init_task and not self.stagehand_filler:
            logger.info("Waiting for browser to initialize before submitting form")
            await self.browser_init_task

        if not self.stagehand_filler:
            return False

        try:
            logger.info("Submitting web form with collected data")
            logger.info(f"Data collected: {self.collected_data}")

            # Ensure StagehandFormFiller has all collected data
            # (it should already have it from fill_field calls,
            # but ensure consistency)
            self.stagehand_filler.collected_data.update(self.collected_data)

            # Use StagehandFormFiller's submit_form method which now
            # uses collected_data
            success = await self.stagehand_filler.submit_form()

            if success:
                logger.info("Form submitted successfully!")
                return True
            else:
                logger.warning("Form submission may have failed")
                return False

        except Exception as e:
            logger.error(f"Error submitting form: {e}")
            return False

    def get_current_question(self) -> Optional[FormQuestion]:
        """Get the current question to ask.

        Returns:
            The current FormQuestion or None if all questions answered.
        """
        if self.current_question_index < len(self.questions):
            return self.questions[self.current_question_index]
        return None

    async def process_context(
        self, context: ConversationContext
    ) -> AsyncGenerator[Union[AgentResponse, EndCall], None]:
        """Process conversation context with real-time form filling.

        Args:
            context: The conversation context with events.

        Yields:
            AgentResponse: Text responses to the user.
            EndCall: Call termination when form is complete.
        """
        # Initialize browser on first call (non-blocking)
        if not self.browser_init_task and not self.stagehand_filler:
            self.browser_init_task = asyncio.create_task(self._initialize_browser())
            logger.info("Browser initialization started in background")

        # Get current question after initialization
        current_question = self.get_current_question()
        question_name = current_question.field_name if current_question else "None"
        logger.info(f"Current question: {question_name}")
        logger.info(f"Question index: {self.current_question_index}/{len(self.questions)}")
        logger.info(f"Events count: {len(context.events)}")

        # Check latest event to determine what to do
        latest_event = context.events[-1] if context.events else None
        is_agent_response = isinstance(latest_event, AgentResponse) if latest_event else False

        # Handle initial greeting - speak first when conversation starts
        if not context.events:
            logger.info("Starting conversation - Agent speaks first")
            initial_greeting = (
                "Hello! I'm here to help you fill out an application "
                "form today. I'll ask you a series of questions and "
                "fill in the form as we go. Ready to get started?"
            )
            yield AgentResponse(content=initial_greeting)
            return

        # If last event was our greeting, and user responded, ask
        # first question
        if len(context.events) == 2 and not is_agent_response and self.current_question_index == 0:
            user_message = context.get_latest_user_transcript_message()
            if user_message and current_question:
                logger.info(f"User ready to start: '{user_message}'")
                logger.info(f"Asking first question: {current_question.field_name}")
                yield AgentResponse(content=f"Great! Let's begin. {current_question.question}")
                return

        # Check if all questions have been answered
        # Only submit if we've actually collected data
        if not current_question and self.current_question_index > 0 and len(self.collected_data) > 0:
            # All questions answered - submit the form
            logger.info(f"All {self.current_question_index} questions answered")
            logger.info(f"Collected data for {len(self.collected_data)} fields")

            submission_success = await self._submit_form()
            self.form_submitted = True

            if submission_success:
                goodbye = "Perfect! I've submitted your application. Thank you!"
            else:
                goodbye = "Thank you for providing all the information. Your responses have been recorded."

            # Clean up
            await self.cleanup_and_submit()

            # End call
            args = EndCallArgs(goodbye_message=goodbye)
            async for item in end_call(args):
                yield item
            return

        # Guard against no questions or empty state
        if not current_question and self.current_question_index == 0:
            logger.warning("No questions available or not properly initialized")
            return

        # Process user response
        messages = convert_messages_to_gemini(context.events, text_events_only=True)

        # Add context about current question
        question_context = f"""
        
        Current form field: {current_question.field_name}
        Question: {current_question.question}
        
        Listen to the user's response and use the record_form_field
        tool to save it. Then acknowledge their answer naturally.
        """

        enhanced_config = gemini_types.GenerateContentConfig(
            system_instruction=(self.generation_config.system_instruction + question_context),
            temperature=self.temperature,
            tools=[RecordFormFieldTool.to_gemini_tool()],
            max_output_tokens=self.generation_config.max_output_tokens,
            thinking_config=gemini_types.ThinkingConfig(thinking_budget=0),
        )

        # Get user's latest message
        user_message = context.get_latest_user_transcript_message()
        if user_message:
            logger.info(f'User response: "{user_message}"')

        # Stream Gemini response
        full_response = ""
        stream = await self.client.aio.models.generate_content_stream(
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
                    if function_call.name == RecordFormFieldTool.name():
                        field_name = function_call.args.get("field_name", current_question.field_name)
                        value = function_call.args.get("value", "")

                        logger.info(f"Recording: {field_name} = {value}")

                        # Store data first
                        self.collected_data[field_name] = value
                        # Fill the form field asynchronously in background
                        # (non-blocking)
                        asyncio.create_task(self._fill_form_field_async(field_name, value))
                        # Log the collected data
                        logger.info(f"Collected: {field_name}={value}")
                        # Move to next question immediately
                        # (don't wait for form filling)
                        self.current_question_index += 1

                        # Clear context
                        self.clear_context()

                        # Get next question
                        next_question = self.get_current_question()
                        if next_question:
                            yield AgentResponse(content=f"Great! {next_question.question}")

                        # Yield tool result immediately
                        yield ToolResult(
                            tool_name="record_form_field",
                            tool_args={"field_name": field_name, "value": value},
                            result=f"Recorded: {field_name}={value}",
                        )

        if full_response:
            logger.info(f'Agent response: "{full_response}"')
