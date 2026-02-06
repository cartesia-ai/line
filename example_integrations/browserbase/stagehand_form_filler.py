"""Browser automation for filling web forms during voice conversations.

This module provides the StagehandFormFiller class which manages browser
automation for filling forms using Stagehand. It handles form field
mapping, field filling, and form submission.

The class provides deterministic conversation flow with passthrough tools
that directly emit the next question without LLM involvement.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
import os
from typing import Annotated, AsyncGenerator, Dict, List, Optional

from loguru import logger

from line.events import AgentEndCall, AgentSendText, OutputEvent
from line.llm_agent import ToolEnv, passthrough_tool


class FieldType(Enum):
    TEXT = "text"
    EMAIL = "email"
    PHONE = "phone"
    SELECT = "select"
    RADIO = "radio"
    CHECKBOX = "checkbox"
    TEXTAREA = "textarea"


@dataclass
class FormField:
    """Represents a form field with its metadata."""

    field_id: str
    field_type: FieldType
    label: str
    question: str  # The question to ask the user
    required: bool = False
    options: Optional[List[str]] = None


# Ordered list of questions to ask (matches v01 behavior)
FORM_QUESTIONS: List[FormField] = [
    FormField(
        field_id="full_name",
        field_type=FieldType.TEXT,
        label="What is your full name?",
        question="What is your full name?",
        required=True,
    ),
    FormField(
        field_id="email",
        field_type=FieldType.EMAIL,
        label="What is your email address?",
        question="What is your email address?",
        required=True,
    ),
    FormField(
        field_id="phone",
        field_type=FieldType.PHONE,
        label="What is your phone number?",
        question="What is your phone number?",
        required=False,
    ),
    FormField(
        field_id="work_eligibility",
        field_type=FieldType.RADIO,
        label="Are you legally eligible to work in this country?",
        question="Are you legally eligible to work in this country?",
        options=["Yes", "No"],
        required=True,
    ),
    FormField(
        field_id="availability_type",
        field_type=FieldType.RADIO,
        label="What's your availability?",
        question="What's your availability - temporary, part-time, or full-time?",
        options=["Temporary", "Part-time", "Full-time"],
        required=True,
    ),
    FormField(
        field_id="role_selection",
        field_type=FieldType.CHECKBOX,
        label="Which of these roles are you applying for?",
        question=(
            "Which role are you applying for? We have openings for "
            "Sales Manager, IT Support, Recruiting, Software Engineer, "
            "or Marketing Specialist."
        ),
        options=[
            "Sales manager",
            "IT Support",
            "Recruiting",
            "Software engineer",
            "Marketing specialist",
        ],
        required=True,
    ),
    FormField(
        field_id="previous_experience",
        field_type=FieldType.RADIO,
        label="Have you worked in a role similar to this one in the past?",
        question="Have you worked in a similar role before?",
        options=["Yes", "No"],
        required=True,
    ),
    FormField(
        field_id="skills_experience",
        field_type=FieldType.TEXTAREA,
        label="What relevant skills and experience do you have?",
        question=(
            "What relevant skills and experience do you have "
            "that make you a strong candidate for this position?"
        ),
        required=True,
    ),
    FormField(
        field_id="additional_info",
        field_type=FieldType.TEXTAREA,
        label="Anything else you'd like to let us know about you?",
        question="Is there anything else you'd like to tell us about yourself?",
        required=False,
    ),
]


class StagehandFormFiller:
    """Manages browser automation for filling forms using Stagehand.

    This class initializes the browser session eagerly on construction and provides
    deterministic conversation flow via passthrough tools. Each tool call
    records the answer and directly emits the next question.
    """

    def __init__(self, form_url: str):
        self.form_url = form_url
        self.client = None
        self.session = None
        self.collected_data: Dict[str, str] = {}
        self._form_submitted = False  # Track if form was submitted to avoid duplicates

        # Question tracking for deterministic flow
        self.questions = FORM_QUESTIONS
        self.current_question_index = 0

        # Start initialization immediately on construction
        self._init_future: asyncio.Task = asyncio.create_task(self._initialize())

    def _get_current_question(self) -> Optional[FormField]:
        """Get the current question to ask."""
        if self.current_question_index < len(self.questions):
            return self.questions[self.current_question_index]
        return None

    def _get_field_by_id(self, field_id: str) -> Optional[FormField]:
        """Get a form field by its ID."""
        for field in self.questions:
            if field.field_id == field_id:
                return field
        return None

    async def _initialize(self) -> None:
        """Initialize Stagehand and open the form."""
        try:
            # Import here to avoid import errors when stagehand is not installed
            from stagehand import AsyncStagehand

            logger.info("Initializing Stagehand browser automation")

            self.client = AsyncStagehand(
                browserbase_api_key=os.environ.get("BROWSERBASE_API_KEY"),
                browserbase_project_id=os.environ.get("BROWSERBASE_PROJECT_ID"),
                model_api_key=os.environ.get("GEMINI_API_KEY"),
            )

            self.session = await self.client.sessions.create(model_name="gemini/gemini-2.5-flash-preview-09-2025")

            logger.info(f"Session started: {self.session.id}")

            # Navigate to form
            logger.info(f"Opening form: {self.form_url}")
            await self.session.navigate(url=self.form_url)

            # Wait for form to load
            await asyncio.sleep(2)

            logger.info("Browser automation initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Stagehand: {e}")
            raise

    async def _submit_form(self) -> bool:
        """Submit the completed form in the browser.

        Returns:
            True if form was submitted successfully, False otherwise.
        """
        if self._form_submitted:
            logger.info("Form has already been submitted, skipping duplicate submission")
            return True

        try:
            await self._init_future
            logger.info("Submitting the form")
            logger.info(f"Form has {len(self.collected_data)} fields filled")

            await self.session.act(input="Find and click the Submit button to submit the form")

            # Wait for submission to process
            await asyncio.sleep(1)

            logger.info("Form submitted successfully!")
            self._form_submitted = True
            return True

        except Exception as e:
            logger.error(f"Error submitting form: {e}")
            return False

    async def _fill_field_background(self, field: FormField, value: str) -> None:
        """Fill a form field asynchronously in the background."""
        try:
            await self._fill_field(field, value)
            logger.info(f"Successfully filled field: {field.field_id} in browser")
        except Exception as e:
            logger.error(f"Error filling field {field.field_id}: {e}")

    async def _fill_field(self, field: FormField, answer: str):
        """Fill a specific form field in the browser.

        Args:
            field: The form field to fill.
            answer: The answer value to fill.

        Returns:
            True if field was filled successfully, False otherwise.
        """
        await self._init_future
        logger.info(f"Filling field '{field.field_id}' with: {answer}")

        # Use Stagehand's natural language API to fill the field
        if field.field_type in [FieldType.TEXT, FieldType.EMAIL, FieldType.PHONE]:
            await self.session.act(input=f"Fill in the '{field.label}' field with: {answer}")

        elif field.field_type == FieldType.TEXTAREA:
            await self.session.act(input=f"Type in the '{field.label}' text area: {answer}")

        elif field.field_type in [FieldType.SELECT, FieldType.RADIO]:
            await self.session.act(input=f"Select '{answer}' for the '{field.label}' field")

        elif field.field_type == FieldType.CHECKBOX:
            # For role selection, check the specific role checkbox
            if field.field_id == "role_selection":
                await self.session.act(input=f"Check the '{answer}' checkbox")
            else:
                # For other checkboxes, check/uncheck based on answer
                if answer.lower() in ["yes", "true"]:
                    await self.session.act(input=f"Check the '{field.label}' checkbox")
                else:
                    await self.session.act(input=f"Uncheck the '{field.label}' checkbox")

    async def cleanup(self) -> None:
        """Clean up browser resources.

        This method safely handles cleanup even if initialization is still in progress.
        It will wait for initialization to complete (or fail) before cleaning up the session.
        """
        try:
            # Wait for initialization to complete (or fail) before cleanup
            await self._init_future

            if self.session:
                session = self.session
                self.session = None  # Prevent double cleanup
                await session.end()
                logger.info("Session ended")
        except Exception as e:
            logger.info(f"Error during cleanup: {e}")

    # =========================================================================
    # Tool methods for use with LlmAgent
    # =========================================================================

    @passthrough_tool
    async def start_questionnaire(
        self,
        ctx: ToolEnv,
    ) -> AsyncGenerator[OutputEvent, None]:
        """Start the questionnaire by asking the first question.

        Call this tool when the user indicates they are ready to begin.
        This will ask the first question in the form.
        """
        logger.info("Starting questionnaire")

        first_question = self._get_current_question()
        if first_question:
            logger.info(f"Asking first question: {first_question.field_id}")
            yield AgentSendText(text=f"Great! Let's begin. {first_question.question}")
        else:
            yield AgentSendText(text="It looks like there are no questions to ask.")
            yield AgentEndCall()

    @passthrough_tool
    async def record_form_field(
        self,
        ctx: ToolEnv,
        field_name: Annotated[
            str,
            "The name of the form field (e.g., 'full_name', 'email', 'phone', 'work_eligibility', "
            "'availability_type', 'role_selection', 'previous_experience', 'skills_experience', "
            "'additional_info')",
        ],
        value: Annotated[str, "The value provided by the user for this field"],
    ) -> AsyncGenerator[OutputEvent, None]:
        """Record a form field value and ask the next question.

        Call this tool after the user provides an answer. The tool will:
        1. Record the value and fill the browser form in the background
        2. Automatically ask the next question (or submit if all questions answered)
        """
        logger.info(f"Recording: {field_name} = {value}")

        # Get the field
        field = self._get_field_by_id(field_name)

        if not field:
            # Field not recognized - don't skip the question, re-ask without revealing technical details
            logger.warning(f"Unknown field: {field_name}, asking current question again")
            current_question = self._get_current_question()
            if current_question:
                yield AgentSendText(text=f"Let me ask that again. {current_question.question}")
            else:
                yield AgentSendText(text="Let me continue with the next question.")
            return

        # Store the answer
        self.collected_data[field_name] = value
        logger.info(f"Collected: {field_name}={value.strip()}")

        # Fill the form field in the background (non-blocking)
        asyncio.create_task(self._fill_field_background(field, value.strip()))

        # Move to next question
        self.current_question_index += 1
        next_question = self._get_current_question()
        if next_question:
            # Ask the next question deterministically
            logger.info(f"Asking next question: {next_question.field_id}")
            yield AgentSendText(text=f"Great! {next_question.question}")
            return

        # All questions answered - submit the form
        logger.info(f"All {self.current_question_index} questions answered")
        logger.info(f"Collected data for {len(self.collected_data)} fields")

        # Give pending fill operations time to complete
        await asyncio.sleep(1)

        success = await self._submit_form()
        if success:
            yield AgentSendText(text="Perfect! I've submitted your application. Thank you!")
        else:
            yield AgentSendText(
                text="Thank you for providing all the information. Your responses have been recorded."
            )

        # End the call
        await self.cleanup()
        yield AgentEndCall()

    async def on_call_ended(self) -> None:
        """Handle call ended event - submit form if needed and cleanup.

        This is called when the call ends (user hangs up, disconnect, etc.)
        to ensure any collected data is submitted and browser resources are cleaned up.
        """
        # Try and submit
        logger.info("Call ending - auto-submitting form with collected data")
        await self._submit_form()
        await self.cleanup()
