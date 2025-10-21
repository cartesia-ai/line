"""Browser automation for filling web forms during voice conversations.

This module provides the StagehandFormFiller class which manages browser
automation for filling forms using Stagehand. It handles form field
mapping, field filling, and form submission.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
import os
from typing import Dict, List, Optional

from loguru import logger
from stagehand import Stagehand, StagehandConfig


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
    """Represents a form field with its metadata"""

    field_id: str
    field_type: FieldType
    label: str
    required: bool = False
    options: Optional[List[str]] = None


class FormFieldMapping:
    """Maps conversation questions to actual form fields"""

    def __init__(self):
        self.field_mappings = {
            "full_name": FormField(
                field_id="full_name",
                field_type=FieldType.TEXT,
                label="What is your full name?",
                required=True,
            ),
            "email": FormField(
                field_id="email",
                field_type=FieldType.EMAIL,
                label="What is your email address?",
                required=True,
            ),
            "phone": FormField(
                field_id="phone",
                field_type=FieldType.PHONE,
                label="What is your phone number?",
                required=False,
            ),
            "work_eligibility": FormField(
                field_id="work_eligibility",
                field_type=FieldType.RADIO,
                label="Are you legally eligible to work in this country?",
                options=["Yes", "No"],
                required=True,
            ),
            "availability_type": FormField(
                field_id="availability",
                field_type=FieldType.RADIO,
                label="What's your availability?",
                options=["Temporary", "Part-time", "Full-time"],
                required=True,
            ),
            "additional_info": FormField(
                field_id="additional_info",
                field_type=FieldType.TEXTAREA,
                label="Anything else you'd like to let us know about you?",
                required=False,
            ),
            "role_selection": FormField(
                field_id="role_selection",
                field_type=FieldType.CHECKBOX,
                label="Which of these roles are you applying for?",
                options=[
                    "Sales manager",
                    "IT Support",
                    "Recruiting",
                    "Software engineer",
                    "Marketing specialist",
                ],
                required=True,
            ),
            "previous_experience": FormField(
                field_id="previous_experience",
                field_type=FieldType.RADIO,
                label=("Have you worked in a role similar to this one in the past?"),
                options=["Yes", "No"],
                required=True,
            ),
            "skills_experience": FormField(
                field_id="skills_experience",
                field_type=FieldType.TEXTAREA,
                label=(
                    "What relevant skills and experience do you have "
                    "that make you a strong candidate for this position?"
                ),
                required=True,
            ),
        }

    def get_form_field(self, question_id: str) -> Optional[FormField]:
        """Get the form field mapping for a question ID.

        Args:
            question_id: The question identifier.

        Returns:
            The FormField object or None if not found.
        """
        return self.field_mappings.get(question_id)


class StagehandFormFiller:
    """Manages browser automation for filling forms using Stagehand"""

    def __init__(self, form_url: str):
        self.form_url = form_url
        self.stagehand: Optional[Stagehand] = None
        self.page = None
        self.is_initialized = False
        self.field_mapper = FormFieldMapping()
        self.collected_data: Dict[str, str] = {}

    async def initialize(self) -> None:
        """Initialize Stagehand and open the form.

        Returns:
            None.
        """
        if self.is_initialized:
            return

        try:
            logger.info("Initializing Stagehand browser automation")

            # Configure Stagehand
            config = StagehandConfig(
                env="BROWSERBASE",
                # Fast model for form filling
                model_name="google/gemini-2.0-flash-exp",
                model_api_key=os.getenv("GEMINI_API_KEY"),
            )

            self.stagehand = Stagehand(config)
            await self.stagehand.init()

            self.page = self.stagehand.page

            # Navigate to form
            logger.info(f"Opening form: {self.form_url}")
            await self.page.goto(self.form_url)

            # Wait for form to load
            await asyncio.sleep(2)

            self.is_initialized = True
            logger.info("Browser automation initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Stagehand: {e}")
            raise

    async def fill_field(self, question_id: str, answer: str) -> bool:
        """Fill a specific form field based on the question ID and answer.

        Args:
            question_id: The question identifier.
            answer: The answer value to fill.

        Returns:
            True if field was filled successfully, False otherwise.
        """
        if not self.is_initialized:
            # Initialize asynchronously without blocking
            init_task = asyncio.create_task(self.initialize())
            await init_task

        try:
            # Get field mapping
            field = self.field_mapper.get_form_field(question_id)
            if not field:
                logger.warning(f"No field mapping found for question: {question_id}")
                return False

            # Store and use the answer directly
            answer = answer.strip()
            self.collected_data[question_id] = answer

            logger.info(f"Async filling field '{field.label}' with: {answer}")

            # Create async task for the actual field filling
            fill_action = None

            # Use Stagehand's natural language API to fill the field
            if field.field_type in [FieldType.TEXT, FieldType.EMAIL, FieldType.PHONE]:
                fill_action = self.page.act(f"Fill in the '{field.label}' field with: {answer}")

            elif field.field_type == FieldType.TEXTAREA:
                fill_action = self.page.act(f"Type in the '{field.label}' text area: {answer}")

            elif field.field_type in [FieldType.SELECT, FieldType.RADIO]:
                fill_action = self.page.act(f"Select '{answer}' for the '{field.label}' field")

            elif field.field_type == FieldType.CHECKBOX:
                # For role selection, check the specific role checkbox
                if question_id == "role_selection":
                    fill_action = self.page.act(f"Check the '{answer}' checkbox")
                else:
                    # For other checkboxes, check/uncheck based on answer
                    if answer.lower() in ["yes", "true"]:
                        fill_action = self.page.act(f"Check the '{field.label}' checkbox")
                    else:
                        fill_action = self.page.act(f"Uncheck the '{field.label}' checkbox")

            # Execute the fill action asynchronously
            if fill_action:
                await fill_action

            return True

        except Exception as e:
            logger.error(f"Error filling field {question_id}: {e}")
            return False

    async def submit_form(self) -> bool:
        """Submit the completed form.

        Returns:
            True if form was submitted successfully, False otherwise.
        """
        try:
            logger.info("Submitting the form")
            logger.info(f"Form has {len(self.collected_data)} fields filled")

            await self.page.act("Find and click the Submit button to submit the form")

            # Wait for submission to process
            await asyncio.sleep(1)

            logger.info("Form submitted successfully!")
            return True

        except Exception as e:
            logger.error(f"Error submitting form: {e}")
            return False

    async def cleanup(self) -> None:
        """Clean up browser resources.

        Returns:
            None.
        """
        if self.stagehand or self.page:
            try:
                await self.stagehand.close()
                logger.info("Browser closed")
            except Exception as e:
                logger.error(f"Error closing browser: {e}")
