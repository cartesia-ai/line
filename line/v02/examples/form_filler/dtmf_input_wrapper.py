"""
DTMF Input Wrapper - Allows users to input phone numbers and dates via DTMF tones.

This wrapper intercepts DTMF events and converts them to text input for specific
form fields (phone numbers, dates of birth), enabling touch-tone input as an
alternative to voice.
"""

from dataclasses import dataclass, field
from typing import AsyncIterable, Optional

from loguru import logger

from line.v02.agent import TurnEnv
from line.v02.events import (
    AddToHistory,
    AgentSendText,
    InputEvent,
    OutputEvent,
    SpecificUserTextSent,
    SpecificUserTurnEnded,
    SpecificUserTurnStarted,
    UserDtmfSent,
    UserTurnEnded,
)


@dataclass
class DtmfFieldConfig:
    """Configuration for a DTMF-enabled field."""

    field_id: str
    # Keywords in the question text that identify this field
    question_keywords: list[str]
    # How to format the collected digits
    format_type: str  # "phone" or "date"
    # Minimum digits required
    min_digits: int = 0
    # Maximum digits to collect
    max_digits: int = 20


@dataclass
class DtmfInputConfig:
    """Configuration for DTMF input handling."""

    # Fields that accept DTMF input
    dtmf_fields: list[DtmfFieldConfig] = field(default_factory=list)
    # Message to prompt user about DTMF option
    dtmf_prompt: str = "You can also enter this using your phone's keypad, then press pound when done."
    # Termination button (ends DTMF collection)
    termination_button: str = "#"


# Default configuration for phone and date fields
DEFAULT_DTMF_CONFIG = DtmfInputConfig(
    dtmf_fields=[
        DtmfFieldConfig(
            field_id="callback_number",
            question_keywords=["phone number", "phone", "callback", "reach you"],
            format_type="phone",
            min_digits=10,
            max_digits=15,
        ),
        DtmfFieldConfig(
            field_id="date_of_birth",
            question_keywords=["date of birth", "birth date", "birthday", "born"],
            format_type="date",
            min_digits=8,
            max_digits=8,
        ),
    ],
)


class DtmfInputWrapper:
    """Wrapper that enables DTMF input for phone numbers and dates.

    This wrapper intercepts events and:
    1. Detects when the agent asks for DTMF-eligible fields
    2. Adds a prompt informing users they can use touch-tone input
    3. Collects DTMF digits into a buffer
    4. Converts collected digits to text format when user finishes
    5. Passes the formatted text to the inner agent as a UserTurnEnded event
    """

    def __init__(
        self,
        inner_agent,
        config: DtmfInputConfig = DEFAULT_DTMF_CONFIG,
    ):
        self.inner_agent = inner_agent
        self.config = config

        # DTMF collection state
        self._dtmf_buffer: list[str] = []
        self._active_field: Optional[DtmfFieldConfig] = None
        self._prompted_for_dtmf: bool = False
        self._last_agent_text: str = ""

    def _detect_dtmf_field(self, text: str) -> Optional[DtmfFieldConfig]:
        """Check if the agent's text is asking for a DTMF-eligible field."""
        text_lower = text.lower()
        for field_config in self.config.dtmf_fields:
            for keyword in field_config.question_keywords:
                if keyword.lower() in text_lower:
                    return field_config
        return None

    def _format_phone_number(self, digits: str) -> str:
        """Format digits as a spoken phone number."""
        # Convert digits to spoken format: "4 1 5 5 5 5 1 2 3 4"
        return " ".join(digits)

    def _format_date(self, digits: str) -> str:
        """Format 8 digits (MMDDYYYY) as a spoken date."""
        if len(digits) != 8:
            # If not exactly 8 digits, just speak them
            return " ".join(digits)

        month = digits[0:2]
        day = digits[2:4]
        year = digits[4:8]

        # Month names
        months = {
            "01": "January",
            "02": "February",
            "03": "March",
            "04": "April",
            "05": "May",
            "06": "June",
            "07": "July",
            "08": "August",
            "09": "September",
            "10": "October",
            "11": "November",
            "12": "December",
        }

        month_name = months.get(month, f"month {month}")
        # Remove leading zero from day
        day_num = str(int(day)) if day.isdigit() else day

        return f"{month_name} {day_num}, {year}"

    def _format_digits(self, digits: str, format_type: str) -> str:
        """Format collected digits based on field type."""
        if format_type == "phone":
            return self._format_phone_number(digits)
        elif format_type == "date":
            return self._format_date(digits)
        else:
            return " ".join(digits)

    def _clear_dtmf_state(self):
        """Reset DTMF collection state."""
        self._dtmf_buffer = []
        self._active_field = None
        self._prompted_for_dtmf = False

    def _create_history_items(self, formatted_text: str) -> list:
        """Create history items for the synthetic user turn."""
        user_text_event = SpecificUserTextSent(content=formatted_text)
        return [
            SpecificUserTurnStarted(),
            user_text_event,
            SpecificUserTurnEnded(content=[user_text_event]),
        ]

    async def _submit_dtmf_input(self, env: TurnEnv, formatted: str, history: list):
        """Submit collected DTMF input to the inner agent.

        Yields an AddToHistory event to add the synthetic user turn to voice_agent_app's
        central history, then forwards a UserTurnEnded to the inner agent.
        """
        logger.info(f"Submitting DTMF input to agent: '{formatted}'")

        # Create synthetic history items
        history_items = self._create_history_items(formatted_text=formatted)

        # Yield AddToHistory to update voice_agent_app's central history
        yield AddToHistory(items=history_items)

        # Create UserTurnEnded for the inner agent with updated history
        user_text_event = SpecificUserTextSent(content=formatted)
        synthetic_event = UserTurnEnded(
            content=[user_text_event],
            history=list(history) + history_items,
        )

        self._clear_dtmf_state()

        async for output in self.inner_agent.process(env, synthetic_event):
            yield output

    async def process(self, env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        """Process an input event, handling DTMF collection."""

        logger.info(f"Processing input event: {event}")
        # Handle DTMF input
        if isinstance(event, UserDtmfSent):
            button = event.button
            logger.info(
                f"DTMF wrapper received button '{button}', "
                f"buffer={self._dtmf_buffer}, "
                f"active_field={self._active_field.field_id if self._active_field else None}"
            )

            # If termination button pressed, finalize DTMF input
            if button == self.config.termination_button:
                if self._dtmf_buffer and self._active_field:
                    digits = "".join(self._dtmf_buffer)
                    formatted = self._format_digits(digits, self._active_field.format_type)
                    logger.info(f"DTMF input complete (# pressed): {digits} -> '{formatted}'")

                    async for output in self._submit_dtmf_input(env, formatted, event.history):
                        yield output
                return

            # Collect the digit
            if button in "0123456789":
                self._dtmf_buffer.append(button)
                logger.debug(f"DTMF digit collected: {button}, buffer: {''.join(self._dtmf_buffer)}")
            return

        # Handle UserTurnEnded - check if there's buffered DTMF to submit
        if isinstance(event, UserTurnEnded):
            if self._dtmf_buffer and self._active_field:
                digits = "".join(self._dtmf_buffer)

                # Check minimum digits requirement
                if len(digits) >= self._active_field.min_digits:
                    formatted = self._format_digits(digits, self._active_field.format_type)
                    logger.info(f"DTMF input submitted on turn end: {digits} -> '{formatted}'")

                    async for output in self._submit_dtmf_input(env, formatted, event.history):
                        yield output
                    return
                else:
                    logger.debug(
                        f"DTMF buffer has insufficient digits ({len(digits)}), "
                        f"need {self._active_field.min_digits}"
                    )
                    self._dtmf_buffer = []

        # Forward event to inner agent
        async for output in self.inner_agent.process(env, event):
            # Check if agent is asking for a DTMF-eligible field
            if isinstance(output, AgentSendText):
                self._last_agent_text = output.text
                detected_field = self._detect_dtmf_field(output.text)

                if detected_field:
                    # Set up DTMF collection for this field
                    self._active_field = detected_field
                    self._dtmf_buffer = []
                    self._prompted_for_dtmf = False
                    logger.info(f"Detected DTMF-eligible field: {detected_field.field_id}")

                    # Yield the original text first
                    yield output

                    # Then yield the DTMF prompt
                    yield AgentSendText(text=self.config.dtmf_prompt)
                    self._prompted_for_dtmf = True
                    continue
                else:
                    # Not a DTMF field, clear state
                    if self._active_field:
                        logger.info("Clearing DTMF state (non-DTMF field detected)")
                        self._clear_dtmf_state()

            yield output
