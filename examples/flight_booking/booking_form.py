"""Booking form and flight change for the flight change agent."""

import asyncio
from typing import Annotated, Any, Optional

from loguru import logger

from line.llm_agent import ToolEnv, loopback_tool

# Form field definitions — 5 questions for flight change
FORM_FIELDS = [
    {"id": "first_name", "text": "What's your first name?", "type": "string"},
    {"id": "last_name", "text": "And your last name?", "type": "string"},
    {"id": "new_date", "text": "What date would you like to change your flight to?", "type": "string"},
    {"id": "time_preference", "text": "Any preference on timing — morning, afternoon, or evening?", "type": "string"},
    {"id": "confirmation_code", "text": "And what's your confirmation code?", "type": "string"},
]


class BookingForm:
    """Manages the booking form state — 5 questions asked one at a time."""

    def __init__(self):
        self._fields = FORM_FIELDS.copy()
        self._answers: dict[str, Any] = {}
        self._current_index: int = 0
        self._is_started: bool = False
        logger.info(f"BookingForm initialized with {len(self._fields)} fields")

    def start_form(self) -> dict:
        """Start the booking form process."""
        if not self._is_started:
            self._is_started = True
            logger.info("Booking form started")
        return self.get_status()

    def get_status(self) -> dict:
        """Get current form status."""
        current = self._get_current_field()
        answered_count = len(self._answers)
        total_count = len(self._fields)

        return {
            "is_started": self._is_started,
            "is_complete": current is None,
            "progress": f"{answered_count}/{total_count}",
            "answered_fields": list(self._answers.keys()),
            "current_question": self._format_question(current) if current else None,
        }

    def _get_current_field(self) -> Optional[dict]:
        """Get the current field to ask about."""
        if self._current_index < len(self._fields):
            return self._fields[self._current_index]
        return None

    def _format_question(self, field: dict) -> str:
        """Format a field as a question for the agent."""
        return field["text"]

    def record_answer(self, answer: str) -> dict:
        """Record an answer to the current field."""
        if not self._is_started:
            return {
                "success": False,
                "error": "Form has not been started yet. Start the form first.",
                "next_question": None,
            }

        field = self._get_current_field()
        if not field:
            return {
                "success": False,
                "error": "All questions have already been answered.",
                "is_complete": True,
            }

        answer = answer.strip()
        if not answer:
            return {
                "success": False,
                "error": "Please provide a non-empty answer.",
                "current_question": self._format_question(field),
            }

        self._answers[field["id"]] = answer
        self._current_index += 1
        logger.info(f"Recorded '{field['id']}': {answer}")

        next_field = self._get_current_field()

        return {
            "success": True,
            "recorded_field": field["id"],
            "recorded_value": answer,
            "next_question": self._format_question(next_field) if next_field else None,
            "is_complete": next_field is None,
            "progress": f"{len(self._answers)}/{len(self._fields)}",
        }

    def get_collected_data(self) -> dict:
        """Return collected flight change details."""
        return {
            "first_name": self._answers.get("first_name", ""),
            "last_name": self._answers.get("last_name", ""),
            "new_date": self._answers.get("new_date", ""),
            "time_preference": self._answers.get("time_preference", ""),
            "confirmation_code": self._answers.get("confirmation_code", ""),
        }


# Global form instance (reset per call)
_form_instance: Optional[BookingForm] = None


def get_form() -> BookingForm:
    """Get or create the form instance."""
    global _form_instance
    if _form_instance is None:
        _form_instance = BookingForm()
    return _form_instance


def reset_form_instance():
    """Reset the global form instance."""
    global _form_instance
    _form_instance = None


# ── Tool definitions ──


@loopback_tool(is_background=True)
async def start_booking_form(ctx: ToolEnv) -> str:
    """Start the booking form to collect the caller's details. Call this at the beginning of the conversation."""
    form = get_form()
    status = form.start_form()

    first_question = status.get("current_question", "")
    return (
        f"Starting booking form. Progress: {status['progress']}. "
        f"First question: {first_question}"
    )


@loopback_tool(is_background=True)
async def record_booking_answer(
    ctx: ToolEnv,
    answer: Annotated[str, "The user's confirmed answer to record (NOT 'yes' or confirmations)"],
) -> str:
    """Record the user's answer to the current booking form question.
    IMPORTANT: Only call this with the actual answer value, never with 'yes' or confirmation words.
    For fields that need confirmation: first say the answer back to confirm, then call this after the user confirms."""
    form = get_form()
    result = form.record_answer(answer)

    if not result["success"]:
        return f"Error: {result.get('error', 'Unknown error')}. Current question: {result.get('current_question', '')}"

    recorded_field = result.get("recorded_field", "")

    if result["is_complete"]:
        return (
            f"Recorded {recorded_field}. All questions answered! "
            "Now call change_flight to process the flight change."
        )

    next_q = result.get("next_question", "")
    return f"Recorded {recorded_field}. Next question: {next_q}"


@loopback_tool(is_background=True)
async def get_booking_form_status(ctx: ToolEnv) -> str:
    """Check the current status of the booking form including progress and next question."""
    form = get_form()
    status = form.get_status()

    if not status["is_started"]:
        return "Booking form has not been started yet."

    if status["is_complete"]:
        return f"Booking form is complete with {status['progress']} questions answered."

    return (
        f"Form in progress: {status['progress']} answered. "
        f"Current question: {status['current_question']}"
    )


@loopback_tool(is_background=True)
async def change_flight(ctx: ToolEnv) -> str:
    """Change the caller's flight using the collected details.
    Call this after all 5 form questions are answered to process the flight change."""
    form = get_form()
    data = form.get_collected_data()

    name = f"{data['first_name'].strip()} {data['last_name'].strip()}"
    code = data["confirmation_code"].strip().upper()
    new_date = data["new_date"].strip()
    time_pref = data["time_preference"].strip()

    logger.info(f"Processing flight change for {name} / {code} to {new_date} ({time_pref})")

    # Simulate making the change
    await asyncio.sleep(2)

    logger.info(f"Flight change completed for {name} / {code}")
    return (
        f"Done! The flight for {name} (confirmation code: {code}) has been changed to {new_date} ({time_pref}). "
        "Tell the caller it's all taken care of and ask if there's anything else you can help with."
    )
