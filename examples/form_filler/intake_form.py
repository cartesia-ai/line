"""Intake form management for doctor visit scheduling."""

import asyncio
from dataclasses import dataclass
import json
from typing import Annotated, Any, Optional

from loguru import logger

from line.llm_agent import ToolEnv, loopback_tool

# Default instruction shown to the agent after recording an answer (before next question).
REPEAT_ANSWER_TO_USER_INSTRUCTION = "Repeat the answer back to the user. Then proceed to next question."

REPEAT_SPELLING_BACK_TO_USER_INSTRUCTION = "IMPORTANT: spell the user's answer back with dashes and ask if it is correct (for example 'That's J-O-H-N, right?' or 'That's A-B-C-1-2-3'). Make sure the user confirms the answer before asking the next question."

# Form field definitions (Doctor Visit Scheduling Intake)
FORM_FIELDS = [
    {
        "id": "first_name",
        "question": "What is your first name?",
        "context": "If the user gave their first name in the conversation, save it directly.",
        "type": "string",
        "section": "greeting",
        "required": True,
        "post_record_instructions": REPEAT_SPELLING_BACK_TO_USER_INSTRUCTION,
    },
    {
        "id": "last_name",
        "question": "What is your last name?",
        "context": "If the user gave their last name in the conversation, save it directly. ",
        "type": "string",
        "section": "greeting",
        "required": True,
        "post_record_instructions": REPEAT_SPELLING_BACK_TO_USER_INSTRUCTION,
    },
    {
        "id": "reason_for_visit",
        "question": "What is the reason for your visit?",
        "context": "Ask what brings them in today. Keep the reason concise and specific. Ask one follow up question about their condition/symptoms/onset/etc.",
        "type": "string",
        "section": "intake",
        "required": True,
        "post_record_instructions": REPEAT_ANSWER_TO_USER_INSTRUCTION,
    },
    {
        "id": "date_of_birth",
        "question": "What is your date of birth?",
        "context": "Format the date as 'Month Day, Year' (e.g., 'January 15, 1990') before saving.",
        "type": "date",
        "section": "intake",
        "required": True,
        "post_record_instructions": REPEAT_ANSWER_TO_USER_INSTRUCTION,
    },
    {
        "id": "phone",
        "question": "What is the best phone number to reach you?",
        "context": "",
        "type": "string",
        "section": "intake",
        "required": True,
        "post_record_instructions": REPEAT_ANSWER_TO_USER_INSTRUCTION,
    },
]


@dataclass
class RecordAnswerResult:
    """Result of recording an intake form answer."""

    success: bool
    error: Optional[str] = None
    current_question: Optional[str] = None
    next_question: Optional[str] = None
    is_complete: Optional[bool] = None
    recorded_field: Optional[str] = None
    recorded_value: Optional[Any] = None
    progress: Optional[str] = None
    record_intake_instructions: Optional[str] = None


class IntakeForm:
    """Manages the doctor visit scheduling intake form state and provides tools for the agent."""

    def __init__(self):
        self._fields = FORM_FIELDS.copy()
        self._answers: dict[str, Any] = {}
        self._current_index: int = 0
        self._is_started: bool = False
        self._is_submitted: bool = False
        logger.info(f"IntakeForm initialized with {len(self._fields)} fields")

    def start_form(self) -> dict:
        """Start the intake form process."""
        if not self._is_started:
            self._is_started = True
            logger.info("Intake form started")
        return self.get_status()

    def get_status(self) -> dict:
        """Get current form status."""
        current = self._get_current_field()
        answered_count = len(self._answers)
        total_count = len(self._fields)

        return {
            "is_started": self._is_started,
            "is_complete": current is None,
            "is_submitted": self._is_submitted,
            "progress": f"{answered_count}/{total_count}",
            "answered_fields": list(self._answers.keys()),
            "current_question": self._format_question(current) if current else None,
            "current_section": current["section"] if current else None,
        }

    def _get_field_by_id(self, field_id: str) -> Optional[dict]:
        """Get a field definition by its ID."""
        for field in self._fields:
            if field["id"] == field_id:
                return field
        return None

    def get_answered_fields_summary(self) -> dict:
        """Get a summary of all answered fields with their values."""
        answered = []
        for field in self._fields:
            if field["id"] in self._answers:
                value = self._answers[field["id"]]
                answered.append(
                    {
                        "field_id": field["id"],
                        "question": field["question"],
                        "answer": str(value),
                        "raw_value": value,
                    }
                )
        return {
            "answered_count": len(answered),
            "total_count": len(self._fields),
            "fields": answered,
        }

    async def record_answer(self, field_id: str, answer: str) -> RecordAnswerResult:
        """Record or edit an answer to the given field and confirm the value with the user and let them know they can correct it if needed.
        If the field is the current question, then advance the form filler internal state to the next question.
        If the field is not the current question, set/update the answer without advancing (allows out-of-order form filling).
        """
        if not self._is_started:
            return RecordAnswerResult(
                success=False,
                error="Form has not been started yet. Start the form first.",
                next_question=None,
            )

        if self._is_submitted:
            return RecordAnswerResult(
                success=False,
                error="Form has already been submitted. Cannot record answers.",
                next_question=None,
            )

        field = self._get_field_by_id(field_id)
        if not field:
            return RecordAnswerResult(
                success=False,
                error=f"Unknown field '{field_id}'. Available fields: {', '.join(f['id'] for f in self._fields)}",
                next_question=None,
            )

        if field != self._get_current_field():
            # If the field is not the current question, edit the answer, and don't advance the form filler internal state.
            success, processed, error = self._process_answer(answer, field)
            if not success:
                return RecordAnswerResult(
                    success=False,
                    error=error,
                    current_question=self._format_question(field),
                    next_question=None,
                )

            self._answers[field_id] = processed
            next_field = self._get_current_field()
            return RecordAnswerResult(
                success=True,
                recorded_field=field_id,
                recorded_value=processed,
                next_question=self._format_question(next_field) if next_field else None,
                is_complete=next_field is None,
                progress=f"{len(self._answers)}/{len(self._fields)}",
                record_intake_instructions=field.get("post_record_instructions"),
            )

        success, processed, error = self._process_answer(answer, field)
        if not success:
            return RecordAnswerResult(
                success=False,
                error=error,
                current_question=self._format_question(field),
                next_question=None,
            )

        self._answers[field_id] = processed
        self._current_index += 1
        logger.info(f"Recorded '{field_id}': {processed}")
        logger.info(f"Current state of the form: {self.get_form_data()}")
        next_field = self._get_current_field()

        return RecordAnswerResult(
            success=True,
            recorded_field=field_id,
            recorded_value=processed,
            next_question=self._format_question(next_field) if next_field else None,
            is_complete=next_field is None,
            progress=f"{len(self._answers)}/{len(self._fields)}",
            record_intake_instructions=field.get("post_record_instructions"),
        )

    def _get_current_field(self) -> Optional[dict]:
        """Get the current field to ask about."""
        if self._current_index < len(self._fields):
            return self._fields[self._current_index]
        return None

    def _format_question(self, field: dict) -> str:
        """Format a field as a question for the agent."""
        text = field["question"]
        if field.get("context"):
            text += f" (context: (do not say this to the user unless instructed) {field['context']})"
        return text

    def _process_answer(self, answer: str, field: dict) -> tuple[bool, Any, str]:
        """Process and validate an answer. Returns (success, processed_value, error_message)."""
        answer = answer.strip()
        ftype = field["type"]

        if ftype == "string":
            if not answer and field.get("required", True):
                return False, None, "Please provide a non-empty answer."
            return True, answer or "", ""

        elif ftype == "date":
            if not answer:
                return False, None, "Please provide a date (month, day, and year)."
            return True, answer, ""

        return True, answer, ""

    def get_form_data(self) -> dict:
        """Get the complete form data as JSON-serializable dict."""
        return {field["id"]: self._answers.get(field["id"]) for field in self._fields}

    def get_contact_info(self) -> Optional[dict]:
        """Get contact info for scheduling. Returns None if any required contact field is missing."""
        first_name = self._answers.get("first_name") or ""
        last_name = self._answers.get("last_name") or ""
        phone = self._answers.get("phone") or ""
        if not first_name.strip() or not last_name.strip() or not phone.strip():
            return None
        return {
            "first_name": first_name,
            "last_name": last_name,
            "phone": phone.strip(),
        }

    async def submit_form(self) -> dict:
        """Submit the completed form to the API."""
        unanswered = [f["id"] for f in self._fields if f["id"] not in self._answers]
        if unanswered:
            return {
                "success": False,
                "error": f"Form is not complete. Missing answers for: {', '.join(unanswered)}",
            }

        if self._is_submitted:
            return {
                "success": False,
                "error": "Form has already been submitted.",
            }

        form_data = self.get_form_data()

        # Mock API call
        await asyncio.sleep(0.5)
        logger.info(f"Submitting form data: {json.dumps(form_data, indent=2)}")

        self._is_submitted = True

        return {
            "success": True,
            "submitted": True,
            "message": "",
            "next_steps": "You can now book an appointment.",
        }


def create_intake_tools(form: IntakeForm):
    """Create intake form tools bound to a specific form instance."""

    @loopback_tool
    async def record_answer(
        ctx: ToolEnv,
        field_id: Annotated[str, "The ID of the field to record an answer for"],
        answer: Annotated[str, "The user's answer to the current form question"],
    ) -> str:
        """Save the user's answer to a field of the intake form. Confirm the value with the user and let them know they can correct it if needed."""
        result = await form.record_answer(field_id, answer)

        if not result.success:
            return (
                f"Error: {result.error or 'Unknown error'}. Current question: {result.current_question or ''}"
            )

        if result.is_complete:
            return f"[{result.recorded_field}: {result.recorded_value}] {result.record_intake_instructions}."

        return f"[{result.recorded_field}: {result.recorded_value}] {result.record_intake_instructions}  Next question: {result.next_question}"

    @loopback_tool
    async def list_answer(ctx: ToolEnv) -> str:
        """List all answers the user has provided so far in the intake form.
        Use when the user asks to review their answers or wants to know what they've entered.
        """
        summary = form.get_answered_fields_summary()

        if summary["answered_count"] == 0:
            return "No answers recorded yet. The form may not have been started."

        parts = [f"Answered {summary['answered_count']}/{summary['total_count']} questions:\n"]
        for field in summary["fields"]:
            parts.append(f"- {field['field_id']}: {field['answer']}\n")

        return "".join(parts)

    return [
        record_answer,
        list_answer,
    ]
