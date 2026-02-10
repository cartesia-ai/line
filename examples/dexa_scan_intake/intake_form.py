"""Intake form management for DEXA scan appointments."""

import asyncio
import json
from typing import Annotated, Any, Optional
from loguru import logger

from line.llm_agent import ToolEnv, loopback_tool

# Form field definitions
FORM_FIELDS = [
    # Personal Information
    {"id": "first_name", "text": "What is your first name?", "type": "string", "section": "personal", "required": True},
    {"id": "last_name", "text": "What is your last name?", "type": "string", "section": "personal", "required": True},
    {"id": "email", "text": "What is your email address?", "type": "string", "section": "personal", "required": True},
    {"id": "phone", "text": "What is your phone number?", "type": "string", "section": "personal", "required": True},
    {"id": "date_of_birth", "text": "What is your date of birth?", "type": "string", "section": "personal", "required": True},
    {
        "id": "ethnicity",
        "text": "What is your ethnicity?",
        "type": "select",
        "section": "personal",
        "required": True,
        "options": [
            {"value": "asian", "text": "Asian"},
            {"value": "black", "text": "Black or African American"},
            {"value": "hispanic", "text": "Hispanic or Latino"},
            {"value": "native", "text": "Native American or Alaska Native"},
            {"value": "pacific", "text": "Native Hawaiian or Pacific Islander"},
            {"value": "white", "text": "White"},
            {"value": "mixed", "text": "Two or more races"},
            {"value": "other", "text": "Other"},
            {"value": "prefer_not", "text": "Prefer not to say"},
        ],
    },
    {
        "id": "gender",
        "text": "For comparative statistics, would you like to be compared to the male or female population?",
        "type": "select",
        "section": "personal",
        "required": True,
        "options": [
            {"value": "male", "text": "Male"},
            {"value": "female", "text": "Female"},
        ],
    },
    {"id": "height_inches", "text": "What is your height in inches? For example, 5 foot 8 would be 68 inches.", "type": "number", "section": "personal", "required": True, "min": 36, "max": 96},
    {"id": "weight_pounds", "text": "What is your weight in pounds?", "type": "number", "section": "personal", "required": True, "min": 50, "max": 700},
    # Qualifying Questions
    {"id": "q_weight_concerns", "text": "Do you currently or have you ever had concerns about your weight?", "type": "boolean", "section": "qualifying", "required": True},
    {"id": "q_reduce_body_fat", "text": "Would you like to reduce your current body fat percentage?", "type": "boolean", "section": "qualifying", "required": True},
    {"id": "q_athlete", "text": "Are you an athlete or fitness enthusiast interested in competing or performing at a higher level?", "type": "boolean", "section": "qualifying", "required": True},
    {"id": "q_family_history", "text": "Do you or any immediate family members have heart disease or diabetes?", "type": "boolean", "section": "qualifying", "required": True},
    {"id": "q_high_blood_pressure", "text": "Do you have high blood pressure?", "type": "boolean", "section": "qualifying", "required": True},
    {"id": "q_injuries", "text": "Are you currently suffering from any joint, muscular, or ligament injuries?", "type": "boolean", "section": "qualifying", "required": True},
    # Disqualifying Questions
    {"id": "disq_barium_xray", "text": "Have you had a barium X-ray in the last 2 weeks?", "type": "boolean", "section": "disqualifying", "required": True},
    {"id": "disq_nuclear_scan", "text": "Have you had a nuclear medicine scan or injection of an X-ray dye in the last week?", "type": "boolean", "section": "disqualifying", "required": True},
]


class IntakeForm:
    """Manages the DEXA scan intake form state and provides tools for the agent."""

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

    def restart_form(self) -> dict:
        """Clear all answers and restart the form."""
        self._answers = {}
        self._current_index = 0
        self._is_started = True
        self._is_submitted = False
        logger.info("Intake form restarted")
        return {
            "success": True,
            "message": "Form has been cleared and restarted",
            "next_question": self._format_question(self._fields[0]),
        }

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

    def _get_current_field(self) -> Optional[dict]:
        """Get the current field to ask about."""
        if self._current_index < len(self._fields):
            return self._fields[self._current_index]
        return None

    def _format_question(self, field: dict) -> str:
        """Format a field as a question for the agent."""
        text = field["text"]
        ftype = field["type"]

        if ftype == "select" and "options" in field:
            opts = ", ".join(o["text"] for o in field["options"])
            text += f" The options are: {opts}."
        elif ftype == "boolean":
            text += " Yes or no?"
        elif ftype == "number":
            if "min" in field and "max" in field:
                text += f" Please provide a number between {field['min']} and {field['max']}."

        return text

    def _process_answer(self, answer: str, field: dict) -> tuple[bool, Any, str]:
        """Process and validate an answer. Returns (success, processed_value, error_message)."""
        answer = answer.strip()
        ftype = field["type"]

        if ftype == "string":
            if not answer:
                return False, None, "Please provide a non-empty answer."
            return True, answer, ""

        elif ftype == "number":
            try:
                # Handle common spoken numbers
                num = float(answer.replace(",", ""))
                if "min" in field and num < field["min"]:
                    return False, None, f"The number should be at least {field['min']}."
                if "max" in field and num > field["max"]:
                    return False, None, f"The number should be no more than {field['max']}."
                return True, int(num) if num.is_integer() else num, ""
            except ValueError:
                return False, None, "Please provide a valid number."

        elif ftype == "boolean":
            lower = answer.lower()
            if lower in ["yes", "true", "y", "1", "yeah", "yep", "correct", "right", "affirmative"]:
                return True, True, ""
            elif lower in ["no", "false", "n", "0", "nope", "nah", "negative"]:
                return True, False, ""
            return False, None, "Please answer yes or no."

        elif ftype == "select":
            lower = answer.lower()
            for opt in field.get("options", []):
                if lower in (opt["text"].lower(), opt["value"].lower()) or lower in opt["text"].lower():
                    return True, opt["value"], ""
            opts = ", ".join(o["text"] for o in field["options"])
            return False, None, f"Please choose from: {opts}."

        return True, answer, ""

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
                "error": "Form is already complete. Submit the form or restart if needed.",
                "is_complete": True,
            }

        success, processed, error = self._process_answer(answer, field)
        if not success:
            return {
                "success": False,
                "error": error,
                "current_question": self._format_question(field),
            }

        self._answers[field["id"]] = processed
        self._current_index += 1
        logger.info(f"Recorded '{field['id']}': {processed}")

        next_field = self._get_current_field()

        # Check section transitions
        section_message = ""
        if next_field and field["section"] != next_field["section"]:
            if next_field["section"] == "qualifying":
                section_message = "Now I need to ask a few qualifying questions. "
            elif next_field["section"] == "disqualifying":
                section_message = "Almost done. Just two more quick questions. "

        return {
            "success": True,
            "recorded_field": field["id"],
            "recorded_value": processed,
            "section_message": section_message,
            "next_question": self._format_question(next_field) if next_field else None,
            "is_complete": next_field is None,
            "progress": f"{len(self._answers)}/{len(self._fields)}",
        }

    def get_form_data(self) -> dict:
        """Get the complete form data as JSON-serializable dict."""
        return {
            "personal_info": {
                "first_name": self._answers.get("first_name"),
                "last_name": self._answers.get("last_name"),
                "email": self._answers.get("email"),
                "phone": self._answers.get("phone"),
                "date_of_birth": self._answers.get("date_of_birth"),
                "ethnicity": self._answers.get("ethnicity"),
                "gender": self._answers.get("gender"),
                "height_inches": self._answers.get("height_inches"),
                "weight_pounds": self._answers.get("weight_pounds"),
            },
            "qualifying_questions": {
                "weight_concerns": self._answers.get("q_weight_concerns"),
                "reduce_body_fat": self._answers.get("q_reduce_body_fat"),
                "athlete": self._answers.get("q_athlete"),
                "family_history": self._answers.get("q_family_history"),
                "high_blood_pressure": self._answers.get("q_high_blood_pressure"),
                "injuries": self._answers.get("q_injuries"),
            },
            "disqualifying_questions": {
                "barium_xray": self._answers.get("disq_barium_xray"),
                "nuclear_scan": self._answers.get("disq_nuclear_scan"),
            },
        }

    def check_eligibility(self) -> dict:
        """Check if the user is eligible for a DEXA scan based on answers."""
        # Must answer YES to at least one qualifying question
        qualifying = [
            self._answers.get("q_weight_concerns"),
            self._answers.get("q_reduce_body_fat"),
            self._answers.get("q_athlete"),
            self._answers.get("q_family_history"),
            self._answers.get("q_high_blood_pressure"),
            self._answers.get("q_injuries"),
        ]
        has_qualifying = any(q is True for q in qualifying if q is not None)

        # Must answer NO to both disqualifying questions
        barium = self._answers.get("disq_barium_xray")
        nuclear = self._answers.get("disq_nuclear_scan")
        has_disqualifying = barium is True or nuclear is True

        eligible = has_qualifying and not has_disqualifying

        reasons = []
        if not has_qualifying:
            reasons.append("You must answer yes to at least one qualifying question.")
        if barium is True:
            reasons.append("A barium X-ray within 2 weeks disqualifies you. Please reschedule.")
        if nuclear is True:
            reasons.append("A nuclear medicine scan or X-ray dye injection within 1 week disqualifies you. Please reschedule.")

        return {
            "eligible": eligible,
            "reasons": reasons,
        }

    async def submit_form(self) -> dict:
        """Submit the completed form to the API."""
        if self._current_index < len(self._fields):
            return {
                "success": False,
                "error": "Form is not complete. Please answer all questions first.",
            }

        if self._is_submitted:
            return {
                "success": False,
                "error": "Form has already been submitted.",
            }

        eligibility = self.check_eligibility()
        form_data = self.get_form_data()

        # Mock API call
        await asyncio.sleep(0.5)
        logger.info(f"Submitting form data: {json.dumps(form_data, indent=2)}")

        self._is_submitted = True

        if not eligibility["eligible"]:
            return {
                "success": True,
                "submitted": True,
                "eligible": False,
                "message": "Form submitted but you are not currently eligible for a scan.",
                "reasons": eligibility["reasons"],
                "contact": "Please contact support@bodyspec.com for assistance.",
            }

        return {
            "success": True,
            "submitted": True,
            "eligible": True,
            "message": "Form submitted successfully! You are eligible for a DEXA scan.",
            "confirmation_number": f"DEXA-{hash(json.dumps(form_data)) % 100000:05d}",
            "next_steps": "You can now book an appointment. We will send a confirmation email shortly.",
        }


# Global form instance (per session - in production this would be per-call)
_form_instance: Optional[IntakeForm] = None


def get_form() -> IntakeForm:
    """Get or create the form instance."""
    global _form_instance
    if _form_instance is None:
        _form_instance = IntakeForm()
    return _form_instance


def reset_form_instance():
    """Reset the global form instance."""
    global _form_instance
    _form_instance = None


# Tool definitions

@loopback_tool
async def start_intake_form(ctx: ToolEnv) -> str:
    """Start the DEXA scan intake form. Use when user wants to book an appointment, get started, or asks how often they should scan."""
    form = get_form()
    status = form.start_form()

    first_question = status.get("current_question", "")
    return (
        f"Starting intake form. Progress: {status['progress']}. "
        f"First question: {first_question}"
    )


@loopback_tool
async def record_intake_answer(
    ctx: ToolEnv,
    answer: Annotated[str, "The user's answer to the current form question"],
) -> str:
    """Record the user's answer to the current intake form question."""
    form = get_form()
    result = form.record_answer(answer)

    if not result["success"]:
        return f"Could not record answer: {result.get('error', 'Unknown error')}. Current question: {result.get('current_question', '')}"

    if result["is_complete"]:
        eligibility = form.check_eligibility()
        if eligibility["eligible"]:
            return "Form complete! The user is eligible for a DEXA scan. Ask if they want to submit the form."
        else:
            reasons = " ".join(eligibility["reasons"])
            return f"Form complete but user may not be eligible: {reasons}. Ask if they want to submit anyway or contact support."

    section_msg = result.get("section_message", "")
    next_q = result.get("next_question", "")
    progress = result.get("progress", "")

    return f"Recorded. Progress: {progress}. {section_msg}Next question: {next_q}"


@loopback_tool
async def get_intake_form_status(ctx: ToolEnv) -> str:
    """Check the current status of the intake form including progress and next question."""
    form = get_form()
    status = form.get_status()

    if not status["is_started"]:
        return "Intake form has not been started yet."

    if status["is_submitted"]:
        return "Intake form has already been submitted."

    if status["is_complete"]:
        return f"Intake form is complete with {status['progress']} questions answered. Ready to submit."

    return (
        f"Form in progress: {status['progress']} answered. "
        f"Current section: {status['current_section']}. "
        f"Current question: {status['current_question']}"
    )


@loopback_tool
async def restart_intake_form(ctx: ToolEnv) -> str:
    """Clear all answers and restart the intake form from the beginning. Only use if the user explicitly asks to start over."""
    form = get_form()
    result = form.restart_form()
    return f"Form restarted. {result['next_question']}"


@loopback_tool
async def submit_intake_form(ctx: ToolEnv) -> str:
    """Submit the completed intake form. Only use after all questions are answered."""
    form = get_form()
    result = await form.submit_form()

    if not result["success"]:
        return f"Could not submit: {result['error']}"

    if not result["eligible"]:
        reasons = " ".join(result["reasons"])
        return f"Form submitted. Unfortunately, {reasons} {result['contact']}"

    return (
        f"Form submitted successfully! Confirmation number: {result['confirmation_number']}. "
        f"{result['next_steps']}"
    )
