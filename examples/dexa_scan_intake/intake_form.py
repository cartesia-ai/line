"""Intake form management for DEXA scan appointments."""

import asyncio
import json
from typing import Annotated, Any, Optional
from loguru import logger

from line.llm_agent import ToolEnv, loopback_tool

# Form field definitions
FORM_FIELDS = [
    # Personal Information
    {"id": "first_name", "text": "What is your first name?", "type": "string", "section": "personal", "required": True, "confirm": True},
    {"id": "last_name", "text": "What is your last name?", "type": "string", "section": "personal", "required": True, "confirm": True},
    {"id": "email", "text": "What is your email address?", "type": "string", "section": "personal", "required": True, "confirm": True},
    {"id": "phone", "text": "What is your phone number?", "type": "string", "section": "personal", "required": True, "confirm": True},
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

    def _get_field_by_id(self, field_id: str) -> Optional[dict]:
        """Get a field definition by its ID."""
        for field in self._fields:
            if field["id"] == field_id:
                return field
        return None

    def _get_field_index(self, field_id: str) -> int:
        """Get the index of a field by its ID. Returns -1 if not found."""
        for i, field in enumerate(self._fields):
            if field["id"] == field_id:
                return i
        return -1

    def get_answered_fields_summary(self) -> dict:
        """Get a summary of all answered fields with their values."""
        answered = []
        for field in self._fields:
            if field["id"] in self._answers:
                value = self._answers[field["id"]]
                # Format the value for display
                if field["type"] == "boolean":
                    display_value = "Yes" if value else "No"
                elif field["type"] == "select":
                    # Find the text for the selected option
                    display_value = value
                    for opt in field.get("options", []):
                        if opt["value"] == value:
                            display_value = opt["text"]
                            break
                else:
                    display_value = str(value)

                answered.append({
                    "field_id": field["id"],
                    "question": field["text"],
                    "answer": display_value,
                    "raw_value": value,
                })
        return {
            "answered_count": len(answered),
            "total_count": len(self._fields),
            "fields": answered,
        }

    def edit_answer(self, field_id: str, new_answer: str) -> dict:
        """Edit a previously answered field without changing the current position."""
        if not self._is_started:
            return {
                "success": False,
                "error": "Form has not been started yet.",
            }

        if self._is_submitted:
            return {
                "success": False,
                "error": "Form has already been submitted. Cannot edit answers.",
            }

        field = self._get_field_by_id(field_id)
        if not field:
            available = [f["id"] for f in self._fields]
            return {
                "success": False,
                "error": f"Unknown field '{field_id}'. Available fields: {', '.join(available)}",
            }

        if field_id not in self._answers:
            return {
                "success": False,
                "error": f"Field '{field_id}' has not been answered yet. Cannot edit.",
            }

        # Validate and process the new answer
        success, processed, error = self._process_answer(new_answer, field)
        if not success:
            return {
                "success": False,
                "error": error,
                "field_id": field_id,
                "question": self._format_question(field),
            }

        old_value = self._answers[field_id]
        self._answers[field_id] = processed
        logger.info(f"Edited '{field_id}': {old_value} -> {processed}")

        # Get current question info
        current = self._get_current_field()

        return {
            "success": True,
            "field_id": field_id,
            "old_value": old_value,
            "new_value": processed,
            "message": f"Updated {field_id} from '{old_value}' to '{processed}'.",
            "current_question": self._format_question(current) if current else None,
            "is_complete": current is None,
        }

    def go_back_to_question(self, field_id: str) -> dict:
        """Go back to a specific question, clearing all answers from that point forward."""
        if not self._is_started:
            return {
                "success": False,
                "error": "Form has not been started yet.",
            }

        if self._is_submitted:
            return {
                "success": False,
                "error": "Form has already been submitted. Use restart_form to start over.",
            }

        field_index = self._get_field_index(field_id)
        if field_index == -1:
            available = [f["id"] for f in self._fields]
            return {
                "success": False,
                "error": f"Unknown field '{field_id}'. Available fields: {', '.join(available)}",
            }

        # Clear answers from this field forward
        cleared_fields = []
        for i in range(field_index, len(self._fields)):
            fid = self._fields[i]["id"]
            if fid in self._answers:
                del self._answers[fid]
                cleared_fields.append(fid)

        self._current_index = field_index
        logger.info(f"Went back to '{field_id}', cleared: {cleared_fields}")

        field = self._fields[field_index]
        return {
            "success": True,
            "message": f"Returned to question: {field['text']}",
            "cleared_fields": cleared_fields,
            "current_question": self._format_question(field),
            "progress": f"{len(self._answers)}/{len(self._fields)}",
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
            "next_field": next_field,
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

@loopback_tool(is_background=True)
async def start_intake_form(ctx: ToolEnv) -> str:
    """Start the DEXA scan intake form. Use when user wants to book an appointment, get started, or asks how often they should scan."""
    form = get_form()
    status = form.start_form()

    first_question = status.get("current_question", "")
    return (
        f"Starting intake form. Progress: {status['progress']}. "
        f"First question: {first_question}"
    )


@loopback_tool(is_background=True)
async def record_intake_answer(
    ctx: ToolEnv,
    answer: Annotated[str, "The user's confirmed answer to record (NOT 'yes' or confirmations)"],
) -> str:
    """Record the user's answer to the current intake form question.
    IMPORTANT: Only call this with the actual answer value, never with 'yes' or confirmation words.
    For fields with confirm=true: first confirm verbally with user, then call this tool after they confirm."""
    form = get_form()
    result = form.record_answer(answer)

    if not result["success"]:
        return f"Error: {result.get('error', 'Unknown error')}. Current question: {result.get('current_question', '')}"

    recorded_field = result.get("recorded_field", "")

    if result["is_complete"]:
        eligibility = form.check_eligibility()
        if eligibility["eligible"]:
            return f"Recorded {recorded_field}. Form complete! User is eligible. Ask if they want to submit."
        else:
            return f"Recorded {recorded_field}. Form complete but eligibility issue. Ask if they want to submit anyway."

    section_msg = result.get("section_message", "")
    next_q = result.get("next_question", "")
    next_field = result.get("next_field", {})
    requires_confirm = next_field.get("confirm", False)

    confirm_note = " (confirm answer before recording)" if requires_confirm else ""
    return f"Recorded {recorded_field}. {section_msg}Next question{confirm_note}: {next_q}"


@loopback_tool(is_background=True)
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


@loopback_tool(is_background=True)
async def restart_intake_form(ctx: ToolEnv) -> str:
    """Clear all answers and restart the intake form from the beginning. Only use if the user explicitly asks to start over."""
    form = get_form()
    result = form.restart_form()
    return f"Form restarted. {result['next_question']}"


@loopback_tool(is_background=True)
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


@loopback_tool(is_background=True)
async def edit_intake_answer(
    ctx: ToolEnv,
    field_id: Annotated[str, "The ID of the field to edit (e.g., 'email', 'first_name', 'phone')"],
    new_answer: Annotated[str, "The new answer to set for this field"],
) -> str:
    """Edit a previous answer in the intake form without changing the current question.
    Use when the user wants to correct a specific answer they gave earlier (e.g., 'actually my email is different').
    The form will continue from where it left off after editing."""
    form = get_form()
    result = form.edit_answer(field_id, new_answer)

    if not result["success"]:
        return f"Could not edit answer: {result['error']}"

    response = f"Updated {result['field_id']}: '{result['old_value']}' → '{result['new_value']}'. "

    if result["is_complete"]:
        return response + "Form is complete and ready to submit."
    else:
        return response + f"Continuing with: {result['current_question']}"


@loopback_tool(is_background=True)
async def go_back_in_intake_form(
    ctx: ToolEnv,
    field_id: Annotated[str, "The ID of the field to go back to (e.g., 'email', 'first_name', 'date_of_birth')"],
) -> str:
    """Go back to a previous question in the intake form to re-answer it and subsequent questions.
    Use when the user wants to go back and redo from a certain point (e.g., 'wait, go back to the email question').
    This will clear answers from that question forward."""
    form = get_form()
    result = form.go_back_to_question(field_id)

    if not result["success"]:
        return f"Could not go back: {result['error']}"

    response = f"Going back. "
    if result["cleared_fields"]:
        response += f"Cleared {len(result['cleared_fields'])} answer(s). "
    response += f"Progress: {result['progress']}. "
    response += f"Question: {result['current_question']}"

    return response


@loopback_tool(is_background=True)
async def list_intake_answers(ctx: ToolEnv) -> str:
    """List all answers the user has provided so far in the intake form.
    Use when the user asks to review their answers or wants to know what they've entered."""
    form = get_form()
    summary = form.get_answered_fields_summary()

    if summary["answered_count"] == 0:
        return "No answers recorded yet. The form may not have been started."

    parts = [f"Answered {summary['answered_count']}/{summary['total_count']} questions:\n"]
    for field in summary["fields"]:
        parts.append(f"- {field['field_id']}: {field['answer']}\n")

    return "".join(parts)
