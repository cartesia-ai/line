"""
FormFiller - Loads questions from YAML and provides a loopback tool for recording answers.
"""

from pathlib import Path
from typing import Annotated, Any, Optional

import yaml
from loguru import logger

from line.v02.llm import loopback_tool
from line.v02.llm.agent import ToolEnv


class FormFiller:
    """
    Form filler that loads questions from YAML and provides a loopback tool
    for recording answers and determining the next question.

    Usage:
        form = FormFiller("form.yaml")
        agent = LlmAgent(
            tools=[form.tool, end_call],
            config=LlmConfig(system_prompt="Your prompt here..."),
        )

    The record_answer tool returns:
        - completed: dict of answered questions
        - remaining: list of remaining question IDs
        - next_question: the next question to ask (or None if complete)
        - is_complete: whether the form is done
    """

    def __init__(self, form_path: str, system_prompt: str):
        self.form_path = form_path
        self.system_prompt = system_prompt
        self._config = self._load_config()
        self._questions = self._flatten_questions(self._config["questionnaire"]["questions"])
        self._answers: dict = {}
        self._current_index: int = 0
        self._tool = self._create_tool()
        logger.info(f"FormFiller initialized with {len(self._questions)} questions")

    def _load_config(self) -> dict:
        with open(Path(self.form_path), "r") as f:
            return yaml.safe_load(f)

    def _flatten_questions(self, questions: list, parent_path: str = "") -> list:
        flattened = []
        for q in questions:
            if q.get("type") == "group":
                group_path = f"{parent_path}.{q['id']}" if parent_path else q["id"]
                flattened.extend(self._flatten_questions(q["questions"], group_path))
            else:
                q = q.copy()
                q["full_id"] = f"{parent_path}.{q['id']}" if parent_path else q["id"]
                flattened.append(q)
        return flattened

    def _should_show_question(self, question: dict) -> bool:
        if "dependsOn" not in question:
            return True

        dep = question["dependsOn"]
        dep_answer = self._answers.get(dep["questionId"])
        if dep_answer is None:
            return False

        op = dep.get("operator", "equals")
        val = dep["value"]

        if op == "equals":
            return dep_answer == val
        elif op == "not_equals":
            return dep_answer != val
        elif op == "in":
            return dep_answer in val if isinstance(val, list) else False
        elif op == "not_in":
            return dep_answer not in val if isinstance(val, list) else True
        return True

    def _get_current_question(self) -> Optional[dict]:
        while self._current_index < len(self._questions):
            q = self._questions[self._current_index]
            if self._should_show_question(q):
                return q
            self._current_index += 1
        return None

    def _format_question(self, q: dict) -> str:
        text = q["text"]
        qtype = q["type"]

        if qtype == "select" and "options" in q:
            opts = ", ".join(o["text"] for o in q["options"])
            text += f" (Options: {opts})"
        elif qtype == "boolean":
            text += " (yes or no)"
        elif qtype == "number":
            if "min" in q and "max" in q:
                text += f" (between {q['min']} and {q['max']})"
            elif "min" in q:
                text += f" (minimum {q['min']})"
            elif "max" in q:
                text += f" (maximum {q['max']})"

        return text

    def _process_answer(self, answer: str, q: dict) -> Optional[Any]:
        answer = answer.strip()
        qtype = q["type"]

        if qtype == "string":
            return answer
        elif qtype == "number":
            try:
                num = float(answer)
                if "min" in q and num < q["min"]:
                    return None
                if "max" in q and num > q["max"]:
                    return None
                return int(num) if num.is_integer() else num
            except ValueError:
                return None
        elif qtype == "boolean":
            lower = answer.lower()
            if lower in ["yes", "true", "y", "1"]:
                return True
            elif lower in ["no", "false", "n", "0"]:
                return False
            return None
        elif qtype == "select":
            for opt in q.get("options", []):
                if answer.lower() in (opt["text"].lower(), opt["value"].lower()):
                    return opt["value"]
            return None
        elif qtype == "date":
            return answer

        return answer

    def _create_tool(self):
        form = self

        @loopback_tool(
            name="record_answer",
            description="Record a VALID answer to the current form question. Only call this when the user has clearly provided an answer that matches the question being asked. Do NOT call if the user said something unrelated, unclear, or did not answer the question - instead ask for clarification.",
        )
        async def record_answer(ctx: ToolEnv, answer: Annotated[str, "The user's actual answer extracted from their response - must be relevant to the current question"]):
            return form.record_answer(answer)

        return record_answer

    @property
    def tool(self):
        """The loopback tool for recording answers."""
        return self._tool

    @property
    def form_title(self) -> str:
        return self._config["questionnaire"].get("text", "Form")

    @property
    def answers(self) -> dict:
        return self._answers.copy()

    @property
    def is_complete(self) -> bool:
        return self._get_current_question() is None

    def get_current_question_text(self) -> str:
        """Get the current question formatted for display."""
        q = self._get_current_question()
        if q:
            return self._format_question(q)
        return ""

    def get_system_prompt(self) -> str:
        """Generate system prompt including form structure and current state."""
        form_title = self._config["questionnaire"].get("text", "Form")

        # Build form overview with all questions
        questions_overview = []
        for i, q in enumerate(self._questions, 1):
            q_desc = f"  {i}. {q['id']}: {q['text']}"
            if q.get("type") == "select" and "options" in q:
                opts = [o["text"] for o in q["options"]]
                q_desc += f" (Options: {', '.join(opts)})"
            elif q.get("type") == "boolean":
                q_desc += " (yes/no)"
            elif q.get("type") == "number":
                if "min" in q and "max" in q:
                    q_desc += f" (between {q['min']} and {q['max']})"
            if q.get("dependsOn"):
                dep = q["dependsOn"]
                q_desc += f" [conditional: only if {dep['questionId']} {dep.get('operator', 'equals')} {dep['value']}]"
            questions_overview.append(q_desc)

        form_prompt = f"""## Form: {form_title}

You are conducting a questionnaire to collect information from the user.

### Questions in this form:
{chr(10).join(questions_overview)}

### How to conduct the form:
1. Ask ONE question at a time and WAIT for the user's response
2. ONLY call record_answer when the user provides a CLEAR, VALID answer to the current question
   - If the user says something unrelated (e.g., "yeah", "okay", "um"), do NOT call record_answer
   - If the answer doesn't make sense for the question (e.g., a name for an email), ask for clarification
   - If you're unsure what the user meant, ask them to repeat or clarify
3. After calling record_answer, the tool returns "next_question" - speak this question to the user
4. When is_complete is True, summarize all answers and ask for confirmation
5. Only call end_call AFTER the user confirms

### IMPORTANT:
- You must ASK the question out loud BEFORE the user can answer it
- Do not assume an answer - wait for the user to clearly state it
- If the user interrupts or says something off-topic, acknowledge and re-ask the current question

### Current state:
- Questions answered: {len(self._answers)}
- Current question to ask: {self.get_current_question_text() or "Form complete"}"""

        if self.system_prompt:
            return f"{self.system_prompt}\n\n{form_prompt}"
        return form_prompt

    def _get_remaining_questions(self) -> list[str]:
        """Get list of remaining question IDs."""
        remaining = []
        temp_index = self._current_index
        while temp_index < len(self._questions):
            q = self._questions[temp_index]
            if self._should_show_question(q):
                remaining.append(q["id"])
            temp_index += 1
        return remaining

    def record_answer(self, answer: str) -> dict:
        """Record an answer and return form status."""
        q = self._get_current_question()
        if not q:
            return {
                "success": False,
                "error": "Form is already complete",
                "completed": self._answers.copy(),
                "remaining": [],
                "next_question": None,
                "is_complete": True,
            }

        processed = self._process_answer(answer, q)
        if processed is None:
            return {
                "success": False,
                "error": f"Invalid answer for {q['type']} question",
                "completed": self._answers.copy(),
                "remaining": self._get_remaining_questions(),
                "next_question": self._format_question(q),
                "is_complete": False,
            }

        self._answers[q["id"]] = processed
        self._current_index += 1
        logger.info(f"Recorded answer for '{q['id']}': {processed}")

        next_q = self._get_current_question()
        logger.info(
            "record_answer response: {\n"
            f"  'success': True,\n"
            f"  'completed': {self._answers.copy()},\n"
            f"  'remaining': {self._get_remaining_questions()},\n"
            f"  'next_question': {self._format_question(next_q) if next_q else None},\n"
            f"  'is_complete': {next_q is None}\n"
            "}"
        )

        return {
            "success": True,
            "completed": self._answers.copy(),
            "remaining": self._get_remaining_questions(),
            "next_question": self._format_question(next_q) if next_q else None,
            "is_complete": next_q is None,
        }
