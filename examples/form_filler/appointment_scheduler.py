"""Appointment scheduling for doctor visits with mock availability data."""

import asyncio
from datetime import datetime, timedelta
import random
import re

# IntakeForm is imported at runtime to avoid circular imports
from typing import TYPE_CHECKING, Annotated, Optional

from loguru import logger

from line.llm_agent import ToolEnv, loopback_tool

if TYPE_CHECKING:
    from intake_form import IntakeForm


def _generate_mock_availability(days_ahead: int = 7) -> list[dict]:
    """Generate mock availability slots."""
    slots = []
    base_date = datetime.now()

    # Generate slots for the next N days
    for day_offset in range(1, days_ahead + 1):
        date = base_date + timedelta(days=day_offset)

        # Skip Sundays
        if date.weekday() == 6:
            continue

        # Generate 2-4 available slots per day
        num_slots = random.randint(2, 4)

        if date.weekday() < 5:  # Weekday
            possible_times = ["9:00 AM", "10:30 AM", "12:00 PM", "2:00 PM", "3:30 PM", "5:00 PM"]
        else:  # Weekend
            possible_times = ["10:00 AM", "11:30 AM", "1:00 PM"]

        selected_times = random.sample(possible_times, min(num_slots, len(possible_times)))
        selected_times.sort()

        for time in selected_times:
            slots.append(
                {
                    "date": date.strftime("%A, %B %d"),
                    "date_iso": date.strftime("%Y-%m-%d"),
                    "time": time,
                    "slot_id": f"{date.strftime('%Y%m%d')}_{time.replace(':', '').replace(' ', '')}",
                }
            )

    return slots


class AppointmentScheduler:
    """Manages appointment scheduling with mock availability."""

    def __init__(self):
        self._selected_slot: Optional[dict] = None
        self._booked_appointment: Optional[dict] = None
        self._all_slots: list[dict] = []

    async def get_availability(self, days_ahead: int = 7) -> dict:
        """Get available appointment slots."""
        # Simulate a slow API call
        await asyncio.sleep(1)
        if len(self._all_slots) == 0:
            self._all_slots = sorted(
                _generate_mock_availability(days_ahead), key=lambda x: (x["date_iso"], x["time"])
            )

        return {
            "success": True,
            "slots": self._all_slots,
            "total_available": len(self._all_slots),
        }

    def select_slot(self, date: str, time: str) -> dict:
        """Select an appointment slot based on date and time.

        Args:
            date: Day of week (e.g., "Thursday") or date (e.g., "February 13", "2025-02-13")
            time: Time like "9:00 AM", "2:00 PM", or period like "morning", "afternoon"
        """
        date_lower = date.lower().strip()
        time_lower = time.lower().strip()

        if not date_lower:
            return {"success": False, "error": "Please provide a date for the appointment."}
        if not time_lower:
            return {"success": False, "error": "Please provide a time for the appointment."}

        # Map time periods to hour ranges
        time_periods = {
            "morning": range(6, 12),
            "afternoon": range(12, 17),
            "evening": range(17, 21),
        }

        def strip_ordinal(s: str) -> str:
            """Strip ordinal suffixes: '24th' -> '24', '1st' -> '1', etc."""
            for suffix in ("st", "nd", "rd", "th"):
                if s.endswith(suffix) and s[: -len(suffix)].isdigit():
                    return s[: -len(suffix)]
            return s

        def slot_matches_date(slot: dict) -> bool:
            slot_date = slot["date"].lower()  # e.g., "thursday, february 13"
            slot_iso = slot["date_iso"]  # e.g., "2025-02-13"

            # Exact ISO match (e.g., "2025-02-13")
            if slot_iso == date_lower:
                return True

            # Parse slot date into tokens: ["thursday", "february", "13"]
            # Remove comma and split
            slot_tokens = slot_date.replace(",", "").split()

            # Parse user input into tokens, stripping ordinal suffixes
            date_tokens = [strip_ordinal(t) for t in date_lower.replace(",", "").split()]

            # All user tokens must match slot tokens (as complete words)
            # E.g., "february 13" matches ["thursday", "february", "13"]
            # But "february 1" does NOT match ["thursday", "february", "13"]
            for user_token in date_tokens:
                # Try to match this token with any slot token
                matched = False
                for slot_token in slot_tokens:
                    # Exact match
                    if user_token == slot_token:
                        matched = True
                        break
                    # Handle zero-padded day: "4" should match "04"
                    # Only do this for numeric tokens to avoid false positives
                    if user_token.isdigit() and slot_token.isdigit():
                        if int(user_token) == int(slot_token):
                            matched = True
                            break

                if not matched:
                    return False

            return True

        def slot_matches_time(slot: dict) -> bool:
            slot_time = slot["time"].lower()  # e.g., "9:00 am"
            # Extract hour from slot time
            try:
                hour = int(slot_time.split(":")[0])
                if "pm" in slot_time and hour != 12:
                    hour += 12
            except ValueError:
                hour = 0

            # Check for time period match
            for period, hour_range in time_periods.items():
                if period in time_lower and hour in hour_range:
                    return True

            match = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)$", time_lower)
            if match:
                h, m, ampm = match.group(1), match.group(2) or "00", match.group(3)
                time_normalized = f"{h}:{m} {ampm}"
                return time_normalized == slot_time

            # Fallback: check if slot starts with the hour (e.g., "2" matches "2:00 pm")
            return slot_time.startswith(f"{time_lower.split()[0]}:")

        # Find matching slots
        matching_slots = [s for s in self._all_slots if slot_matches_date(s) and slot_matches_time(s)]

        if not matching_slots:
            # Try date-only match if no exact match
            date_matches = [s for s in self._all_slots if slot_matches_date(s)]
            if date_matches:
                return {
                    "success": False,
                    "error": f"No slot at {time} on that date. Available times: {', '.join(s['time'] for s in date_matches)}",
                    "available_on_date": date_matches,
                }
            return {
                "success": False,
                "error": f"No availability found for {date}. Please check available dates.",
            }

        # Select the first matching slot
        self._selected_slot = matching_slots[0]
        return {
            "success": True,
            "selected": self._selected_slot,
            "message": f"Selected {self._selected_slot['time']} on {self._selected_slot['date']}",
        }

    async def book_appointment(self, first_name: str, last_name: str, phone: str) -> dict:
        """Book the selected appointment slot."""
        if not self._selected_slot:
            return {
                "success": False,
                "error": "No slot selected. Please select a time slot first.",
            }

        # Simulate API call
        await asyncio.sleep(0.5)

        self._booked_appointment = {
            "slot": self._selected_slot,
            "patient": {
                "first_name": first_name,
                "last_name": last_name,
                "phone": phone,
            },
        }

        logger.info(f"Booked appointment: {self._booked_appointment}")

        return {
            "success": True,
            "appointment": {
                "date": self._selected_slot["date"],
                "time": self._selected_slot["time"],
            },
            "message": "Appointment booked successfully! A confirmation will be sent to your phone.",
        }


def create_scheduler_tools(scheduler: AppointmentScheduler, form: "IntakeForm"):
    """Create scheduler tools bound to specific scheduler and form instances."""

    @loopback_tool
    async def check_availability(ctx: ToolEnv) -> str:
        """Check available appointment times."""
        result = await scheduler.get_availability()

        if not result["success"]:
            return result["error"]

        slots = result["slots"]
        if not slots:
            return "No available appointments found in the next week."

        output = f"I found {result['total_available']} available slots. Here are some options:\n"

        current_date = None
        for slot in slots:
            if slot["date"] != current_date:
                current_date = slot["date"]
                output += f"\n{slot['date']}:\n"
            output += f"- {slot['time']}\n"

        return output

    @loopback_tool
    async def select_appointment_slot(
        ctx: ToolEnv,
        date: Annotated[str, "The date - day of week (e.g., 'Thursday') or date (e.g., 'February 13')"],
        time: Annotated[str, "The time (e.g., '9:00 AM', '2pm') or period (e.g., 'morning', 'afternoon')"],
    ) -> str:
        """Select an appointment slot based on the user's preferred date and time.
        Also use this when the user wants to change their selected time before booking.
        """
        result = scheduler.select_slot(date, time)

        if not result["success"]:
            return result["error"]

        slot = result["selected"]
        return (
            f"Selected {slot['time']} on {slot['date']}. "
            f"Ready to book using the contact info from the intake form."
        )

    @loopback_tool(is_background=True)
    async def book_appointment_and_submit_form(ctx: ToolEnv) -> str:
        """Book the selected appointment slot using contact info from the completed intake form."""
        contact = form.get_contact_info()
        if not contact:
            yield (
                "Contact info is missing from the intake form. "
                "Please complete the form (name and phone) before booking."
            )
            return

        # Submit the intake form now that the appointment is booked
        submit_result = await form.submit_form()
        if not submit_result["success"]:
            logger.warning(f"Failed to submit intake form: {submit_result.get('error')}")
        else:
            logger.info(f"Intake form submitted: {form.get_answered_fields_summary()}")

        result = await scheduler.book_appointment(
            contact["first_name"],
            contact["last_name"],
            contact["phone"],
        )

        if not result["success"]:
            yield result["error"]
            return

        appt = result["appointment"]
        yield (f"Your appointment is confirmed! You're scheduled for {appt['time']} on {appt['date']}. ")

    return [
        check_availability,
        select_appointment_slot,
        book_appointment_and_submit_form,
    ]
