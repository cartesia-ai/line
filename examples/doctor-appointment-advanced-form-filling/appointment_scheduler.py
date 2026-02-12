"""Appointment scheduling for doctor visits with mock availability data."""

import asyncio
from datetime import datetime, timedelta
import random
from typing import Annotated, Optional

from intake_form import get_form
from loguru import logger

from line.llm_agent import ToolEnv, loopback_tool


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
        self._all_slots: list[dict] = sorted(_generate_mock_availability(), key=lambda x: (x["date_iso"], x["time"]))
        logger.info("AppointmentScheduler initialized")

    async def get_availability(self, days_ahead: int = 7) -> dict:
        """Get available appointment slots."""

        # Simulate a slow API call
        await asyncio.sleep(1)

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

        # Map time periods to hour ranges
        time_periods = {
            "morning": range(6, 12),
            "afternoon": range(12, 17),
            "evening": range(17, 21),
        }

        def slot_matches_date(slot: dict) -> bool:
            slot_date = slot["date"].lower()  # e.g., "thursday, february 13"
            slot_iso = slot["date_iso"]  # e.g., "2025-02-13"
            return (
                date_lower in slot_date
                or slot_iso == date_lower
                or any(part in slot_date for part in date_lower.split())
            )

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

            # Check for exact/partial time match
            # Normalize time_lower: "2pm" -> "2:00 pm", "2:00" -> "2:00"
            time_normalized = time_lower.replace("am", " am").replace("pm", " pm").strip()
            return time_normalized in slot_time or slot_time.startswith(time_lower.split()[0])

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

    async def book_appointment(self, first_name: str, last_name: str, email: str, phone: str) -> dict:
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
                "email": email,
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
            "message": "Appointment booked successfully! An email will be sent shortly.",
        }


# Global scheduler instance
_scheduler_instance: Optional[AppointmentScheduler] = None


def get_scheduler() -> AppointmentScheduler:
    """Get or create the scheduler instance."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = AppointmentScheduler()
    return _scheduler_instance


def reset_scheduler_instance():
    """Reset the global scheduler instance."""
    global _scheduler_instance
    _scheduler_instance = None


# Tool definitions


@loopback_tool
async def check_availability(ctx: ToolEnv) -> str:
    """Check available appointment times."""
    scheduler = get_scheduler()
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
    Also use this when the user wants to change their selected time before booking."""
    scheduler = get_scheduler()
    result = scheduler.select_slot(date, time)

    if not result["success"]:
        return result["error"]

    slot = result["selected"]
    return (
        f"Selected {slot['time']} on {slot['date']}. "
        f"Ready to book using the contact info from the intake form."
    )


@loopback_tool(is_background=True)
async def book_appointment(ctx: ToolEnv) -> str:
    """Book the selected appointment slot using contact info from the completed intake form."""
    form = get_form()
    contact = form.get_contact_info()
    if not contact:
        return (
            "Contact info is missing from the intake form. "
            "Please complete the form (name, email, and phone) before booking."
        )
    scheduler = get_scheduler()
    result = await scheduler.book_appointment(
        contact["first_name"],
        contact["last_name"],
        contact["email"],
        contact["phone"],
    )

    if not result["success"]:
        return result["error"]

    appt = result["appointment"]
    return (
        f"Your appointment is confirmed! "
        f"You're scheduled for {appt['time']} on {appt['date']}. "
        f"An email is on its way to {contact['email']}."
    )
