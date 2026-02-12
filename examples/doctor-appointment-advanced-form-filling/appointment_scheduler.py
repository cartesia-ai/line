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
        self._availability_cache: dict[str, list[dict]] = {}
        logger.info("AppointmentScheduler initialized")

    def get_availability(self, days_ahead: int = 7) -> dict:
        """Get available appointment slots."""
        all_slots = _generate_mock_availability(days_ahead)
        all_slots.sort(key=lambda x: (x["date_iso"], x["time"]))

        return {
            "success": True,
            "slots": all_slots[:10],  # Limit to 10 for voice
            "total_available": len(all_slots),
            "showing": min(10, len(all_slots)),
        }

    def select_slot(self, slot_description: str) -> dict:
        """Select an appointment slot based on user description."""
        all_slots = _generate_mock_availability(14)
        slot_description_lower = slot_description.lower()

        for slot in all_slots:
            if (
                slot["date"].lower() in slot_description_lower
                or slot["time"].lower() in slot_description_lower
                or slot["date_iso"] in slot_description_lower
            ):
                day_match = any(
                    day in slot_description_lower
                    for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                )
                time_match = any(
                    t in slot_description_lower
                    for t in ["morning", "afternoon", "9", "10", "11", "12", "1", "2", "3", "4", "5"]
                )

                if day_match or time_match:
                    self._selected_slot = slot
                    return {
                        "success": True,
                        "selected": slot,
                        "message": f"Selected {slot['time']} on {slot['date']}",
                    }

        return {
            "success": False,
            "error": "Could not find a matching slot. Please specify date and/or time more clearly.",
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
    result = scheduler.get_availability()

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

    if result["total_available"] > result["showing"]:
        output += f"\nThere are {result['total_available'] - result['showing']} more slots available."

    return output


@loopback_tool
async def select_appointment_slot(
    ctx: ToolEnv,
    selection: Annotated[str, "The user's selection - date and/or time"],
) -> str:
    """Select an appointment slot based on user's preference (date, time)."""
    scheduler = get_scheduler()
    result = scheduler.select_slot(selection)

    if not result["success"]:
        return result["error"]

    slot = result["selected"]
    return (
        f"Got it! I've selected {slot['time']} on {slot['date']}. "
        f"Ready to confirm the booking using the contact info from the intake form."
    )


@loopback_tool
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
