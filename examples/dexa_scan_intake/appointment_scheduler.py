"""Appointment scheduling for DEXA scans with mock availability data."""

import asyncio
import random
from datetime import datetime, timedelta
from typing import Annotated, Optional
from loguru import logger

from line.llm_agent import ToolEnv, loopback_tool

# Mock BodySpec locations in San Francisco
LOCATIONS = {
    "financial_district": {
        "name": "Financial District",
        "address": "123 Market Street, Suite 400, San Francisco, CA 94105",
        "hours": "Mon-Fri 8am-6pm, Sat 9am-2pm",
    },
    "soma": {
        "name": "SoMa",
        "address": "456 Howard Street, San Francisco, CA 94103",
        "hours": "Mon-Fri 7am-7pm, Sat-Sun 9am-4pm",
    },
    "marina": {
        "name": "Marina District",
        "address": "789 Chestnut Street, San Francisco, CA 94123",
        "hours": "Mon-Fri 8am-5pm, Sat 10am-3pm",
    },
    "castro": {
        "name": "Castro",
        "address": "321 Castro Street, San Francisco, CA 94114",
        "hours": "Mon-Fri 9am-6pm, Sat 10am-2pm",
    },
    "sunset": {
        "name": "Sunset District",
        "address": "555 Irving Street, San Francisco, CA 94122",
        "hours": "Mon-Fri 8am-5pm, Sat 9am-1pm",
    },
}


def _generate_mock_availability(location_id: str, days_ahead: int = 7) -> list[dict]:
    """Generate mock availability slots for a location."""
    slots = []
    base_date = datetime.now()

    # Generate slots for the next N days
    for day_offset in range(1, days_ahead + 1):
        date = base_date + timedelta(days=day_offset)

        # Skip Sundays for most locations
        if date.weekday() == 6 and location_id not in ["soma"]:
            continue

        # Generate 2-4 available slots per day
        num_slots = random.randint(2, 4)

        # Available time slots based on location hours
        if date.weekday() < 5:  # Weekday
            possible_times = ["9:00 AM", "10:30 AM", "12:00 PM", "2:00 PM", "3:30 PM", "5:00 PM"]
        else:  # Weekend
            possible_times = ["10:00 AM", "11:30 AM", "1:00 PM"]

        selected_times = random.sample(possible_times, min(num_slots, len(possible_times)))
        selected_times.sort()

        for time in selected_times:
            slots.append({
                "date": date.strftime("%A, %B %d"),
                "date_iso": date.strftime("%Y-%m-%d"),
                "time": time,
                "location_id": location_id,
                "location_name": LOCATIONS[location_id]["name"],
                "slot_id": f"{location_id}_{date.strftime('%Y%m%d')}_{time.replace(':', '').replace(' ', '')}",
            })

    return slots


class AppointmentScheduler:
    """Manages appointment scheduling with mock availability."""

    def __init__(self):
        self._selected_slot: Optional[dict] = None
        self._booked_appointment: Optional[dict] = None
        self._availability_cache: dict[str, list[dict]] = {}
        self._contact_for_link: Optional[dict] = None
        logger.info("AppointmentScheduler initialized")

    def get_locations(self) -> list[dict]:
        """Get list of all locations."""
        return [
            {"id": loc_id, **loc_data}
            for loc_id, loc_data in LOCATIONS.items()
        ]

    def get_availability(self, location_id: Optional[str] = None, days_ahead: int = 7) -> dict:
        """Get available appointment slots."""
        if location_id and location_id not in LOCATIONS:
            return {
                "success": False,
                "error": f"Unknown location. Available locations: {', '.join(LOCATIONS.keys())}",
            }

        all_slots = []

        if location_id:
            # Get availability for specific location
            slots = _generate_mock_availability(location_id, days_ahead)
            all_slots.extend(slots)
        else:
            # Get availability for all locations
            for loc_id in LOCATIONS:
                slots = _generate_mock_availability(loc_id, days_ahead)
                all_slots.extend(slots[:3])  # Limit to 3 per location for voice readability

        # Sort by date then time
        all_slots.sort(key=lambda x: (x["date_iso"], x["time"]))

        return {
            "success": True,
            "slots": all_slots[:10],  # Limit to 10 for voice
            "total_available": len(all_slots),
            "showing": min(10, len(all_slots)),
        }

    def select_slot(self, slot_description: str) -> dict:
        """Select an appointment slot based on user description."""
        # Get fresh availability
        all_slots = []
        for loc_id in LOCATIONS:
            slots = _generate_mock_availability(loc_id, 14)
            all_slots.extend(slots)

        # Try to match the description
        slot_description_lower = slot_description.lower()

        for slot in all_slots:
            # Check if description matches date, time, or location
            if (slot["date"].lower() in slot_description_lower or
                slot["time"].lower() in slot_description_lower or
                slot["location_name"].lower() in slot_description_lower or
                slot["date_iso"] in slot_description_lower):

                # Additional matching for day of week
                day_match = any(day in slot_description_lower for day in
                              ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])
                time_match = any(t in slot_description_lower for t in
                               ["morning", "afternoon", "9", "10", "11", "12", "1", "2", "3", "4", "5"])

                if day_match or time_match or slot["location_name"].lower() in slot_description_lower:
                    self._selected_slot = slot
                    return {
                        "success": True,
                        "selected": slot,
                        "message": f"Selected {slot['time']} on {slot['date']} at {slot['location_name']}",
                    }

        return {
            "success": False,
            "error": "Could not find a matching slot. Please specify date, time, and/or location more clearly.",
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

        confirmation_number = f"BS-{random.randint(100000, 999999)}"

        self._booked_appointment = {
            "confirmation_number": confirmation_number,
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
            "confirmation_number": confirmation_number,
            "appointment": {
                "date": self._selected_slot["date"],
                "time": self._selected_slot["time"],
                "location": self._selected_slot["location_name"],
                "address": LOCATIONS[self._selected_slot["location_id"]]["address"],
            },
            "message": "Appointment booked successfully! A confirmation email will be sent shortly.",
        }

    async def send_availability_link(self, first_name: str, last_name: str, email: str, phone: str) -> dict:
        """Send a link to check more availabilities via email."""
        # Simulate API call
        await asyncio.sleep(0.3)

        self._contact_for_link = {
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "phone": phone,
        }

        logger.info(f"Sending availability link to {email}")

        return {
            "success": True,
            "message": f"A link to view all available appointments has been sent to {email}. "
                      "The link will be valid for 48 hours.",
        }

    def reset(self):
        """Reset scheduler state."""
        self._selected_slot = None
        self._booked_appointment = None
        self._availability_cache = {}
        self._contact_for_link = None


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
async def list_locations(ctx: ToolEnv) -> str:
    """List all BodySpec DEXA scan locations in San Francisco."""
    scheduler = get_scheduler()
    locations = scheduler.get_locations()

    result = "We have 5 locations in San Francisco:\n"
    for loc in locations:
        result += f"- {loc['name']}: {loc['address']}. Hours: {loc['hours']}\n"

    return result


@loopback_tool
async def check_availability(
    ctx: ToolEnv,
    location: Annotated[str, "Optional location name or ID. Leave empty for all locations."] = "",
) -> str:
    """Check available appointment times. Can filter by location or show all."""
    scheduler = get_scheduler()

    # Map common location names to IDs
    location_map = {
        "financial": "financial_district",
        "financial district": "financial_district",
        "downtown": "financial_district",
        "soma": "soma",
        "south of market": "soma",
        "marina": "marina",
        "castro": "castro",
        "sunset": "sunset",
    }

    location_id = None
    if location:
        location_lower = location.lower()
        location_id = location_map.get(location_lower, location_lower)
        if location_id not in LOCATIONS:
            location_id = None  # Fall back to all locations

    result = scheduler.get_availability(location_id)

    if not result["success"]:
        return result["error"]

    slots = result["slots"]
    if not slots:
        return "No available appointments found in the next week. Would you like me to send you a link to check more dates?"

    # Format for voice
    output = f"I found {result['total_available']} available slots. Here are some options:\n"

    current_date = None
    for slot in slots:
        if slot["date"] != current_date:
            current_date = slot["date"]
            output += f"\n{slot['date']}:\n"
        output += f"- {slot['time']} at {slot['location_name']}\n"

    if result["total_available"] > result["showing"]:
        output += f"\nThere are {result['total_available'] - result['showing']} more slots available. "
        output += "Let me know if you'd like to see a specific location or I can send you a link to view all options."

    return output


@loopback_tool
async def select_appointment_slot(
    ctx: ToolEnv,
    selection: Annotated[str, "The user's selection - can include date, time, and/or location"],
) -> str:
    """Select an appointment slot based on user's preference (date, time, location)."""
    scheduler = get_scheduler()
    result = scheduler.select_slot(selection)

    if not result["success"]:
        return result["error"]

    slot = result["selected"]
    return (
        f"Got it! I've selected {slot['time']} on {slot['date']} at our {slot['location_name']} location. "
        f"To confirm this booking, I'll need your name, email, and phone number."
    )


@loopback_tool
async def book_appointment(
    ctx: ToolEnv,
    first_name: Annotated[str, "Patient's first name"],
    last_name: Annotated[str, "Patient's last name"],
    email: Annotated[str, "Patient's email address"],
    phone: Annotated[str, "Patient's phone number"],
) -> str:
    """Book the selected appointment slot with patient information."""
    scheduler = get_scheduler()
    result = await scheduler.book_appointment(first_name, last_name, email, phone)

    if not result["success"]:
        return result["error"]

    appt = result["appointment"]
    return (
        f"Your appointment is confirmed! Confirmation number: {result['confirmation_number']}. "
        f"You're scheduled for {appt['time']} on {appt['date']} at our {appt['location']} location, "
        f"{appt['address']}. A confirmation email is on its way to {email}."
    )


@loopback_tool
async def send_availability_link(
    ctx: ToolEnv,
    first_name: Annotated[str, "User's first name"],
    last_name: Annotated[str, "User's last name"],
    email: Annotated[str, "User's email address"],
    phone: Annotated[str, "User's phone number"],
) -> str:
    """Send a link to view all available appointments when the shown times don't work."""
    scheduler = get_scheduler()
    result = await scheduler.send_availability_link(first_name, last_name, email, phone)

    if not result["success"]:
        return "There was an issue sending the link. Please try again."

    return result["message"]
