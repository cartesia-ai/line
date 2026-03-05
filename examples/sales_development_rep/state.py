"""
Lead state management for the Sales Development Rep agent.

Tracks all information gathered about a potential customer during the call.
"""

from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class LeadState:
    """Stateful lead information accumulated during the call."""

    # Contact info
    name: str = ""
    company: str = ""
    email: str = ""
    phone: str = ""
    role: str = ""

    # Interest and qualification
    interest_area: str = ""  # e.g., "voice agents", "TTS API", "enterprise"
    interest_level: str = "unknown"  # high, medium, low, unknown
    pain_points: list[str] = field(default_factory=list)
    timeline: str = ""  # "immediate", "1-3 months", "exploring"

    # Additional context
    notes: str = ""
    call_id: str = ""
    call_start_time: str = ""

    def merge(self, extracted: dict) -> list[str]:
        """Merge extracted data into state. Returns list of newly updated fields."""
        updated = []

        # Contact info
        if extracted.get("name") and not self.name:
            self.name = extracted["name"]
            updated.append("name")

        if extracted.get("company") and not self.company:
            self.company = extracted["company"]
            updated.append("company")

        if extracted.get("email") and not self.email:
            self.email = extracted["email"]
            updated.append("email")

        if extracted.get("phone") and not self.phone:
            self.phone = extracted["phone"]
            updated.append("phone")

        if extracted.get("role") and not self.role:
            self.role = extracted["role"]
            updated.append("role")

        # Interest and qualification
        if extracted.get("interest_area") and not self.interest_area:
            self.interest_area = extracted["interest_area"]
            updated.append("interest_area")

        if extracted.get("interest_level") and extracted["interest_level"] != "unknown":
            self.interest_level = extracted["interest_level"]
            if "interest_level" not in updated:
                updated.append("interest_level")

        if extracted.get("timeline") and not self.timeline:
            self.timeline = extracted["timeline"]
            updated.append("timeline")

        # Pain points - accumulate
        for pain_point in extracted.get("pain_points", []):
            if pain_point and pain_point not in self.pain_points:
                self.pain_points.append(pain_point)
                if "pain_points" not in updated:
                    updated.append("pain_points")

        # Notes - append
        if extracted.get("notes"):
            if self.notes:
                self.notes += f"; {extracted['notes']}"
            else:
                self.notes = extracted["notes"]
            updated.append("notes")

        return updated

    def get_missing_required(self) -> list[str]:
        """Get list of required fields not yet collected."""
        missing = []
        if not self.name:
            missing.append("name")
        if not self.company:
            missing.append("company")
        if not self.phone and not self.email:
            missing.append("phone or email")
        return missing

    def is_complete(self) -> bool:
        """Check if we have minimum required info for a qualified lead."""
        return bool(self.name and self.company and (self.phone or self.email))

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "company": self.company,
            "email": self.email,
            "phone": self.phone,
            "role": self.role,
            "interest_area": self.interest_area,
            "interest_level": self.interest_level,
            "pain_points": self.pain_points,
            "timeline": self.timeline,
            "notes": self.notes,
            "call_id": self.call_id,
            "call_start_time": self.call_start_time,
        }

    def to_json(self) -> str:
        """Convert to JSON string for API submission."""
        return json.dumps(self.to_dict(), indent=2)

    def to_api_payload(self) -> dict:
        """
        Convert to API payload format for CRM submission.
        This is the format you'd send to your actual CRM/sales API.
        """
        return {
            "lead": {
                "contact": {
                    "name": self.name,
                    "company": self.company,
                    "email": self.email,
                    "phone": self.phone,
                    "role": self.role,
                },
                "qualification": {
                    "interest_area": self.interest_area,
                    "interest_level": self.interest_level,
                    "pain_points": self.pain_points,
                    "timeline": self.timeline,
                },
                "metadata": {
                    "notes": self.notes,
                    "call_id": self.call_id,
                    "call_start_time": self.call_start_time,
                    "submitted_at": datetime.utcnow().isoformat(),
                    "source": "sdr_voice_agent",
                },
            }
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        parts = []
        if self.name:
            parts.append(f"Name: {self.name}")
        if self.company:
            parts.append(f"Company: {self.company}")
        if self.role:
            parts.append(f"Role: {self.role}")
        if self.email:
            parts.append(f"Email: {self.email}")
        if self.phone:
            parts.append(f"Phone: {self.phone}")
        if self.interest_area:
            parts.append(f"Interest: {self.interest_area}")
        if self.interest_level != "unknown":
            parts.append(f"Interest Level: {self.interest_level}")
        if self.pain_points:
            parts.append(f"Pain Points: {', '.join(self.pain_points)}")
        if self.timeline:
            parts.append(f"Timeline: {self.timeline}")

        return " | ".join(parts) if parts else "No lead info collected yet"
