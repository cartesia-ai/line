"""Tools for the DEXA Scan Intake Agent."""

import asyncio
import os
from typing import Annotated

from exa_py import Exa
from loguru import logger

from line.llm_agent import ToolEnv, loopback_tool

# Mock database of users and their past appointments
MOCK_USER_DATABASE = {
    ("lucy", "liu", "1990-05-15"): {
        "user_id": "USR001",
        "email": "lucy.liu@example.com",
        "appointments": [
            {
                "date": "2025-01-15",
                "time": "10:30 AM",
                "location": "San Francisco - Financial District",
                "type": "Full Body DEXA Scan",
                "summary": {
                    "body_fat_percentage": 22.3,
                    "lean_mass_lbs": 98.5,
                    "bone_density_tscore": 1.2,
                    "visceral_fat_lbs": 1.8,
                },
            },
            {
                "date": "2024-10-08",
                "time": "2:00 PM",
                "location": "San Francisco - Financial District",
                "type": "Full Body DEXA Scan",
                "summary": {
                    "body_fat_percentage": 24.1,
                    "lean_mass_lbs": 95.2,
                    "bone_density_tscore": 1.1,
                    "visceral_fat_lbs": 2.1,
                },
            },
        ],
    },
    ("john", "smith", "1985-08-22"): {
        "user_id": "USR002",
        "email": "john.smith@example.com",
        "appointments": [
            {
                "date": "2025-02-01",
                "time": "9:00 AM",
                "location": "Los Angeles - Santa Monica",
                "type": "Full Body DEXA Scan",
                "summary": {
                    "body_fat_percentage": 18.5,
                    "lean_mass_lbs": 145.3,
                    "bone_density_tscore": 0.8,
                    "visceral_fat_lbs": 1.2,
                },
            },
        ],
    },
    ("sarah", "johnson", "1992-03-10"): {
        "user_id": "USR003",
        "email": "sarah.j@example.com",
        "appointments": [
            {
                "date": "2024-12-20",
                "time": "11:15 AM",
                "location": "San Diego - Downtown",
                "type": "Full Body DEXA Scan",
                "summary": {
                    "body_fat_percentage": 26.8,
                    "lean_mass_lbs": 88.4,
                    "bone_density_tscore": 0.5,
                    "visceral_fat_lbs": 2.4,
                },
            },
            {
                "date": "2024-09-15",
                "time": "3:30 PM",
                "location": "San Diego - Downtown",
                "type": "Full Body DEXA Scan",
                "summary": {
                    "body_fat_percentage": 28.2,
                    "lean_mass_lbs": 86.1,
                    "bone_density_tscore": 0.4,
                    "visceral_fat_lbs": 2.7,
                },
            },
            {
                "date": "2024-06-01",
                "time": "10:00 AM",
                "location": "San Diego - Downtown",
                "type": "Full Body DEXA Scan",
                "summary": {
                    "body_fat_percentage": 29.5,
                    "lean_mass_lbs": 84.8,
                    "bone_density_tscore": 0.3,
                    "visceral_fat_lbs": 2.9,
                },
            },
        ],
    },
}


async def _mock_api_call(first_name: str, last_name: str, date_of_birth: str) -> dict:
    """Simulate an async API call to the patient records system."""
    # Simulate network latency
    await asyncio.sleep(0.5)

    # Normalize inputs for lookup
    key = (first_name.lower().strip(), last_name.lower().strip(), date_of_birth.strip())

    if key in MOCK_USER_DATABASE:
        return {"success": True, "data": MOCK_USER_DATABASE[key]}
    else:
        return {"success": False, "error": "No matching records found"}


# DEXA Knowledge Base - used by lookup_dexa_knowledge tool
DEXA_KNOWLEDGE = {
    "what_is_dexa": "DEXA (Dual-Energy X-ray Absorptiometry) uses two X-ray beams to measure body composition and bone density with high precision, distinguishing bone, lean tissue, and fat.",
    "how_it_works": "You lie on an open table while a scanning arm passes over you, emitting low-dose X-rays. Different tissues absorb different amounts, allowing precise measurement of bone, muscle, and fat.",
    "what_it_measures": "Total body fat percentage, lean muscle mass by region, bone mineral density, visceral fat (around organs), and left/right symmetry.",
    "accuracy": "DEXA is the gold standard with 1-2% margin of error for body fat. Much more accurate than scales, calipers, or underwater weighing.",
    "safety": "Very safe - uses 1/10th the radiation of a chest X-ray (0.001 mSv), less than daily background radiation.",
    "preparation": "Wear comfortable clothes without metal. Avoid calcium supplements 24hrs before. Stay hydrated. No fasting needed but avoid large meals. Remove jewelry.",
    "what_to_expect": "Takes 7-10 minutes. Lie still on your back. Painless and non-invasive. The arm passes over but doesn't touch you.",
    "frequency": "Every 3-6 months for tracking changes. More frequent scans may not show meaningful differences.",
    "visceral_fat": "Fat around internal organs. High visceral fat increases risk of diabetes, heart disease, and metabolic syndrome. DEXA measures this directly.",
    "results": "Shows body fat percentage (essential/athletic/fit/average/obese ranges), lean mass, bone density vs peers, and regional breakdown.",
    "who_should_get": "Athletes, fitness trackers, bone health concerns, weight management, older adults, anyone wanting baseline health metrics.",
}


@loopback_tool
async def lookup_dexa_knowledge(
    ctx: ToolEnv,
    topic: Annotated[
        str,
        "The topic to look up. Options: what_is_dexa, how_it_works, what_it_measures, accuracy, safety, preparation, what_to_expect, frequency, visceral_fat, results, who_should_get",
    ],
) -> str:
    """Look up information about DEXA scans from the knowledge base. Use this to answer questions about DEXA."""
    topic_key = topic.lower().replace(" ", "_").replace("-", "_")

    # Try exact match first
    if topic_key in DEXA_KNOWLEDGE:
        return DEXA_KNOWLEDGE[topic_key]

    # Try partial match
    for key, value in DEXA_KNOWLEDGE.items():
        if topic_key in key or key in topic_key:
            return value

    # Return all topics if no match
    return f"Topic '{topic}' not found. Available: {', '.join(DEXA_KNOWLEDGE.keys())}"


@loopback_tool(is_background=True)
async def search_dexa_info(
    ctx: ToolEnv,
    query: Annotated[
        str,
        "The search query about DEXA scans, body composition, or related health topics.",
    ],
) -> str:
    """Search the web for DEXA scan information, providers, or related health topics."""
    logger.info(f"Performing Exa search for DEXA info: '{query}'")

    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        return "Web search is unavailable. I can still answer based on my knowledge base."

    try:
        client = Exa(api_key=api_key)
        results = await asyncio.to_thread(
            client.search_and_contents,
            f"DEXA scan {query}",
            num_results=5,
            type="auto",
            text={"max_characters": 800},
        )

        if not results or not results.results:
            return "I couldn't find specific information about that. Let me help with what I know."

        content_parts = [f"Search results for: '{query}'\n"]
        for i, result in enumerate(results.results[:5]):
            content_parts.append(f"\n--- Source {i + 1}: {result.title} ---\n")
            if result.text:
                content_parts.append(f"{result.text}\n")
            content_parts.append(f"URL: {result.url}\n")

        logger.info(f"Search completed: {len(results.results)} sources found")
        return "".join(content_parts)

    except Exception as e:
        logger.error(f"Exa search failed: {e}")
        return "Search encountered an issue. I can still help with my existing knowledge."


@loopback_tool
async def lookup_past_appointments(
    ctx: ToolEnv,
    first_name: Annotated[str, "The patient's first name."],
    last_name: Annotated[str, "The patient's last name."],
    date_of_birth: Annotated[str, "The patient's date of birth in YYYY-MM-DD format."],
) -> str:
    """Look up a patient's past DEXA scan appointments and results after verifying their identity."""
    logger.info(f"Looking up appointments for {first_name} {last_name}, DOB: {date_of_birth}")

    # Make async API call to patient records system
    response = await _mock_api_call(first_name, last_name, date_of_birth)

    if not response["success"]:
        return (
            "I wasn't able to find any records matching that information. "
            "Please double-check the name spelling and date of birth format, which should be "
            "year, month, day, like 1990-05-15. If you're a new patient, you may not have any "
            "records in our system yet."
        )

    data = response["data"]
    appointments = data["appointments"]

    if not appointments:
        return f"I found your account, but you don't have any past appointments on record yet."

    # Format the appointments for voice output
    result_parts = [f"I found {len(appointments)} appointment(s) on record for {first_name.title()}.\n\n"]

    for i, appt in enumerate(appointments, 1):
        result_parts.append(f"Appointment {i}:\n")
        result_parts.append(f"- Date: {appt['date']} at {appt['time']}\n")
        result_parts.append(f"- Location: {appt['location']}\n")
        result_parts.append(f"- Type: {appt['type']}\n")

        summary = appt["summary"]
        result_parts.append(f"- Results summary: Body fat {summary['body_fat_percentage']}%, ")
        result_parts.append(f"Lean mass {summary['lean_mass_lbs']} lbs, ")
        result_parts.append(f"Bone density T-score {summary['bone_density_tscore']}, ")
        result_parts.append(f"Visceral fat {summary['visceral_fat_lbs']} lbs\n\n")

    result_parts.append(
        "For full detailed reports with charts and complete regional breakdowns, "
        "the patient can visit their dashboard at bodyspec.com and log in with their account."
    )

    return "".join(result_parts)
