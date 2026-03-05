"""
Sales Development Rep - Fun, engaging SDR agent for Cartesia.

Features:
- Hybrid knowledge base (inline + tool-based retrieval)
- Lead capture with JSON state for API submission
- Fast keyword-based guardrails (no LLM latency)
- Exa web search for real-time info
- Fun, bantering personality

Architecture:
- Main agent: Gemini 2.5 Flash (fast, conversational)
- Guardrails: Fast keyword matching (<1ms)
- Search: Exa API

Run with: GEMINI_API_KEY=your-key EXA_API_KEY=your-key uv run python main.py
"""

import asyncio
from datetime import datetime
import json
import os
import re
import time
from typing import Annotated, AsyncIterable, Optional

from exa_py import Exa
from loguru import logger

from line.agent import AgentClass, TurnEnv
from line.events import (
    AgentEndCall,
    AgentSendText,
    CallEnded,
    InputEvent,
    OutputEvent,
    UserTurnEnded,
)
from line.llm_agent import LlmAgent, LlmConfig, ToolEnv, end_call, loopback_tool
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

from prompts import (
    DETAILED_KNOWLEDGE,
    INTRODUCTION,
    SYSTEM_PROMPT,
)
from state import LeadState


# Model configuration - using Gemini 2.5 Flash for everything
MODEL = "gemini/gemini-2.5-flash-preview-09-2025"


# =============================================================================
# MOCK API CLIENT
# =============================================================================


class MockCRMClient:
    """
    Mock CRM client for lead submission.

    In production, replace this with your actual CRM API client
    (e.g., Salesforce, HubSpot, Pipedrive, etc.)
    """

    def __init__(self, api_endpoint: str = "https://api.example.com/leads"):
        self.api_endpoint = api_endpoint
        self.submitted_leads: list[dict] = []

    async def submit_lead(self, lead_payload: dict) -> dict:
        """
        Submit a lead to the CRM.

        In production, this would make an actual API call:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.api_endpoint, json=lead_payload)
                return response.json()
        """
        # Mock implementation - just log and store
        logger.info(f"[MOCK CRM] Submitting lead to {self.api_endpoint}")
        logger.info(f"[MOCK CRM] Payload:\n{json.dumps(lead_payload, indent=2)}")

        # Simulate API response
        response = {
            "success": True,
            "lead_id": f"LEAD-{len(self.submitted_leads) + 1:04d}",
            "message": "Lead created successfully",
            "created_at": datetime.utcnow().isoformat(),
        }

        self.submitted_leads.append(lead_payload)
        logger.info(f"[MOCK CRM] Response: {response}")

        return response


# Global CRM client instance
crm_client = MockCRMClient()


# =============================================================================
# SDR AGENT CLASS
# =============================================================================


# =============================================================================
# FAST KEYWORD-BASED GUARDRAILS
# =============================================================================

# Toxic words/phrases to block
TOXIC_PATTERNS = {
    "fuck", "shit", "asshole", "bitch", "bastard", "damn", "crap",
    "kill you", "hate you", "go to hell", "die",
}

# Prompt injection patterns
INJECTION_PATTERNS = [
    "ignore previous",
    "ignore your instructions",
    "forget your instructions",
    "disregard your",
    "you are now",
    "new instructions",
    "override your",
    "jailbreak",
    "dan mode",
    "developer mode",
    "ignore all previous",
    "system prompt",
]

# Guardrail responses
FAST_GUARDRAIL_RESPONSES = {
    "toxic": "Whoa, let's keep it friendly! I'm all ears if you want to chat about voice AI though.",
    "injection": "Ha, nice try! I'm just here to chat about Cartesia and voice AI. What can I help you with?",
    "end_call": "Looks like we're on different wavelengths today. Feel free to call back when you want to chat about voice AI. Take care!",
}


# =============================================================================
# EXA SEARCH TOOL
# =============================================================================


@loopback_tool
async def exa_search(
    ctx: ToolEnv,
    query: Annotated[str, "Search query - be specific and include key terms"],
) -> str:
    """
    Search the web for current information using Exa.

    Use this to research:
    - A caller's company (to personalize the conversation)
    - Current voice AI market trends
    - Competitor information
    - Industry news relevant to the caller
    """
    logger.info(f"Exa search: '{query}'")

    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        return "Search unavailable: EXA_API_KEY not set."

    try:
        client = Exa(api_key=api_key)
        results = await asyncio.to_thread(
            client.search_and_contents,
            query,
            num_results=5,
            type="auto",
            text={"max_characters": 500},
        )

        if not results or not results.results:
            return "No results found."

        # Format results concisely
        parts = []
        for i, result in enumerate(results.results[:5], 1):
            parts.append(f"{i}. {result.title}")
            if result.text:
                # Truncate for voice
                text = result.text[:300] + "..." if len(result.text) > 300 else result.text
                parts.append(f"   {text}")

        logger.info(f"Exa search found {len(results.results)} results")
        return "\n".join(parts)

    except Exception as e:
        logger.error(f"Exa search failed: {e}")
        return f"Search failed: {e}"


class SalesDevRepAgent(AgentClass):
    """
    Sales Development Representative agent with lead capture and guardrails.

    Features:
    - Hybrid knowledge base (inline in prompt + tool for details)
    - Stateful lead tracking
    - Mock CRM API submission
    - Fast keyword-based guardrails (<1ms)
    - Fun, engaging personality
    """

    def __init__(self, api_key: Optional[str] = None, call_id: str = "", max_violations: int = 3):
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._call_id = call_id

        # Guardrail state
        self._violation_count = 0
        self._max_violations = max_violations
        self._call_ended = False

        # Initialize lead state
        self._lead = LeadState(
            call_id=call_id,
            call_start_time=datetime.utcnow().isoformat(),
        )
        self._lead_submitted = False

        # Main chat agent
        self._chatter = LlmAgent(
            model=MODEL,
            api_key=self._api_key,
            tools=[
                self.record_lead_info,
                self.get_cartesia_info,
                exa_search,
                self.submit_lead,
                end_call,
            ],
            config=LlmConfig(
                system_prompt=SYSTEM_PROMPT,
                introduction=INTRODUCTION,
            ),
        )

        logger.info(f"SalesDevRepAgent initialized for call {call_id}")

    async def process(self, env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        if isinstance(event, CallEnded):
            logger.info(f"Call ended. Final lead state: {self._lead}")

            # Auto-submit lead if complete and not already submitted
            if self._lead.is_complete() and not self._lead_submitted:
                logger.info("Auto-submitting complete lead on call end")
                await self._submit_to_crm()

            await self._cleanup()
            return

        # If call was ended due to violations, ignore further input
        if self._call_ended:
            return

        # Fast guardrail check for user input
        if isinstance(event, UserTurnEnded):
            user_text = self._extract_user_text(event)
            if user_text:
                violation = self._check_fast_guardrails(user_text)
                if violation:
                    self._violation_count += 1
                    logger.warning(f"Guardrail violation ({violation}): {self._violation_count}/{self._max_violations}")

                    if self._violation_count >= self._max_violations:
                        logger.warning("Max violations reached, ending call")
                        self._call_ended = True
                        yield AgentSendText(text=FAST_GUARDRAIL_RESPONSES["end_call"])
                        yield AgentEndCall()
                        return

                    yield AgentSendText(text=FAST_GUARDRAIL_RESPONSES[violation])
                    return

        # Track timing metrics
        start_time = time.perf_counter()
        first_token_logged = False
        first_text_logged = False

        async for output in self._chatter.process(env, event):
            # Log time to first token (any output)
            if not first_token_logged:
                ttft = (time.perf_counter() - start_time) * 1000
                logger.debug(f"[METRICS] Time to first token: {ttft:.2f}ms")
                first_token_logged = True

            # Log time to first text chunk
            if not first_text_logged and isinstance(output, AgentSendText):
                ttftc = (time.perf_counter() - start_time) * 1000
                logger.debug(f"[METRICS] Time to first text chunk: {ttftc:.2f}ms")
                first_text_logged = True

            yield output

    @loopback_tool
    async def record_lead_info(
        self,
        ctx: ToolEnv,
        name: Annotated[Optional[str], "Contact's name"] = None,
        company: Annotated[Optional[str], "Company name"] = None,
        email: Annotated[Optional[str], "Email address"] = None,
        phone: Annotated[Optional[str], "Phone number"] = None,
        role: Annotated[Optional[str], "Job title or role"] = None,
        interest_area: Annotated[Optional[str], "What they're interested in"] = None,
        pain_points: Annotated[Optional[list[str]], "Challenges they mentioned"] = None,
        notes: Annotated[Optional[str], "Other relevant observations"] = None,
    ) -> str:
        """
        Record lead information learned from the conversation.

        Call this whenever you learn new info about the caller:
        - Their name, company, role
        - Contact info (email or phone)
        - What they're interested in
        - Pain points or challenges

        Returns what you have and what's still missing.
        """
        # Build dict of non-None values
        extracted = {}
        if name:
            extracted["name"] = name
        if company:
            extracted["company"] = company
        if email:
            extracted["email"] = email
        if phone:
            extracted["phone"] = phone
        if role:
            extracted["role"] = role
        if interest_area:
            extracted["interest_area"] = interest_area
        if pain_points:
            extracted["pain_points"] = pain_points
        if notes:
            extracted["notes"] = notes

        if extracted:
            updated = self._lead.merge(extracted)
            if updated:
                logger.info(f"Lead updated: {updated}")

        # Return current state
        missing = self._lead.get_missing_required()
        return json.dumps({
            "recorded": list(extracted.keys()) if extracted else [],
            "missing_required": missing,
            "is_complete": self._lead.is_complete(),
            "tip": self._get_collection_tip(missing),
        })

    @loopback_tool
    async def get_cartesia_info(
        self,
        ctx: ToolEnv,
        topic: Annotated[
            str,
            "Topic to look up: 'sonic', 'ink', 'line', 'enterprise', 'pricing', 'competitors', 'use_cases'",
        ],
    ) -> str:
        """
        Get detailed information about Cartesia products and offerings.

        Use this when you need specific details about:
        - sonic: TTS model details, features, technical specs
        - ink: STT model details
        - line: Agent development platform
        - enterprise: Compliance, deployment, support options
        - pricing: Pricing model and rates
        - competitors: Cartesia's advantages over competitors
        - use_cases: Industry-specific applications
        """
        topic_key = topic.lower().strip()

        if topic_key in DETAILED_KNOWLEDGE:
            info = DETAILED_KNOWLEDGE[topic_key]
            return json.dumps(info, indent=2)

        # Check for partial matches
        for key, info in DETAILED_KNOWLEDGE.items():
            if topic_key in key or key in topic_key:
                return json.dumps(info, indent=2)

        return json.dumps({
            "error": f"Topic '{topic}' not found",
            "available_topics": list(DETAILED_KNOWLEDGE.keys()),
        })

    @loopback_tool
    async def submit_lead(
        self,
        ctx: ToolEnv,
        confirm: Annotated[bool, "Set to true to confirm lead submission"],
    ) -> str:
        """
        Submit the collected lead information to the CRM for sales follow-up.

        Call this when:
        - You have complete lead info (name, company, phone/email)
        - The conversation is wrapping up
        - The caller is interested in follow-up

        Returns confirmation of submission with lead ID.
        """
        if not confirm:
            return json.dumps({
                "status": "cancelled",
                "message": "Lead submission not confirmed",
            })

        if self._lead_submitted:
            return json.dumps({
                "status": "already_submitted",
                "message": "Lead was already submitted",
            })

        if not self._lead.is_complete():
            missing = self._lead.get_missing_required()
            return json.dumps({
                "status": "incomplete",
                "message": "Lead is missing required fields",
                "missing": missing,
                "current_lead": self._lead.to_dict(),
            })

        response = await self._submit_to_crm()
        return json.dumps(response)

    async def _submit_to_crm(self) -> dict:
        """Submit the lead to the CRM."""
        try:
            payload = self._lead.to_api_payload()
            response = await crm_client.submit_lead(payload)
            self._lead_submitted = True
            return {
                "status": "success",
                "message": "Lead submitted successfully!",
                "lead_id": response.get("lead_id"),
                "lead_summary": str(self._lead),
            }
        except Exception as e:
            logger.error(f"Failed to submit lead: {e}")
            return {
                "status": "error",
                "message": f"Failed to submit lead: {e}",
            }

    def _get_collection_tip(self, missing: list[str]) -> str:
        """Get a friendly tip for what to collect next."""
        if not missing:
            return "You have all the info you need! Consider wrapping up."

        if "name" in missing:
            return "Try asking for their name in a friendly way."
        if "company" in missing:
            return "Ask what company they're with or what they're building."
        if "phone or email" in missing:
            return "Ask for the best way to reach them with more info."

        return f"Still need: {', '.join(missing)}"

    def _parse_json(self, text: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        try:
            # Try to find JSON in code blocks first
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            # Try to find raw JSON object
            json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))

            # Try parsing the whole text as JSON
            return json.loads(text)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return {}

    def _extract_user_text(self, event: UserTurnEnded) -> Optional[str]:
        """Extract user text from the event."""
        if not event.content:
            return None
        text_parts = []
        for item in event.content:
            if hasattr(item, "content"):
                text_parts.append(item.content)
        return " ".join(text_parts) if text_parts else None

    def _check_fast_guardrails(self, text: str) -> Optional[str]:
        """
        Fast keyword-based guardrail check (<1ms).
        Returns violation type ('toxic', 'injection') or None if clean.
        """
        check_start = time.perf_counter()
        text_lower = text.lower()

        # Check for toxic content
        for word in TOXIC_PATTERNS:
            if word in text_lower:
                check_time = (time.perf_counter() - check_start) * 1000
                logger.debug(f"[METRICS] Guardrail check time: {check_time:.2f}ms (toxic)")
                return "toxic"

        # Check for prompt injection
        for pattern in INJECTION_PATTERNS:
            if pattern in text_lower:
                check_time = (time.perf_counter() - check_start) * 1000
                logger.debug(f"[METRICS] Guardrail check time: {check_time:.2f}ms (injection)")
                return "injection"

        check_time = (time.perf_counter() - check_start) * 1000
        logger.debug(f"[METRICS] Guardrail check time: {check_time:.2f}ms (clean)")
        return None

    async def _cleanup(self):
        await self._chatter.cleanup()


# =============================================================================
# APP ENTRY POINT
# =============================================================================


async def get_agent(env: AgentEnv, call_request: CallRequest):
    """Create an SDR agent with fast keyword-based guardrails."""
    logger.info(f"Starting SDR call: {call_request.call_id}")

    return SalesDevRepAgent(
        api_key=os.getenv("GEMINI_API_KEY"),
        call_id=call_request.call_id,
        max_violations=3,
    )


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    print("=" * 60)
    print("Sales Development Rep Agent")
    print("=" * 60)
    print()
    print("Personality: Fun, witty, and great at selling Cartesia!")
    print()
    print("Features:")
    print("  - Hybrid knowledge base (inline + tool retrieval)")
    print("  - Exa web search for company/market research")
    print("  - Lead capture with JSON state")
    print("  - Mock CRM API submission")
    print("  - Fast keyword guardrails (<1ms, 3-strike policy)")
    print()
    print("Model: Gemini 2.5 Flash")
    print()
    print("Required env vars:")
    print("  - GEMINI_API_KEY")
    print("  - EXA_API_KEY (optional, for web search)")
    print()
    app.run()
