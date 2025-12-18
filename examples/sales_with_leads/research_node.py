"""
ResearchNode - Research company information using Google Search via Gemini Live API.
"""

from datetime import datetime, timezone
import json
import re
from typing import AsyncGenerator

from config import RESEARCH_PROMPT, LeadsAnalysis, ResearchAnalysis
from google.genai import Client, types
from google.genai.types import LiveConnectConfig, LiveServerMessage
from loguru import logger

from line.nodes.conversation_context import ConversationContext
from line.nodes.reasoning import ReasoningNode


class ResearchNode(ReasoningNode):
    """
    Node that researches company information using Google Search via Gemini Live API.

    Triggers on LeadsAnalysis events and uses Google Search to find relevant
    company information, recent news, and business insights to help the sales agent.
    """

    def __init__(
        self,
        gemini_client,
        model_id: str = "gemini-2.5-flash",
        max_context_length: int = 50,
    ):
        super().__init__(system_prompt=RESEARCH_PROMPT, max_context_length=max_context_length)

        self.gemini_client = gemini_client
        self.model_id = model_id

        # Create Gemini Live API client for Google Search
        self.live_client = Client(http_options={"api_version": "v1alpha"})

        # Configure Live API with Google Search tool
        self.live_config = LiveConnectConfig(
            system_instruction=self.system_prompt,
            tools=[{"google_search": {}}],
            response_modalities=["TEXT"],
        )

        # Cache to prevent duplicate research
        self.previous_leads: dict[str, dict] = {}

        logger.info(f"🔬 ResearchNode initialized with model: {model_id}, node_id: {self.id}")

    def _should_research(self, new_leads: dict) -> bool:
        """
        Check if research is needed based on leads changes.

        Args:
            new_leads: New leads information from LeadsAnalysis

        Returns:
            bool: True if research should be performed
        """
        company = new_leads.get("company", "").strip().lower()
        if not company:
            logger.info("🔬 No company name found, skipping research")
            return False

        if company not in self.previous_leads:
            logger.info(f"🔬 New company '{company}', research needed")
            return True

        # Compare key fields to detect meaningful changes
        prev = self.previous_leads[company]
        key_fields = ["company", "name", "interest_level"]

        has_changes = any(new_leads.get(field) != prev.get(field) for field in key_fields)

        if has_changes:
            logger.info(f"🔬 Company '{company}' has changes, research needed")
        else:
            logger.info(f"🔬 Company '{company}' unchanged, skipping research")

        return has_changes

    def _parse_search_queries(self, msg: LiveServerMessage) -> set[str]:
        """Parse search queries from message content."""
        queries = set()

        # Parse queries from grounding metadata
        if (
            msg.server_content
            and msg.server_content.grounding_metadata
            and msg.server_content.grounding_metadata.web_search_queries
        ):
            queries.update(msg.server_content.grounding_metadata.web_search_queries)

        # Parse queries from executable_code parts
        if msg.server_content and msg.server_content.model_turn:
            for part in msg.server_content.model_turn.parts:
                if hasattr(part, "executable_code") and part.executable_code:
                    code = part.executable_code.code
                    # Extract queries from google_search.search(queries=[...]) pattern
                    pattern = r"google_search\.search\(queries=\[(.*?)\]"
                    match = re.search(pattern, code)
                    if match:
                        queries_str = match.group(1)
                        # Extract individual quoted strings
                        query_pattern = r'"([^"]*)"'
                        extracted_queries = re.findall(query_pattern, queries_str)
                        queries.update(extracted_queries)

        return queries

    def _parse_search_pages(self, msg: LiveServerMessage) -> set[str]:
        """Parse search page URLs from message grounding metadata."""
        pages = set()

        # Parse page URLs from grounding metadata
        if (
            msg.server_content
            and msg.server_content.grounding_metadata
            and msg.server_content.grounding_metadata.grounding_chunks
        ):
            for chunk in msg.server_content.grounding_metadata.grounding_chunks:
                if chunk.web and chunk.web.uri:
                    pages.add(chunk.web.uri)

        return pages

    def _extract_json_from_research(self, research_text: str) -> dict:
        """
        Extract structured JSON from research response.

        Args:
            research_text: Raw research response from Gemini

        Returns:
            dict: Parsed company information or fallback structure
        """
        try:
            # Look for JSON pattern at end of response
            json_pattern = r'\{[^}]*"company_overview"[^}]*\}(?:\s*\})*'
            matches = re.findall(json_pattern, research_text, re.DOTALL)

            if matches:
                # Get the last/longest match
                json_str = max(matches, key=len)
                company_info = json.loads(json_str)
                logger.debug("🔬 Successfully parsed structured research JSON")
                return company_info
            else:
                logger.warning("🔬 No structured JSON found in research response")

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"🔬 Failed to parse research JSON: {e}")

        # Fallback: create basic structure from research text
        return {
            "company_overview": "Research completed but structured data unavailable",
            "pain_points": [],
            "key_people": [],
            "sales_opportunities": [],
            "raw_research": research_text[:500],  # First 500 chars
        }

    async def _perform_research(self, leads_info: dict) -> tuple[dict, str]:
        """
        Perform Google Search research using Gemini Live API.

        Args:
            leads_info: Leads information to research

        Returns:
            tuple: (company_info dict, research_summary string)
        """
        company = leads_info.get("company", "Unknown Company")
        person_name = leads_info.get("name", "")

        # Build research query based on available information
        search_prompt = f"""Research the company "{company}" to help our sales agent."""

        if person_name:
            search_prompt += f" Contact person: {person_name}."

        search_prompt += """

        Find information about:
        1. Company size, industry, and business model
        2. Potential business challenges or pain points
        3. Key executives and leadership

        Focus on official sources and recent information. End with a brief structured JSON summary.
        """

        research_summary = ""
        search_queries = set()
        search_pages = set()

        try:
            async with self.live_client.aio.live.connect(
                model=self.model_id, config=self.live_config
            ) as stream:
                # Send research prompt
                search_content = types.Content(role="user", parts=[types.Part(text=search_prompt)])

                await stream.send_client_content(turns=[search_content], turn_complete=True)

                # Collect research response
                async for msg in stream.receive():
                    if msg.text:
                        research_summary += msg.text

                    # Extract search metadata
                    search_queries.update(self._parse_search_queries(msg))
                    search_pages.update(self._parse_search_pages(msg))

        except Exception as e:
            logger.error(f"🔬 Error during Google Search research: {e}")
            research_summary = f"Research failed: {str(e)}"

        # Log search information
        if search_queries:
            logger.debug(f"🔍 Research queries used: {list(search_queries)}")
        if search_pages:
            logger.debug(f"📄 Research pages found: {len(search_pages)} sources")

        # Extract structured information
        company_info = self._extract_json_from_research(research_summary)

        return company_info, research_summary

    async def process_context(self, context: ConversationContext) -> AsyncGenerator[ResearchAnalysis, None]:
        """
        Process LeadsAnalysis events and perform company research.

        Args:
            context: Conversation context containing LeadsAnalysis events

        Yields:
            ResearchAnalysis: Research results for the company
        """
        logger.info("🔬 Starting research analysis")

        if not context.events:
            logger.info("No conversation events for research analysis")
            return

        # Find the most recent LeadsAnalysis event
        latest_leads = None
        for event in reversed(context.events):
            if isinstance(event, LeadsAnalysis):
                latest_leads = event
                break

        if not latest_leads:
            logger.info("No LeadsAnalysis event found for research")
            return

        leads_info = latest_leads.leads_info

        # Check if research is needed (avoid duplicates)
        if not self._should_research(leads_info):
            return

        try:
            logger.info(f"🔬 Researching company: {leads_info.get('company', 'Unknown')}")

            # Perform Google Search research
            company_info, research_summary = await self._perform_research(leads_info)

            # Update cache
            company_name = leads_info.get("company", "").strip().lower()
            if company_name:
                self.previous_leads[company_name] = leads_info.copy()

            # Yield research results
            yield ResearchAnalysis(
                company_info=company_info,
                research_summary=research_summary,
                confidence="high",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            logger.info(f"🔬 Research completed for {leads_info.get('company', 'Unknown')}")

        except Exception as e:
            logger.exception(f"🔬 Error during research processing: {e}")

            # Yield a low-confidence result with error info
            yield ResearchAnalysis(
                company_info={
                    "company_overview": f"Research failed: {str(e)}",
                    "pain_points": [],
                    "key_people": [],
                    "sales_opportunities": [],
                },
                research_summary=f"Research error: {str(e)}",
                confidence="low",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
