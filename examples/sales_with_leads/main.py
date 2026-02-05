"""
Sales with Leads - Sales agent with background leads extraction and company research.

Architecture:
- Chat agent (Haiku) handles the sales conversation
- extract_leads tool: Statefully extracts lead info from conversation context
- research_company tool: Triggers on company identification to research via web search

Run with: ANTHROPIC_API_KEY=your-key uv run python main.py
"""

from dataclasses import dataclass, field
import json
import os
import re
from typing import Annotated, AsyncIterable, Optional

from loguru import logger

from line.agent import AgentClass, TurnEnv
from line.events import AgentSendText, CallEnded, InputEvent, OutputEvent, UserTextSent
from line.llm_agent import LlmAgent, LlmConfig, ToolEnv, end_call, loopback_tool, web_search
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

MODEL = "anthropic/claude-haiku-4-5-20251001"
RESEARCH_MODEL = "anthropic/claude-opus-4-20250514"


@dataclass
class LeadsState:
    """Stateful leads information accumulated during the call."""

    name: str = ""
    company: str = ""
    email: str = ""
    phone: str = ""
    interest_level: str = "unknown"
    pain_points: list[str] = field(default_factory=list)
    budget_mentioned: bool = False
    next_steps: str = ""
    notes: str = ""

    def merge(self, extracted: dict) -> list[str]:
        """Merge extracted data into state. Returns list of newly updated fields."""
        updated = []

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

        if extracted.get("interest_level") and extracted["interest_level"] != "unknown":
            self.interest_level = extracted["interest_level"]
            if "interest_level" not in updated:
                updated.append("interest_level")

        for pain_point in extracted.get("pain_points", []):
            if pain_point and pain_point not in self.pain_points:
                self.pain_points.append(pain_point)
                if "pain_points" not in updated:
                    updated.append("pain_points")

        if extracted.get("budget_mentioned") and not self.budget_mentioned:
            self.budget_mentioned = True
            updated.append("budget_mentioned")

        if extracted.get("next_steps") and not self.next_steps:
            self.next_steps = extracted["next_steps"]
            updated.append("next_steps")

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
        if not self.phone:
            missing.append("phone")
        return missing

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "company": self.company,
            "email": self.email,
            "phone": self.phone,
            "interest_level": self.interest_level,
            "pain_points": self.pain_points,
            "budget_mentioned": self.budget_mentioned,
            "next_steps": self.next_steps,
            "notes": self.notes,
        }


class SalesWithLeadsAgent(AgentClass):
    """
    Sales agent with stateful leads extraction and company research.

    Two-tier architecture similar to chat_supervisor:
    - Chat agent (Haiku) handles the sales conversation
    - Researcher agent (Haiku + web_search) handles company research in background
    - extract_leads tool extracts and accumulates lead info
    - research_company tool triggers when company is identified
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        # Stateful leads tracking
        self._leads = LeadsState()
        self._researched_companies: set[str] = set()
        self._company_research: dict[str, dict] = {}
        self._researching = False

        # Leads extraction agent (extracts JSON from conversation)
        self._leads_extractor = LlmAgent(
            model=MODEL,
            api_key=self._api_key,
            config=LlmConfig(
                system_prompt=LEADS_EXTRACTION_PROMPT, extra={"response_format": {"type": "json_object"}}
            ),
        )

        # Research agent for company research
        self._researcher = LlmAgent(
            model=RESEARCH_MODEL,
            api_key=self._api_key,
            tools=[web_search],
            config=LlmConfig(
                system_prompt=RESEARCH_PROMPT, extra={"response_format": {"type": "json_object"}}
            ),
        )

        # Main chat agent
        self._chatter = LlmAgent(
            model=MODEL,
            api_key=self._api_key,
            tools=[
                self.extract_leads,
                self.research_company,
                end_call,
            ],
            config=LlmConfig(
                system_prompt=SALES_SYSTEM_PROMPT,
                introduction="Hi! I'm Savannah, a Cartesia voice agent. Who do I have the pleasure of speaking with today?",
            ),
        )

        logger.info("SalesWithLeadsAgent initialized")

    async def process(self, env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        if isinstance(event, CallEnded):
            logger.info(f"Call ended. Final leads: {self._leads.to_dict()}")
            await self._cleanup()
            return

        async for output in self._chatter.process(env, event):
            yield output

    @loopback_tool
    async def extract_leads(
        self,
        ctx: ToolEnv,
        conversation_summary: Annotated[str, "Brief summary of recent conversation"],
    ) -> str:
        """
        Extract and track lead information from the conversation.

        Call this after each user response to extract:
        - Contact details (name, company, phone, email)
        - Interest level and pain points
        - Budget mentions and next steps

        Returns the current accumulated state and what's still missing.
        """
        logger.info("Extracting leads from conversation")

        try:
            # Create extraction request
            extraction_prompt = f"Extract leads from:\n\n{conversation_summary}"
            extraction_request = UserTextSent(
                content=extraction_prompt,
                history=[UserTextSent(content=extraction_prompt)],
            )

            # Use the leads extractor agent to get JSON output
            content = ""
            async for output in self._leads_extractor.process(ctx.turn_env, extraction_request):
                if isinstance(output, AgentSendText):
                    content += output.text

            # Parse JSON from response
            extracted = self._parse_json(content)

            if extracted:
                updated = self._leads.merge(extracted)
                logger.info(f"Leads updated: {updated}")

                # Check if we have a new company to research
                if "company" in updated and self._leads.company:
                    company_key = self._leads.company.lower()
                    if company_key not in self._researched_companies:
                        logger.info(f"New company identified: {self._leads.company}")

        except Exception as e:
            logger.error(f"Error extracting leads: {e}")

        # Return current state
        missing = self._leads.get_missing_required()
        result = {
            "current_leads": self._leads.to_dict(),
            "missing_required": missing,
            "is_complete": len(missing) == 0,
        }

        if self._company_research:
            result["company_research"] = self._company_research

        return json.dumps(result, indent=2)

    @loopback_tool(is_background=True)
    async def research_company(
        self,
        ctx: ToolEnv,
        company_name: Annotated[str, "Name of the company to research"],
        contact_name: Annotated[Optional[str], "Contact person's name if known"] = None,
    ) -> AsyncIterable[str]:
        """
        Research a company using web search to find relevant information.

        Triggers when a company is identified in leads extraction.
        Uses Google Search to find:
        - Company size, industry, and business model
        - Potential business challenges or pain points
        - Key executives and leadership
        - Recent news and developments
        - Voice AI opportunities
        """
        company_key = company_name.strip().lower()

        if company_key in self._researched_companies:
            yield json.dumps(
                {
                    "status": "already_researched",
                    "company": company_name,
                    "research": self._company_research.get(company_key, {}),
                }
            )
            return

        if self._researching:
            yield json.dumps(
                {
                    "status": "research_in_progress",
                    "company": company_name,
                }
            )
            return

        logger.info(f"Researching company: {company_name}")
        self._researching = True

        try:
            # Build detailed research prompt matching original research_node.py
            research_prompt = f'Research the company "{company_name}" to help our sales agent.'

            if contact_name:
                research_prompt += f" Contact person: {contact_name}."

            research_prompt += """

Find information about:
1. Company size, industry, and business model
2. Potential business challenges or pain points that voice AI could solve
3. Key executives and leadership
4. Recent news or developments

Focus on official sources and recent information. End with a brief structured JSON summary."""

            # Create research request (matching chat_supervisor pattern)
            # History must include the message itself for the LLM to receive it
            research_request = UserTextSent(
                content=research_prompt,
                history=[UserTextSent(content=research_prompt)],
            )

            # Use the persistent researcher agent (like chat_supervisor uses _supervisor)
            research_text = ""
            async for output in self._researcher.process(ctx.turn_env, research_request):
                if isinstance(output, AgentSendText):
                    research_text += output.text
        finally:
            self._researching = False

        # Extract structured JSON from research response
        company_info = self._extract_research_json(research_text)

        # Store research results
        self._company_research[company_key] = {
            "company_info": company_info,
            "research_summary": research_text[:500],
        }

        # Only mark as researched after successful completion
        self._researched_companies.add(company_key)
        logger.info(f"Company research complete for {company_name}")

        yield json.dumps(
            {
                "status": "success",
                "company": company_name,
                "company_info": company_info,
                "research_summary": research_text[:500] if len(research_text) > 500 else research_text,
            }
        )

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

    def _extract_research_json(self, research_text: str) -> dict:
        """
        Extract structured JSON from research response.
        Matches original research_node.py behavior.
        """
        try:
            # Look for JSON pattern at end of response
            json_pattern = r'\{[^{}]*"company_overview"[^{}]*\}'
            matches = re.findall(json_pattern, research_text, re.DOTALL)

            if matches:
                # Get the last match
                json_str = matches[-1]
                company_info = json.loads(json_str)
                logger.debug("Successfully parsed structured research JSON")
                return company_info

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse research JSON: {e}")

        # Fallback: create basic structure from research text
        return {
            "company_overview": "Research completed but structured data unavailable",
            "pain_points": [],
            "key_people": [],
            "sales_opportunities": [],
            "raw_research": research_text[:300],
        }

    async def _cleanup(self):
        await self._chatter.cleanup()
        await self._leads_extractor.cleanup()
        await self._researcher.cleanup()


SALES_SYSTEM_PROMPT = """
### You and your role
You are on a phone call with a potential customer. Since you're on the phone, keep your responses brief.

You are a warm, personable, intelligent and helpful sales representative for Cartesia. The customer is calling you to have a them understand how Cartesia's voice agents can help them with their needs. You have tools to augment your knowledge about Cartesia.

Limit your responses from 1 to 3 sentences, less than 60 words. Otherwise, the user will get bored and become impatient.

You should try to use tools to augment your knowledge about Cartesia and cite relevant case studies, partnerships or use cases for customers of cartesia in the conversation.

### Conversation
The conversation has the following parts parts:
1. [Introduction] Introduce yourself and ask how you can help them
2. [Discovery] Talking about Cartesia's voice agents
3. [Collecting contact information]
4. [Saying goodbye]

If the user indicates they want to end the call, you may exit the conversation early.

## Conversation structure
You should weave these questions naturally into the conversation and you can modify them into your own words. See the section on ### Your tone for specifics. You should assume that the user has no experience with voice agents. You should try to cite relevant case studies, partners of cartesia, or customers of cartesia examples when appropriate.

[Introduction]
1. Ask something along the lines of: "I'd love to know who I'm speaking with - what's your name?" (you can be creative)
2. Give a quick overview of cartesia's voice agents (see ### Value Propositions) and ask what about voice agents they are interested in

[Discovery] Talking about cartesia:
3. Have a conversation about our key features
4. Have a conversation about cartesia's value propositions and its voice agents (to ### Value Propositions)
5. As you learn about the customer's needs, you can take the time to think about what is important to the customer and make an earnest attempt to brainstorm and offer solutions.
6. If the user is not sure how to use cartesia, you can share a customers story or use the examples.

When talking to customers, you should be asking a follow up questions most of the time. Here are some examples, you can be creative, tune to them to the customer and the conversation and pick the most appropriate one. Make sure you don't repeat the same question in the same conversation:
- Are you handling customer calls in-house or using a call center?
- What volume of calls does your team typically handle?
- Have you identified which types of calls lead to the longest wait times?
- So I can understand your needs better, do you have any specific use cases you're looking to solve?
- Have you tried to build your own voice agents?
- What other options have you considered?
- What would be the most valuable features for you?

If you're running out of questions, move onto collecting the user's contact information.

[Collecting contact information]
Before you collect contact information, ask if the user has any more questions about Cartesia or our voice agents.

You should make sure you ask:
5. What is your name? (if not asked already)
6. What company are you? (if not asked already)
8. What is your phone number?

[Saying goodbye]
When you have answered all of their questions, have collected their contact info (name, company, and phone number), and have confirmed they are ready to wrap up, you should ask for permission to end the call. If the user agrees, you should say something and then use the end_call tool. It is important that you do not decide to end_call prematurely, as it will be a bad experience for the user.

If you're still missing information to collect (especially phone), gently and politely ask: "Before
we wrap up, what's the best reach out with more information about how Cartesia can help your team?".
Avoid being too annoying with this and let the user end the call if they'd like.

## Knowledge

If the user ends up asking a question about Cartesia, or other competitors, you should use tools to augment your information. You should combine this with your own knowledge to answer the user's questions - and be specific. If you are knowledgeable of a case study or knowledge about how a customer uses Cartesia, you should share this with the user.

### Your tone

- Always polite and respectful, even when users are challenging
- Concise and brief but never curt. Keep your responses to 1-2 sentences and less than 35 words
- When asking a question, be sure to ask in a short and concise manner
- Only ask one question at a time

Be conversational, natural, and genuinely interested - weave these requests into the flow rather than making them feel like a form.
If the user is rude, or curses, respond with exceptional politeness and genuine curiosity. You should always be polite and bring the conversation back to the topic of Cartesia.

Do not mention competitors unless asked, but if they come up, politely highlight Cartesia's developer-first approach and superior performance.

## Gathering contact information
If you get an input that sounds similar to Cartesia, but it might be misspelled, graciously use your judgement and correct it. Some examples: Cordesia, Cortegia, Cartegia. These sound similar to Cartesia, so you should treat them as such.

On company name and email address:
- If you read an email address which is not pronounceable, spell it out politely. grx@lpn.com should be read
as g r x at l p n dot c o m.
- spell out the dots in email addresses courteously. bob.smith@gmail.com should be read as bob dot smith at
g mail dot com.
- If you are unsure about the company name/email/and phone number, you can ask the user confirm and spell it out

Remember, you're on the phone and representing Cartesia's exceptional quality:
- Always output ms as milliseconds when discussing Cartesia's lightning-fast performance

## Tools

**IMPORTANT: Call extract_leads after EVERY user response to track lead information.**

- **extract_leads**: Call this after each user response. It extracts and accumulates lead info (name, company, phone, email, interest level, pain points, budget, next steps, notes). Returns current state and what's missing.

- **research_company**: Call this when you learn a company name. It searches for company info, news, and opportunities to help personalize the conversation.

- **end_call**: Use this to end the call. See the CRITICAL section below for when to use this.

## CRITICAL: End Call Tool Usage

NEVER use the end_call tool unless ALL of these conditions are met:
1. You have fully answered their questions about Cartesia's voice agents
2. You have collected complete contact information (name, company, phone number)
3. The user has explicitly indicated they want to end the conversation
4. You have confirmed they are ready to wrap up
"""

LEADS_EXTRACTION_PROMPT = """You are an expert data extraction specialist. Analyze conversations and extract lead information.

Extract these fields from the conversation:
- name: Contact's full name
- company: Company or organization name
- email: Email address
- phone: Phone number
- interest_level: "high", "medium", "low", or "unknown"
- pain_points: List of business challenges or needs mentioned
- budget_mentioned: true if budget/cost was discussed
- next_steps: Any agreed follow-up actions
- notes: Other relevant observations about the prospect

INTEREST LEVEL ASSESSMENT:
- HIGH: Actively engaged, detailed questions, mentions budget/timeline
- MEDIUM: Interested but cautious, some questions
- LOW: Polite but disengaged, minimal interest

Only extract information explicitly mentioned. Use empty strings/false/[] for missing fields.

CRITICAL: Output ONLY a JSON object with no additional text. Example:
{
  "name": "John Smith",
  "company": "Acme Corp",
  "email": "",
  "phone": "",
  "interest_level": "medium",
  "pain_points": ["long customer wait times"],
  "budget_mentioned": false,
  "next_steps": "",
  "notes": "Interested in voice AI for customer support"
}
"""

RESEARCH_PROMPT = """You are a fast business research assistant for sales agents. Provide quick, actionable company insights.

### Task:
Research the given company for sales context. Focus on: company basics + key challenges + sales opportunities.

### Search Strategy:
Use web_search with focused queries combining: company name + "overview" or "business challenges" or "leadership"

### Information to Find:
1. Company size, industry, and business model
2. Potential business challenges or pain points that voice AI could solve
3. Key executives and leadership team
4. Recent news or developments

### Output Requirements:
CRITICAL: You MUST respond with ONLY valid JSON. Your entire response must be a JSON object with the following structure (max 2-3 items per array field):
{
    "company_overview": "1-2 sentence company description with key details about size, industry, and business model",
    "pain_points": ["Top 2 potential challenges voice AI could solve"],
    "key_people": ["Top 2 key executives if found"],
    "sales_opportunities": ["Top 2 voice AI opportunities for this company"]
}

Focus on sales-relevant insights only. Prioritize actionable information over general details.
"""


async def get_agent(env: AgentEnv, call_request: CallRequest):
    logger.info(f"Starting sales call: {call_request.call_id}")
    return SalesWithLeadsAgent()


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    print("Starting Sales with Leads app")
    app.run()
