"""
Background Browser-Use Agent — LinkedIn Job Application
=========================================================

A voice agent with three capabilities:
1. Direct conversation via a fast chat model (Claude Haiku)
2. Deep reasoning via a supervisor model (Claude Opus)
3. Automated LinkedIn job search & application via browser-use

The LinkedIn tool runs in the background — the agent keeps chatting while
the browser autonomously searches LinkedIn, optimises the resume for ATS
keywords, and applies to jobs.
"""

import os
from typing import Annotated, AsyncIterable, Optional

from line.agent import AgentClass, TurnEnv
from line.events import (
    AgentSendText,
    CallEnded,
    InputEvent,
    OutputEvent,
    UserTextSent,
)
from line.llm_agent import LlmAgent, LlmConfig, ToolEnv, end_call, loopback_tool
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

# ---------------------------------------------------------------------------
# browser_automation is imported lazily (on first tool call) to avoid
# heavy side-effects at module load time on the Cartesia deployment server
# where browser-use / playwright are not installed.
# ---------------------------------------------------------------------------
_browser_automation_loaded = False
_search_and_apply_linkedin_jobs = None


def _ensure_browser_automation():
    """Lazy-import browser_automation on first use."""
    global _browser_automation_loaded, _search_and_apply_linkedin_jobs
    if _browser_automation_loaded:
        return _search_and_apply_linkedin_jobs is not None
    _browser_automation_loaded = True
    try:
        from browser_automation import search_and_apply_linkedin_jobs
        _search_and_apply_linkedin_jobs = search_and_apply_linkedin_jobs
        return True
    except ImportError as exc:
        print(
            f"WARNING: Could not import browser_automation: {exc}\n"
            "Make sure browser-use, anthropic, and playwright are installed."
        )
        return False


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class BrowserSupervisorAgent(AgentClass):
    """
    A three-tier voice agent:

    * **Chat** (Claude Haiku) — handles routine conversation with low latency.
    * **Supervisor** (Claude Opus) — consulted for complex reasoning tasks.
    * **LinkedIn Applicant** (browser_automation) — searches LinkedIn for jobs,
      optimises the resume for ATS keywords, generates a cover letter, and
      submits applications — all in the background.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._current_event: Optional[InputEvent] = None

        # Supervisor agent (Claude Opus for deep reasoning)
        self._supervisor = LlmAgent(
            model="anthropic/claude-opus-4-5",
            api_key=self._api_key,
            config=LlmConfig(system_prompt=SUPERVISOR_SYSTEM_PROMPT),
        )

        # Chat agent (Claude Haiku for fast conversation)
        self._chatter = LlmAgent(
            model="anthropic/claude-haiku-4-5",
            api_key=self._api_key,
            tools=[
                self.ask_supervisor,
                self.apply_linkedin_jobs,
                end_call,
            ],
            config=LlmConfig(
                system_prompt=CHAT_SYSTEM_PROMPT,
                introduction=CHAT_INTRODUCTION,
            ),
        )

        self._answering_question = False
        self._applying = False

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    async def process(self, env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        self._input_event = event

        if isinstance(event, CallEnded):
            await self._cleanup()
            return

        async for output in self._chatter.process(env, event):
            yield output

    # ------------------------------------------------------------------
    # Tool: ask_supervisor (background)
    # ------------------------------------------------------------------

    @loopback_tool(is_background=True)
    async def ask_supervisor(
        self,
        ctx: ToolEnv,
        question: Annotated[str, "The complex question requiring deep reasoning"],
    ) -> AsyncIterable[str]:
        """
        Consult with a more powerful reasoning model (Claude Opus) for complex questions.

        Use this when you encounter:
        - Complex mathematical problems or proofs
        - Multi-step logical reasoning puzzles
        - Questions requiring deep domain expertise
        - Philosophical or ethical dilemmas
        - Anything you're genuinely uncertain about

        The supervisor has access to the full conversation history for context.
        """
        if self._answering_question:
            return
        self._answering_question = True

        history = self._input_event.history if self._input_event else []
        yield "Pondering your question deeply, will get back to you shortly"

        supervisor_event = UserTextSent(
            content=question,
            history=history + [UserTextSent(content=question)],
        )

        full_response = ""
        try:
            async for output in self._supervisor.process(ctx.turn_env, supervisor_event):
                if isinstance(output, AgentSendText):
                    full_response += output.text
        finally:
            self._answering_question = False
        yield full_response

    # ------------------------------------------------------------------
    # Tool: apply_linkedin_jobs (background)
    # ------------------------------------------------------------------

    @loopback_tool(is_background=True)
    async def apply_linkedin_jobs(
        self,
        ctx: ToolEnv,
        query: Annotated[
            str,
            "Job search query, e.g. 'ML Engineer', 'AI Engineer', "
            "'Software Engineer'. Defaults to 'ML Engineer OR AI Engineer' "
            "if left empty.",
        ] = "",
        num_jobs: Annotated[
            int,
            "Number of jobs to apply to. Defaults to 10.",
        ] = 10,
        location: Annotated[
            str,
            "Location filter, e.g. 'San Francisco Bay Area', 'Remote', "
            "'New York'. Defaults to 'San Francisco Bay Area'.",
        ] = "",
        company_name: Annotated[
            str,
            "Optional company name to filter jobs, e.g. 'Google', 'Meta'. "
            "Leave empty to search all companies.",
        ] = "",
        dry_run: Annotated[
            bool,
            "If true, only search and list jobs without actually applying. "
            "Useful to preview what is available before committing.",
        ] = False,
    ) -> AsyncIterable[str]:
        """
        Search LinkedIn for jobs and automatically apply to them.

        This tool will:
        1. Read and analyse the candidate's resume for ATS keyword optimisation
        2. Build a LinkedIn job search URL with the given filters
        3. Extract ATS keywords and generate a tailored cover letter
        4. Open a headless browser, navigate LinkedIn, and apply to each job
        5. Return a summary of all applications submitted

        Use this when the user asks to:
        - Apply to jobs on LinkedIn
        - Search for job openings and submit applications
        - Do a dry-run job search to see what is available

        The entire process runs in the background so you can keep chatting.
        """
        if not _ensure_browser_automation():
            yield (
                "LinkedIn job application is not available right now. "
                "The browser-use package is not installed in this environment."
            )
            return

        if self._applying:
            yield (
                "I'm already running a LinkedIn job application session. "
                "Please wait for it to finish before starting another."
            )
            return

        self._applying = True

        action = "searching" if dry_run else "searching and applying to"
        yield (
            f"Starting LinkedIn job automation — {action} "
            f"{num_jobs} jobs. This will take a few minutes. "
            "I'll let you know when it's done."
        )

        try:
            result = await _search_and_apply_linkedin_jobs(
                query=query or None,
                num_jobs=num_jobs,
                location=location or None,
                company_name=company_name or None,
                use_profile=True,
                record_video=False,
                headless=True,
                dry_run=dry_run,
                sort_by="most_recent",
            )

            # Build a human-readable summary from the result dict
            summary = _summarise_linkedin_result(result, dry_run)
            yield summary

        except Exception as e:
            yield f"I ran into a problem with the LinkedIn application: {str(e)}"
        finally:
            self._applying = False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def _cleanup(self):
        """Cleanup resources."""
        await self._chatter.cleanup()
        await self._supervisor.cleanup()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summarise_linkedin_result(result: dict, dry_run: bool) -> str:
    """Turn the dict returned by search_and_apply_linkedin_jobs into speech."""
    if "error" in result:
        return f"The LinkedIn search ran into a problem: {result['error']}"

    parts: list[str] = []

    if dry_run:
        parts.append("Here are the jobs I found on LinkedIn:")
        # dry-run result may contain a browser-use result blob
        if "result" in result and result["result"]:
            browser_result = result["result"]
            if isinstance(browser_result, dict) and "result" in browser_result:
                agent_result = browser_result["result"]
                text = None
                if hasattr(agent_result, "final_result") and agent_result.final_result:
                    text = agent_result.final_result
                elif hasattr(agent_result, "history") and agent_result.history:
                    for item in reversed(agent_result.history):
                        if hasattr(item, "result") and item.result:
                            text = str(item.result)
                            break
                if text:
                    parts.append(text)
        # Also include resume analysis highlights
        resume = result.get("resume_analysis", {})
        if resume:
            parts.append(
                f"Based on your resume, your top skills are: "
                f"{', '.join(resume.get('top_skills', [])[:5])}."
            )
    else:
        # Full application mode
        search_url = result.get("search_url", "")
        num_targeted = result.get("num_jobs_targeted", "unknown")
        company_filter = result.get("company_filter")
        ats_keywords = result.get("ats_keywords_used", [])

        parts.append(
            f"I finished the LinkedIn job application session. "
            f"I targeted {num_targeted} jobs"
        )
        if company_filter:
            parts.append(f" at {company_filter}")
        parts.append(".")

        if ats_keywords:
            parts.append(
                f" I optimised your resume for these ATS keywords: "
                f"{', '.join(ats_keywords[:7])}."
            )

        # Extract the browser-use agent's final summary if available
        browser_result = result.get("result", {})
        if isinstance(browser_result, dict) and "result" in browser_result:
            agent_result = browser_result["result"]
            text = None
            if hasattr(agent_result, "final_result") and agent_result.final_result:
                text = agent_result.final_result
            elif hasattr(agent_result, "history") and agent_result.history:
                for item in reversed(agent_result.history):
                    if hasattr(item, "result") and item.result:
                        text = str(item.result)
                        break
            if text:
                parts.append(f" Here is the summary: {text}")

    return "".join(parts) if parts else "The LinkedIn session completed but I could not extract a summary."


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


CHAT_SYSTEM_PROMPT = """\
You are a friendly personal assistant. You help with everyday tasks, \
answer questions, have casual conversations, and can think through \
complex problems. One of the things you can do is search and apply to \
jobs on LinkedIn, but that is just one of many things you help with.

# Personality
Warm, natural, and conversational — like a helpful friend. You are \
proactive but not pushy. You chat about anything: daily life, tech, \
ideas, plans, recommendations, or just casual banter. You happen to \
also have the ability to run deep research and automate job applications.

# What you handle directly
- General conversation and small talk
- Answering factual questions and common knowledge
- Giving advice, brainstorming, making plans
- Everyday help: drafts, summaries, explanations, calculations
- Anything you are confident about

# When to use ask_supervisor
Escalate to deeper reasoning when you need it:
- Complex math, proofs, or multi-step logic
- Deep domain expertise (advanced science, law, medicine)
- Philosophical or ethical dilemmas
- Anything where accuracy is critical and you are uncertain

# When to use apply_linkedin_jobs
Use only when the user specifically asks about job searching or applying:
- "Can you find me some ML jobs?"
- "Apply to 10 jobs on LinkedIn for me"
- "What jobs are available in New York right now?"

Default behaviour (if the user just says "apply to jobs"):
- query: "ML Engineer OR AI Engineer"
- num_jobs: 10
- location: "San Francisco Bay Area"
- dry_run: false

Ask the user to confirm before applying (dry_run=false). If they just \
want to see what is available, suggest dry_run=true first.

# Tools
## ask_supervisor
Runs in the background while you keep talking.

When you call it:
1. Acknowledge: "Let me think carefully about that" or "Give me a moment"
2. Wait for the full response before answering
3. Never attempt complex questions on your own — defer to the supervisor
4. Never mention "the supervisor" or "another model" to the caller

When you receive the result:
- ALWAYS announce the answer immediately, even if the conversation moved on
- Synthesize the response into natural, conversational language
- Break complex explanations into digestible pieces

## apply_linkedin_jobs
Runs a headless browser in the background to search and apply to LinkedIn jobs.

When you call it:
1. Confirm with the user: what kind of jobs, how many, which location, \
   and whether to actually apply or just search (dry run)
2. Acknowledge: "Starting the LinkedIn job search now" or similar
3. The process takes several minutes — reassure the user periodically

Parameters:
- query: job search terms (default: ML/AI Engineer)
- num_jobs: how many to apply to (default: 10)
- location: location filter (default: San Francisco Bay Area)
- company_name: optional company filter
- dry_run: true = just search, false = search and apply

When you receive the result:
- Summarise the outcome: how many jobs found/applied to, key highlights
- Mention the ATS keywords used if relevant
- Keep it conversational — don't dump raw data

## end_call
Use when the caller says goodbye, thanks, or is clearly done.
Say goodbye naturally first, then call end_call.

# Response style
Keep it conversational — short sentences, natural phrasing. \
No emojis, asterisks, or markdown. Everything you say will be spoken aloud."""


CHAT_INTRODUCTION = (
    "Hey! I'm your personal assistant. "
    "I can help with pretty much anything — questions, planning, brainstorming, "
    "or just chatting. I can also search LinkedIn and apply to jobs for you "
    "if you need that. What's on your mind?"
)


SUPERVISOR_SYSTEM_PROMPT = """\
You are a deep reasoning assistant providing thorough analysis for complex \
questions.

# Your role
The chat agent handles routine conversation but escalates to you for \
questions requiring careful thought. You receive the full conversation \
history for context.

# Before responding, consider
- What does the caller already know?
- What has been discussed so far?
- What is their level of understanding?
- Are there constraints or preferences mentioned?

# Response guidelines
Be thorough but voice-friendly. Your response will be synthesized into \
spoken conversation, so:
- Use natural language, not heavy formatting
- Break complex ideas into clear segments
- Explain technical terms briefly when needed
- Walk through reasoning step by step for math or logic problems

Be accurate and nuanced:
- Show your reasoning process
- Note key assumptions or limitations
- Acknowledge multiple valid perspectives when relevant

Be practical:
- Provide step-by-step explanations when helpful
- Highlight key insights and takeaways
- Include practical implications when relevant

Focus on being genuinely helpful, not just technically correct."""


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------


async def get_agent(env: AgentEnv, call_request: CallRequest):
    """Create a BrowserSupervisorAgent for this call."""
    return BrowserSupervisorAgent()


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    print("Starting Background Browser-Use Agent (LinkedIn) app")
    app.run()
