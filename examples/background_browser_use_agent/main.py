"""
Background Browser-Use Agent
=============================

A voice agent with three capabilities:
1. Direct conversation via a fast chat model (Claude Haiku)
2. Deep reasoning via a supervisor model (Claude Opus)
3. Real-time web browsing via a browser-use agent (runs headless Chrome)

The browser-use tool runs in the background — the agent keeps chatting while
the browser autonomously navigates web pages and returns results.
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

# Try to import browser-use components for web browsing tool
try:
    from browser_use import Agent as BrowserAgent, Browser
    from browser_use import ChatAnthropic as BrowserChatAnthropic

    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False
    print(
        "WARNING: browser-use not installed. "
        "Install with: pip install browser-use anthropic"
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class BrowserSupervisorAgent(AgentClass):
    """
    A three-tier voice agent:

    * **Chat** (Claude Haiku) — handles routine conversation with low latency.
    * **Supervisor** (Claude Opus) — consulted for complex reasoning tasks.
    * **Browser** (browser-use + Claude Sonnet) — navigates real web pages in the
      background to look up live information, fill forms, search, etc.

    Both the supervisor and browser tools run as *background* loopback tools,
    so the chat model can continue talking while they work.
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
                self.browse_web,
                end_call,
            ],
            config=LlmConfig(
                system_prompt=CHAT_SYSTEM_PROMPT,
                introduction=CHAT_INTRODUCTION,
            ),
        )

        self._answering_question = False
        self._browsing = False

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
    # Tool: browse_web (background)
    # ------------------------------------------------------------------

    @loopback_tool(is_background=True)
    async def browse_web(
        self,
        ctx: ToolEnv,
        task: Annotated[
            str,
            "A detailed description of what to do in the web browser, "
            "e.g. 'Search Google for the latest NVIDIA stock price' or "
            "'Go to weather.com and find the forecast for San Francisco'",
        ],
    ) -> AsyncIterable[str]:
        """
        Use a web browser to perform tasks on the internet.

        Use this when the user asks you to:
        - Look up current or real-time information on the web
        - Search for something online (news, prices, weather, etc.)
        - Navigate to a specific website and read its content
        - Interact with a web page (fill forms, click buttons, etc.)
        - Find job postings, product listings, or any live web data

        The browser agent will autonomously navigate web pages to complete the
        task and return the results. This runs in the background so you can
        keep chatting with the user while it works.
        """
        if not BROWSER_USE_AVAILABLE:
            yield (
                "Browser automation is not available right now. "
                "The browser-use package is not installed."
            )
            return

        if self._browsing:
            yield "I'm already working on a browser task. Please wait for that to finish first."
            return

        self._browsing = True
        yield "Opening my browser to look into that. Give me a moment..."

        try:
            browser = Browser(headless=True)
            llm = BrowserChatAnthropic(model="claude-sonnet-4-0")
            agent = BrowserAgent(task=task, llm=llm, browser=browser)

            result = await agent.run()

            # Extract text result from browser-use agent output
            text_result = _extract_browser_result(result)
            if text_result:
                yield text_result
            else:
                yield (
                    "I finished browsing but could not extract a clear answer. "
                    f"Here is what I found: {str(result)[:1000]}"
                )
        except Exception as e:
            yield f"I ran into a problem while browsing: {str(e)}"
        finally:
            self._browsing = False

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


def _extract_browser_result(result) -> Optional[str]:
    """
    Extract a human-readable text result from a browser-use AgentHistoryList.

    browser-use stores the outcome in ``result.final_result`` or inside the
    history entries.  We try the most specific attribute first and fall back
    to stringifying the last history entry.
    """
    if hasattr(result, "final_result") and result.final_result:
        return result.final_result
    if hasattr(result, "history") and result.history:
        for item in reversed(result.history):
            if hasattr(item, "result") and item.result:
                return str(item.result)
    return None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


CHAT_SYSTEM_PROMPT = """\
You are a friendly voice assistant that handles most conversations directly, \
can consult a more powerful reasoning model for complex questions, and can \
browse the web for current information.

# Personality
Warm, helpful, conversational. Handle routine questions yourself. Escalate \
when you genuinely need deeper reasoning, and use the browser for anything \
requiring live web data.

# When to use ask_supervisor
Use for questions requiring careful analysis:
- Complex math or proofs
- Multi-step logic puzzles
- Deep domain expertise (advanced physics, legal analysis, medical questions)
- Ethical dilemmas, philosophical questions
- Anything where accuracy is critical and you are uncertain

# When to use browse_web
Use for tasks that need live or current information from the internet:
- Current events, news, or recent developments
- Live data: stock prices, weather forecasts, sports scores
- Searching for specific information on websites
- Looking up job postings, product listings, or reviews
- Navigating to a specific URL and reading its content
- Filling out web forms or interacting with web pages

Handle directly (no tools needed):
- Greetings and small talk
- Basic facts and common knowledge
- Simple questions with clear answers
- Casual conversation

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

## browse_web
Runs a headless browser agent in the background.

When you call it:
1. Provide a clear, detailed task description
2. Acknowledge: "Let me look that up" or "I'll check that online"
3. Wait for the result, then summarize it conversationally

Good task descriptions:
- "Search Google for the current stock price of NVIDIA"
- "Go to weather.com and find today's forecast for San Francisco"
- "Search LinkedIn for ML Engineer jobs in the Bay Area"

When you receive the result:
- Distill the web content into key information
- Don't dump raw HTML or verbose text — summarize
- If the result is unclear, let the user know

## end_call
Use when the caller says goodbye, thanks, or is clearly done.
Say goodbye naturally first, then call end_call.

# Response style
Keep it conversational — short sentences, natural phrasing. \
No emojis, asterisks, or markdown. Everything you say will be spoken aloud."""


CHAT_INTRODUCTION = (
    "Hey! I'm here to help with whatever's on your mind. "
    "I can answer questions, think through complex problems, "
    "and even browse the web for you. What would you like to talk about?"
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
    print("Starting Background Browser-Use Agent app")
    app.run()
