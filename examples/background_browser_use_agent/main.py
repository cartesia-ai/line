"""
Personal Voice Assistant
========================

A voice agent with four capabilities:
1. Direct conversation via a fast chat model (Claude Haiku)
2. Deep reasoning via a supervisor model (Claude Opus)
3. Automated LinkedIn job search & application via browser-use
4. Music / song generation via ACE-Step model server

Tools 3 and 4 run in the background — the agent keeps chatting while
the long-running task completes.
"""

import os
import time
from pathlib import Path
from typing import Annotated, AsyncIterable, Optional

import httpx

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
# Music model server configuration
# ---------------------------------------------------------------------------
_MUSIC_SERVER_URL = os.getenv("MUSIC_MODEL_SERVER_URL", "http://localhost:8190")
_SONG_OUTPUT_DIR = Path(__file__).resolve().parent / "song-generated"
# ngrok free-tier serves an interstitial HTML page unless this header is set
_MUSIC_HEADERS = {"ngrok-skip-browser-warning": "true"}

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
    A four-tier voice agent:

    * **Chat** (Claude Haiku) — handles routine conversation with low latency.
    * **Supervisor** (Claude Opus) — consulted for complex reasoning tasks.
    * **LinkedIn Applicant** (browser_automation) — searches LinkedIn for jobs,
      optimises the resume for ATS keywords, generates a cover letter, and
      submits applications — all in the background.
    * **Song Generator** (ACE-Step model server) — generates music / singing
      audio from caption + lyrics via the ACE-Step 1.5 model server.
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
                self.generate_song,
                end_call,
            ],
            config=LlmConfig(
                system_prompt=CHAT_SYSTEM_PROMPT,
                introduction=CHAT_INTRODUCTION,
            ),
        )

        self._answering_question = False
        self._applying = False
        self._generating_song = False

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
    # Tool: generate_song (background)
    # ------------------------------------------------------------------

    @loopback_tool(is_background=True)
    async def generate_song(
        self,
        ctx: ToolEnv,
        caption: Annotated[
            str,
            "Text description of the desired music, e.g. "
            "'female vocalist singing a catchy pop melody with soft acoustic "
            "guitar accompaniment, warm and intimate, clear vocals, English'. "
            "Be descriptive about genre, mood, instruments, and language.",
        ],
        lyrics: Annotated[
            str,
            "Song lyrics with section markers. Use markers like [Verse], "
            "[Chorus], [Bridge], [Outro]. Use '[Instrumental]' for "
            "instrumental-only music. Example:\n"
            "[Verse]\\nHello hello\\nHello you stranger\\n\\n"
            "[Chorus]\\nHello hello\\nHello you stranger",
        ] = "",
        instrumental: Annotated[
            bool,
            "If true, generate instrumental music without vocals. "
            "Overrides lyrics to '[Instrumental]'.",
        ] = False,
        duration: Annotated[
            float,
            "Duration of the generated audio in seconds. "
            "Range: 5 to 600. Default is 30 seconds.",
        ] = 30.0,
        bpm: Annotated[
            int,
            "Beats per minute. Range 30-300. "
            "Leave as 0 for automatic BPM detection based on the caption.",
        ] = 0,
        seed: Annotated[
            int,
            "Random seed for reproducibility. Use -1 for a random seed.",
        ] = -1,
    ) -> AsyncIterable[str]:
        """
        Generate a song or music track using the ACE-Step AI music model.

        This tool will:
        1. Send the caption and lyrics to the ACE-Step 1.5 model server
        2. Generate high-quality FLAC audio
        3. Save the audio file to the song-generated directory
        4. Return a summary with the file path and generation details

        Use this when the user asks you to:
        - Sing a song or generate music
        - Create a melody, beat, or instrumental track
        - Compose something musical

        The generation runs in the background so you can keep chatting.
        """
        if self._generating_song:
            yield (
                "I'm already generating a song. "
                "Please wait for it to finish before requesting another."
            )
            return

        self._generating_song = True

        yield (
            f"Starting music generation — this usually takes 15 to 60 seconds "
            f"depending on the duration. I'll let you know as soon as it's ready."
        )

        try:
            # Check server health first
            async with httpx.AsyncClient(timeout=10.0, headers=_MUSIC_HEADERS) as client:
                try:
                    health = await client.get(f"{_MUSIC_SERVER_URL}/health")
                    health_data = health.json()
                    if health_data.get("status") != "ok":
                        yield (
                            "The music model server is still loading. "
                            "Please try again in a minute."
                        )
                        return
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    yield (
                        "I can't reach the music model server right now. "
                        "Make sure it's running on "
                        f"{_MUSIC_SERVER_URL} and try again."
                    )
                    return

            # Build the request payload
            payload = {
                "caption": caption,
                "lyrics": lyrics if lyrics else ("[Instrumental]" if instrumental else ""),
                "instrumental": instrumental,
                "duration": duration,
                "seed": seed,
            }
            if bpm and bpm > 0:
                payload["bpm"] = bpm

            # Make the generation request (long timeout — generation can take minutes)
            async with httpx.AsyncClient(timeout=300.0, headers=_MUSIC_HEADERS) as client:
                response = await client.post(
                    f"{_MUSIC_SERVER_URL}/generate",
                    json=payload,
                )

            if response.status_code != 200:
                error_detail = response.text[:200]
                yield f"The music server returned an error (HTTP {response.status_code}): {error_detail}"
                return

            # Save the audio file
            _SONG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            # Sanitise caption for filename (first 40 chars, alphanumeric + underscore)
            safe_caption = "".join(
                c if c.isalnum() or c in " _-" else "" for c in caption[:40]
            ).strip().replace(" ", "_") or "song"
            filename = f"{timestamp}_{safe_caption}.flac"
            filepath = _SONG_OUTPUT_DIR / filename

            filepath.write_bytes(response.content)

            # Extract metadata from response headers
            sample_rate = response.headers.get("X-Sample-Rate", "48000")
            used_seed = response.headers.get("X-Seed", "unknown")
            gen_time = response.headers.get("X-Generation-Time", "unknown")
            file_size_kb = len(response.content) / 1024

            summary_parts = [
                f"Your song is ready! I generated {duration:.0f} seconds of audio",
            ]
            if not instrumental and lyrics:
                summary_parts.append(" with vocals")
            else:
                summary_parts.append(" instrumental")
            summary_parts.append(
                f". It took {gen_time} seconds to generate"
                f" and the file is {file_size_kb:.0f} KB."
                f" Saved to {filepath.name}."
            )
            if used_seed != "unknown":
                summary_parts.append(
                    f" The seed was {used_seed} if you want to reproduce it."
                )

            yield "".join(summary_parts)

        except httpx.TimeoutException:
            yield (
                "The music generation timed out. The song might be too long "
                "or the server is overloaded. Try a shorter duration."
            )
        except Exception as e:
            yield f"I ran into a problem generating the song: {str(e)}"
        finally:
            self._generating_song = False

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
complex problems. You can also search and apply to jobs on LinkedIn, \
and generate songs or music on demand. These are just some of the \
many things you help with.

# Personality
Warm, natural, and conversational — like a helpful friend. You are \
proactive but not pushy. You chat about anything: daily life, tech, \
ideas, plans, recommendations, or just casual banter. You happen to \
also have the ability to run deep research, automate job applications, \
and create music.

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

# When to use generate_song
Use when the user asks you to sing, make music, generate a song, \
compose a track, or anything music-related:
- "Sing me a song"
- "Generate a pop song about hello"
- "Make me an instrumental beat"
- "Can you sing happy birthday?"

When the user asks you to "sing", compose a suitable caption describing \
the musical style and write appropriate lyrics yourself. Be creative! \
For example if they say "sing me a lullaby", you would set:
- caption: "soft female vocalist singing a gentle lullaby with piano \
accompaniment, soothing and calm, clear vocals, English"
- lyrics: "[Verse]\\nHush little baby...\\n[Chorus]\\n..."

Default behaviour:
- duration: 30 seconds (increase if user asks for a longer song)
- instrumental: false (unless specifically requested)
- bpm: 0 (auto-detect from caption)
- seed: -1 (random)

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

## generate_song
Generates music / singing audio using the ACE-Step AI model. \
Runs in the background.

When you call it:
1. Craft a descriptive caption (genre, mood, instruments, vocal style, language)
2. Write lyrics with section markers ([Verse], [Chorus], etc.) or use \
   '[Instrumental]' for no vocals
3. Acknowledge: "Let me generate that song for you" or similar
4. It takes 15-60 seconds depending on duration

Parameters:
- caption: descriptive text about the music style (REQUIRED)
- lyrics: song lyrics with section markers (default: empty)
- instrumental: true for no vocals (default: false)
- duration: length in seconds, 5-600 (default: 30)
- bpm: beats per minute, 0 for auto (default: 0)
- seed: random seed, -1 for random (default: -1)

When you receive the result:
- Tell the user their song is ready
- Mention the filename and how long it took
- Offer to generate another variation if they want

## end_call
Use when the caller says goodbye, thanks, or is clearly done.
Say goodbye naturally first, then call end_call.

# Response style
Keep it conversational — short sentences, natural phrasing. \
No emojis, asterisks, or markdown. Everything you say will be spoken aloud."""


CHAT_INTRODUCTION = (
    "Hey! I'm your personal assistant. "
    "I can help with pretty much anything — questions, planning, brainstorming, "
    "or just chatting. I can also search LinkedIn and apply to jobs for you, "
    "and if you want me to sing or generate some music, I can do that too. "
    "What's on your mind?"
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
