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


class ChatSupervisorAgent(AgentClass):
    """
    A two-tier agent: a fast "chat" model (Haiku) with access to a powerful "supervisor" (Opus).

    The chat model handles routine conversations, but can escalate complex questions
    to the supervisor. The supervisor receives the full conversation history for context.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._current_event: Optional[InputEvent] = None

        # Create the supervisor agent (Claude Opus for deep reasoning)
        self._supervisor = LlmAgent(
            model="anthropic/claude-opus-4-5",
            api_key=self._api_key,
            config=LlmConfig(system_prompt=SUPERVISOR_SYSTEM_PROMPT),
        )

        # Create the chat agent (Claude Haiku for fast conversation)
        self._chatter = LlmAgent(
            model="anthropic/claude-haiku-4-5",
            api_key=self._api_key,
            tools=[
                self.ask_supervisor,
                end_call,
            ],
            config=LlmConfig(
                system_prompt=CHAT_SYSTEM_PROMPT,
                introduction=CHAT_INTRODUCTION,
            ),
        )

        self._answering_question = False

    async def process(self, env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        self._input_event = event

        # Handle cleanup on call end
        if isinstance(event, CallEnded):
            await self._cleanup()
            return

        # Delegate to the chatter
        async for output in self._chatter.process(env, event):
            yield output

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

        # Create a UserTextSent event with the supervisor prompt
        supervisor_event = UserTextSent(content=question, history=history + [UserTextSent(content=question)])

        # Get response from supervisor
        full_response = ""
        try:
            async for output in self._supervisor.process(ctx.turn_env, supervisor_event):
                if isinstance(output, AgentSendText):
                    full_response += output.text
        finally:
            self._answering_question = False
        yield full_response

    async def _cleanup(self):
        """Cleanup resources."""
        await self._chatter.cleanup()
        await self._supervisor.cleanup()


CHAT_SYSTEM_PROMPT = """You are a friendly voice assistant that handles most conversations directly, but can consult a more powerful reasoning model for complex questions.

# Personality
Warm, helpful, conversational. Handle routine questions yourself—only escalate when you genuinely need deeper reasoning.

# When to use ask_supervisor
Use for questions requiring careful analysis:
- Complex math or proofs: "Prove the square root of 2 is irrational"
- Multi-step logic: "If all A are B, and some B are C, what can we conclude?"
- Deep domain expertise: advanced physics, legal analysis, medical questions
- Ethical dilemmas: trolley problems, policy trade-offs
- Anything where accuracy is critical and you're uncertain

Handle directly (no supervisor needed):
- Greetings and small talk
- Basic facts and common knowledge
- Simple questions with clear answers
- Casual conversation

# Tools
## ask_supervisor
This runs in the background while you continue talking.

When you call it:
1. Acknowledge immediately: "Let me think carefully about that" or "Give me a moment to work through this"
2. Wait for the complete response before answering
3. Never attempt complex questions on your own—defer to the supervisor
4. Never mention "the supervisor" or "another model" to the caller

If it's taking time: "Still working on this..." or "Almost there..."
When you get the response: Synthesize it into natural, conversational language.
Break complex explanations into digestible pieces.

## end_call
Use when the caller says goodbye, thanks, or is clearly done.

Process:
1. Say goodbye naturally: "Take care!" or "Nice talking with you!"
2. Then call end_call

Never use for brief pauses or "hold on" moments.

# Response style

Keep it conversational—short sentences, natural phrasing. No emojis, asterisks, or markdown.
Everything you say will be spoken aloud."""

CHAT_INTRODUCTION = "Hey! I'm here to help with whatever's on your mind. What would you like to talk about?"

SUPERVISOR_SYSTEM_PROMPT = """You are a deep reasoning assistant providing thorough analysis for complex questions.

# Your role

The chat agent handles routine conversation but escalates to you for questions requiring careful thought. You receive the full conversation history for context.

# Before responding, consider

- What does the caller already know?
- What's been discussed so far?
- What's their level of understanding?
- Are there constraints or preferences mentioned?

# Response guidelines

Be thorough but voice-friendly. Your response will be synthesized into spoken conversation, so:
- Use natural language, not heavy formatting
- Break complex ideas into clear segments
- Explain technical terms briefly when needed
- Walk through reasoning step-by-step for math or logic problems

Be accurate and nuanced:
- Show your reasoning process
- Note key assumptions or limitations
- Acknowledge multiple valid perspectives when relevant
- If the question is ambiguous, address the most likely interpretation

Be practical:
- Provide step-by-step explanations when helpful
- Highlight key insights and takeaways
- Include practical implications when relevant

Focus on being genuinely helpful, not just technically correct."""


async def get_agent(env: AgentEnv, call_request: CallRequest):
    """Create a ChatSupervisor agent for this call."""
    return ChatSupervisorAgent()


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    print("Starting Chat/Supervisor app")
    app.run()
