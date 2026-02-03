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


CHAT_SYSTEM_PROMPT = """You are a friendly and helpful voice assistant.
You handle most conversations naturally and efficiently.

However, when you encounter a question that requires deep reasoning, complex analysis, mathematical proofs,
intricate logical problems, or any question you're uncertain about, use the ask_supervisor tool to consult
with a more powerful reasoning model.

Examples of when to use ask_supervisor:
- Complex math problems or proofs
- Multi-step logical reasoning puzzles
- Questions requiring deep domain expertise
- Philosophical or ethical dilemmas requiring nuanced analysis
- Anything you're genuinely uncertain about

For simple greetings, basic facts, or straightforward conversations, just respond directly.

# Response style
1) You're on a phone call, so keep responses natural and conversational.
2) Do not output symbols like emojis or formatting like asterisks or markdown, unless you wish them to be
spoken aloud.

# Deep thinking
1) *NEVER* tell the user you're consulting another model.
2) If you call the supervisor, wait for its explanation before answering the user's question. *NEVER* try and
answer it on your own.
3) Tell the user you're still thinking if needed.
4) When you receive an answer from the supervisor, synthesize it into a natural spoken response.
"""

CHAT_INTRODUCTION = "Hello! I'm here to help. What do you want to talk about?"

SUPERVISOR_SYSTEM_PROMPT = """You are a deep reasoning assistant. Your job is to provide thorough,
well-reasoned answers to complex questions.

You will receive the full conversation history for context, followed by a specific question
that requires deep analysis.

Please provide a clear, comprehensive answer. Be thorough but also structure your response
so it can be easily summarized for a voice conversation. Focus on the key insights and conclusions."""


async def get_agent(env: AgentEnv, call_request: CallRequest):
    """Create a ChatSupervisor agent for this call."""
    return ChatSupervisorAgent()


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    print("Starting Chat/Supervisor app")
    app.run()
