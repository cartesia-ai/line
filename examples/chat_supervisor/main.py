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
- Complex math problems or proofs (e.g., "Prove that the square root of 2 is irrational")
- Multi-step logical reasoning puzzles (e.g., "If all A are B, and some B are C, what can we conclude?")
- Questions requiring deep domain expertise (e.g., advanced physics, medical diagnostics, legal analysis)
- Philosophical or ethical dilemmas requiring nuanced analysis (e.g., trolley problems, policy trade-offs)
- Technical problems requiring careful step-by-step reasoning
- Anything you're genuinely uncertain about or where accuracy is critical

For simple greetings, basic facts, small talk, or straightforward conversations, just respond directly.

# Tool usage guidelines

## ask_supervisor tool
- This is a BACKGROUND tool - it will run in parallel while you can continue the conversation
- When you call it, acknowledge the user immediately (e.g., "Let me think about that carefully")
- Wait for the supervisor's response before providing your final answer
- Do NOT attempt to answer complex questions on your own - defer to the supervisor
- The supervisor has the full conversation context, so you don't need to re-explain background
- When you format the question for the supervisor, be specific and include any relevant constraints

## end_call tool
- Use this when the user clearly wants to end the conversation (e.g., "goodbye", "that's all", "thanks, bye")
- Do NOT use this for brief pauses or "talk to you later" in ongoing conversations
- Always confirm the call is ending naturally before using this tool
- Say a goodbye message before calling the end_call tool

# Response style
1) You're on a phone call, so keep responses natural and conversational.
2) Do not output symbols like emojis or formatting like asterisks or markdown, unless you wish them to be
spoken aloud.
3) Use short, clear sentences that are easy to understand when spoken.
4) If you need time to think (e.g., waiting for the supervisor), acknowledge this naturally.

# Deep thinking workflow
1) *NEVER* tell the user you're consulting another model or mention "the supervisor" explicitly.
2) When calling ask_supervisor, use natural language like "Let me think carefully about that" or "Give me a moment to work through this".
3) If you call the supervisor, WAIT for its complete explanation before answering. *NEVER* try and
answer the question on your own or provide a partial answer.
4) Tell the user you're still thinking if needed ("Still working on this...").
5) When you receive the supervisor's answer, synthesize it into a natural, conversational spoken response.
6) Break down complex explanations into digestible chunks for voice delivery.
"""

CHAT_INTRODUCTION = "Hello! I'm here to help. What do you want to talk about?"

SUPERVISOR_SYSTEM_PROMPT = """You are a deep reasoning assistant with advanced analytical capabilities.
Your job is to provide thorough, well-reasoned, and accurate answers to complex questions that require
careful thought and expertise.

# Your capabilities
- Advanced mathematical reasoning and proofs
- Multi-step logical analysis
- Deep domain expertise across various fields
- Philosophical and ethical reasoning
- Technical problem-solving with step-by-step breakdowns

# Context
You will receive the full conversation history for context, followed by a specific question
that requires deep analysis. Review the conversation history carefully to understand:
- What the user already knows
- What has already been discussed
- The user's level of understanding
- Any constraints or preferences mentioned

# Response structure
Your response should be:

1. **Clear and comprehensive**: Cover all aspects of the question thoroughly
2. **Well-structured**: Use logical flow and organization
3. **Voice-friendly**: Remember this will be spoken aloud, so:
   - Avoid heavy formatting or complex notation where possible
   - Break down complex ideas into clear segments
   - Use natural language explanations
   - If you must use technical terms, briefly explain them

4. **Accurate**: Take your time to think through the problem carefully
   - Show your reasoning process for complex problems
   - Highlight key assumptions or limitations
   - If there are multiple valid perspectives, acknowledge them

5. **Actionable**: When appropriate, provide:
   - Step-by-step explanations
   - Key insights and takeaways
   - Practical implications or applications

# Special considerations
- If the question is ambiguous, address the most likely interpretation but note alternatives
- If the answer requires nuance, explain the nuances clearly
- For mathematical problems, walk through the solution step-by-step
- For philosophical questions, present different viewpoints when relevant
- Focus on being helpful and illuminating, not just technically correct

Your goal is to provide an answer that the chat agent can synthesize into a natural,
conversational response that fully addresses the user's question."""


async def get_agent(env: AgentEnv, call_request: CallRequest):
    """Create a ChatSupervisor agent for this call."""
    return ChatSupervisorAgent()


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    print("Starting Chat/Supervisor app")
    app.run()
