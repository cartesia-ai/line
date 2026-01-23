import os

from line.v02.events import AgentHandedOff, AgentSendText, CallStarted
from line.v02.llm import LlmAgent, LlmConfig, handoff_tool
from line.v02.llm.agent import ToolEnv
from line.v02.tools import end_call


class SpanishAgentTransfer:
    """
    Handoff tool that transfers to a Spanish-speaking agent.

    Uses a class to maintain state (the agent instance) across handoff calls.
    """

    def __init__(self):
        self._agent: LlmAgent | None = None

    def _create_agent(self) -> LlmAgent:
        """Create the Spanish-speaking agent."""
        return LlmAgent(
            model="gemini/gemini-2.0-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            tools=[end_call],
            config=LlmConfig(
                system_prompt="You are a friendly and helpful assistant. Have a natural conversation with the user. You speak only in Spanish.",
                introduction="¡Hola! Soy tu asistente de IA. ¿Cómo puedo ayudarte hoy?",
            ),
        )

    @property
    def tool(self):
        """Return the handoff tool that transfers to this agent."""
        transfer = self  # Capture self in closure

        @handoff_tool(
            name="transfer_to_spanish",
            description="Transfer the call to a Spanish-speaking agent. Use this when the user requests to speak in Spanish.",
        )
        async def transfer_to_spanish(ctx: ToolEnv, event):
            """Transfer to the Spanish speaking agent."""
            if isinstance(event, AgentHandedOff):
                # Create the Spanish agent on first handoff
                transfer._agent = transfer._create_agent()
                yield AgentSendText(text="Transferring you to our Spanish-speaking agent now...")

                # Trigger the Spanish agent's introduction
                async for output in transfer._agent.process(ctx.turn_env, CallStarted()):
                    yield output
                return

            # Delegate all subsequent events to the Spanish agent
            if transfer._agent:
                async for output in transfer._agent.process(ctx.turn_env, event):
                    yield output

        return transfer_to_spanish
