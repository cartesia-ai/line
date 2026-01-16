"""Chat Node for Personal Banking Customer Support Agent.

This module contains the main ChatNode that manages conversation flow between
specialized sub-agents through handoff mechanisms.
"""

from enum import Enum
from typing import AsyncGenerator, Union
import uuid

from context import BankContext
from loguru import logger
from sub_agents import FAQAgent, SubAgent, TransactionAgent, VerificationAgent, WelcomeAgent

from line import ConversationContext, ReasoningNode
from line.events import AgentResponse, ToolCall, ToolResult
from line.tools.system_tools import EndCallArgs, EndCallTool, end_call
from line.utils.gemini_utils import convert_messages_to_gemini


class AgentState(Enum):
    """Enumeration of possible agent states."""

    WELCOME = "welcome"
    VERIFICATION = "verification"
    TRANSACTION = "transaction"
    FAQ = "faq"


class ChatNode(ReasoningNode):
    """Main chat node that orchestrates conversation flow between sub-agents.

    This node manages:
    1. Agent state tracking
    2. Sub-agent initialization and selection
    3. Handoff processing between agents
    4. Context forwarding to appropriate agents
    """

    def __init__(self, max_context_length: int = 100):
        """Initialize the chat node.

        Args:
            max_context_length: Maximum conversation turns to keep.
        """
        super().__init__("", max_context_length)

        # Create context for this session
        self.context = BankContext()

        # Generate unique session ID
        self.session_id = str(uuid.uuid4())
        self.context.session_id = self.session_id
        logger.info(f"🎯 New session: {self.session_id}")

        # Start with welcome agent
        self.current_state = AgentState.WELCOME

        # Initialize all agents with the context
        self.agents: dict[AgentState, SubAgent] = {
            AgentState.WELCOME: WelcomeAgent(self.context),
            AgentState.VERIFICATION: VerificationAgent(self.context),
            AgentState.TRANSACTION: TransactionAgent(self.context),
            AgentState.FAQ: FAQAgent(self.context),
        }

        logger.info(f"✅ Chat node initialized. Starting in {self.current_state.value} state")

    def initial_message(self):
        """Return the initial message for the Welcome agent."""
        # Welcome agent returns a simple message, so we can return the constant directly
        from sub_agents.welcome_agent import WELCOME_INITIAL_MESSAGE

        return WELCOME_INITIAL_MESSAGE

    def _get_current_agent(self):
        """Get the current active agent."""
        return self.agents[self.current_state]

    def _handoff_to_transaction(self) -> AgentState:
        """Handoff to transaction agent.

        Returns:
            AgentState.TRANSACTION
        """
        logger.info("🔄 Handoff to transaction agent")
        return AgentState.TRANSACTION

    def _handoff_to_faq(self) -> AgentState:
        """Handoff to FAQ agent.

        Returns:
            AgentState.FAQ
        """
        logger.info("🔄 Handoff to FAQ agent")
        return AgentState.FAQ

    def _handoff_to_verification(self) -> AgentState:
        """Handoff to verification agent.

        Returns:
            AgentState.VERIFICATION
        """
        logger.info("🔄 Handoff to verification agent")
        return AgentState.VERIFICATION

    def _process_handoff_tool_call(self, tool_call: ToolCall) -> AgentState:
        """Process handoff tool calls and return new agent state.

        This method handles verification overriding - if a handoff_to_transaction
        is requested but the user is not verified, it automatically routes to
        verification instead and modifies the tool call in place.

        Args:
            tool_call: The tool call to process (may be modified in place)

        Returns:
            New AgentState
        """
        # Check for transaction handoffs when user is not verified
        if tool_call.tool_name == "handoff_to_transaction":
            if not self.context.user_details.verified:
                logger.info(
                    "🔄 User not verified, overriding handoff_to_transaction with handoff_to_verification"
                )
                # Modify the tool call in place to reflect the actual routing
                tool_call.tool_name = "handoff_to_verification"
                return self._handoff_to_verification()
            else:
                return self._handoff_to_transaction()
        elif tool_call.tool_name == "handoff_to_faq":
            return self._handoff_to_faq()
        elif tool_call.tool_name == "handoff_to_verification":
            return self._handoff_to_verification()
        else:
            raise ValueError(f"🔄 Unknown handoff tool call: {tool_call.tool_name}")

    async def process_context(
        self, context: ConversationContext
    ) -> AsyncGenerator[Union[AgentResponse, ToolResult], None]:
        """Process conversation context through current agent.

        Args:
            context: ConversationContext with messages and metadata

        Yields:
            AgentResponse: Messages from the current agent
            ToolResult: Tool calls from the current agent or handoff results
        """
        messages = convert_messages_to_gemini(context.get_committed_events())
        user_message = context.get_latest_user_transcript_message()

        if user_message:
            logger.info(f'📋 Processing with {self.current_state.value} agent: "{user_message}"')

        # Get current agent and generate response
        current_agent = self._get_current_agent()

        # Process the agent's response stream
        async for msg in current_agent.generate(user_message, messages):
            if isinstance(msg, ToolCall) and msg.tool_name.startswith("handoff_to_"):
                # Intercept handoff tool calls and process them
                new_state = self._process_handoff_tool_call(msg)

                # Update the current state
                old_state = self.current_state
                self.current_state = new_state

                # Generate initial message from new agent
                logger.info(f"🔄 State transition: {old_state.value} → {new_state.value}")

                # Create a tool result to indicate successful handoff and yield it
                tool_result = ToolResult(
                    tool_name=msg.tool_name,
                    tool_args=msg.tool_args,
                    tool_call_id=msg.tool_call_id,
                )
                yield tool_result

                # Stream initial message from new agent
                async for initial_msg in self.agents[new_state].initial_message(user_message, messages):
                    yield initial_msg
            elif isinstance(msg, ToolCall) and msg.tool_name == EndCallTool.name():
                logger.info("🔄 End call requested")
                args = EndCallArgs.model_validate(msg.tool_args)
                async for item in end_call(args):
                    yield item
            else:
                yield msg
