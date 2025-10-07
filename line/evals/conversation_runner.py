"""
ConversationRunner - A testing wrapper around ReasoningNode for conversation flow validation.

This class allows testing conversation flows by providing expected conversation traces
and validating that the ReasoningNode produces similar responses.
"""

from typing import List, Optional

from line.evals.similarity_utils import is_similar_str
from line.evals.turn import Turn
from line.events import EventInstance
from line.nodes.conversation_context import ConversationContext
from line.nodes.reasoning import ReasoningNode


class ConversationRunner:
    """
    A testing wrapper for ReasoningNode that validates conversation flows.

    This class takes an expected conversation trace and validates that a ReasoningNode
    produces similar responses when given the same user inputs.
    """

    def __init__(
        self,
        reasoning_node: ReasoningNode,
        expected_conversation: List[Turn],
        initial_agent_message: Optional[str] = None,
        test_note: Optional[str] = None,
    ):
        """
        Initialize the test conversation.

        Args:
            reasoning_node: The ReasoningNode to test
            expected_conversation: List of Turn objects representing the expected conversation flow,
                                 alternating between user and agent turns
            initial_agent_message: Optional initial message from agent to verify against first AgentTurn
        """
        self.reasoning_node = reasoning_node
        self.expected_conversation = expected_conversation
        self.initial_agent_message = initial_agent_message
        self.test_note = test_note

    def _verify_initial_agent_message(self) -> Optional[List[EventInstance]]:
        """
        Verify the initial agent message and return its events if it exists.

        Returns:
            List of EventInstance if conversation starts with agent turn, None otherwise

        Raises:
            AssertionError: If initial agent message doesn't match expected first AgentTurn
        """
        if not self.expected_conversation:
            return None

        first_turn = self.expected_conversation[0]
        if not first_turn.is_agent:
            return None

        # If initial_agent_message is provided, verify it matches
        if self.initial_agent_message is None:
            return first_turn.to_events()

        if first_turn.text == self.initial_agent_message:
            return first_turn.to_events()

        results = is_similar_str(self.initial_agent_message, first_turn.text)
        if results.is_success:
            return first_turn.to_events()

        error_str = (
            f"Initial agent message doesn't match expected first AgentTurn.\n"
            f"Provided initial_agent_message: '{self.initial_agent_message}'\n"
            f"Expected first AgentTurn text: '{first_turn.text}'\n"
            f"Similarity error: {results.error}"
        )

        if self.test_note is not None:
            error_str += f"\nTest notes: {self.test_note}"

        raise AssertionError(error_str)

    def _verify_conversation_pattern(self) -> None:
        """
        Validate that the conversation follows proper alternating user-assistant pattern.

        Raises:
            ValueError: If the conversation pattern is invalid
        """
        if not self.expected_conversation:
            return

        # Ensure conversation ends with agent turn
        last_turn = self.expected_conversation[-1]
        if not last_turn.is_agent:
            error_str = "Conversation must end with agent turn."
            if self.test_note is not None:
                error_str += f"\nTest notes: {self.test_note}"
            raise ValueError(error_str)

        # Validate alternating pattern
        for i in range(1, len(self.expected_conversation)):
            current_turn = self.expected_conversation[i]
            previous_turn = self.expected_conversation[i - 1]

            same_type = (current_turn.is_user and previous_turn.is_user) or (
                current_turn.is_agent and previous_turn.is_agent
            )
            if same_type:
                error_str = (
                    f"Invalid conversation pattern at position {i}: "
                    f"Two consecutive '{current_turn.role}' turns. "
                    f"Expected alternating user-assistant pattern."
                )

                if self.test_note is not None:
                    error_str += f"\nTest notes: {self.test_note}"
                raise ValueError(error_str)

    async def run(self) -> None:
        """
        Run the conversation test, validating each agent response against expected.

        This method processes the expected conversation turn by turn:
        1. Process user turns by adding them to conversation history
        2. For each user turn, get the expected agent response
        3. Build ConversationContext and call process_context() on ReasoningNode
        4. Convert actual response to Turn and validate similarity
        5. Continue with next turn

        Raises:
            ValueError: If conversation pattern is invalid (non-alternating user-assistant turns)
            AssertionError: If any agent response doesn't match expected
        """
        # Validate conversation pattern first
        self._verify_conversation_pattern()

        # Track conversation history
        conversation_history: List[EventInstance] = []

        # Handle initial agent message
        initial_events = self._verify_initial_agent_message()
        i = 0
        if initial_events is not None:
            # Add the first agent turn to conversation history and skip it
            conversation_history.extend(initial_events)
            i = 1

        while i < len(self.expected_conversation):
            user_turn = self.expected_conversation[i]

            # Add user turn events to history
            user_events = user_turn.to_events()
            conversation_history.extend(user_events)
            i += 1

            # Get expected agent response from following turn
            expected_agent_turn = self.expected_conversation[i]

            # Build conversation context from history
            ctx = ConversationContext(
                events=conversation_history.copy(),
                system_prompt=self.reasoning_node.system_prompt,
            )

            # Get actual response from reasoning node
            actual_events = []
            async for event in self.reasoning_node.process_context(ctx):
                actual_events.append(event)

            # Convert actual events to Turn
            actual_turn = Turn.from_events(actual_events)

            # Validate similarity
            similarity_error = expected_agent_turn.is_similar(actual_turn)
            if similarity_error is not None:
                error_str = (
                    f"Agent turn doesn't match expected.\n"
                    f"  User message: {user_turn.text}\n"
                    f"  Expected:     {expected_agent_turn}\n"
                    f"  Actual:       {actual_turn}\n"
                    f"  Reason:       {similarity_error}\n"
                )

                if self.test_note is not None:
                    error_str += f"\nTest notes: {self.test_note}"

                raise AssertionError(error_str)

            # Add actual agent turn events to history for next iteration
            conversation_history.extend(actual_events)
            i += 1
