"""
Unit tests for ConversationContext.get_committed_events()

Tests the matching of AgentResponse events against AgentSpeechSent events,
particularly handling interruptions where speech is cut short.
"""

import pytest

from line.events import AgentResponse, AgentSpeechSent, UserTranscriptionReceived
from line.nodes.conversation_context import ConversationContext


class TestGetCommittedEvents:
    """Test cases for get_committed_events method."""

    def test_full_match_single_response(self):
        """Test when AgentResponse is fully spoken (no interruption)."""
        events = [
            AgentResponse(content="Hello world!"),
            AgentSpeechSent(content="Helloworld!"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Hello world!"

    def test_interruption_partial_match(self):
        """Test when AgentSpeechSent is interrupted mid-response."""
        events = [
            AgentResponse(content="Hello world! How are you today?"),
            AgentSpeechSent(content="Helloworld!How"),  # Interrupted after "How"
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        # Should return AgentResponse with only what was actually spoken
        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        # The committed text should preserve formatting from AgentResponse
        # "Hello world! How" matches "Helloworld!How"
        assert committed[0].content == "Hello world! How"

    def test_multiple_responses_with_full_match(self):
        """Test multiple AgentResponse events concatenated before speech."""
        events = [
            AgentResponse(content="Hello"),
            AgentResponse(content=" world!"),
            AgentSpeechSent(content="Helloworld!"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Hello world!"

    def test_with_newlines_and_formatting(self):
        """Test matching with newlines and complex formatting."""
        events = [
            AgentResponse(content="Hello!\n\nHow are you?"),
            AgentSpeechSent(content="Hello!How"),  # Interrupted after "How"
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Hello!\n\nHow"

    def test_real_world_conversation(self):
        """Test real-world conversation with multiple interruptions and continuations."""
        events = [
            AgentResponse(
                content="Let's play 20 questions! When you have your item in mind, just say start."
            ),
            AgentSpeechSent(content="Let's"),  # Interrupted!
            UserTranscriptionReceived(content="Yeah."),
            AgentSpeechSent(content=" play 20 questions! When you have your item in mind, just say start."),
            AgentResponse(
                content=(
                    "Alright, I'm ready to play! I'll try my best to guess what you're "
                    "thinking of.\n\nQuestion 1: Is it an animal?"
                )
            ),
            AgentSpeechSent(
                content="Alright,I'mreadytoplay!I'lltrymybesttoguesswhatyou'rethinkingof.Question1:Isitananimal?"
            ),
            UserTranscriptionReceived(content="No. It's not an animal."),
            AgentResponse(
                content=(
                    "Okay, not an animal! That narrows it down a bit.\n\nQuestion 2: Is it a physical object?"
                )
            ),
            AgentSpeechSent(content="Okay,notananimal!"),  # Interrupted!
            UserTranscriptionReceived(content="Good call to go."),
            AgentResponse(content="Question 2: Is it a physical object?"),
            AgentSpeechSent(content="Question2:Isitaphysicalobject?"),
            UserTranscriptionReceived(content="No. It's not a physical object."),
            AgentResponse(
                content=(
                    "Interesting! Not a physical object.\n\nQuestion 3: Is it an abstract concept or idea?"
                )
            ),
            AgentSpeechSent(content="Interesting!Notaphysicalobject."),
            UserTranscriptionReceived(content="What was question"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        # Expected: 10 events total
        # 1. AgentResponse: "Let's play 20 questions! When you have your item in mind, just say start."
        # 2. AgentSpeechSent: "Let's" (matched from first speech)
        # 2. UserTranscription: 'Yeah.'
        # 3. AgentResponse: full second response
        # 4. UserTranscription: "No. It's not an animal."
        # 5. AgentResponse: 'Okay, not an animal!' (partial from third response)
        # 6. UserTranscription: 'Good call to go.'
        # 7. AgentResponse: 'Question 2: Is it a physical object?'
        # 8. UserTranscription: "No. It's not a physical object."
        # 9. AgentResponse: 'Interesting! Not a physical object.'
        # 10. UserTranscription: 'What was question'
        assert len(committed) == 11

        # Check first committed response (interrupted)
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Let's"

        # Check first user message
        assert isinstance(committed[1], UserTranscriptionReceived)
        assert committed[1].content == "Yeah."

        assert isinstance(committed[2], AgentResponse)
        assert committed[2].content == "play 20 questions! When you have your item in mind, just say start."

        # Check second committed response (full)
        assert isinstance(committed[3], AgentResponse)
        assert committed[3].content == (
            "Alright, I'm ready to play! I'll try my best to guess what you're "
            "thinking of.\n\nQuestion 1: Is it an animal?"
        )

        # Check second user message
        assert isinstance(committed[4], UserTranscriptionReceived)
        assert committed[4].content == "No. It's not an animal."

        # Check third committed response (interrupted - only "Okay, not an animal!")
        assert isinstance(committed[5], AgentResponse)
        assert committed[5].content == "Okay, not an animal!"

        # Check third user message
        assert isinstance(committed[6], UserTranscriptionReceived)
        assert committed[6].content == "Good call to go."

        # Check fourth committed response (continuation from pending)
        assert isinstance(committed[7], AgentResponse)
        assert committed[7].content == "Question 2: Is it a physical object?"

        # Check fourth user message
        assert isinstance(committed[8], UserTranscriptionReceived)
        assert committed[8].content == "No. It's not a physical object."

        # Check fifth user message (no agent response committed yet)
        assert isinstance(committed[9], AgentResponse)
        assert committed[9].content == "Interesting! Not a physical object."

        # Check sixth user message
        assert isinstance(committed[10], UserTranscriptionReceived)
        assert committed[10].content == "What was question"

    def test_user_transcription_passed_through(self):
        """Test that UserTranscriptionReceived events are passed through unchanged."""
        events = [
            UserTranscriptionReceived(content="Hi there"),
            AgentResponse(content="Hello!"),
            AgentSpeechSent(content="Hello!"),
            UserTranscriptionReceived(content="How are you?"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 3
        assert isinstance(committed[0], UserTranscriptionReceived)
        assert committed[0].content == "Hi there"
        assert isinstance(committed[1], AgentResponse)
        assert committed[1].content == "Hello!"
        assert isinstance(committed[2], UserTranscriptionReceived)
        assert committed[2].content == "How are you?"

    def test_multiple_speech_events(self):
        """Test multiple speech events in conversation."""
        events = [
            AgentResponse(content="Hello!"),
            AgentSpeechSent(content="Hello!"),
            UserTranscriptionReceived(content="Hi"),
            AgentResponse(content="How are you?"),
            AgentSpeechSent(content="Howareyou?"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 3
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Hello!"
        assert isinstance(committed[1], UserTranscriptionReceived)
        assert committed[1].content == "Hi"
        assert isinstance(committed[2], AgentResponse)
        assert committed[2].content == "How are you?"

    def test_interruption_preserves_pending_for_next_speech(self):
        """Test that unspoken text remains pending for next speech event."""
        events = [
            AgentResponse(content="Hello world!"),
            AgentSpeechSent(content="Hello"),  # Only "Hello" spoken
            # In real scenario, there would be another speech event later
            # but pending text should carry over
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        # Should only commit what was actually spoken (with formatting preserved)
        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Hello"

    def test_empty_events(self):
        """Test with no events."""
        context = ConversationContext(events=[], system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 0

    def test_only_user_events(self):
        """Test with only user transcription events."""
        events = [
            UserTranscriptionReceived(content="Hello"),
            UserTranscriptionReceived(content="How are you?"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 2
        assert all(isinstance(e, UserTranscriptionReceived) for e in committed)

    def test_response_without_speech(self):
        """Test AgentResponse without corresponding AgentSpeechSent."""
        events = [
            AgentResponse(content="Hello"),
            UserTranscriptionReceived(content="Hi"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        # AgentResponse without speech should not be committed
        assert len(committed) == 1
        assert isinstance(committed[0], UserTranscriptionReceived)

    def test_pending_text_carries_over_multiple_responses(self):
        """Test that pending text accumulates across multiple AgentResponse events."""
        events = [
            AgentResponse(content="Hello"),
            AgentResponse(content=" world"),
            AgentResponse(content="! How are you?"),
            AgentSpeechSent(content="Helloworld!How"),  # Matches across all three responses
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Hello world! How"

    def test_chinese_characters_full_match(self):
        """Test matching with Chinese characters (no spaces between words)."""
        events = [
            AgentResponse(content="你好！今天天气怎么样？"),  # "Hello! How's the weather today?"
            AgentSpeechSent(content="你好！今天天气怎么样？"),  # TTS with all text
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "你好！今天天气怎么样？"

    def test_chinese_characters_partial_match(self):
        """Test matching with Chinese characters when interrupted."""
        events = [
            AgentResponse(content="你好！今天天气怎么样？"),  # "Hello! How's the weather today?"
            AgentSpeechSent(content="你好！今天"),  # TTS interrupted after "Hello! Today"
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "你好！今天"

    def test_mixed_language_with_spaces(self):
        """Test matching with mixed English and Chinese with spaces."""
        events = [
            AgentResponse(content="Hello 你好! How are you 今天好吗?"),
            AgentSpeechSent(content="Hello你好!Howareyou今天好吗?"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Hello 你好! How are you 今天好吗?"

    def test_chinese_with_interruption_and_continuation(self):
        """Test Chinese text with interruption and continuation like real conversation."""
        events = [
            AgentResponse(content="我想问你一个问题"),  # "I want to ask you a question"
            AgentSpeechSent(content="我想问你"),  # Interrupted after "I want to ask you"
            UserTranscriptionReceived(content="等一下"),  # "Wait a moment"
            AgentResponse(content="好的，你准备好了吗？"),  # "Okay, are you ready?"
            AgentSpeechSent(content="好的，你准备好了吗？"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 3
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "我想问你"
        assert isinstance(committed[1], UserTranscriptionReceived)
        assert committed[1].content == "等一下"
        assert isinstance(committed[2], AgentResponse)
        assert committed[2].content == "好的，你准备好了吗？"

    def test_multiple_responses_concatenation_with_space(self):
        """Test that multiple AgentResponse events are concatenated with space separator."""
        events = [
            AgentResponse(content="First response."),
            AgentResponse(content="Second response."),
            AgentResponse(content="Third response."),
            AgentSpeechSent(content="Firstresponse.Second"),  # Interrupted in second response
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        # Should preserve the space separator added during concatenation
        assert committed[0].content == "First response.Second"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
