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


class TestLatinScriptsWithDiacritics:
    """Test cases for European languages with diacritics."""

    def test_french_accents_full_match(self):
        """Test French text with various accents (é, è, ê, à, ç)."""
        events = [
            AgentResponse(content="Ça va? Comment ça s'est passé?"),
            AgentSpeechSent(content="Çava?Commentças'estpassé?"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Ça va? Comment ça s'est passé?"

    def test_french_interruption(self):
        """Test French text interrupted mid-sentence."""
        events = [
            AgentResponse(content="Bonjour! J'espère que tu vas bien aujourd'hui."),
            AgentSpeechSent(content="Bonjour!J'espèrequetuvas"),  # Interrupted
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Bonjour! J'espère que tu vas"

    def test_spanish_special_characters(self):
        """Test Spanish punctuation and accents (ñ, á, í, ¿, ¡)."""
        events = [
            AgentResponse(content="¿Cómo estás? ¡Muy bien! Mañana es España."),
            AgentSpeechSent(content="¿Cómoestás?¡Muybien!MañanaesEspaña."),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "¿Cómo estás? ¡Muy bien! Mañana es España."

    def test_spanish_interruption(self):
        """Test Spanish text with interruption."""
        events = [
            AgentResponse(content="¿Cómo estás? ¡Muy bien!"),
            AgentSpeechSent(content="¿Cómoestás?"),  # Interrupted after first sentence
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "¿Cómo estás?"

    def test_german_umlauts(self):
        """Test German text with umlauts (ä, ö, ü, ß)."""
        events = [
            AgentResponse(content="Guten Tag! Schönes Wetter. Straße und Äpfel."),
            AgentSpeechSent(content="GutenTag!SchönesWetter.StraßeundÄpfel."),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Guten Tag! Schönes Wetter. Straße und Äpfel."

    def test_portuguese_tildes(self):
        """Test Portuguese text with tildes and accents (ã, õ, â, ê)."""
        events = [
            AgentResponse(content="Não tenho irmão. São Paulo é bonito!"),
            AgentSpeechSent(content="Nãotenhoirmão.SãoPauloé"),  # Interrupted
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Não tenho irmão. São Paulo é"

    def test_multiple_languages_in_conversation(self):
        """Test conversation switching between multiple European languages."""
        events = [
            AgentResponse(content="Bonjour! Comment ça va?"),
            AgentSpeechSent(content="Bonjour!Commentçava?"),
            UserTranscriptionReceived(content="Très bien!"),
            AgentResponse(content="¿Y cómo está el tiempo?"),
            AgentSpeechSent(content="¿Ycómo"),  # Interrupted
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 3
        assert committed[0].content == "Bonjour! Comment ça va?"
        assert committed[1].content == "Très bien!"
        assert committed[2].content == "¿Y cómo"


class TestOtherNonLatinScripts:
    """Test cases for Japanese, Arabic, Korean, and other scripts."""

    def test_japanese_hiragana_katakana_kanji(self):
        """Test Japanese with hiragana, katakana, and kanji."""
        events = [
            AgentResponse(content="こんにちは。今日はいい天気ですね。ありがとう。"),
            AgentSpeechSent(content="こんにちは。今日はいい天気ですね。ありがとう。"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "こんにちは。今日はいい天気ですね。ありがとう。"

    def test_japanese_interruption(self):
        """Test Japanese text with interruption."""
        events = [
            AgentResponse(content="こんにちは。今日はいい天気ですね。"),
            AgentSpeechSent(content="こんにちは。今日は"),  # Interrupted
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "こんにちは。今日は"

    def test_japanese_mixed_with_english(self):
        """Test Japanese mixed with English words."""
        events = [
            AgentResponse(content="Hello! 今日はいい天気ですね。Thank you!"),
            AgentSpeechSent(content="Hello!今日はいい天気ですね。Thankyou!"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Hello! 今日はいい天気ですね。Thank you!"

    def test_arabic_rtl_text(self):
        """Test Arabic right-to-left text."""
        events = [
            AgentResponse(content="مرحبا! كيف حالك اليوم؟"),
            AgentSpeechSent(content="مرحبا!كيفحالكاليوم؟"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "مرحبا! كيف حالك اليوم؟"

    def test_arabic_interruption(self):
        """Test Arabic text with interruption."""
        events = [
            AgentResponse(content="مرحبا! كيف حالك اليوم؟ أنا بخير."),
            AgentSpeechSent(content="مرحبا!كيفحالك"),  # Interrupted
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "مرحبا! كيف حالك"

    def test_korean_hangul(self):
        """Test Korean Hangul characters."""
        events = [
            AgentResponse(content="안녕하세요! 오늘 날씨가 좋네요."),
            AgentSpeechSent(content="안녕하세요!오늘날씨가좋네요."),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "안녕하세요! 오늘 날씨가 좋네요."

    def test_korean_interruption(self):
        """Test Korean text with interruption."""
        events = [
            AgentResponse(content="안녕하세요! 오늘 날씨가 좋네요. 감사합니다."),
            AgentSpeechSent(content="안녕하세요!오늘"),  # Interrupted
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "안녕하세요! 오늘"

    def test_thai_script(self):
        """Test Thai script (no spaces between words)."""
        events = [
            AgentResponse(content="สวัสดีครับ วันนี้อากาศดีมาก"),
            AgentSpeechSent(content="สวัสดีครับวันนี้อากาศดีมาก"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "สวัสดีครับ วันนี้อากาศดีมาก"

    def test_hindi_devanagari(self):
        """Test Hindi Devanagari script."""
        events = [
            AgentResponse(content="नमस्ते! आप कैसे हैं?"),
            AgentSpeechSent(content="नमस्ते!आपकैसेहैं?"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "नमस्ते! आप कैसे हैं?"


class TestEmojisAndSpecialCharacters:
    """Test cases for emojis and special characters."""

    def test_emojis_in_response_full_match(self):
        """Test responses containing emojis."""
        events = [
            AgentResponse(content="Hello! 👋 How are you? 😊"),
            AgentSpeechSent(content="Hello!👋Howareyou?😊"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Hello! 👋 How are you? 😊"

    def test_emoji_interruption(self):
        """Test interruption at emoji boundary."""
        events = [
            AgentResponse(content="Great! 🎉 Let's celebrate! 🎊"),
            AgentSpeechSent(content="Great!🎉"),  # Interrupted after first emoji
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Great! 🎉"

    def test_multiple_emojis_consecutively(self):
        """Test multiple emojis in a row."""
        events = [
            AgentResponse(content="Wow! 🎉🎊🎈 Amazing!"),
            AgentSpeechSent(content="Wow!🎉🎊🎈Amazing!"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Wow! 🎉🎊🎈 Amazing!"

    def test_emoji_skin_tone_modifiers(self):
        """Test emojis with skin tone modifiers."""
        events = [
            AgentResponse(content="Hello! 👋🏽 Nice to meet you! 👍🏾"),
            AgentSpeechSent(content="Hello!👋🏽Nicetomeetyou!👍🏾"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Hello! 👋🏽 Nice to meet you! 👍🏾"

    def test_numbers_and_symbols(self):
        """Test responses with numbers and mathematical symbols."""
        events = [
            AgentResponse(content="The answer is 42! That's 100% correct. 2+2=4."),
            AgentSpeechSent(content="Theansweris42!That's100%correct.2+2=4."),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "The answer is 42! That's 100% correct. 2+2=4."

    def test_currency_symbols(self):
        """Test various currency symbols."""
        events = [
            AgentResponse(content="It costs $100 or €85 or £75 or ¥10000."),
            AgentSpeechSent(content="Itcosts$100or€85or£75or¥10000."),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "It costs $100 or €85 or £75 or ¥10000."


class TestWhitespaceAndFormatting:
    """Test cases for various whitespace and formatting scenarios."""

    def test_multiple_spaces(self):
        """Test multiple consecutive spaces."""
        events = [
            AgentResponse(content="Hello    world!    How   are you?"),
            AgentSpeechSent(content="Helloworld!Howareyou?"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Hello    world!    How   are you?"

    def test_multiple_newlines(self):
        """Test multiple consecutive newlines."""
        events = [
            AgentResponse(content="Hello\n\n\nworld!\n\nHow are you?"),
            AgentSpeechSent(content="Helloworld!Howareyou?"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Hello\n\n\nworld!\n\nHow are you?"

    def test_leading_trailing_whitespace(self):
        """Test with leading/trailing whitespace in content."""
        events = [
            AgentResponse(content="  Hello world!  "),
            AgentSpeechSent(content="Helloworld!"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        # Strip is applied in the implementation
        assert committed[0].content == "Hello world!"

    def test_tab_characters(self):
        """Test with tab characters."""
        events = [
            AgentResponse(content="Hello\tworld!\tHow\tare\tyou?"),
            AgentSpeechSent(content="Helloworld!Howareyou?"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Hello\tworld!\tHow\tare\tyou?"

    def test_mixed_whitespace_types(self):
        """Test mixed spaces, tabs, and newlines."""
        events = [
            AgentResponse(content="Hello \t world!\n How  \tare\tyou?"),
            AgentSpeechSent(content="Helloworld!Howareyou?"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Hello \t world!\n How  \tare\tyou?"


class TestPunctuationVariations:
    """Test cases for various punctuation marks and their variations."""

    def test_smart_quotes_and_apostrophes(self):
        """Test 'smart' typographic quotes vs straight quotes."""
        events = [
            AgentResponse(content="""It's a "test" of 'quotes' and "more"."""),
            AgentSpeechSent(content="""It'sa"test"of'quotes'and"more"."""),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == """It's a "test" of 'quotes' and "more"."""

    def test_ellipsis_variations(self):
        """Test three dots vs ellipsis character."""
        events = [
            AgentResponse(content="Well... I think… maybe?"),
            AgentSpeechSent(content="Well...Ithink…maybe?"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Well... I think… maybe?"

    def test_dashes_and_hyphens(self):
        """Test em-dash, en-dash, and hyphen."""
        events = [
            AgentResponse(content="Hello—world! It's 2020–2025 or well-known."),
            AgentSpeechSent(content="Hello—world!It's2020–2025orwell-known."),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Hello—world! It's 2020–2025 or well-known."

    def test_various_brackets(self):
        """Test different types of brackets."""
        events = [
            AgentResponse(content="Test (parentheses) [brackets] {braces} <angles>."),
            AgentSpeechSent(content="Test(parentheses)[brackets]{braces}<angles>."),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Test (parentheses) [brackets] {braces} <angles>."

    def test_special_punctuation(self):
        """Test special punctuation marks."""
        events = [
            AgentResponse(content="Wow! Really? Yes... Maybe; no, never: always."),
            AgentSpeechSent(content="Wow!Really?Yes...Maybe;no,never:always."),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Wow! Really? Yes... Maybe; no, never: always."


class TestEdgeCases:
    """Test cases for edge cases and boundary conditions."""

    def test_empty_response_content(self):
        """Test with empty AgentResponse content."""
        events = [
            AgentResponse(content=""),
            AgentSpeechSent(content=""),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        # Empty content after strip returns no committed events
        assert len(committed) == 0

    def test_only_whitespace_content(self):
        """Test response with only whitespace."""
        events = [
            AgentResponse(content="   \n\t  "),
            AgentSpeechSent(content=""),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        # Whitespace-only content gets stripped and returns empty
        assert len(committed) == 0

    def test_only_punctuation(self):
        """Test response with only punctuation."""
        events = [
            AgentResponse(content="...!?!"),
            AgentSpeechSent(content="...!?!"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "...!?!"

    def test_very_long_response(self):
        """Test with extremely long response text."""
        long_text = "Hello world! " * 100  # 1300 chars
        long_speech = long_text.replace(" ", "")[:650]  # Interrupted halfway

        events = [
            AgentResponse(content=long_text),
            AgentSpeechSent(content=long_speech),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        # Should have committed about half of the responses
        assert len(committed[0].content) < len(long_text)

    def test_single_character_response(self):
        """Test single character responses."""
        events = [
            AgentResponse(content="A"),
            AgentSpeechSent(content="A"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "A"

    def test_single_word_response(self):
        """Test single word response."""
        events = [
            AgentResponse(content="Hello"),
            AgentSpeechSent(content="Hello"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Hello"


class TestComplexMultilingual:
    """Test cases for complex multilingual scenarios."""

    def test_complex_multilingual_mix(self):
        """Test mixing multiple scripts in one response."""
        events = [
            AgentResponse(content="Hello 你好 مرحبا こんにちは 안녕하세요! How are you?"),
            AgentSpeechSent(content="Hello你好مرحباこんにちは안녕하세요!Howareyou?"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Hello 你好 مرحبا こんにちは 안녕하세요! How are you?"

    def test_multilingual_with_emojis(self):
        """Test multilingual text with emojis."""
        events = [
            AgentResponse(content="Hello 👋 你好 😊 مرحبا 🌟"),
            AgentSpeechSent(content="Hello👋你好😊مرحبا🌟"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Hello 👋 你好 😊 مرحبا 🌟"

    def test_code_snippet_in_response(self):
        """Test code snippets in response."""
        events = [
            AgentResponse(content="Use the function: print('hello world')"),
            AgentSpeechSent(content="Usethefunction:print('helloworld')"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Use the function: print('hello world')"

    def test_url_in_response(self):
        """Test URLs in responses."""
        events = [
            AgentResponse(content="Visit https://example.com for more info!"),
            AgentSpeechSent(content="Visithttps://example.comformoreinfo!"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Visit https://example.com for more info!"

    def test_email_in_response(self):
        """Test email addresses in responses."""
        events = [
            AgentResponse(content="Contact us at support@example.com today!"),
            AgentSpeechSent(content="Contactusatsupport@example.comtoday!"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "Contact us at support@example.com today!"

    def test_case_sensitivity_preserved(self):
        """Test that case is preserved correctly."""
        events = [
            AgentResponse(content="HELLO World! HoW ArE yOu?"),
            AgentSpeechSent(content="HELLOWorld!HoWArEyOu?"),
        ]

        context = ConversationContext(events=events, system_prompt="")
        committed = context.get_committed_events()

        assert len(committed) == 1
        assert isinstance(committed[0], AgentResponse)
        assert committed[0].content == "HELLO World! HoW ArE yOu?"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
