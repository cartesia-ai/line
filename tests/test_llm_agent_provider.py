"""Tests for line.llm_agent.provider — specifically _feed_tool_args."""

from line.llm_agent.provider import _feed_tool_args

# ---------------------------------------------------------------------------
# OpenAI / Anthropic: incremental fragments → concatenate
# ---------------------------------------------------------------------------


class TestIncrementalConcatenation:
    """OpenAI and Anthropic stream arguments as small JSON fragments."""

    def test_single_fragment_complete_object(self):
        """A single fragment that is already a complete JSON object."""
        state = _feed_tool_args(None, '{"city": "Tokyo"}')
        assert state.args == '{"city": "Tokyo"}'
        assert state.depth == 0

    def test_two_fragments(self):
        """Two incremental fragments that together form a complete object."""
        s1 = _feed_tool_args(None, '{"ci')
        assert s1.args == '{"ci'
        assert s1.depth == 1

        s2 = _feed_tool_args(s1, 'ty": "Tokyo"}')
        assert s2.args == '{"city": "Tokyo"}'
        assert s2.depth == 0

    def test_many_small_fragments(self):
        """Simulates typical OpenAI streaming with many tiny fragments."""
        fragments = ["{", '"name"', ": ", '"', "Alice", '"', ", ", '"age"', ": ", "30", "}"]
        state = None
        for frag in fragments:
            state = _feed_tool_args(state, frag)

        assert state.args == '{"name": "Alice", "age": 30}'
        assert state.depth == 0
        assert state.in_string is False
        assert state.escape_next is False

    def test_empty_object(self):
        """Tool with no parameters: {}."""
        s1 = _feed_tool_args(None, "{")
        assert s1.depth == 1

        s2 = _feed_tool_args(s1, "}")
        assert s2.args == "{}"
        assert s2.depth == 0


# ---------------------------------------------------------------------------
# Gemini: complete objects repeated → replace
# ---------------------------------------------------------------------------


class TestGeminiReplace:
    """Gemini sends complete JSON objects, possibly repeated or growing."""

    def test_identical_resend(self):
        """Same complete object sent twice — should replace, not double."""
        s1 = _feed_tool_args(None, '{"city": "Tokyo"}')
        assert s1.depth == 0

        s2 = _feed_tool_args(s1, '{"city": "Tokyo"}')
        assert s2.args == '{"city": "Tokyo"}'
        assert s2.depth == 0

    def test_progressive_update(self):
        """Gemini sends progressively larger complete objects."""
        s1 = _feed_tool_args(None, '{"city": "Tokyo"}')
        assert s1.depth == 0

        s2 = _feed_tool_args(s1, '{"city": "Tokyo", "date": "2025-01-01"}')
        assert s2.args == '{"city": "Tokyo", "date": "2025-01-01"}'
        assert s2.depth == 0

    def test_three_resends(self):
        """Multiple consecutive replacements."""
        s = _feed_tool_args(None, '{"a": 1}')
        s = _feed_tool_args(s, '{"a": 1, "b": 2}')
        s = _feed_tool_args(s, '{"a": 1, "b": 2, "c": 3}')
        assert s.args == '{"a": 1, "b": 2, "c": 3}'
        assert s.depth == 0


# ---------------------------------------------------------------------------
# Nested objects
# ---------------------------------------------------------------------------


class TestNestedObjects:
    def test_nested_braces_incremental(self):
        """Nested objects tracked correctly across incremental fragments."""
        s = _feed_tool_args(None, '{"a": {')
        assert s.depth == 2

        s = _feed_tool_args(s, '"b": 1}')
        assert s.depth == 1

        s = _feed_tool_args(s, "}")
        assert s.args == '{"a": {"b": 1}}'
        assert s.depth == 0

    def test_deeply_nested(self):
        """Three levels of nesting in one fragment."""
        obj = '{"a": {"b": {"c": 1}}}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0


# ---------------------------------------------------------------------------
# Braces and special chars inside strings
# ---------------------------------------------------------------------------


class TestBracesInStrings:
    def test_braces_inside_string_value(self):
        """Braces inside a JSON string must not affect depth tracking."""
        obj = '{"template": "{hello}"}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0

    def test_braces_in_string_across_fragments(self):
        s = _feed_tool_args(None, '{"t": "{')
        assert s.depth == 1  # only the outer { counts
        assert s.in_string is True

        s = _feed_tool_args(s, 'x}"}')
        assert s.args == '{"t": "{x}"}'
        assert s.depth == 0
        assert s.in_string is False


# ---------------------------------------------------------------------------
# Escape sequences
# ---------------------------------------------------------------------------


class TestEscapeSequences:
    def test_escaped_quote_in_value(self):
        r"""Value containing \" — must not close the string."""
        obj = r'{"msg": "say \"hi\""}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0
        assert s.in_string is False

    def test_escaped_backslash_before_quote(self):
        r"""Value ending with \\ followed by closing quote."""
        # JSON: {"p": "\\"} — the value is a single backslash
        obj = '{"p": "\\\\"}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0
        assert s.in_string is False

    def test_escaped_backslash_then_escaped_quote(self):
        r"""\\\" inside a string: literal backslash + literal quote."""
        # JSON: {"p": "\\\""} — the value is \"
        obj = '{"p": "\\\\\\""}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0
        assert s.in_string is False

    def test_unicode_escape(self):
        r"""\uXXXX escapes must not confuse the parser."""
        obj = '{"ch": "\\u0022"}'  # \u0022 is "
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0
        assert s.in_string is False

    def test_unicode_escape_brace_codepoint(self):
        r"""\u007B is { — must not count as a real brace."""
        obj = '{"ch": "\\u007B"}'
        s = _feed_tool_args(None, obj)
        assert s.depth == 0
        assert s.in_string is False

    def test_escape_spanning_fragments(self):
        r"""Backslash at end of one fragment, escaped char in next."""
        s = _feed_tool_args(None, '{"m": "a\\')
        assert s.in_string is True
        assert s.escape_next is True

        s = _feed_tool_args(s, '"b"}')
        assert s.args == '{"m": "a\\"b"}'
        assert s.depth == 0
        assert s.in_string is False
        assert s.escape_next is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_first_fragment_is_empty_string(self):
        """Empty first fragment should be a no-op, next fragment concatenates."""
        s = _feed_tool_args(None, "")
        assert s.args == ""
        assert s.depth == 0

        # Next real fragment should concatenate (args is falsy → else branch)
        s = _feed_tool_args(s, '{"a": 1}')
        assert s.args == '{"a": 1}'
        assert s.depth == 0

    def test_state_none_always_starts_fresh(self):
        s = _feed_tool_args(None, '{"x": 1}')
        assert s.args == '{"x": 1}'
        assert s.depth == 0

    def test_negative_depth_does_not_trigger_replace(self):
        """Malformed JSON with extra } — depth goes negative, no false replace."""
        s = _feed_tool_args(None, '{"a": 1}}')
        assert s.depth == -1
        # Next fragment should concatenate (depth != 0)
        s = _feed_tool_args(s, "extra")
        assert s.args == '{"a": 1}}extra'
        assert s.depth == -1

    def test_boolean_and_null_values(self):
        obj = '{"flag": true, "empty": null, "off": false}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0

    def test_array_values(self):
        """Arrays inside object values — [ ] should not affect brace depth."""
        obj = '{"items": [1, 2, {"nested": true}]}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0
