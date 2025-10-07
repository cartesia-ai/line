import os

import pytest

from line.evals.similarity_utils import is_similar_dict, is_similar_str, is_similar_text

# Skip tests if GEMINI_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY environment variable not set",
)


def test_wildcard_matching():
    """Test wildcard (*) matches any string."""
    result = is_similar_str("*", "Hello world")
    assert result.is_success
    assert result.error is None


def test_statement_pattern_matching():
    """Test statement patterns like <mentions something>."""
    result = is_similar_str("<mentions compliance>", "We are SOC-2 compliant")
    assert result.is_success
    assert result.error is None

    result = is_similar_str("<mentions SOC-2>", "We follow general security practices")
    assert not result.is_success
    assert result.error is not None


def test_semantic_similarity():
    """Test AI-powered semantic matching."""
    result = is_similar_str("What's your name?", "Can you tell me your name?")
    assert result.is_success
    assert result.error is None

    result = is_similar_str("What's your name?", "What's your age?")
    assert not result.is_success
    assert result.error is not None


def test_dict_similarity():
    """Test dictionary comparison with nested structures."""
    actual = {"user": {"name": "John"}, "status": "active"}
    expected = {"user": {"name": "John"}, "status": "active"}
    result = is_similar_dict(actual, expected)
    assert result.is_success
    assert result.error is None

    # Test mismatch
    actual_bad = {"user": {"name": "Jane"}, "status": "active"}
    result = is_similar_dict(actual_bad, expected)
    assert not result.is_success
    assert result.error is not None


def test_list_similarity():
    """Test list comparison with at least one matching element."""
    # Matches itself, via str
    result = is_similar_text("Hello", "Hello")
    assert result.is_success
    assert result.error is None

    # Matches itself via list
    result = is_similar_text(["Hello"], ["Hello"])
    assert result.is_success
    assert result.error is None

    # Should not match itself
    result = is_similar_text("Hello", "Bye")
    assert not result.is_success
    assert result.error is not None

    # Matches similarity
    result = is_similar_text(["Hello"], ["Hi"])
    assert result.is_success
    assert result.error is None

    # Tests matches wildcard
    result = is_similar_text(["Hello"], ["*"])
    assert result.is_success
    assert result.error is None

    # Matches similar with multiple
    result = is_similar_text(
        ["Hello", "oranges"],
        [
            "apples",
            "Hi",
        ],
    )
    assert result.is_success
    assert result.error is None

    # Test no match
    result = is_similar_text(["apples", "oranges"], ["cats", "dogs"])
    assert not result.is_success
    assert result.error is not None
