"""
Similarity checking utilities for conversation evaluation.

This module provides functions for comparing strings and dictionaries with semantic
similarity checking using AI models.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union  # noqa: F401

from google.genai import Client
from google.genai.types import GenerateContentConfig


@dataclass
class SimilarityResult:
    is_success: Optional[bool]  # None = if not applicable
    error: Optional[str]  # Error message if not successful


def is_statement_pattern(s: str) -> bool:
    """Check if string is a statement pattern like <mentions something>."""
    return s.strip().startswith("<") and s.strip().endswith(">")


def extract_statement(s: str) -> str:
    """Extract statement content from pattern by removing < and >."""
    return s.strip()[1:-1]


def check_string_statement(statement: str, actual_text: str) -> SimilarityResult:
    """Check if actual text matches a statement pattern.

    Args:
        statement: The statement description (without < >)
        actual_text: The actual text to check against the statement

    Returns:
        None if text matches statement, error message string if not
    """
    client = Client()

    prompt = f"""
    Check if the following text matches this statement/requirement:

    Statement: "{statement}"
    Text: "{actual_text}"

    Instructions:
    - Respond with "YES" if the text matches the statement, or "NO: [reason]" if it doesn't.
    - The text should contain or express the concept described in the statement.

    Examples:
    - Statement: "mentions SOC-2 compliance" vs Text: "Our security audit passed SOC-2 requirements" → YES
    - Statement: "mentions SOC-2 compliance" vs Text: "We follow security best practices" → NO:
        Doesn't mention SOC-2
    - Statement: "asks for user name" vs Text: "What's your name?" → YES
    - Statement: "asks for user name" vs Text: "How old are you?" → NO: Asks for age, not name
    """

    config = GenerateContentConfig(
        temperature=0.1,
    )

    response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt, config=config)
    response_text = response.text.strip() if response.text else ""

    if response_text.upper().startswith("YES"):
        return SimilarityResult(is_success=True, error=None)
    elif response_text.upper().startswith("NO"):
        reason = response_text[3:].strip().lstrip(":").strip()
        return SimilarityResult(is_success=False, error=reason)
    else:
        return SimilarityResult(
            is_success=False,
            error=f"Unexpected response format from statement check: {response_text}",
        )


def is_similar_str(a: str, b: str) -> SimilarityResult:
    """Check if two strings have the same meaning using Gemini with special rule support.

    Special Rules:
        - "*" wildcard: Matches any string content (either a or b can be "*")
        - Statement patterns: Strings like "<mentions SOC-2 compliance>" match text containing that concept

    Args:
        a: First string to compare
        b: Second string to compare

    Returns:
        None if strings are similar, error message string if not
    """
    # * means any string is allowed
    if a == "*" or b == "*":
        return SimilarityResult(is_success=True, error=None)

    # Handle statement patterns
    result = is_similar_via_statement_pattern(a, b)
    if result.is_success is not None:
        return result

    # Handle single text comparision
    return is_similar_via_single_text_comparison(a, b)


def is_similar_via_statement_pattern(a: str, b: str) -> SimilarityResult:
    a_is_statement = is_statement_pattern(a)
    b_is_statement = is_statement_pattern(b)

    if a_is_statement or b_is_statement:
        # At least one is a statement pattern
        if a_is_statement and b_is_statement:
            # Both are statement patterns - compare the statements themselves
            statement_a = extract_statement(a)
            statement_b = extract_statement(b)
            return is_similar_str(statement_a, statement_b)  # Recursive call without < >

        # One is a statement, one is actual text
        statement = extract_statement(a) if a_is_statement else extract_statement(b)
        actual_text = b if a_is_statement else a

        return check_string_statement(statement, actual_text)

    return SimilarityResult(is_success=None, error=None)


def is_similar_via_single_text_comparison(a: str, b: str) -> SimilarityResult:
    # First check if strings are equal after basic normalization
    if a.lower().strip() == b.lower().strip():
        return SimilarityResult(is_success=True, error=None)

    client = Client()

    prompt = f"""
    Compare these two strings and determine if they have the same or very similar meaning:

    String A: "{a}"
    String B: "{b}"

    Rules:
    - Respond with "YES" if they have the same meaning, or "NO: [reason]" if they don't.
    - Consider paraphrasing, synonyms, and different ways of expressing the same concept.
    - Ignore filler prefixes like "Now", "Okay", "Got it", "Thank you", "Finally", "Sounds good", etc.
    - Affirmative phrases like "yes", "that is correct" or "correct" are similar
    - For alphanumeric matching, you may allow mismatches on spacing
    - For alphanumeric matching, you may allow matching when spelled out (e.g. 1 is equivalent to "one", 2 is equivalent to "two", etc.)
    - For alphanumeric matching, you may allow semantic matching between spelled out numbers with spaces or concatenated string of digits

    Examples:
    - "What's your name?" vs "Can you tell me your name?" → YES
    - "What's your name?" vs "What's your age?" → NO: Different information being requested
    - "You are verified" vs "Your identity is confirmed" → YES
    - "Now, what's your Name?" vs "Thank you, what's your name?" → YES
    - "Hello" vs "Goodbye" → NO: Opposite greetings with different meanings
    - "one    two three  four" versus "1234" → YES
    """  # noqa: E501

    config = GenerateContentConfig(
        temperature=0.1,  # Low temperature for consistent results
    )

    response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt, config=config)

    response_text = response.text.strip() if response.text else ""

    if response_text.upper().startswith("YES"):
        return SimilarityResult(is_success=True, error=None)
    elif response_text.upper().startswith("NO"):
        # Extract and return reason
        reason = response_text[3:].strip().lstrip(":").strip()
        return SimilarityResult(is_success=False, error=reason)
    else:
        # Fallback in case of unexpected response format
        return SimilarityResult(
            is_success=False,
            error=f"Unexpected response format from similarity check: {response_text}\n"
            f'String A: "{a}"\nString B: "{b}"',
        )


def is_similar_text(a: Union[List[str], str], b: Union[List[str], str]) -> SimilarityResult:
    """Given two texts that are lists, check that at least one element from a is similar to one element from b.

    Args:
        a: First list of strings to compare
        b: Second list of strings to compare

    Returns:
        SimilarityResult indicating if the lists are similar
    """  # noqa: E501
    a = [a] if isinstance(a, str) else a
    b = [b] if isinstance(b, str) else b

    if not a and not b:
        raise RuntimeError("Both lists are empty")
    if not a or not b:
        return SimilarityResult(is_success=False, error=f"One list is empty: a={a}, b={b}")

    # Check if any element from 'a' is similar to any element from 'b'
    for a_item in a:
        for b_item in b:
            result = is_similar_str(a_item, b_item)
            if result.is_success:
                return SimilarityResult(is_success=True, error=None)

    if len(a) == 1 and len(b) == 1:
        return SimilarityResult(is_success=False, error=f"{a} != {b}")
    else:
        return SimilarityResult(
            is_success=False, error=f"No similar elements found the following two lists: a={a}, b={b}"
        )


def is_similar_dict(actual: Dict, expected: Dict) -> SimilarityResult:
    """Recursively check if two dictionaries are similar.

    Uses string similarity checking for string values and recursive comparison for nested dicts.

    Args:
        actual: The actual dictionary
        expected: The expected dictionary

    Returns:
        None if dictionaries are similar, error message string if not
    """
    # Check if keys match
    actual_keys = set(actual.keys())
    expected_keys = set(expected.keys())

    if actual_keys != expected_keys:
        missing_keys = expected_keys - actual_keys
        extra_keys = actual_keys - expected_keys
        error_parts = []
        if missing_keys:
            error_parts.append(f"missing keys: {list(missing_keys)}")
        if extra_keys:
            error_parts.append(f"extra keys: {list(extra_keys)}")
        return SimilarityResult(
            is_success=False,
            error=f"Key mismatch - {', '.join(error_parts)}",
        )

    # Check each key-value pair
    for key in expected_keys:
        actual_value = actual[key]
        expected_value = expected[key]

        # Skip validation if expected value is None
        if expected_value is None:
            continue

        # Handle string values with similarity checking
        if isinstance(expected_value, str) and isinstance(actual_value, str):
            result = is_similar_str(actual_value, expected_value)
            if result.is_success is False:
                return SimilarityResult(
                    is_success=False,
                    error=f"String value mismatch for key '{key}': {result.error}",
                )

        # Handle nested dictionaries
        elif isinstance(expected_value, dict) and isinstance(actual_value, dict):
            error = is_similar_dict(actual_value, expected_value)
            if error.is_success is False:
                return SimilarityResult(
                    is_success=False,
                    error=f"Nested dict mismatch for key '{key}': {error}",
                )

        # Handle other types with exact comparison
        else:
            if actual_value != expected_value:
                return SimilarityResult(
                    is_success=False,
                    error=f"Value mismatch for key '{key}': expected {expected_value}, got {actual_value}",
                )

    return SimilarityResult(is_success=True, error=None)
