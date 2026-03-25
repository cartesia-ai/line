"""
Unit tests for schema_converter module.

Tests TypedDict support, nested objects, and strict mode handling.
"""

from typing import Annotated, TypedDict
import warnings

import pytest

from line.llm_agent.schema_converter import (
    _is_typeddict,
    _json_schema_violates_openai_strict,
    function_tool_to_litellm,
    python_type_to_json_schema,
)
from line.llm_agent.tools.decorators import loopback_tool
from line.llm_agent.tools.utils import ToolEnv

# =============================================================================
# TypedDict Definitions for Testing
# =============================================================================


class SimpleItem(TypedDict):
    """A simple TypedDict with basic types."""

    name: str
    quantity: int


class ItemWithOptional(TypedDict, total=False):
    """A TypedDict with all optional fields."""

    name: str
    notes: str


class NestedItem(TypedDict):
    """A TypedDict containing another TypedDict."""

    item: SimpleItem
    tags: list[str]


class ComplexItem(TypedDict):
    """A TypedDict with various field types."""

    id: str
    count: int
    price: float
    active: bool
    tags: list[str]
    metadata: dict  # This should generate a warning


# =============================================================================
# Tests for _is_typeddict
# =============================================================================


class TestIsTypedDict:
    """Tests for the _is_typeddict helper function."""

    def test_detects_typeddict(self):
        """Should return True for TypedDict classes."""
        assert _is_typeddict(SimpleItem) is True
        assert _is_typeddict(ItemWithOptional) is True
        assert _is_typeddict(NestedItem) is True

    def test_rejects_regular_dict(self):
        """Should return False for regular dict type."""
        assert _is_typeddict(dict) is False

    def test_rejects_regular_class(self):
        """Should return False for regular classes."""

        class RegularClass:
            name: str

        assert _is_typeddict(RegularClass) is False

    def test_rejects_basic_types(self):
        """Should return False for basic types."""
        assert _is_typeddict(str) is False
        assert _is_typeddict(int) is False
        assert _is_typeddict(list) is False


# =============================================================================
# Tests for python_type_to_json_schema with TypedDict
# =============================================================================


class TestTypedDictSchema:
    """Tests for TypedDict schema generation."""

    def test_simple_typeddict(self):
        """Should generate proper schema for simple TypedDict."""
        schema = python_type_to_json_schema(SimpleItem)

        assert schema["type"] == "object"
        assert "properties" in schema
        assert schema["properties"]["name"] == {"type": "string"}
        assert schema["properties"]["quantity"] == {"type": "integer"}
        assert schema["required"] == ["name", "quantity"]
        assert schema["additionalProperties"] is False

    def test_typeddict_without_strict(self):
        """Should not add additionalProperties when strict=False."""
        schema = python_type_to_json_schema(SimpleItem, strict=False)

        assert schema["type"] == "object"
        assert "additionalProperties" not in schema

    def test_optional_fields_typeddict(self):
        """Should handle TypedDict with total=False (all optional)."""
        schema = python_type_to_json_schema(ItemWithOptional)

        assert schema["type"] == "object"
        assert "properties" in schema
        # With total=False, no fields are required
        assert "required" not in schema or schema.get("required") == []
        # additionalProperties: false is invalid with OpenAI strict unless every key is required
        assert "additionalProperties" not in schema
        assert _json_schema_violates_openai_strict(schema)

    def test_nested_typeddict(self):
        """Should handle nested TypedDict correctly."""
        schema = python_type_to_json_schema(NestedItem)

        assert schema["type"] == "object"
        assert "properties" in schema

        # Check nested item
        item_schema = schema["properties"]["item"]
        assert item_schema["type"] == "object"
        assert item_schema["properties"]["name"] == {"type": "string"}
        assert item_schema["properties"]["quantity"] == {"type": "integer"}
        assert item_schema["additionalProperties"] is False

        # Check tags
        assert schema["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}

    def test_list_of_typeddict(self):
        """Should generate proper schema for list[TypedDict]."""
        schema = python_type_to_json_schema(list[SimpleItem])

        assert schema["type"] == "array"
        assert schema["items"]["type"] == "object"
        assert schema["items"]["properties"]["name"] == {"type": "string"}
        assert schema["items"]["additionalProperties"] is False


# =============================================================================
# Tests for dict type warnings
# =============================================================================


class TestDictWarnings:
    """Tests for warnings when using dict types."""

    def test_plain_dict_warning(self):
        """Should warn when using plain dict type."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            schema = python_type_to_json_schema(dict)

            assert len(w) == 1
            assert "TypedDict" in str(w[0].message)
            assert schema == {"type": "object"}

    def test_list_dict_warning(self):
        """Should warn when using list[dict] type."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            schema = python_type_to_json_schema(list[dict])

            assert len(w) == 1
            assert "TypedDict" in str(w[0].message)
            assert schema["type"] == "array"
            assert schema["items"] == {"type": "object"}


# =============================================================================
# Tests for function_tool_to_litellm with TypedDict
# =============================================================================


class TestFunctionToolWithTypedDict:
    """Tests for converting tools with TypedDict parameters."""

    def test_tool_with_typeddict_list(self):
        """Should generate proper schema for tool with list[TypedDict]."""

        @loopback_tool
        async def add_items(
            ctx: ToolEnv,
            items: Annotated[list[SimpleItem], "Items to add"],
        ):
            """Add items to order."""
            pass

        schema = function_tool_to_litellm(add_items)

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "add_items"
        assert schema["function"]["strict"] is True

        params = schema["function"]["parameters"]
        assert params["additionalProperties"] is False

        items_schema = params["properties"]["items"]
        assert items_schema["type"] == "array"
        assert items_schema["items"]["type"] == "object"
        assert items_schema["items"]["additionalProperties"] is False
        assert "name" in items_schema["items"]["properties"]
        assert "quantity" in items_schema["items"]["properties"]

    def test_tool_with_nested_typeddict(self):
        """Should handle tools with nested TypedDict parameters."""

        @loopback_tool
        async def process_nested(
            ctx: ToolEnv,
            data: Annotated[NestedItem, "Nested data structure"],
        ):
            """Process nested data."""
            pass

        schema = function_tool_to_litellm(process_nested)
        params = schema["function"]["parameters"]

        data_schema = params["properties"]["data"]
        assert data_schema["type"] == "object"
        assert data_schema["additionalProperties"] is False

        # Check nested item has additionalProperties: false
        item_schema = data_schema["properties"]["item"]
        assert item_schema["additionalProperties"] is False

    def test_tool_with_required_optional_typeddict_strict_raises(self):
        """Required param typed as total=False TypedDict cannot satisfy OpenAI strict."""

        @loopback_tool
        async def with_opts(
            ctx: ToolEnv,
            payload: Annotated[ItemWithOptional, "Optional keys only"],
        ):
            """Use optional-key TypedDict."""
            pass

        with pytest.raises(ValueError, match="cannot satisfy OpenAI strict mode"):
            function_tool_to_litellm(with_opts)

    def test_tool_with_required_optional_typeddict_non_strict_ok(self):
        """Same tool succeeds when strict=False."""

        @loopback_tool
        async def with_opts(
            ctx: ToolEnv,
            payload: Annotated[ItemWithOptional, "Optional keys only"],
        ):
            """Use optional-key TypedDict."""
            pass

        spec = function_tool_to_litellm(with_opts, strict=False)
        assert spec["function"].get("strict") is not True
        params = spec["function"]["parameters"]
        assert "additionalProperties" not in params
        payload = params["properties"]["payload"]
        assert "additionalProperties" not in payload

    def test_tool_with_list_dict_strict_raises(self):
        """list[dict] cannot satisfy strict mode."""

        @loopback_tool
        async def add_items(
            ctx: ToolEnv,
            items: Annotated[list[dict], "Items"],
        ):
            """Add items."""
            pass

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="cannot satisfy OpenAI strict mode"):
                function_tool_to_litellm(add_items)
