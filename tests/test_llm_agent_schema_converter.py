"""
Unit tests for schema_converter module.

Tests TypedDict support, nested objects, and strict mode handling.
"""

from typing import Annotated, TypedDict

import pytest

from line.llm_agent.schema_converter import (
    _is_typeddict,
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


class MixedItem(TypedDict):
    """A TypedDict with both required and optional fields."""

    name: str
    notes: str


# Simulate mixed required/optional (NotRequired not available in Python 3.10)
MixedItem.__required_keys__ = frozenset({"name"})
MixedItem.__optional_keys__ = frozenset({"notes"})


class NestedItem(TypedDict):
    """A TypedDict containing another TypedDict."""

    item: SimpleItem
    tags: list[str]


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
        schema = python_type_to_json_schema(ItemWithOptional, strict=False)

        assert schema["type"] == "object"
        assert "properties" in schema
        # With total=False, no fields are required
        assert "required" not in schema or schema.get("required") == []
        # additionalProperties: false is not set in non-strict mode
        assert "additionalProperties" not in schema

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
# Tests for Annotated support in TypedDict and standalone
# =============================================================================


class TestAnnotatedSchema:
    """Tests for Annotated type handling in schema generation."""

    def test_annotated_standalone(self):
        """Annotated[str, 'desc'] produces string schema with description."""
        schema = python_type_to_json_schema(Annotated[str, "A city name"])
        assert schema == {"type": "string", "description": "A city name"}

    def test_annotated_integer(self):
        """Annotated[int, 'desc'] produces integer schema with description."""
        schema = python_type_to_json_schema(Annotated[int, "Count of items"])
        assert schema == {"type": "integer", "description": "Count of items"}

    def test_annotated_no_string_metadata(self):
        """Annotated with non-string metadata produces schema without description."""
        schema = python_type_to_json_schema(Annotated[str, 42])
        assert schema == {"type": "string"}

    def test_annotated_first_string_wins(self):
        """Only the first string in Annotated metadata becomes the description."""
        schema = python_type_to_json_schema(Annotated[str, "first", "second"])
        assert schema["description"] == "first"

    def test_typeddict_with_annotated_fields(self):
        """TypedDict fields with Annotated carry descriptions into the schema."""

        class Ticket(TypedDict):
            subject: Annotated[str, "Short summary of the issue"]
            priority: str

        schema = python_type_to_json_schema(Ticket)
        assert schema["properties"]["subject"] == {
            "type": "string",
            "description": "Short summary of the issue",
        }
        # Non-annotated field has no description
        assert schema["properties"]["priority"] == {"type": "string"}

    def test_nested_typeddict_with_annotated_fields(self):
        """Annotated descriptions survive nested TypedDict recursion."""

        class Address(TypedDict):
            street: Annotated[str, "Street address"]
            city: Annotated[str, "City name"]

        class Person(TypedDict):
            name: str
            address: Address

        schema = python_type_to_json_schema(Person)
        address_props = schema["properties"]["address"]["properties"]
        assert address_props["street"] == {
            "type": "string",
            "description": "Street address",
        }
        assert address_props["city"] == {
            "type": "string",
            "description": "City name",
        }

    def test_annotated_list_of_typeddict(self):
        """Annotated works on list items too."""

        class Item(TypedDict):
            name: Annotated[str, "Item name"]

        schema = python_type_to_json_schema(list[Item])
        item_props = schema["items"]["properties"]
        assert item_props["name"] == {"type": "string", "description": "Item name"}

    def test_annotated_does_not_override_existing_description(self):
        """If the inner type already has a description, Annotated doesn't overwrite."""
        # Annotated[Annotated[str, "inner"], "outer"] — inner wins
        inner = Annotated[str, "inner desc"]
        outer = Annotated[inner, "outer desc"]
        schema = python_type_to_json_schema(outer)
        assert schema["description"] == "inner desc"


# =============================================================================
# Tests for optional TypedDict strict mode auto-disable
# =============================================================================


class TestOptionalTypedDictStrict:
    """Optional TypedDict fields auto-disable strict instead of raising."""

    def test_all_optional_typeddict_strict_does_not_raise(self):
        """TypedDict with total=False no longer raises in strict mode."""
        schema = python_type_to_json_schema(ItemWithOptional, strict=True)
        assert schema["type"] == "object"
        # Strict auto-disabled — no additionalProperties
        assert "additionalProperties" not in schema

    def test_mixed_typeddict_strict_does_not_raise(self):
        """TypedDict with mixed required/optional auto-disables strict."""
        schema = python_type_to_json_schema(MixedItem, strict=True)
        assert schema["type"] == "object"
        assert schema["required"] == ["name"]
        assert "additionalProperties" not in schema

    def test_all_required_typeddict_strict_still_works(self):
        """TypedDict with all required fields still gets strict constraints."""
        schema = python_type_to_json_schema(SimpleItem, strict=True)
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {"name", "quantity"}

    def test_nested_optional_typeddict_auto_disables(self):
        """Strict is disabled for nested optional TypedDicts too."""

        class Inner(TypedDict, total=False):
            x: str
            y: str

        class Outer(TypedDict):
            inner: Inner

        schema = python_type_to_json_schema(Outer, strict=True)
        # Outer is all-required → strict
        assert schema["additionalProperties"] is False
        # Inner is all-optional → strict auto-disabled
        inner_schema = schema["properties"]["inner"]
        assert "additionalProperties" not in inner_schema


# =============================================================================
# Tests for dict type warnings
# =============================================================================


class TestDictErrors:
    """Tests for errors when using dict types in strict mode."""

    def test_plain_dict_raises_in_strict_mode(self):
        """Should raise ValueError when using plain dict type in strict mode."""
        with pytest.raises(ValueError) as exc_info:
            python_type_to_json_schema(dict, strict=True)
        assert "TypedDict" in str(exc_info.value)

    def test_plain_dict_ok_in_non_strict_mode(self):
        """Should not raise when using plain dict type in non-strict mode."""
        schema = python_type_to_json_schema(dict, strict=False)
        assert schema == {"type": "object"}

    def test_list_dict_raises_in_strict_mode(self):
        """Should raise ValueError when using list[dict] type in strict mode."""
        with pytest.raises(ValueError) as exc_info:
            python_type_to_json_schema(list[dict], strict=True)
        assert "TypedDict" in str(exc_info.value)

    def test_list_dict_ok_in_non_strict_mode(self):
        """Should not raise when using list[dict] type in non-strict mode."""
        schema = python_type_to_json_schema(list[dict], strict=False)
        assert schema["type"] == "array"
        assert schema["items"] == {"type": "object"}

    def test_pep604_dict_union_raises_in_strict_mode(self):
        """dict | None follows Optional[dict] strict-mode behavior."""
        with pytest.raises(ValueError) as exc_info:
            python_type_to_json_schema(dict | None, strict=True)
        assert "TypedDict" in str(exc_info.value)

    def test_pep604_dict_union_ok_in_non_strict_mode(self):
        """dict | None converts to the non-None schema in non-strict mode."""
        schema = python_type_to_json_schema(dict | None, strict=False)
        assert schema == {"type": "object"}


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

    def test_tool_with_optional_typeddict_auto_disables_strict(self):
        """Optional-key TypedDict auto-disables strict instead of raising."""

        @loopback_tool
        async def with_opts(
            ctx: ToolEnv,
            payload: Annotated[ItemWithOptional, "Optional keys only"],
        ):
            """Use optional-key TypedDict."""
            pass

        # Should NOT raise — strict is auto-disabled for the optional TypedDict
        spec = function_tool_to_litellm(with_opts)
        params = spec["function"]["parameters"]
        payload = params["properties"]["payload"]
        # The TypedDict itself should not have additionalProperties (strict disabled)
        assert "additionalProperties" not in payload
        # All fields are optional so no required list
        assert "required" not in payload or payload.get("required") == []

    def test_tool_with_list_dict_auto_disables_strict(self):
        """list[dict] disables strict mode instead of raising."""

        @loopback_tool
        async def add_items(
            ctx: ToolEnv,
            items: Annotated[list[dict], "Items"],
        ):
            """Add items."""
            pass

        schema = function_tool_to_litellm(add_items)

        fn = schema["function"]
        assert fn.get("strict") is not True
        assert "additionalProperties" not in fn["parameters"]
        items_schema = fn["parameters"]["properties"]["items"]
        assert items_schema == {
            "type": "array",
            "items": {"type": "object"},
            "description": "Items",
        }

    def test_tool_with_optional_param_dict_type_succeeds(self):
        """Tool with optional param containing dict should NOT raise.

        When a tool has optional parameters, strict mode is disabled at the
        top-level because OpenAI strict mode requires all properties to be
        in 'required'. This test ensures that strict checks for nested types
        (like dict) are also disabled - strict validation should not run
        if strict mode will ultimately be disabled.
        """
        from typing import Optional

        @loopback_tool
        async def with_optional_dict(
            ctx: ToolEnv,
            name: Annotated[str, "Name"],
            options: Annotated[Optional[dict], "Optional options"] = None,
        ):
            """A tool with optional dict parameter."""
            pass

        # This should NOT raise - strict mode is disabled due to optional param
        schema = function_tool_to_litellm(with_optional_dict)

        # Verify strict mode is disabled (no "strict": True in payload)
        assert schema["function"].get("strict") is not True
        # Verify additionalProperties is not set at top level
        params = schema["function"]["parameters"]
        assert "additionalProperties" not in params

    def test_tool_with_pep604_list_dict_auto_disables_strict(self):
        """list[dict] | None disables strict mode like Optional[list[dict]]."""

        @loopback_tool
        async def add_items(
            ctx: ToolEnv,
            items: Annotated[list[dict] | None, "Items"],
        ):
            """Add items."""
            pass

        schema = function_tool_to_litellm(add_items)

        fn = schema["function"]
        assert fn.get("strict") is not True
        items_schema = fn["parameters"]["properties"]["items"]
        assert items_schema == {
            "type": "array",
            "items": {"type": "object"},
            "description": "Items",
        }


# =============================================================================
# Tests: json_schema_to_python_type round-trip
# =============================================================================


class TestJsonSchemaToPythonType:
    """Verify json_schema_to_python_type produces types that round-trip correctly."""

    def _round_trip(self, schema, strict=False):
        """Convert schema → Python type → back to schema."""
        from line.llm_agent.tools.http_server_tool_utils import json_schema_to_python_type

        py_type = json_schema_to_python_type(schema)
        return python_type_to_json_schema(py_type, strict=strict)

    def test_string(self):
        assert self._round_trip({"type": "string"}) == {"type": "string"}

    def test_integer(self):
        assert self._round_trip({"type": "integer"}) == {"type": "integer"}

    def test_number(self):
        assert self._round_trip({"type": "number"}) == {"type": "number"}

    def test_boolean(self):
        assert self._round_trip({"type": "boolean"}) == {"type": "boolean"}

    def test_string_with_description(self):
        result = self._round_trip({"type": "string", "description": "A name"})
        assert result == {"type": "string", "description": "A name"}

    def test_string_enum(self):
        result = self._round_trip({"type": "string", "enum": ["low", "high"]})
        assert result == {"type": "string", "enum": ["low", "high"]}

    def test_string_enum_with_description(self):
        result = self._round_trip({"type": "string", "enum": ["a", "b"], "description": "Pick one"})
        assert result["enum"] == ["a", "b"]
        assert result["description"] == "Pick one"

    def test_array_bare(self):
        assert self._round_trip({"type": "array"}) == {"type": "array"}

    def test_array_with_items(self):
        result = self._round_trip({"type": "array", "items": {"type": "string"}})
        assert result == {"type": "array", "items": {"type": "string"}}

    def test_array_with_described_items(self):
        result = self._round_trip({"type": "array", "items": {"type": "integer", "description": "An ID"}})
        assert result == {
            "type": "array",
            "items": {"type": "integer", "description": "An ID"},
        }

    def test_freeform_object(self):
        result = self._round_trip({"type": "object"}, strict=False)
        assert result == {"type": "object"}

    def test_object_with_properties(self):
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string", "description": "Full name"},
                "age": {"type": "integer"},
            },
        }
        result = self._round_trip(schema)
        assert result["type"] == "object"
        assert result["required"] == ["name"]
        assert result["properties"]["name"] == {
            "type": "string",
            "description": "Full name",
        }
        assert result["properties"]["age"] == {"type": "integer"}

    def test_nested_object(self):
        schema = {
            "type": "object",
            "required": ["address"],
            "properties": {
                "address": {
                    "type": "object",
                    "required": ["street"],
                    "properties": {
                        "street": {"type": "string", "description": "Street address"},
                        "city": {"type": "string"},
                    },
                },
            },
        }
        result = self._round_trip(schema)
        addr = result["properties"]["address"]
        assert addr["type"] == "object"
        assert addr["required"] == ["street"]
        assert addr["properties"]["street"] == {
            "type": "string",
            "description": "Street address",
        }

    def test_object_all_optional_no_strict(self):
        schema = {
            "type": "object",
            "properties": {
                "x": {"type": "string"},
                "y": {"type": "integer"},
            },
        }
        result = self._round_trip(schema, strict=True)
        # All optional → strict auto-disabled
        assert "additionalProperties" not in result


# =============================================================================
# Tests: http_server_tool schema conversion
# =============================================================================


class TestWebhookToolSchema:
    """Verify http_server_tool FunctionTool converts to valid provider schemas."""

    @pytest.fixture
    def ticket_tool(self, monkeypatch):
        from line.llm_agent.tools.system import http_server_tool

        monkeypatch.setenv("SUPPORT_API_KEY", "test-key")
        return http_server_tool(
            name="create_ticket",
            description="Creates a support ticket.",
            url="https://example.com/api/tickets",
            method="POST",
            request_body_schema={
                "type": "object",
                "required": ["subject"],
                "properties": {
                    "subject": {"type": "string", "description": "Short summary."},
                    "source": {"type": "string", "constant_value": "voice_agent"},
                },
            },
            auth={"Authorization": "Bearer ${SUPPORT_API_KEY}"},
        )

    @pytest.fixture
    def multi_param_tool(self):
        from line.llm_agent.tools.system import http_server_tool

        return http_server_tool(
            name="update_order",
            description="Update an order.",
            url="https://example.com/orders/{order_id}",
            method="PATCH",
            request_body_schema={
                "type": "object",
                "required": ["status"],
                "properties": {
                    "status": {"type": "string", "description": "New status."},
                    "note": {"type": "string", "description": "Optional note."},
                },
            },
            query_params_schema={
                "type": "object",
                "required": [],
                "properties": {
                    "notify": {"type": "boolean", "description": "Send notification."},
                },
            },
        )

    def test_litellm_format_structure(self, ticket_tool):
        schema = function_tool_to_litellm(ticket_tool)
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "create_ticket"
        assert fn["description"] == "Creates a support ticket."
        assert "parameters" in fn

    def test_litellm_constant_value_excluded(self, ticket_tool):
        schema = function_tool_to_litellm(ticket_tool)
        props = schema["function"]["parameters"]["properties"]
        assert "subject" in props
        assert "source" not in props

    def test_litellm_nested_object_constants_preserve_object_schema(self):
        from line.llm_agent.tools.system import http_server_tool

        tool = http_server_tool(
            name="create_ticket",
            description="Create a ticket.",
            url="https://example.com/api/tickets",
            method="POST",
            request_body_schema={
                "type": "object",
                "required": ["ticket"],
                "properties": {
                    "ticket": {
                        "type": "object",
                        "required": ["subject", "channel"],
                        "properties": {
                            "subject": {
                                "type": "string",
                                "description": "Ticket summary.",
                            },
                            "channel": {
                                "type": "string",
                                "constant_value": "voice",
                            },
                        },
                    },
                },
            },
        )

        schema = function_tool_to_litellm(tool)
        ticket_schema = schema["function"]["parameters"]["properties"]["ticket"]

        assert ticket_schema["type"] == "object"
        assert ticket_schema["required"] == ["subject"]
        assert set(ticket_schema["properties"]) == {"subject"}
        assert ticket_schema["properties"]["subject"]["type"] == "string"
        assert ticket_schema["properties"]["subject"]["description"] == "Ticket summary."
        assert ticket_schema["additionalProperties"] is False

    def test_litellm_nested_freeform_object_disables_strict(self):
        from line.llm_agent.tools.system import http_server_tool

        tool = http_server_tool(
            name="create_ticket",
            description="Create a ticket.",
            url="https://example.com/api/tickets",
            method="POST",
            request_body_schema={
                "type": "object",
                "required": ["ticket"],
                "properties": {
                    "ticket": {
                        "type": "object",
                        "required": ["metadata"],
                        "properties": {
                            "metadata": {
                                "type": "object",
                                "description": "Arbitrary customer metadata.",
                            },
                        },
                    },
                },
            },
        )

        schema = function_tool_to_litellm(tool)
        fn = schema["function"]
        assert fn.get("strict") is not True
        assert "additionalProperties" not in fn["parameters"]
        metadata_schema = fn["parameters"]["properties"]["ticket"]["properties"]["metadata"]
        assert metadata_schema == {
            "type": "object",
            "description": "Arbitrary customer metadata.",
        }

    def test_litellm_nested_freeform_object_array_disables_strict(self):
        from line.llm_agent.tools.system import http_server_tool

        tool = http_server_tool(
            name="create_ticket",
            description="Create a ticket.",
            url="https://example.com/api/tickets",
            method="POST",
            request_body_schema={
                "type": "object",
                "required": ["ticket"],
                "properties": {
                    "ticket": {
                        "type": "object",
                        "required": ["events"],
                        "properties": {
                            "events": {
                                "type": "array",
                                "description": "Arbitrary event payloads.",
                                "items": {"type": "object"},
                            },
                        },
                    },
                },
            },
        )

        schema = function_tool_to_litellm(tool)
        fn = schema["function"]
        assert fn.get("strict") is not True
        events_schema = fn["parameters"]["properties"]["ticket"]["properties"]["events"]
        assert events_schema == {
            "type": "array",
            "items": {"type": "object"},
            "description": "Arbitrary event payloads.",
        }

    def test_litellm_required_correct(self, ticket_tool):
        schema = function_tool_to_litellm(ticket_tool)
        params = schema["function"]["parameters"]
        assert params["required"] == ["subject"]

    def test_litellm_strict_all_required(self, ticket_tool):
        schema = function_tool_to_litellm(ticket_tool, strict=True)
        fn = schema["function"]
        # All params are required → strict should be enabled
        assert fn.get("strict") is True
        assert fn["parameters"].get("additionalProperties") is False

    def test_litellm_strict_disabled_with_optional(self, multi_param_tool):
        schema = function_tool_to_litellm(multi_param_tool, strict=True)
        fn = schema["function"]
        # Has optional params (note, notify) → strict auto-disabled
        assert fn.get("strict") is not True

    def test_openai_responses_api_format(self, ticket_tool):
        from line.llm_agent.schema_converter import function_tool_to_openai

        schema = function_tool_to_openai(ticket_tool, responses_api=True)
        # Responses API: type/name/description at top level
        assert schema["type"] == "function"
        assert schema["name"] == "create_ticket"
        assert "parameters" in schema
        assert "source" not in schema["parameters"]["properties"]

    def test_openai_chat_completions_format(self, ticket_tool):
        from line.llm_agent.schema_converter import function_tool_to_openai

        schema = function_tool_to_openai(ticket_tool, responses_api=False)
        # Chat Completions: nested under "function" key
        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "create_ticket"
        assert "source" not in schema["function"]["parameters"]["properties"]

    def test_multi_source_params_all_present(self, multi_param_tool):
        schema = function_tool_to_litellm(multi_param_tool)
        props = schema["function"]["parameters"]["properties"]
        # URL template, body, and query params all present
        assert "order_id" in props
        assert "status" in props
        assert "note" in props
        assert "notify" in props

    def test_param_types_preserved(self, multi_param_tool):
        schema = function_tool_to_litellm(multi_param_tool)
        props = schema["function"]["parameters"]["properties"]
        assert props["order_id"]["type"] == "string"
        assert props["status"]["type"] == "string"
        assert props["notify"]["type"] == "boolean"

    def test_param_descriptions_preserved(self, multi_param_tool):
        schema = function_tool_to_litellm(multi_param_tool)
        props = schema["function"]["parameters"]["properties"]
        assert props["status"]["description"] == "New status."
        assert props["notify"]["description"] == "Send notification."
