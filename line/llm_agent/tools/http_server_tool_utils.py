"""Validation and schema helpers for http_server_tool.

Extracted from system.py to keep http_server_tool() concise.
"""

import os
import re
from typing import Annotated, Any, Dict, Literal, Type

from line.llm_agent.tools.utils import ParameterInfo

_URL_PARAM_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

# Maps JSON schema types to (python_type_for_params, allowed_constant_types).
TYPE_MAP: Dict[str, tuple] = {
    "string": (str, str),
    "integer": (int, int),
    "number": (float, (int, float)),
    "boolean": (bool, bool),
    "array": (list, list),
    "object": (dict, dict),
}

VALID_METHODS = {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}

_QUERY_SCALAR_TYPES = {"string", "integer", "number", "boolean"}


def error(name: str, msg: str) -> ValueError:
    return ValueError(f"http_server_tool(name={name!r}): {msg}")


def has_object_properties(schema: Dict[str, Any]) -> bool:
    """Return True for object schemas that declare nested properties."""
    return "properties" in schema and schema.get("type") in (None, "object")


def resolve_env_vars(name: str, value: str) -> str:
    """Replace ``${ENV_VAR}`` placeholders in *value* with ``os.environ`` values."""

    def _replace(m: re.Match) -> str:
        var = m.group(1)
        try:
            return os.environ[var]
        except KeyError:
            raise error(name, f"environment variable ${{{var}}} required by auth is not set.") from None

    return re.sub(r"\$\{(\w+)\}", _replace, value)


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


def parse_path_params(name: str, url: str) -> list[str]:
    """Return validated URL template parameter names from the URL."""
    brace_depth = 0
    for ch in url:
        if ch == "{":
            brace_depth += 1
            if brace_depth > 1:
                raise error(name, f"url has nested braces, which is not supported: {url!r}")
        elif ch == "}":
            brace_depth -= 1
            if brace_depth < 0:
                raise error(name, f"url has unmatched closing brace: {url!r}")
    if brace_depth != 0:
        raise error(name, f"url has unmatched opening brace: {url!r}")

    params = re.findall(r"\{([^{}]*)\}", url)
    seen: set[str] = set()
    for param in params:
        if not _URL_PARAM_NAME_RE.fullmatch(param):
            raise error(
                name,
                f"url template variable {param!r} is invalid. "
                "Expected 1-64 characters matching [A-Za-z0-9_.-].",
            )
        if param in seen:
            raise error(name, f"url template variable {param!r} appears more than once.")
        seen.add(param)
    return params


# ---------------------------------------------------------------------------
# Schema validation — shared structure checks
# ---------------------------------------------------------------------------


def _validate_schema_structure(
    name: str, schema: Dict[str, Any], label: str, *, allow_omitted_type: bool = False
) -> tuple[Dict[str, Any], list[str]]:
    """Validate top-level schema shape. Returns (properties, required)."""
    if not isinstance(schema, dict):
        raise error(name, f"{label} must be a dict, got {type(schema).__name__}.")
    schema_type = schema.get("type")
    if schema_type != "object" and not (allow_omitted_type and schema_type is None):
        raise error(name, f'{label} must have "type": "object", got "type": {schema_type!r}.')
    props = schema.get("properties")
    if props is None:
        raise error(name, f'{label} must have a "properties" key.')
    if not isinstance(props, dict):
        raise error(name, f'{label}["properties"] must be a dict, got {type(props).__name__}.')
    required = schema.get("required", [])
    if not isinstance(required, list):
        raise error(name, f'{label}["required"] must be a list, got {type(required).__name__}.')
    unknown = set(required) - set(props)
    if unknown:
        raise error(name, f'{label}["required"] lists fields not in properties: {unknown}.')
    return props, required


def _validate_constant_value_type(name: str, prop_def: Dict[str, Any], path: str) -> None:
    """Check that constant_value matches the declared JSON schema type."""
    json_type = prop_def.get("type")
    cv = prop_def["constant_value"]
    if json_type is None:
        return
    if json_type in ("integer", "number"):
        valid = isinstance(cv, (int, float)) and not isinstance(cv, bool)
        if json_type == "integer":
            valid = type(cv) is int
    else:
        _, allowed = TYPE_MAP[json_type]
        valid = isinstance(cv, allowed)
    if not valid:
        raise error(
            name,
            f"{path} declares type={json_type!r} but constant_value={cv!r} is {type(cv).__name__}.",
        )


# ---------------------------------------------------------------------------
# Body schema validation — supports nesting, objects, arrays
# ---------------------------------------------------------------------------


def validate_body_schema(name: str, schema: Dict[str, Any], label: str) -> None:
    """Validate a request_body_schema dict."""
    props, _ = _validate_schema_structure(name, schema, label)
    for prop_name, prop_def in props.items():
        _validate_body_property(name, prop_def, f"{label}.properties.{prop_name}")


def _validate_body_property(name: str, prop_def: Dict[str, Any], path: str) -> None:
    """Validate a single body property definition. Recurses into nested objects."""
    if not isinstance(prop_def, dict):
        raise error(name, f"{path} must be a dict, got {type(prop_def).__name__}.")
    json_type = prop_def.get("type")
    if json_type is not None and json_type not in TYPE_MAP:
        raise error(
            name,
            f"{path} has unknown type {json_type!r}. Expected one of: {', '.join(sorted(TYPE_MAP))}.",
        )
    if "constant_value" in prop_def:
        _validate_constant_value_type(name, prop_def, path)
    if has_object_properties(prop_def):
        props, _ = _validate_schema_structure(name, prop_def, path, allow_omitted_type=True)
        for prop_name, child_def in props.items():
            _validate_body_property(name, child_def, f"{path}.properties.{prop_name}")


# ---------------------------------------------------------------------------
# Query schema validation — flat scalars only
# ---------------------------------------------------------------------------


def validate_query_schema(name: str, schema: Dict[str, Any], label: str) -> None:
    """Validate a query_params_schema dict. Properties must be scalar."""
    props, _ = _validate_schema_structure(name, schema, label)
    for prop_name, prop_def in props.items():
        path = f"{label}.properties.{prop_name}"
        if not isinstance(prop_def, dict):
            raise error(name, f"{path} must be a dict, got {type(prop_def).__name__}.")
        json_type = prop_def.get("type")
        # Must be a scalar type (or omitted for constant_value-only fields).
        if json_type is not None and json_type not in _QUERY_SCALAR_TYPES:
            raise error(
                name,
                f"{path} has type={json_type!r}, which is not supported in "
                f"query_params_schema. Query parameters must be scalar "
                f"(string, integer, number, or boolean).",
            )
        # No nested or compound structures allowed.
        if any(k in prop_def for k in ("properties", "items", "anyOf", "oneOf", "allOf")):
            raise error(
                name,
                f"{path} contains nested structure, which is not supported in "
                f"query_params_schema. Query parameters must be scalar.",
            )
        if "constant_value" in prop_def:
            cv = prop_def["constant_value"]
            if cv is None or isinstance(cv, (dict, list)):
                raise error(name, f"{path}.constant_value must be a scalar string, number, or boolean.")
            _validate_constant_value_type(name, prop_def, path)


# ---------------------------------------------------------------------------
# Constant stripping
# ---------------------------------------------------------------------------


def strip_constants(
    properties: Dict[str, Any], required: list[str]
) -> tuple[Dict[str, Any], list[str], Dict[str, Any]]:
    """Separate visible properties from constant_value properties.

    Returns (visible_props, visible_required, constants) where constants
    is a nested dict mirroring the structure for injection at call time.
    """
    visible: Dict[str, Any] = {}
    constants: Dict[str, Any] = {}
    for prop_name, prop_def in properties.items():
        if "constant_value" in prop_def:
            constants[prop_name] = prop_def["constant_value"]
        elif has_object_properties(prop_def):
            nested_req = list(prop_def.get("required", []))
            child_vis, child_req, child_const = strip_constants(prop_def["properties"], nested_req)
            if child_const:
                constants[prop_name] = child_const
            if child_vis:
                cleaned = dict(prop_def)
                cleaned.setdefault("type", "object")
                cleaned["properties"] = child_vis
                if child_req:
                    cleaned["required"] = child_req
                else:
                    cleaned.pop("required", None)
                visible[prop_name] = cleaned
        else:
            visible[prop_name] = prop_def
    visible_required = [r for r in required if r in visible]
    return visible, visible_required, constants


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def build_param_info(prop_name: str, prop_def: Dict[str, Any], required_list: list[str]) -> ParameterInfo:
    """Build a ParameterInfo from a JSON schema property definition.

    Converts the JSON schema to a Python type via json_schema_to_python_type,
    which round-trips through python_type_to_json_schema to produce the
    provider schema. Description and enum are carried by the Python type
    (via Annotated and Literal), so they don't need separate fields.
    """
    py_type = json_schema_to_python_type(prop_def)
    return ParameterInfo(
        name=prop_name,
        type_annotation=py_type,
        required=prop_name in required_list,
        description="",
    )


_BASIC_TYPE_MAP: Dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}

_typeddict_counter = 0


def _next_typeddict_name() -> str:
    global _typeddict_counter
    _typeddict_counter += 1
    return f"_schema_{_typeddict_counter}"


def _with_description(base: Type, description: str | None) -> Type:
    return Annotated[base, description] if description else base


def json_schema_to_python_type(schema: Dict[str, Any]) -> Type:
    """Convert a JSON schema property dict to a Python type.

    The returned type round-trips through python_type_to_json_schema
    to reproduce the original schema (including nested descriptions).
    """
    from typing import TypedDict

    json_type = schema.get("type", "string")
    description = schema.get("description")
    enum = schema.get("enum")

    if enum is not None:
        base = Literal[tuple(enum)]  # type: ignore[valid-type]
        return _with_description(base, description)

    if has_object_properties(schema):
        props = schema.get("properties", {})
        required_keys = set(schema.get("required", []))
        fields = {name: json_schema_to_python_type(defn) for name, defn in props.items()}
        td = TypedDict(_next_typeddict_name(), fields)  # type: ignore[call-overload]
        td.__required_keys__ = frozenset(required_keys & set(fields))
        td.__optional_keys__ = frozenset(set(fields) - required_keys)
        return _with_description(td, description)

    if json_type == "array":
        items = schema.get("items")
        if items and isinstance(items, dict):
            base = list[json_schema_to_python_type(items)]  # type: ignore[misc]
        else:
            base = list
        return _with_description(base, description)

    if json_type == "object":
        return _with_description(dict, description)

    base = _BASIC_TYPE_MAP.get(json_type, str)
    return _with_description(base, description)


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Merge *overlay* into *base*, recursing into nested dicts."""
    merged = dict(base)
    for k, v in overlay.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged
