"""Validation and schema helpers for webhook_tool.

Extracted from system.py to keep webhook_tool() concise.
"""

import os
import re
from typing import Any, Dict

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


def error(name: str, msg: str) -> ValueError:
    return ValueError(f"webhook_tool(name={name!r}): {msg}")


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
    """Return validated URL template parameter names from a webhook URL."""
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
# Schema validation
# ---------------------------------------------------------------------------


def validate_schema(
    name: str,
    schema: Dict[str, Any],
    label: str,
    *,
    allow_omitted_type: bool = False,
    is_query: bool = False,
) -> None:
    """Validate a body_schema or query_params_schema dict."""
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
    unknown_required = set(required) - set(props)
    if unknown_required:
        raise error(name, f'{label}["required"] lists fields not in properties: {unknown_required}.')
    for prop_name, prop_def in props.items():
        _validate_property(name, prop_def, f"{label}.properties.{prop_name}", is_query=is_query)


def _validate_property(
    name: str, prop_def: Dict[str, Any], path: str, *, is_query: bool
) -> None:
    """Validate a single property definition."""
    if not isinstance(prop_def, dict):
        raise error(name, f"{path} must be a dict, got {type(prop_def).__name__}.")
    json_type = prop_def.get("type")
    if json_type is not None and json_type not in TYPE_MAP:
        raise error(
            name,
            f"{path} has unknown type {json_type!r}. "
            f"Expected one of: {', '.join(sorted(TYPE_MAP))}.",
        )

    # Query params must be scalars — objects and arrays can't be serialized
    # to query strings.
    if is_query and json_type in ("object", "array"):
        raise error(
            name,
            f"{path} has type={json_type!r}, which is not supported in "
            f"query_params_schema. Query parameters must be scalar "
            f"(string, integer, number, or boolean).",
        )

    # Validate constant_value type. bool is a subclass of int in Python,
    # but JSON schema treats them separately.
    if "constant_value" in prop_def:
        cv = prop_def["constant_value"]
        if is_query and (cv is None or isinstance(cv, (dict, list))):
            raise error(
                name, f"{path}.constant_value must be a scalar string, number, or boolean."
            )
        if json_type is not None:
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
                    f"{path} declares type={json_type!r} but "
                    f"constant_value={cv!r} is {type(cv).__name__}.",
                )

    # Reject constant_value inside array items or union branches.
    if "constant_value" not in prop_def:
        for key in ("items", "anyOf", "oneOf", "allOf"):
            nested = prop_def.get(key)
            if nested is None:
                continue
            entries = [nested] if isinstance(nested, dict) else nested
            for entry in entries:
                if isinstance(entry, dict) and "constant_value" in entry:
                    raise error(
                        name,
                        f"{path}.{key} contains constant_value, which is "
                        f"only supported on direct object properties.",
                    )

    # Recurse into nested object schemas.
    if has_object_properties(prop_def):
        validate_schema(name, prop_def, path, allow_omitted_type=True, is_query=is_query)


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
            child_vis, child_req, child_const = strip_constants(
                prop_def["properties"], nested_req
            )
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


def build_param_info(
    prop_name: str, prop_def: Dict[str, Any], required_list: list[str]
) -> "ParameterInfo":
    """Build a ParameterInfo from a JSON schema property definition."""
    from line.llm_agent.tools.utils import ParameterInfo

    json_type = "object" if has_object_properties(prop_def) else prop_def.get("type", "string")
    py_type = TYPE_MAP[json_type][0] if json_type in TYPE_MAP else str
    return ParameterInfo(
        name=prop_name,
        type_annotation=py_type,
        description=prop_def.get("description", ""),
        required=prop_name in required_list,
        enum=prop_def.get("enum"),
        json_schema=prop_def,
    )


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Merge *overlay* into *base*, recursing into nested dicts."""
    merged = dict(base)
    for k, v in overlay.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged
