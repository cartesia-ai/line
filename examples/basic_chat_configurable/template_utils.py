import re


def substitute_template_variables(template: str | None, metadata: dict | None) -> str | None:
    """Substitute {{key}} patterns with metadata[key] if found, otherwise preserve original.

    Args:
        template: String containing {{variable}} patterns to substitute, or None
        metadata: Dict of key-value pairs for substitution

    Returns:
        None if template is None, otherwise template with matched patterns replaced.
    """
    if template is None:
        return None

    if not metadata:
        return template

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        return str(metadata.get(var_name, match.group(0)))

    return re.sub(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}", replacer, template)
