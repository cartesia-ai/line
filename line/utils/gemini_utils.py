"""
Utility functions for converting agent tools and messages to Gemini format
"""

import json
from typing import Any, Callable, Dict, List
import warnings

from google.genai import types
from loguru import logger

from line.events import (
    AgentResponse,
    EventInstance,
    EventType,
    ToolResult,
    UserTranscriptionReceived,
)

# Suppress aiohttp ResourceWarnings about unclosed sessions
# These are caused by the Gemini client's internal session management
warnings.filterwarnings("ignore", message="unclosed", category=ResourceWarning)


def convert_messages_to_gemini(
    events: List[EventInstance],
    handlers: Dict[EventType, Callable[[EventInstance], Dict[str, Any]]] = None,
    text_events_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convert conversation events to Gemini format.

    Note:
        This method only handles these event types:
            - `AgentResponse`
            - `UserTranscriptionReceived`
            - `ToolCall`
            - `ToolResult`
        To convert other events, add handlers.

    Args:
        messages: List of events.
        handlers: Dictionary of event type to handler function.
            The handler function should return a dictionary of Gemini-formatted messages.
        text_events_only: Whether to only include User and Agent messages in the output.
    Returns:
        List of Gemini-formatted messages
    """
    handlers = handlers or {}

    if not types:
        raise ImportError("google.genai is required for Gemini integration")

    gemini_messages = []

    for event in events:
        event_type = type(event)

        if text_events_only and event_type not in (AgentResponse, UserTranscriptionReceived):
            continue

        if event_type in handlers:
            gemini_messages.append(handlers[event_type](event))
        elif isinstance(event, AgentResponse):
            gemini_messages.append(types.ModelContent(parts=[types.Part.from_text(text=event.content)]))
        elif isinstance(event, UserTranscriptionReceived):
            gemini_messages.append(types.UserContent(parts=[types.Part.from_text(text=event.content)]))
        elif isinstance(event, ToolResult):
            # Gemini 400s if a ToolResult is the first message in the context, so don't add it:
            if not gemini_messages:
                logger.debug(f"Skipping orphaned ToolResult at start of context: {event.tool_name}")
                continue

            tool_name = event.tool_name or "unknown_tool"

            function_response = types.Part.from_function_response(
                name=tool_name, response={"output": event.result}
            )

            gemini_messages.append(types.ModelContent(parts=[function_response]))

    logger.debug(f"Converted {len(events)} messages to {len(gemini_messages)} Gemini messages")
    return gemini_messages


def log_gemini_messages(message: str, gemini_messages, statistics=None):
    """
    Log Gemini messages in a nice formatted way.
    Analogous to log_conversation_history but for Gemini message format.
    """
    gemini_messages_str = message_to_str(gemini_messages)

    to_log = f"\n====== BEGIN {message} ======\n"
    to_log += gemini_messages_str
    if statistics:
        to_log += f"\n[Statistics: {statistics}]\n"
    to_log += f"======== END {message} =========\n"

    logger.info(f"Logging Gemini messages: {to_log}")


def message_to_str(gemini_messages) -> str:
    """
    Convert a list of Gemini messages to a formatted string representation.
    Similar to _get_conversation_history_str but for Gemini message format.
    """
    ans = ""
    prev_message = None

    for message in gemini_messages:
        # Get the role from the message type
        if isinstance(message, types.UserContent):
            role = "user"
        elif isinstance(message, types.ModelContent):
            role = "model"
        else:
            role = "unknown"

        # Serialize all parts in the message
        serialized_parts = [serialize_part(part) for part in message.parts]
        content = (
            serialized_parts[0] if len(serialized_parts) == 1 else json.dumps(serialized_parts, indent=2)
        )

        ans += f"{role}: {content}\n"

        # Extra line between user and model turns
        if role == "user" and prev_message is not None and (isinstance(prev_message, types.ModelContent)):
            ans += "\n"
        prev_message = message

    return ans


def serialize_part(part: types.Part) -> str:
    """
    Serialize a Gemini Part object to a readable string representation.
    Handles text, function calls, and function responses.
    """
    if hasattr(part, "text") and part.text is not None and part.text.strip() != "":
        return part.text
    elif hasattr(part, "function_call") and part.function_call is not None:
        func_call = part.function_call
        args_str = json.dumps(dict(func_call.args), indent=2) if func_call.args else "{}"
        return f"[function call] {func_call.name}({args_str})"
    elif hasattr(part, "function_response") and part.function_response is not None:
        func_resp = part.function_response
        response_str = json.dumps(dict(func_resp.response), indent=2) if func_resp.response else "{}"

        return f"[function response] {func_resp.name}: {response_str}"
    else:
        return f"[unsupported part type] {str(part)}"
