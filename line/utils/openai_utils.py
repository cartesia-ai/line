import json
from typing import Any, Callable, Dict, List

from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
)

from line.events import (
    AgentResponse,
    EventInstance,
    EventType,
    ToolCall,
    ToolResult,
    UserTranscriptionReceived,
)


def convert_messages_to_openai(
    events: List[EventInstance],
    handlers: Dict[EventType, Callable[[EventInstance], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Convert conversation messages to OpenAI format.

    With OpenAI, all messages need to be in the context.

    Args:
        events: List of events.
        handlers: Dictionary of event type to handler function.
            The handler function should return a dictionary of OpenAI-formatted messages.

    Returns:
        List of messages in OpenAI format
    """
    handlers = handlers or {}

    openai_messages = []
    for event in events:
        event_type = type(event)
        if event_type in handlers:
            openai_messages.append(handlers[event_type](event))
            continue

        if isinstance(event, AgentResponse):
            openai_messages.append({"role": "assistant", "content": event.content})
        elif isinstance(event, UserTranscriptionReceived):
            if event.language:
                content = f"({event.language}) {event.content}"
            else:
                content = event.content
            openai_messages.append({"role": "user", "content": content})
        elif isinstance(event, ToolCall):
            if event.raw_response:
                openai_messages.append(event.raw_response)
        elif isinstance(event, ToolResult):
            if event.tool_call_id:
                openai_messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": event.tool_call_id,
                        "output": event.result_str,
                    }
                )

    return openai_messages


def extract_text_from_response(response: Response) -> str:
    """Extract all text content from OpenAI response output.

    Args:
        response: OpenAI response object

    Returns:
        Combined text content from the response
    """
    text_content = ""
    for msg in response.output:
        if isinstance(msg, ResponseOutputMessage):
            for content in msg.content:
                if isinstance(content, ResponseOutputText):
                    text_content += content.text
                elif isinstance(content, ResponseOutputRefusal):
                    text_content += content.refusal
    return text_content


def extract_tool_calls_from_response(response: Response) -> List[ToolCall]:
    """Extract function tool calls from OpenAI response output.

    Args:
        response: OpenAI response object

    Returns:
        List of tool calls with name and arguments
    """
    tool_calls = []
    for msg in response.output:
        if isinstance(msg, ResponseFunctionToolCall):
            tool_calls.append(
                ToolCall(
                    tool_name=msg.name,
                    tool_args=json.loads(msg.arguments),
                    tool_call_id=msg.id,
                    raw_response=msg.model_dump(),
                )
            )
    return tool_calls


def has_tool_calls(response: Response) -> bool:
    """Check if response contains any tool calls.

    Args:
        response: OpenAI response object

    Returns:
        True if response contains tool calls, False otherwise
    """
    for msg in response.output:
        if isinstance(msg, ResponseFunctionToolCall):
            return True
    return False
