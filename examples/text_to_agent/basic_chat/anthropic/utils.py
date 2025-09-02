from typing import Dict, Any, List, Callable
from line.events import EventInstance, EventType
from line.tools.system_tools import ToolDefinition
from line.events import AgentResponse, UserTranscriptionReceived, ToolCall, ToolResult


def to_anthropic_tool(tool: ToolDefinition) -> Dict[str, object]:
    oai_tool = tool.to_openai_tool()
    return {
        "name": tool.name(),
        "description": tool.description(),
        "input_schema": oai_tool.get("parameters"),
    }

def convert_messages_to_anthropic(
    events: List[EventInstance],
    handlers: Dict[EventType, Callable[[EventInstance], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Convert conversation messages to Anthropic format.

    With Anthropic, all messages need to be in the context.

    Args:
        events: List of events.
        handlers: Dictionary of event type to handler function.
            The handler function should return a dictionary of Anthropic-formatted messages.

    Returns:
        List of messages in Anthropic format
    """
    handlers = handlers or {}

    anthropic_messages = []
    for event in events:
        event_type = type(event)
        if event_type in handlers:
            anthropic_messages.append(handlers[event_type](event))
            continue

        if isinstance(event, AgentResponse):
            anthropic_messages.append({"role": "assistant", "content": event.content})
        elif isinstance(event, UserTranscriptionReceived):
            anthropic_messages.append({"role": "user", "content": event.content})
        elif isinstance(event, ToolCall):
            if event.raw_response:
                anthropic_messages.append(event.raw_response)
        elif isinstance(event, ToolResult):
            if event.tool_call_id:
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": event.tool_call_id,
                                "content": event.result_str,
                            }
                        ],
                    }
                )

    return anthropic_messages