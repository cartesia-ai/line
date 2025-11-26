"""
UserBridge - Event routing bridge for user communication via harness.

Treats user as a regular bus participant with WebSocket transport.
Handles bidirectional communication and authorization:
- Inbound: WebSocket messages → bus events (via input routing)
- Outbound: Bus events → WebSocket messages (via event handlers)
- Authorization: Only authorized agent can send to user

Input routing eliminates manual async task management by automatically
converting continuous WebSocket input into bus events.

Examples:
    Basic user bridge setup::

        user_bridge = create_user_bridge(harness, authorized_node="coordinator")
        bus.register_bridge("user", user_bridge)
        await user_bridge.start_input_routing()  # Start WebSocket → bus routing

    Sending to user::

        # Type-safe event creation with validation
        message_event = UserMessageEvent(node_id="coordinator", content="Hello!")
        await bus.broadcast("coordinator", message_event)

        tool_event = UserToolCallEvent(
            node_id="coordinator",
            name="search",
            args={},
            result={"status": "success"}
        )
        await bus.broadcast("coordinator", tool_event)
"""

import dataclasses
from typing import TYPE_CHECKING, Union

from loguru import logger
from pydantic import BaseModel

from line.bridge import Bridge
from line.bus import Message
from line.events import (
    AgentError,
    AgentResponse,
    Authorize,
    DTMFOutputEvent,
    EndCall,
    EventType,
    LogMetric,
    ToolCall,
    ToolResult,
    TransferCall,
)

if TYPE_CHECKING:
    from line.harness import ConversationHarness


def create_user_bridge(harness: "ConversationHarness", authorized_node: str) -> Bridge:
    """
    Create event routing bridge for user communication.

    Sets up bidirectional user communication with automatic input routing.
    Uses NodeBridge input routing to eliminate manual WebSocket task management.

    Args:
        harness: ConversationHarness instance with get() method for WebSocket input.
        authorized_node: Agent ID authorized to communicate with user.

    Returns:
        Configured NodeBridge with input routing and user communication handlers.

    Examples:
        >>> harness = ConversationHarness(websocket, shutdown_event)
        >>> bridge = create_user_bridge(harness, "coordinator")
        >>> bus.register_bridge("user", bridge)
        >>> await bridge.start_input_routing()  # Start WebSocket input routing
    """

    async def send_message(message: Message):
        """Send text message to user."""
        event: AgentResponse = message.event
        logger.debug(f"Sending user message: {event.content}")
        return await harness.send_message(event.content)

    async def send_tool_call(message: Message):
        """Send tool call result to user."""
        event: Union[ToolCall, ToolResult] = message.event
        if isinstance(event, ToolResult):
            result = event.result_str if event.result_str is not None else event.error
        else:
            result = None
        return await harness.send_tool_call(event.tool_name, event.tool_args, event.tool_call_id, result)

    async def send_end_call(message: Message):
        """End the call."""
        return await harness.end_call()

    async def send_error(message: Message):
        """Send error message to user."""
        event: AgentError = message.event
        return await harness.send_error(event.error)

    async def send_transfer_call(message: Message):
        """Transfer call to destination."""
        event: TransferCall = message.event
        return await harness.transfer_call(event.target_phone_number, event.timeout_s)

    async def send_log_metric(message: Message):
        """Log metric via harness."""
        event: LogMetric = message.event
        return await harness.log_metric(event.name, event.value)

    async def send_dtmf(message: Message):
        """Send DTMF event to harness."""
        event: DTMFOutputEvent = message.event
        return await harness.send_dtmf(event.button)

    bridge = (
        Bridge(harness)
        .with_input_routing(harness)  # Enable WebSocket → bus event routing
        .authorize(authorized_node, "tools")  # Allow both conversation and tools agents
    )

    (
        bridge.on(AgentResponse)
        .map(send_message)
        .on(ToolCall)
        .map(send_tool_call)
        .on(ToolResult)
        .map(send_tool_call)
        .on(EndCall)
        .map(send_end_call)
        .on(AgentError)
        .map(send_error)
        .on(TransferCall)
        .map(send_transfer_call)
        .on(LogMetric)
        .map(send_log_metric)
        .on(DTMFOutputEvent)
        .map(send_dtmf)
    )

    # Add authorization handler after creation.
    bridge.on(Authorize).map(lambda msg: bridge.authorize(msg.event.agent))

    return bridge


def register_observability_event(bridge: Bridge, harness: "ConversationHarness", event_type: EventType):
    """
    Register an event type for observability logging.

    For Pydantic BaseModel types, automatically uses the class name as event name
    and model_dump() as metadata. For dataclass types, uses the class name as event
    name and asdict() as metadata. For other types, validates that the event type
    has a `to_log_event` method and sets up routing to send log events to the harness.

    Args:
        bridge: The bridge to register the event on
        harness: The ConversationHarness to send log events to
        event_type: The event type to register

    Raises:
        ValueError: If the event type is not a BaseModel/dataclass and doesn't have a `to_log_event` method

    Examples:
        >>> bridge = create_user_bridge(harness, "coordinator")
        >>> register_observability_event(bridge, harness, MyBaseModelEvent)  # Uses class name + model_dump()
        >>> register_observability_event(bridge, harness, MyDataclassEvent)  # Uses class name + asdict()
        >>> register_observability_event(bridge, harness, MyCustomEvent)     # Uses to_log_event() method
    """
    # Check if the event type is a BaseModel subclass or dataclass
    is_base_model = isinstance(event_type, type) and issubclass(event_type, BaseModel)
    is_dataclass = isinstance(event_type, type) and dataclasses.is_dataclass(event_type)

    if not is_base_model and not is_dataclass and not hasattr(event_type, "to_log_event"):
        raise ValueError(
            f"Event type {event_type} must be a pydantic BaseModel subclass, "
            f"dataclass, or have a 'to_log_event' method."
        )

    async def send_log_event(message: Message):
        """Convert event to log format and send to harness."""
        if isinstance(message.event, BaseModel):
            # For BaseModel types, use class name as event and model_dump as metadata
            event_name = type(message.event).__name__
            metadata = message.event.model_dump()
            await harness.log_event(event_name, metadata)
        elif dataclasses.is_dataclass(message.event):
            # For dataclass types, use class name as event and asdict as metadata
            event_name = type(message.event).__name__
            metadata = dataclasses.asdict(message.event)
            await harness.log_event(event_name, metadata)
        else:
            # For other types, use the to_log_event method
            event_data = message.event.to_log_event()
            if not isinstance(event_data, dict) or "event" not in event_data:
                logger.error(f"Invalid log event data from {type(message.event).__name__}: {event_data}")
                return

            event_name = event_data.get("event")
            metadata = event_data.get("metadata", None)
            await harness.log_event(event_name, metadata)

    # Register the event handler
    bridge.on(event_type).map(send_log_event)
