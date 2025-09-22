"""
Bus - Typed event routing system for agent communication.

Routes typed events between agents using broadcast or request-response patterns.
Provides type-safe event handling with Pydantic validation and fluent subscription API.

Examples:
    Create and start bus::

        bus = Bus()
        await bus.start()

    Register agents::

        bus.register_bridge("clipboard", clipboard_bridge)

    Send typed events::

        await bus.broadcast(FormCompleted(node_id="intake", status="done"))
        response = await bus.call(ToolCall(node_id="agent", tool_name="calculator"))

    Subscribe to events::
        bridge.on(UserTranscriptionReceived, source="intake").map(handler)
"""

import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import uuid

from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from line.bridge import Bridge

from line.events import AgentHandoff, Authorize, ToolCall


class Message(BaseModel):
    """Message sent between agents through the bus."""

    # The unique identifier for the message.
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # The source of the message.
    source: str
    # The event being sent.
    event: Any

    # The timestamp when the message was created.
    timestamp: float = Field(default_factory=time.time)

    def __str__(self) -> str:
        return (
            f"f{type(self)}(id={self.id}, "
            f"source={self.source}, "
            f"event={type(self.event)}({self.event}), "
            f"timestamp={self.timestamp})"
        )


class Bus:
    """
    Routes messages between agents in memory.

    Handles broadcast events and request-response calls.
    Provides event-driven communication between agents and bridges.
    """

    # ================================================================
    # Lifecycle Methods
    # ================================================================

    def __init__(self, max_queue_size: int = 1000):
        """
        Create agent bus.

        Args:
            max_queue_size: Max messages before dropping.
        """
        self.max_queue_size = max_queue_size  # Prevents memory overflow during high message volume.

        self.message_queue = asyncio.Queue(
            maxsize=max_queue_size
        )  # Decouples message sending from processing for async handling.
        self.running = False  # Prevents router from processing messages during shutdown.
        self.router_task: Optional[asyncio.Task] = None  # Allows graceful cancellation during cleanup.

        self.bridges: Dict[str, "Bridge"] = {}  # node_id → Bridge instance.

        self.pending_requests: Dict[
            str, asyncio.Future
        ] = {}  # Allows synchronous-style calls over async message bus.

        self.shutdown_event = asyncio.Event()  # Ensures all tasks stop together.

    async def start(self) -> None:
        """
        Start the message router.

        Initializes background task for message routing between bridges.
        """
        if self.running:
            logger.warning("Bus already running")
            return

        self.running = True

        # Log system state summary before starting router
        self._log_system_summary()

        self.router_task = asyncio.create_task(self._message_router())

        logger.info("Bus message router started")

    async def cleanup(self) -> None:
        """
        Stop message routing and clean up resources.

        Cancels all background tasks, cleans up pending requests,
        and ensures graceful shutdown of all bus components.
        """
        logger.info("Cleaning up Bus")
        self.running = False
        self.shutdown_event.set()

        # Cancel router task.
        if self.router_task and not self.router_task.done():
            self.router_task.cancel()
            try:
                await self.router_task
            except asyncio.CancelledError:
                pass

        # Cancel pending requests to prevent hanging futures.
        for future in self.pending_requests.values():
            if not future.done():
                future.cancel()

        self.pending_requests.clear()
        logger.info("Bus cleanup completed")

    def _log_system_summary(self) -> None:
        """Log a clean visual summary of the Bus state before starting."""
        bridge_count = len(self.bridges)

        # Build visual representation
        summary_lines = [
            "🚌 Bus System Ready",
            "=" * 60,
            f"📊 System Overview: {bridge_count} bridges registered",
            "",
            "🏗️  System Architecture:",
        ]

        # Group bridges by type for better visualization
        node_bridges = []
        system_bridges = []

        for name in sorted(self.bridges.keys()):
            bridge = self.bridges[name]
            route_count = len(getattr(bridge, "routes", {}))

            # Get route patterns for this bridge
            route_patterns = []
            if hasattr(bridge, "routes"):
                route_patterns = list(bridge.routes.keys())[:3]  # Show first 3 routes
                if len(bridge.routes) > 3:
                    route_patterns.append("...")

            auth_info = ""
            if hasattr(bridge, "authorized_nodes") and bridge.authorized_nodes:
                auth_nodes = sorted(bridge.authorized_nodes)
                auth_info = f" 🔐[{', '.join(auth_nodes)}]"

            route_info = f"({route_count} routes)" if route_count else "(no routes)"
            pattern_info = f" → {route_patterns}" if route_patterns else ""

            bridge_line = f"   📡 {name:<12} {route_info}{auth_info}{pattern_info}"

            if name in ["user", "tools", "state"]:
                system_bridges.append(bridge_line)
            else:
                node_bridges.append(bridge_line)

        # Add system bridges
        if system_bridges:
            summary_lines.append("   🔧 System Bridges:")
            summary_lines.extend(system_bridges)
            summary_lines.append("")

        # Add node bridges
        if node_bridges:
            summary_lines.append("   🧠 Agent Bridges:")
            summary_lines.extend(node_bridges)
            summary_lines.append("")

        summary_lines.extend(
            [
                "   🔄 Message Flow: Bridges ↔ Bus ↔ All Bridges",
                "=" * 60,
                "🎯 Starting message router...",
            ]
        )

        # Log as single message
        logger.info("\n" + "\n".join(summary_lines))

    # ================================================================
    # Registration & Configuration
    # ================================================================

    def register_bridge(self, node_id: str, bridge: "Bridge") -> None:
        """
        Register event bridge.

        Args:
            node_id: Node identifier.
            bridge: Bridge for event routing.
        """
        self.bridges[node_id] = bridge
        bridge.set_bus(self)

    # ================================================================
    # Public Messaging API
    # ================================================================

    async def broadcast(self, message: Message) -> None:
        """
        Send message to all matching bridges.

        Args:
            message: Message to broadcast.

        Examples:
            Basic message broadcast::

                await bus.broadcast(Message(
                    source="intake",
                    event=ConversationTurn(role="user", content="Hello"),
                ))

            Form completion broadcast::

                await bus.broadcast(Message(
                    source="intake",
                    event=FormCompleted(status="done", fields={"name": "John"}),
                ))

        See Also:
            :meth:`call` - For responses
            :meth:`request` - For specific agents
        """
        logger.debug(f"Bus: Broadcasting message: {message}")
        await self._queue_message(message)

    # ================================================================
    # Message Routing
    # ================================================================

    async def _queue_message(self, message: Message) -> None:
        """
        Queue message for the router to process.

        Args:
            message: Message to queue.
        """
        try:
            await self.message_queue.put(message)
            logger.debug(f"Bus (_queue_message): Message queued: {message}")
        except asyncio.QueueFull:
            logger.error("Bus message queue is full, dropping message")

    def _get_queue_info_synchronous(self) -> Dict[str, Any]:
        """Get information about what is on the queue.

        Note:
            This is incredibly expensive to do (in the land of low latency).
            So call this method only for debugging purposes.
            Never push code that calls this method to production.

        Returns:
            A dictionary of the `self.message_queue` information.
        """
        return {
            "queue_size": self.message_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "is_full": self.message_queue.full(),
            "is_empty": self.message_queue.empty(),
            "messages": self._peek_queue_contents(),
        }

    def _peek_queue_contents(self) -> List[Message]:
        """
        Synchronously peek at all messages in the queue.
        Note: This creates a copy of the queue contents and is not thread-safe.

        Returns:
            List of Message objects currently in the queue.
        """
        messages = []
        temp_queue = asyncio.Queue()

        # Drain the original queue into a temporary queue
        while not self.message_queue.empty():
            try:
                # Use get_nowait() to avoid blocking
                message = self.message_queue.get_nowait()
                messages.append(message)
                temp_queue.put_nowait(message)
            except asyncio.QueueEmpty:
                break

        # Restore the original queue
        while not temp_queue.empty():
            try:
                message = temp_queue.get_nowait()
                self.message_queue.put_nowait(message)
            except asyncio.QueueFull:
                break

        return messages

    async def _message_router(self) -> None:
        """Main message routing loop that processes queued messages."""
        logger.info("Bus message router started")

        try:
            while self.running:
                try:
                    # Timeout allows checking shutdown flag periodically.
                    # logger.debug(f"Bus: Waiting for message in the _message_router")
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                    logger.debug(f"Bus: Message received: {message}")

                    # Delegate to specific routing logic based on pattern.
                    await self._route_message(message)

                except asyncio.TimeoutError:
                    # Prevents infinite blocking during shutdown.
                    continue
                except Exception as e:
                    logger.exception(f"Error in message router: {e}")

        except asyncio.CancelledError:
            logger.info("Message router cancelled")
        except Exception as e:
            logger.exception(f"Unexpected error in message router: {e}")

    async def _route_message(self, message: Message) -> None:
        """Route message to bridges based on direct or broadcast pattern."""
        logger.debug(f"Bus: Routing message: {message}")
        try:
            # Handle agent handoff events or transfer_to_* tool calls directly.
            # TODO: Consider adding certain types of events as being high priority to handle.
            # For example, it would be reasonable to prioritize these events:
            # - AgentResponse should be sent to the user bridge immediately.
            # - Interruption events should be processed by the interruption routes first.
            event = message.event
            if isinstance(event, AgentHandoff):
                logger.info(f"Bus: Handling handoff to {event.target_agent}")
                await self._handle_handoff(message, event.target_agent)
                return
            elif isinstance(event, ToolCall) and event.tool_name.startswith("transfer_to_"):
                target_agent = event.tool_name.replace("transfer_to_", "")
                logger.info(f"Bus: Handling handoff to {target_agent}")
                await self._handle_handoff(message, target_agent)
                return
            else:
                # Broadcast to all bridges
                tasks = []
                for _, bridge in self.bridges.items():
                    # NOTE: Do not await this. We want to fire and forget.
                    # We want the bridges to process the tasks in the background.
                    # This is to ensure that future messages are not blocked by the task.
                    # NOTE: There is an implicit ordering here determined by the order of the bridges.
                    # TODO: Consider making this explicit.
                    tasks.append(asyncio.create_task(bridge.handle_event(message)))

        except Exception as e:
            logger.exception(f"Error routing message {message.id}: {e}")

    async def _handle_handoff(self, message: Message, target_agent: str) -> None:
        """Handle handoff from transfer_to_* tool calls."""
        from_agent = message.source

        # Get reason from the event
        reason = ""
        if isinstance(message.event, AgentHandoff):
            reason = message.event.reason
        elif isinstance(message.event, ToolCall):
            reason = message.event.tool_args.get("reason", "")

        logger.info(f"Processing handoff: {from_agent} -> {target_agent} ({reason})")

        # Tell user bridge to change authorization
        user_auth_event = Authorize(node_id="system", agent=target_agent)
        await self.broadcast(user_auth_event)

        logger.info(f"Handoff completed: {from_agent} -> {target_agent}")
