"""
Bridge - Route events between nodes and bus.

Provides stateless routing between node methods and bus events.
"""

import asyncio
from collections import defaultdict
from fnmatch import fnmatch
import inspect
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Type, TypeVar, Union

from loguru import logger

from line.bus import Bus, Message
from line.events import (
    EventInstance,
    EventsRegistry,
    EventTypeOrAlias,
)
from line.routes import RouteBuilder, RouteHandler

if TYPE_CHECKING:
    from line.nodes.reasoning import Node

T = TypeVar("T")


class Bridge:
    """Route bus events to node methods and broadcast results.

    This class is responsible for managing and executing routes when triggered by bus events.
    When the bus receives an event, it forwards the event to all bridges registered with the bus.
    The bridge determines whether there is a route that should be triggered by the event.
    A route is a sequence of operations that is performed on the message associated with the event.

    When a route is executed, the bridge will construct the :class:`RouteHandler` corresponding to the
    :class:`RouteBuilder` and execute it.
    """

    def __init__(self, node: Union["Node", str]):
        """Create bridge for node.

        Args:
            node: Processing node instance.
        """
        self.node = node  # Processing node instance.

        # Quietly set the node's bridge if possible.
        try:
            node._bridge = self  # type: ignore[attr-defined]
        except AttributeError:
            pass

        # event_pattern → (method, broadcast_event).
        self.routes: dict[EventTypeOrAlias, List[RouteHandler]] = defaultdict(list)
        self.scheduled_tasks: List[asyncio.Task] = []  # Background tasks for periodic execution.
        self.bus = None  # Bus instance for broadcasting results.
        self._signature_cache = {}  # method → takes_parameters for performance.
        self.authorized_nodes = set()  # None = open access, set = restricted access.

        # Input routing state - enables continuous external input → bus event conversion.
        self.input_source = None  # Source object with get() method for input messages.
        self.input_task = None  # Background task for input routing.
        self.input_shutdown = None  # Event to signal input routing shutdown.

        # Track all active route execution tasks
        self.active_route_tasks: List[asyncio.Task] = []
        self._route_tasks_lock = asyncio.Lock()

    @property
    def node_id(self) -> str:
        """Get the node identifier."""
        if isinstance(self.node, str):
            return self.node
        elif hasattr(self.node, "id"):
            return self.node.id
        elif hasattr(self.node, "__class__"):
            return self.node.__class__.__name__
        return "unknown"

    def authorize(self, *node_ids) -> "Bridge":
        """Restrict bridge access to specific nodes.

        TODO (AD): What does this mean? Every bridge has a node right?
        """
        self.authorized_nodes = set(node_ids)
        return self

    def set_bus(self, bus: Bus) -> "Bridge":
        """Set bus for broadcasting."""
        self.bus = bus
        return self

    def with_input_routing(self, source: Any = None) -> "Bridge":
        """
        Enable continuous input routing from external source to bus events.

        Input routing solves the bidirectional communication problem by automatically
        converting external input streams (WebSocket messages, file changes, etc.) into
        bus events. Without this, you need manual async tasks and coordination.

        Args:
            source: Object with async get() method. Defaults to self.node.
                   Examples: ConversationHarness, FileWatcher, asyncio.Queue.

        Returns:
            Self for method chaining.

        Examples:
            >>> # Route WebSocket messages to bus events
            >>> bridge.with_input_routing(harness)
            >>>
            >>> # Route file system changes to bus events
            >>> bridge.with_input_routing(file_watcher)
        """
        self.input_source = source or self.node
        return self

    def on(
        self,
        event_pattern: Union[str, Type[T]],
        *,
        id: Union[str, Callable[[str], bool]] = None,
        source: Union[str, Callable[[str], bool]] = None,
        timestamp: Union[float, Callable[[float], bool]] = None,
        filter_fn: Optional[Callable[["Message"], bool]] = None,
        **event_property_filters,
    ) -> "RouteBuilder":
        """Start building route for event pattern or typed event with filtering.

        Args:
            event_pattern: The event type we want to filter on.
            id: The message id to filter by. Can be a string or a callable that returns bool.
            source: The message source to filter by. Can be a string or a callable that returns bool.
            timestamp: The message timestamp to filter by. Can be a float or a callable that returns bool.
            filter_fn: Custom filter function that takes a BusMessage and returns bool.
            **event_property_filters: Additional filters for properties of the event object.
                Keys should be attribute names of the event, values can be the expected value
                or a callable that takes the property value and returns bool.

        Note:
            When a handler triggers (i.e. the route in .on runs), we spin up a new asyncio.Task.
            Spawning too many tasks will impact the performance of the system.
            We provide filtering directly in the `on` to avoid spawning new asyncio tasks.
        """
        if not inspect.isclass(event_pattern) and not isinstance(event_pattern, str):
            raise ValueError(f"Event pattern must be a string or a class type: {event_pattern}")

        logger.debug(f"Bridge {self.node_id}: Adding route for {event_pattern}")

        # RouteBuilder and RouteHandler have a 1:1 relationship.
        route_builder = RouteBuilder(self)
        route_handler = RouteHandler(route_builder, self)

        # Set up filtering configuration
        route_handler.route_config.id_filter = id
        route_handler.route_config.source = source
        route_handler.route_config.timestamp_filter = timestamp
        route_handler.route_config.filter_fn = filter_fn
        route_handler.route_config.event_property_filters = event_property_filters

        self.routes[event_pattern].append(route_handler)

        return route_builder

    async def handle_event(self, message: "Message") -> None:
        """Route incoming event to appropriate handler."""
        # Check authorization first - empty set means open access.
        if self.authorized_nodes and message.source not in self.authorized_nodes:
            return

        handlers: List[RouteHandler] = self._find_matching_routes(message.event)
        handlers = [handler for handler in handlers if handler.should_process_message(message)]
        if not handlers:
            return

        # Execute all matching routes concurrently.
        tasks = []
        for handler in handlers:
            tasks.append(asyncio.create_task(handler.handle(message)))

        # Track the task for cleanup
        async with self._route_tasks_lock:
            self.active_route_tasks.extend(tasks)

        if tasks:
            # NOTE: We are awaiting here. This is blocking.
            # If you need to run this in the background, make sure to spawn a task and not await that task.
            # See `Bus._route_message` for an example on how this is done.
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log results for each route.
            for route_index, (_, result) in enumerate(zip(tasks, results)):  # type: ignore[arg-type]
                if isinstance(result, Exception):
                    logger.opt(exception=result).error(
                        "Bridge {}: Route {} failed for event {}: {}",
                        self.node_id,
                        route_index,
                        type(message.event).__name__,
                        str(result),
                    )
                else:
                    logger.debug(
                        f"Bridge {self.node_id}: Route {route_index} "
                        f"completed successfully for event {type(message.event)}"
                    )

        # Cleanup tasks.
        async with self._route_tasks_lock:
            self.active_route_tasks = [t for t in self.active_route_tasks if not t.done()]

    async def start_input_routing(self) -> None:
        """
        Start input routing from source to bus events.

        Creates background task that continuously polls input source and converts
        messages to bus events. Essential for bridging external input streams.
        """
        if not self.input_source:
            logger.warning(f"Bridge {self.node_id}: No input source configured")
            return

        if not hasattr(self.input_source, "get"):
            logger.warning(f"Bridge {self.node_id}: Input source missing get() method")
            return

        self.input_shutdown = asyncio.Event()
        self.input_task = asyncio.create_task(self._input_router())
        logger.debug(f"Bridge {self.node_id}: Input routing started")

    async def _input_router(self) -> None:
        """
        Route input from source to bus events continuously.

        Uses input source's map_to_events method to convert messages to specific events.
        Handles timeouts gracefully to avoid blocking on empty sources.
        """
        while not self.input_shutdown.is_set():
            try:
                # Get next message from input source - blocks until available.
                message = await asyncio.wait_for(self.input_source.get(), timeout=1.0)

                # Convert message to events using source's mapping logic.
                if hasattr(self.input_source, "map_to_events"):
                    events = self.input_source.map_to_events(message)
                    for event in events:
                        await self.bus.broadcast(Message(source=self.node_id, event=event))
                else:
                    raise ValueError(f"Input source {self.input_source} has no map_to_events method.")
            except asyncio.TimeoutError:
                # No input available - continue polling.
                continue
            except Exception as e:
                logger.exception(f"Bridge {self.node_id}: Input routing error: {e}")

    async def stop_input_routing(self) -> None:
        """Stop input routing task and cleanup resources."""
        if self.input_shutdown:
            self.input_shutdown.set()

        if self.input_task and not self.input_task.done():
            self.input_task.cancel()
            try:
                await self.input_task
            except asyncio.CancelledError:
                pass

        logger.debug(f"Bridge {self.node_id}: Input routing stopped")

    async def start(self) -> None:
        """Start scheduled tasks and input routing if configured."""
        # Start input routing if configured.
        if self.input_source:
            await self.start_input_routing()

    async def stop(self) -> None:
        """Stop scheduled tasks and input routing."""
        # Stop input routing first.
        await self.stop_input_routing()

        # Cancel all active route tasks.
        await self._cancel_all_route_tasks()

        # Stop scheduled tasks.
        for task in self.scheduled_tasks:
            if not task.done():
                task.cancel()

        if self.scheduled_tasks:
            await asyncio.gather(*self.scheduled_tasks, return_exceptions=True)

        self.scheduled_tasks.clear()
        logger.debug("Bridge stopped all scheduled tasks")

    async def _cancel_all_route_tasks(self):
        """Cancel all active route execution tasks."""
        async with self._route_tasks_lock:
            if not self.active_route_tasks:
                return

            logger.info(
                f"Bridge {self.node_id}: Cancelling {len(self.active_route_tasks)} active route tasks"
            )

            # Cancel all active route tasks
            for task in self.active_route_tasks:
                if not task.done():
                    task.cancel()

            # Wait for all tasks to complete cancellation
            if self.active_route_tasks:
                await asyncio.gather(*self.active_route_tasks, return_exceptions=True)

            self.active_route_tasks.clear()
            logger.info(f"Bridge {self.node_id}: All route tasks cancelled")

    # TODO: Do we need this if we are doing the filtering anyway on each event?
    def can_handle(self, event: Any) -> bool:
        """Check if this bridge can handle the event type."""
        # TODO: This is a very expensive check (maybe?). Just feels too heavy.
        return len(self._find_matching_routes(event)) > 0

    def _find_matching_routes(self, event: EventInstance) -> List[RouteHandler]:
        """Find all routes matching event type (supports wildcards)."""
        handlers: List[RouteHandler] = []
        pattern: EventTypeOrAlias

        # Find all the routes that match the event (glob-style pattern matching).
        for pattern, handler in self.routes.items():
            if inspect.isclass(pattern):
                if isinstance(event, pattern):
                    handlers.extend(handler)

            elif isinstance(pattern, str):
                # This is a catch-all pattern.
                if pattern == "*":
                    handlers.extend(handler)
                    continue

                alias = EventsRegistry.get(type(event))
                if not alias:
                    continue

                # Use glob-style pattern matching.
                if fnmatch(alias, pattern):
                    handlers.extend(handler)
            else:
                # We should never reach here if we ran `.on` correctly.
                raise AssertionError(f"Invalid pattern: {pattern}")

        return handlers
