import asyncio
from dataclasses import dataclass, field
from enum import Enum
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)
import weakref

from loguru import logger

from line.bus import Message
from line.events import EventType
from line.utils.aio import await_tasks_safe

if TYPE_CHECKING:
    from line.bridge import Bridge

# We do not support async handlers here because these event handlers should be synchronous.
OnEventHandler = Union[
    # def handler() -> None
    Callable[[], None],
    # def handler(message: BusMessage) -> None
    Callable[[Message], None],
]


class _EventHandlerDict:
    """Event handler dictionary."""

    fn: OnEventHandler
    # Whether the method takes an argument.
    # This should be True only if the function takes an argument.
    # self is not considered an argument here and will be filtered out.
    has_argument: bool

    def __init__(self, fn: OnEventHandler):
        self.fn = fn

        signature = inspect.signature(fn)
        arguments = [param for param in signature.parameters.values() if param.name != "self"]
        if len(arguments) > 1:
            raise ValueError(
                f"Event handler {fn} takes more than one argument. "
                "Handlers can take 0 or 1 arguments. "
                "See OnEventHandler for function signatures that are supported."
            )
        self.has_argument = len(arguments) > 0

    def __call__(self, *args, **kwargs) -> None:
        if self.has_argument:
            self.fn(*args, **kwargs)
        else:
            self.fn()


class RouteState(Enum):
    """State of a route."""

    RUNNING = "running"
    INTERRUPTED = "interrupted"
    SUSPENDED = "suspended"
    EXITED = "exited"


@dataclass
class RouteConfig:
    """Configuration for a route execution.

    Args:
        operations: List of operations to perform on the event.
        suspended: Whether the route is suspended. If `True`, this route won't execute.
        source: The source node to filter events from. TODO: Deprecate this.
        interrupt_events: List of all events that can interrupt this route.
            If *any* of these events are received, the route will be interrupted.
        suspend_on_events: List of events that will suspend the route.
            If *any* of these events are received, the route will be suspended.
        resume_on_events: List of events that will resume the route.
            If *any* of these events are received, the route will be resumed.
        interrupt_handlers: Dictionary of event handlers for interrupt events.
        filter_fn: Optional custom filter function that takes a BusMessage.
        event_property_filters: Dictionary of event property filters.
    """

    operations: List[Dict] = field(default_factory=list)

    state: RouteState = RouteState.RUNNING
    source: Optional[str] = None
    max_concurrent_tasks: Optional[int] = None

    interrupt_handlers: Dict[str, _EventHandlerDict] = field(default_factory=dict)
    suspend_handlers: Dict[str, _EventHandlerDict] = field(default_factory=dict)
    resume_handlers: Dict[str, _EventHandlerDict] = field(default_factory=dict)

    # These filters operate on the message object.
    event_property_filters: Dict[str, Union[Any, Callable[[Any], bool]]] = field(default_factory=dict)
    filter_fn: Optional[Callable[[Message], bool]] = None


class RouteBuilder:
    """Builder for event routes.

    Unlike :class:`RouteBuilder`, this class handles not just how to process the event,
    but also how to emit results. There is no distinction between the two.

    Simpler interface
    ------------------
    This means that we can do the following:
    ```
    bridge.on(Events.A).broadcast(Events.B)
    ```

    instead of:
    ```
    bridge.on(Events.A).for_each(lambda msg: None).broadcast(Events.B)
    ```

    This is a more natural way to think about the problem instead of forcing an emission.

    Using primitives
    ----------------
    There are 4 primitives that all routes support:
    - `map: Callable[[Any], Any]`: Apply a function to the event.
    - `filter: Callable[[Any], bool]`: Filter the event.
    - `reduce: Callable[[Any], Any]`: Reduce the event.
    - `broadcast: Callable[[Any], None]`: Broadcast the event.
    """

    def __init__(self, bridge: "Bridge"):
        self.bridge = weakref.proxy(bridge)
        self.route_config = RouteConfig()
        self.route_handler: "RouteHandler" = None

    def _set_route_handler(self, route_handler: "RouteHandler") -> None:
        """Creates a weak reference to the route_handler."""
        self.route_handler = weakref.proxy(route_handler)

    def _has_control_operation(self) -> bool:
        """Check if the route has a control operation."""
        return (
            len(self.route_config.operations) > 0 and self.route_config.operations[0]["_fn_type"] == "control"
        )

    def _validate_pre(self):
        """Validate the route configuration before adding an operation."""
        if self._has_control_operation():
            raise ValueError("Control operations must be the first and only operation in a route.")

    def _add_control_operation(self, fn: Callable[[Message], None]) -> "RouteBuilder":
        """Add a control operation to the route.

        Control operations are special because they run synchronously.

        Args:
            fn: Function to run.

        Returns:
            self
        """
        if len(self.route_config.operations) > 0:
            raise ValueError("Control operations must be the first and only operation in a route.")

        self.route_config.operations.append({"_fn_type": "control", "fn": fn})

    def map(self, fn: Callable[[Any], Any]) -> "RouteBuilder":
        """Call a function on the current data.

        Args:
            fn: The function to call on the data.
        """
        self._validate_pre()
        self.route_config.operations.append({"_fn_type": "map", "fn": fn})
        return self

    def stream(
        self,
        generator_fn: Optional[Callable[[Any], Union[Any, AsyncIterable]]] = None,
    ) -> "RouteBuilder":
        """Stream results from a generator function through the remaining pipeline.

        Args:
            generator_fn: Optional generator function. If None, assumes the previous
                         operation's output is already an async generator.
        """
        self._validate_pre()
        self.route_config.operations.append({"_fn_type": "stream", "fn": generator_fn})
        return self

    def filter(self, fn: Callable[[Any], bool]) -> "RouteBuilder":
        """Filter the event - continue only if function returns True."""
        self._validate_pre()
        self.route_config.operations.append({"_fn_type": "filter", "fn": fn})
        return self

    def broadcast(self, event_type: Optional[EventType] = None) -> "RouteBuilder":
        """Broadcast current data to specified event.

        Args:
            event_type: The type of event to broadcast the result of the previous operation to.
                If `None`, we assume that the previous operation is returning or yielding an `EventInstance`.
                If provided, the previous operation must return or yield a mapping, which will be used to
                construct the event object: `event_type(**mapping)`.

        Note:
            It is strongly recommended to have the operation before `broadcast`
            return or yield :class:`EventInstance` objects.
            This is the preferred design as it allows you (the user) to specify the event type and
            data that is the result of the previous operation.

        Note:
            This method does not return a value. It should be treated as a terminal operation.
        """
        self._validate_pre()
        self.route_config.operations.append({"_fn_type": "broadcast", "event": event_type})
        return self

    def _add_on_event_handler(
        self,
        event_type: EventType,
        handler: OnEventHandler,
        method_name: Literal["suspend", "resume", "interrupt"],
    ) -> None:
        assert self.route_handler is not None, (
            f"self._set_route_handler is not initialized. It is required for configuring {method_name}."
        )

        if method_name == "suspend":
            handlers = self.route_config.suspend_handlers
        elif method_name == "resume":
            handlers = self.route_config.resume_handlers
        elif method_name == "interrupt":
            handlers = self.route_config.interrupt_handlers

        if event_type in handlers:
            raise ValueError(f"Event {event_type} already registered for {method_name}")

        handler = _EventHandlerDict(handler) if handler is not None else None
        handlers[event_type] = handler

        # Add implicit handler to the bridge to handle the suspend.
        if method_name == "suspend":
            self.bridge.on(event_type)._add_control_operation(
                lambda message: self.route_handler._suspend(message, handler)
            )
        elif method_name == "resume":
            self.bridge.on(event_type)._add_control_operation(
                lambda message: self.route_handler._resume(message, handler)
            )
        elif method_name == "interrupt":
            self.bridge.on(event_type)._add_control_operation(
                lambda message: self.route_handler._interrupt(message, handler)
            )
        return self

    def suspend_on(self, event_type: EventType, handler: OnEventHandler = None) -> "RouteBuilder":
        """Suspend route from running when event is received.

        Args:
            event_type: EventType that should suspend this route.
            handler: Function that runs after the route is suspended.

        Raises:
            ValueError: If the event is already registered for suspend.
        """
        return self._add_on_event_handler(event_type, handler, "suspend")

    def resume_on(self, event_type: EventType, handler: OnEventHandler = None) -> "RouteBuilder":
        """Resume route execution when any of these events are received."""
        return self._add_on_event_handler(event_type, handler, "resume")

    def interrupt_on(self, event_type: EventType, handler: OnEventHandler = None) -> "RouteBuilder":
        """Interrupt this route execution when any of these events are received.

        Args:
            event: EventType that should interrupt this route.
            handler: Optional callable that runs after the route is cancelled but before the lock is released.
                Receives the interrupt event type as argument.

        Raises:
            ValueError: If the event is already registered for interrupt.
        """
        return self._add_on_event_handler(event_type, handler, "interrupt")

    def on(self, event: str) -> "RouteBuilder":
        """Start a new route on a different event."""
        return self.bridge.on(event)


class RouteHandler:
    """Handles execution of a configured route.

    This class is responsible for executing the route built by :class:`RouteBuilder`.
    """

    def __init__(self, route_builder: RouteBuilder, bridge: "Bridge"):
        self.route_builder = route_builder
        self.route_builder._set_route_handler(self)

        self.bridge = bridge

        # Task management for concurrent task limiting.
        self._active_tasks: list[asyncio.Task] = []  # Track all active tasks.
        self._task_lock = asyncio.Lock()  # Thread-safe access to active_tasks.
        # Task for cancelling all active tasks. There can only be one cancel task at a time.
        self._task_cancel_all_tasks: Optional[asyncio.Task] = None

    @property
    def route_config(self) -> RouteConfig:
        return self.route_builder.route_config

    def should_process_message(self, message: Message) -> bool:
        """Check if the message should be processed based on all configured filters.

        Args:
            message: The message to check.
        """
        if self.route_config.state == RouteState.SUSPENDED or len(self.route_config.operations) == 0:
            return False

        # Apply custom filter function if provided.
        if self.route_config.filter_fn is not None:
            if not self.route_config.filter_fn(message):
                return False

        # Apply event property filters.
        event_filters = self.route_config.event_property_filters or {}
        for prop_name, expected_value in event_filters.items():
            if not hasattr(message.event, prop_name):
                logger.debug(f"Event {message.event} does not have property {prop_name}")
                return False

            actual_value = getattr(message.event, prop_name)
            if callable(expected_value):
                if not expected_value(actual_value):
                    return False
            elif expected_value != actual_value:
                return False

        return True

    async def handle(self, message: Message) -> Any:
        """Handle an incoming message through the route.

        Returns:
            None: if the route is suspended or has no operations.
            Any: Output result of the route.
        """
        if self.route_config.state == RouteState.SUSPENDED or len(self.route_config.operations) == 0:
            return None

        # We do not check self.should_process_message because we assume the bridge does the check.
        # And we don't want to do the check twice.
        # if not self.should_process_message(message):
        #     return None

        # Check source filter (legacy support).
        # TODO: We have to filter out events that are not from the source node instead of a .filter() method
        # because we don't know what the return type of the previous operation is.
        # This is more of a utility method.
        if self.route_config.source and message.source != self.route_config.source:
            return None

        # Wait for the current cancel task to complete before running cancel again.
        if self._task_cancel_all_tasks is not None:
            await await_tasks_safe(self._task_cancel_all_tasks)

        # Control operations are special because they run synchronously.
        # This allows us to run them without releasing the event lock.
        # This will prevent tasks from being spawned on another route while
        # we are running a control operation.
        if self.route_builder._has_control_operation():
            return self.route_config.operations[0]["fn"](message)

        # Create new task for this execution.
        try:
            async with self._task_lock:
                if self.route_config.max_concurrent_tasks is not None:
                    self._active_tasks = [task for task in self._active_tasks if not task.done()]
                    if len(self._active_tasks) >= self.route_config.max_concurrent_tasks:
                        oldest_task = self._active_tasks.pop(0)
                        # NOTE: We are waiting for the task to be cancelled before creating a new task.
                        # Usually this is what we want, but should we expose this as an option to the user?
                        await self._cancel_task_with_cleanup(oldest_task)

                task = asyncio.create_task(self._process_operations(message, self.route_config.operations))
                self._active_tasks.append(task)

            result = await task
            # TODO: Do we really want to wait to clean up rather than just returning.
            await self._clean_active_tasks_safe()
            return result
        except asyncio.CancelledError:
            # TODO (AD): Improve the debug log here to include information
            # about which route is being cancelled.
            logger.debug(f"Route execution (handler {self}) cancelled")
            return None

    def _interrupt(
        self,
        message: Message,
        handler: Optional[_EventHandlerDict] = None,
    ) -> None:
        """Request interruption of this route if it's currently executing.

        Execution Order:
            1. Cancel all active tasks.
               Tasks that come in after this event will not be cancelled, but they will not be executed
               until the interrupt is complete.
            2. After *all* active tasks are cancelled, run the handler.
            3. Clean up canceled tasks from `self._active_tasks`.

        Note:
            The `handler` is only executed after all active tasks are cancelled.
            It is called regardless of whether the tasks were cancelled or completed.

        Args:
            message: The message that triggered the interrupt.
            handler: The handler to run after all active tasks are cancelled.
        """
        event = message.event

        if not self._active_tasks:
            logger.debug("No active tasks to interrupt.")
            return

        logger.debug(f"Interrupting route due to event {type(event)}")

        # Cancel all active tasks.
        async def await_cancel_active_tasks(
            canceled_tasks: List[asyncio.Task], prev_cancel_task: Optional[asyncio.Task]
        ):
            # Wait for the current cancel task to complete.
            if prev_cancel_task is not None:
                await await_tasks_safe(prev_cancel_task)

            await await_tasks_safe(canceled_tasks)

            if handler is not None:
                handler(message)

            # Clear these tasks from the active tasks list.
            await self._clean_active_tasks_safe()

        # NOTE: There might be a race condition here where the task list is being modified somewhere else.
        # We do an unsafe check because we want to cancel as soon as possible.
        active_tasks = [task for task in self._active_tasks if task and not task.done()]
        for task in active_tasks:
            task.cancel()

        prev_cancel_task = self._task_cancel_all_tasks
        self._task_cancel_all_tasks = asyncio.create_task(
            await_cancel_active_tasks(active_tasks, prev_cancel_task)
        )

    def _suspend(
        self,
        message: Message,
        handler: Optional[_EventHandlerDict] = None,
    ) -> None:
        """Suspend this route.

        This means that future messages will not be processed by this route
        until the route is resumed.

        Execution order:
            1. Suspend the route. We do this synchronously to prevent future tasks from being created.
            2. Interrupt all active tasks. Do not run the interrupt handler,
               as there is no guaranteed handler for this event.
            3. Run the suspend handler.
        """
        self.route_config.state = RouteState.SUSPENDED

        self._interrupt(message, handler)
        assert len(self._active_tasks) == 0, (
            "All tasks should have been cancelled and no remaining tasks should be active."
        )

    def _resume(
        self,
        message: Message,
        handler: Optional[_EventHandlerDict] = None,
    ) -> None:
        """Resume this route.

        Execution order:
            1. Run the resume handler.
            2. Resume the route.

        Args:
            message: The message that triggered the resume.
            handler: The handler to run before the route is resumed.

        Note:
            The `handler` is only executed after all active tasks are cancelled.
            It is called regardless of whether the tasks were cancelled or completed.
        """
        self.route_config.state = RouteState.RUNNING
        if handler is not None:
            handler(message)

    async def _clean_active_tasks_safe(self) -> None:
        """Clear the active tasks list."""
        async with self._task_lock:
            self._active_tasks = [task for task in self._active_tasks if not task.done()]

    # TODO(noah): this is claude-generated and kind of sus to me.
    async def _cancel_task_with_cleanup(self, task: asyncio.Task) -> None:
        """Cancel a task and wait for it to complete."""
        if task.done():
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _process_operations(self, data: Any, operations: List[dict]) -> Any:
        """Process data through a sequence of operations."""
        if not operations:
            return data

        current_data = data

        for i, operation in enumerate(operations):
            fn_type = operation["_fn_type"]
            fn = operation.get("fn")
            remaining_ops = operations[i + 1 :]

            if fn_type == "map":
                if asyncio.iscoroutinefunction(fn):
                    current_data = await fn(current_data)
                else:
                    current_data = fn(current_data)

            elif fn_type == "stream":
                # Explicit streaming operation
                await self._handle_stream(current_data, fn, remaining_ops)
                return None  # Stream consumed

            elif fn_type == "filter":
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(current_data)
                else:
                    result = fn(current_data)
                if not result:
                    # Filtered out, stop processing.
                    return None

            elif fn_type == "broadcast":
                event_cls: Optional[EventType] = operation["event"]
                event = current_data
                if current_data is not None and self.bridge.bus:
                    if event_cls is not None:
                        try:
                            event = event_cls(**current_data)
                        except Exception as e:
                            logger.error(
                                f"Error coercing data to {event_cls.__name__} with input {current_data}: {e}",
                                exc_info=True,
                            )
                            raise e
                    await self.bridge.bus.broadcast(Message(source=self.bridge.node_id, event=event))

        return None

    async def _handle_stream(
        self,
        data: Any,
        generator_fn: Optional[Callable],
        remaining_ops: List[dict],
    ) -> None:
        """Handle explicit streaming operation."""
        try:
            if generator_fn is None:
                # Assume data is already an async generator
                if hasattr(data, "__aiter__"):
                    # data is async iterable
                    async for yielded in data:
                        await self._process_operations(yielded, remaining_ops)
                elif hasattr(data, "__iter__") and not isinstance(data, (str, bytes)):
                    # data is regular iterable (but not string/bytes)
                    for yielded in data:
                        await self._process_operations(yielded, remaining_ops)
                else:
                    # data is not iterable, treat as single result
                    await self._process_operations(data, remaining_ops)
            elif self._is_async_generator_function(generator_fn):
                async for yielded in generator_fn(data):
                    await self._process_operations(yielded, remaining_ops)
            elif self._is_generator_function(generator_fn):
                for yielded in generator_fn(data):
                    await self._process_operations(yielded, remaining_ops)
            else:
                # Not a generator, treat as single result and continue
                result = generator_fn(data)
                if asyncio.iscoroutine(result):
                    result = await result
                await self._process_operations(result, remaining_ops)
        except asyncio.CancelledError as e:
            logger.debug("Stream operation cancelled")
            raise e  # Re-raise to propagate cancellation

    def _is_async_generator_function(self, fn: Callable) -> bool:
        """Check if function returns an async generator."""
        return inspect.isasyncgenfunction(fn)

    def _is_generator_function(self, fn: Callable) -> bool:
        """Check if function returns a generator."""
        return inspect.isgeneratorfunction(fn)
