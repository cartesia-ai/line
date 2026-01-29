import asyncio
from dataclasses import dataclass
import time
from typing import Callable, Optional, Tuple, Union
from unittest.mock import AsyncMock, Mock, patch

from helpers import DummyEvent, NumberEvent
from pydantic import BaseModel, ValidationError
import pytest

from line import Bridge, Message, ReasoningNode, RouteBuilder
from line.routes import RouteHandler, RouteState, _EventHandlerDict

# Methods in RouteBuilder that add control operations.
CONTROL_METHOD_TYPES = ["suspend", "resume", "interrupt"]
OPERATION_TYPES = ["map", "stream", "filter", "broadcast"]


class ControlEvent(BaseModel):
    pass


@dataclass
class RouteData:
    node: ReasoningNode
    bridge: Bridge
    builder: RouteBuilder
    handler: RouteHandler

    def as_tuple(
        self,
    ) -> Tuple[ReasoningNode, Bridge, RouteBuilder, Optional[RouteHandler]]:
        return (self.node, self.bridge, self.builder, self.handler)


def get_route_data(skip_handler: bool = False) -> RouteData:
    """Helper function to construct a RouteData object.

    Args:
        skip_handler: If True, do not create a RouteHandler.
    """
    node = ReasoningNode("")
    bridge = Bridge(node)
    builder = RouteBuilder(bridge)
    handler = None if skip_handler else RouteHandler(builder, bridge)
    return RouteData(node=node, bridge=bridge, builder=builder, handler=handler)


# _EventHandlerDict tests.


def test_event_handler_init_no_args():
    def handler():
        return None

    handler_dict = _EventHandlerDict(handler)
    assert not handler_dict.has_argument, "Handler should have no arguments."
    assert handler_dict.fn == handler, "Handler should be equivalent to the defined handler."


def test_event_handler_init_with_args():
    def handler(x: int):
        return None

    handler_dict = _EventHandlerDict(handler)
    assert handler_dict.has_argument, "Handler should have one argument."
    assert handler_dict.fn == handler, "Handler should be equivalent to the defined handler."


def test_event_handler_init_with_multiple_args():
    def handler(x: int, y: int):
        return None

    # Handler cannot have multiple arguments.
    with pytest.raises(ValueError):
        _EventHandlerDict(handler)


# RouteBuilder tests.


def test_chaining_returns_same_builder():
    """Test that chaining methods returns the same builder."""
    route_data = get_route_data()
    rb = route_data.builder
    assert rb.map(lambda x: x) is rb
    assert rb.stream(lambda x: x) is rb
    assert rb.filter(lambda x: True) is rb
    assert rb.broadcast() is rb


@pytest.mark.parametrize("value", (0, 1, 2))
@pytest.mark.asyncio
async def test_filter(value: int):
    """Test that filtering an event calls the handler.

    It should only call the handler if the filter returns True.
    """
    route_data = get_route_data()
    bridge = route_data.bridge

    fn_to_invoke = Mock()
    should_invoke_fn = value % 2 == 0

    def filter_fn(x: Message) -> bool:
        event: NumberEvent = x.event
        return event.value % 2 == 0

    bridge.on(NumberEvent).filter(filter_fn).map(fn_to_invoke)

    await bridge.handle_event(Message(source="test", event=NumberEvent(value=value)))
    if should_invoke_fn:
        fn_to_invoke.assert_called_once()
    else:
        fn_to_invoke.assert_not_called()


# Unit tests


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.parametrize("is_generator", [True, False])
@pytest.mark.asyncio
async def test_add_stream(is_async: bool, is_generator: bool):
    """Test adding a stream operation."""
    builder = get_route_data().builder

    async def stream_fn_async(x):
        await asyncio.sleep(0)
        return (1, 2, 3)

    def stream_fn_sync(x):
        return (1, 2, 3)

    async def stream_gen_async():
        for i in range(3):
            await asyncio.sleep(0)
            yield i

    def stream_gen_sync():
        for i in range(3):
            yield i

    # (is_async, is_generator) -> fn.
    fn = {
        (False, False): stream_fn_sync,
        (False, True): stream_gen_sync,
        (True, False): stream_fn_async,
        (True, True): stream_gen_async,
    }[(is_async, is_generator)]

    builder.stream(fn)

    operations = builder.route_config.operations
    assert len(operations) == 1
    assert operations[0]["_fn_type"] == "stream"
    assert operations[0]["fn"] == fn


def test_add_map():
    builder = get_route_data().builder

    def fn(x):
        return x

    builder.map(fn)

    map_ops = [op for op in builder.route_config.operations if op["_fn_type"] == "map"]
    assert len(map_ops) == 1
    assert map_ops[0]["fn"] == fn


def test_add_broadcast():
    builder = get_route_data().builder

    builder.broadcast(DummyEvent)

    broadcast_ops = [op for op in builder.route_config.operations if op["_fn_type"] == "broadcast"]
    assert len(broadcast_ops) == 1
    assert broadcast_ops[0]["event"] == DummyEvent


def test_add_broadcast_none():
    builder = get_route_data().builder
    builder.broadcast()

    broadcast_ops = [op for op in builder.route_config.operations if op["_fn_type"] == "broadcast"]
    assert len(broadcast_ops) == 1
    assert broadcast_ops[0]["event"] is None


def test_add_filter():
    builder = get_route_data().builder

    def filter_fn(x):
        return True

    builder.filter(filter_fn)

    operations = builder.route_config.operations
    assert len(operations) == 1
    assert operations[0]["_fn_type"] == "filter"
    assert operations[0]["fn"] == filter_fn


def test_control_operation_must_be_first():
    # Add control operation after map.
    builder = get_route_data().builder
    builder.map(lambda x: x)
    with pytest.raises(ValueError):
        builder._add_control_operation(lambda x: x)


@pytest.mark.parametrize("fn_type", OPERATION_TYPES)
def test_control_operation_add_op_after_control(fn_type: str):
    # Add some operation after control.
    builder = get_route_data().builder
    builder._add_control_operation(lambda x: x)

    with pytest.raises(ValueError):
        if fn_type == "map":
            builder.map(lambda x: x)
        elif fn_type == "stream":
            builder.stream(lambda x: x)
        elif fn_type == "filter":
            builder.filter(lambda x: x)
        elif fn_type == "broadcast":
            builder.broadcast()
        else:
            raise RuntimeError(f"Unknown operation type: {fn_type}")


@pytest.mark.parametrize("method_name", CONTROL_METHOD_TYPES)
@pytest.mark.parametrize("handler", [None, lambda: None, lambda x: None])
def test_add_on_event_handler(method_name: str, handler: Optional[Callable]):
    _, bridge, builder, _ = get_route_data().as_tuple()

    fn = getattr(builder, f"{method_name}_on")
    fn(DummyEvent, handler)

    # Verify bridge has a control operation.
    assert len(bridge.routes[DummyEvent]) == 1, "Bridge should have one control route."
    control_route = bridge.routes[DummyEvent][0]
    assert control_route.route_builder is not builder, "Control route should be a different."
    assert control_route.route_builder._has_control_operation(), (
        "Control route should have a control operation."
    )
    assert len(control_route.route_config.operations) == 1, "Control route should have one operation."
    assert control_route.route_config.operations[0]["_fn_type"] == "control", (
        "Control route should have a control operation."
    )


@pytest.mark.parametrize("method_name", CONTROL_METHOD_TYPES)
def test_add_on_event_handler_double_register_event(method_name: str):
    """Test that you cannot double register an on_event handler for the same control type."""
    # We need to keep the handler alive to avoid garbage collection of the weakly-referenced RouteHandler.
    _, bridge, builder, handler = get_route_data().as_tuple()

    fn = getattr(builder, f"{method_name}_on")

    fn(DummyEvent, lambda x: x)
    with pytest.raises(ValueError):
        fn(DummyEvent, lambda x: x)


@pytest.mark.parametrize("method_name", CONTROL_METHOD_TYPES)
@pytest.mark.parametrize("add_handler", [True, False])
def test_control_operation_add_handler(method_name: str, add_handler: bool):
    # We need to keep the route_handler alive to avoid garbage collection of the
    # weakly-referenced RouteHandler.
    _, bridge, builder, route_handler = get_route_data().as_tuple()

    def handler():
        return None

    # Add the handler.
    fn = getattr(builder, f"{method_name}_on")
    fn(DummyEvent, handler if add_handler else None)

    # Get the handler routes (e.g. builder.route_config.suspend_handlers).
    handlers = getattr(builder.route_config, f"{method_name}_handlers")
    assert DummyEvent in handlers, "Expected suspend handler to be added."
    if add_handler:
        assert handlers[DummyEvent].fn == handler
    else:
        assert handlers[DummyEvent] is None


def test_on_alias_to_bridge_on():
    """Test that on is an alias for bridge.on."""
    _, bridge, builder, _ = get_route_data().as_tuple()

    with patch.object(bridge, "on") as mock_on:
        builder.on(DummyEvent)

    mock_on.assert_called_once_with(DummyEvent)


# RouteHandler.should_process_message tests


def test_should_process_message_no_operations():
    """Test that routes with no operations return False"""
    _, _, _, handler = get_route_data().as_tuple()

    message = Message(source="test", event=DummyEvent())
    assert not handler.should_process_message(message)


def test_should_process_message_suspended_route():
    """Test that suspended routes return False"""
    _, _, _, handler = get_route_data().as_tuple()

    # Suspend the route
    handler.route_config.state = RouteState.SUSPENDED

    message = Message(source="test", event=DummyEvent())
    assert not handler.should_process_message(message)


def test_should_process_message_running_route():
    """Test that running routes return True when no filters are set"""
    _, _, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    # Route should be RUNNING by default
    assert handler.route_config.state == RouteState.RUNNING

    message = Message(source="test", event=DummyEvent())
    assert handler.should_process_message(message)


def test_should_process_message_custom_filter_passes():
    """Test custom filter function that returns True"""
    _, _, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    # Set custom filter that allows messages from "test" source
    handler.route_config.filter_fn = lambda msg: msg.source == "test"

    message = Message(source="test", event=DummyEvent())
    assert handler.should_process_message(message)


def test_should_process_message_custom_filter_fails():
    """Test custom filter function that returns False"""
    _, _, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    # Set custom filter that blocks messages from "blocked" source
    handler.route_config.filter_fn = lambda msg: msg.source != "blocked"

    message = Message(source="blocked", event=DummyEvent())
    assert not handler.should_process_message(message)


def test_should_process_message_no_custom_filter():
    """Test that message passes when no custom filter is set"""
    _, _, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    # No custom filter set (should be None by default)
    assert handler.route_config.filter_fn is None

    message = Message(source="any", event=DummyEvent())
    assert handler.should_process_message(message)


@pytest.mark.parametrize(
    "value,filter_value,expected",
    [
        (0, 1, False),  # value is 0, filter is 1, should fail.
        (1, 1, True),  # value is 1, filter is 1, should pass.
        (0, lambda x: x >= 1, False),  # value is 0, filter is x >= 1, should fail.
        (1, lambda x: x >= 1, True),  # value is 1, filter is x >= 1, should pass.
        (2, lambda x: x >= 1, True),  # value is 2, filter is x >= 1, should pass.
    ],
)
def test_should_process_event_property_filter(
    value: int, filter_value: Union[int, Callable[[int], bool]], expected: bool
):
    """Test that event property filters work."""

    class MyEvent(BaseModel):
        value: int

    _, _, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    # pass when value is 1.
    handler.route_config.event_property_filters = {"value": filter_value}

    message = Message(source="any", event=MyEvent(value=value))
    assert_msg = f"Expected {'fail' if not expected else 'pass'}: value {value}, filter {filter_value}."
    assert handler.should_process_message(message) is expected, assert_msg


def test_should_process_message_event_property_filter_missing_field():
    """Test that event property filters work when the field is missing.

    This is not a strict type so we should return False and continue.
    """

    class MyEvent(BaseModel):
        value: int

    _, _, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    handler.route_config.event_property_filters = {"some_field": 1}

    message = Message(source="any", event=MyEvent(value=1))
    assert not handler.should_process_message(message), "Message should not pass when some_field is missing."


# RouteHandler.handle() tests


@pytest.mark.asyncio
async def test_handle_suspended_route():
    """Test that suspended routes immediately return None"""

    _, _, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    # Suspend the route
    handler.route_config.state = RouteState.SUSPENDED

    with patch.object(handler, "_process_operations") as mock_process:
        message = Message(source="test", event=DummyEvent())
        result = await handler.handle(message)

        assert result is None
        mock_process.assert_not_called()


@pytest.mark.asyncio
async def test_handle_source_filter_legacy():
    """Test legacy source filtering behavior"""
    _, _, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    # Set source filter
    handler.route_config.source = "allowed_source"

    with patch.object(handler, "_process_operations") as mock_process:
        # Non-matching source should return None
        message = Message(source="blocked_source", event=DummyEvent())
        result = await handler.handle(message)

        assert result is None
        mock_process.assert_not_called()


@pytest.mark.asyncio
async def test_handle_source_filter_legacy_passes():
    """Test legacy source filtering when source matches"""
    _, _, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    # Set source filter
    handler.route_config.source = "allowed_source"

    with patch.object(handler, "_process_operations", return_value="test_result") as mock_process:
        with patch.object(handler, "_clean_active_tasks_safe", new_callable=AsyncMock) as mock_cleanup:
            # Matching source should proceed
            message = Message(source="allowed_source", event=DummyEvent())
            result = await handler.handle(message)

            assert result == "test_result"
            mock_process.assert_called_once()
            mock_cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_handle_normal_flow():
    """Test basic successful message processing"""
    _, _, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    with patch.object(handler, "_process_operations", return_value="test_result") as mock_process:
        with patch.object(handler, "_clean_active_tasks_safe", new_callable=AsyncMock) as mock_cleanup:
            message = Message(source="test", event=DummyEvent())
            result = await handler.handle(message)

            assert result == "test_result"
            mock_process.assert_called_once_with(message, handler.route_config.operations)
            mock_cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_handle_task_creation_and_cleanup():
    """Test that tasks are properly created and added to active_tasks"""
    _, _, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    with patch.object(handler, "_process_operations", return_value="test_result"):
        with patch.object(handler, "_clean_active_tasks_safe", new_callable=AsyncMock) as mock_cleanup:
            message = Message(source="test", event=DummyEvent())

            # Initially no active tasks
            assert len(handler._active_tasks) == 0

            result = await handler.handle(message)

            # Task should be cleaned up after completion
            assert result == "test_result"
            mock_cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_handle_max_concurrent_tasks_under_limit():
    """Test handling when under concurrent task limit"""
    _, _, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    # Set max concurrent tasks
    handler.route_config.max_concurrent_tasks = 3

    with patch.object(handler, "_process_operations", return_value="result") as mock_process:
        with patch.object(handler, "_clean_active_tasks_safe", new_callable=AsyncMock):
            with patch.object(handler, "_cancel_task_with_cleanup", new_callable=AsyncMock) as mock_cancel:
                message = Message(source="test", event=DummyEvent())
                result = await handler.handle(message)

                assert result == "result"
                # Should not cancel any tasks since under limit
                mock_cancel.assert_not_called()
                # Should call _process_operations once
                mock_process.assert_called_once_with(message, handler.route_config.operations)


@pytest.mark.asyncio
async def test_handle_awaits_existing_cancel_task():
    """Test that method waits for existing cancel task before proceeding"""
    _, _, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    # Create a mock cancel task
    mock_cancel_task = AsyncMock()
    handler._task_cancel_all_tasks = mock_cancel_task

    with patch("line.routes.await_tasks_safe", new_callable=AsyncMock) as mock_await:
        with patch.object(handler, "_process_operations", return_value="result"):
            with patch.object(handler, "_clean_active_tasks_safe", new_callable=AsyncMock):
                message = Message(source="test", event=DummyEvent())
                await handler.handle(message)

                # Should wait for existing cancel task
                mock_await.assert_called_once_with(mock_cancel_task)


@pytest.mark.asyncio
async def test_handle_cancellation_exception():
    """Test CancelledError handling"""
    _, _, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    # Mock _process_operations to raise CancelledError
    mock_cancelled_error = asyncio.CancelledError()

    with patch.object(handler, "_process_operations", side_effect=mock_cancelled_error):
        message = Message(source="test", event=DummyEvent())
        result = await handler.handle(message)

        # Should return None and not propagate exception
        assert result is None


@pytest.mark.asyncio
async def test_handle_process_operations_called_correctly():
    """Test that _process_operations is called with correct parameters"""
    _, _, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    with patch.object(handler, "_process_operations", return_value="result") as mock_process:
        with patch.object(handler, "_clean_active_tasks_safe", new_callable=AsyncMock):
            message = Message(source="test", event=DummyEvent())
            await handler.handle(message)

            # Verify correct parameters passed
            mock_process.assert_called_once_with(message, handler.route_config.operations)


@pytest.mark.asyncio
async def test_handle_no_operations():
    """Test behavior when route_config.operations is empty"""
    _, _, builder, handler = get_route_data().as_tuple()

    # Ensure operations list is empty
    assert handler.route_config.operations == [], "Expected 0 operations in the route config."

    with patch.object(handler, "_process_operations", return_value="empty_result") as mock_process:
        with patch.object(handler, "_clean_active_tasks_safe", new_callable=AsyncMock):
            message = Message(source="test", event=DummyEvent())
            result = await handler.handle(message)

            assert result is None, "Expected route to return None when no operations are set."
            # Do not call RouteHandler._process_operations when no operations are set.
            mock_process.assert_not_called()


@pytest.mark.asyncio
async def test_handle_task_lock_prevents_race_conditions():
    """Test that _task_lock properly synchronizes access to _active_tasks"""
    _, _, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    # Set max concurrent tasks to test the lock
    handler.route_config.max_concurrent_tasks = 1

    call_count = 0
    cancel_called = False

    async def slow_process_operations(message, operations):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.2)  # Longer sleep to ensure race condition
        return f"result_{call_count}"

    async def mock_cancel_task_with_cleanup(task):
        nonlocal cancel_called
        cancel_called = True
        # Actually cancel the task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    with patch.object(handler, "_process_operations", side_effect=slow_process_operations):
        with patch.object(handler, "_clean_active_tasks_safe", new_callable=AsyncMock):
            with patch.object(
                handler,
                "_cancel_task_with_cleanup",
                side_effect=mock_cancel_task_with_cleanup,
            ):
                # Start two concurrent handle calls
                message1 = Message(source="test1", event=DummyEvent())
                message2 = Message(source="test2", event=DummyEvent())

                task1 = asyncio.create_task(handler.handle(message1))
                task2 = asyncio.create_task(handler.handle(message2))

                results = await asyncio.gather(task1, task2, return_exceptions=True)

                # Verify cancellation was called
                assert cancel_called, "Second task should have triggered cancellation of first task."

                # First task might be cancelled, second should complete
                assert len(results) == 2
                # The first task should be cancelled, the second should complete successfully.
                assert results[0] is None, "Expected first task should be cancelled."
                assert results[1] is not None, "Expected second task to complete successfully."


# RouteHandler control operation tests.


@pytest.mark.asyncio
@pytest.mark.parametrize("num_handler_args", [None, 0, 1])
async def test_handle_interrupt(num_handler_args: Optional[int]):
    """Test RouteHandler._interrupt().

    Make sure it gets triggered properly.
    """
    call_count = 0

    def interrupt_handler_0arg():
        nonlocal call_count
        call_count += 1
        return f"result_{call_count}"

    def interrupt_handler_1arg(message):
        nonlocal call_count
        call_count += 1
        return f"result_{call_count}"

    async def map_fn_slow(x):
        await asyncio.sleep(0.5)
        return "should not be reached"

    _, bridge, builder, handler = get_route_data().as_tuple()
    builder.map(map_fn_slow)

    # Add interrupt handler.
    interrupt_handler = {
        None: None,
        0: interrupt_handler_0arg,
        1: interrupt_handler_1arg,
    }[num_handler_args]
    call_count = 0
    builder.interrupt_on(ControlEvent, interrupt_handler)
    interrupt_route = bridge.routes[ControlEvent][0]

    # Fire a message.
    message = Message(source="test", event=DummyEvent())
    task_main = asyncio.create_task(handler.handle(message))

    # Fire an interrupt.
    interrupt_message = Message(source="interrupt", event=ControlEvent())
    task_interrupt = asyncio.create_task(interrupt_route.handle(interrupt_message))

    # Wait for the main task to complete.
    results = await asyncio.gather(task_main, task_interrupt, return_exceptions=True)

    assert results[0] is None, "Main task should be cancelled."
    if num_handler_args is not None:
        assert call_count == 1, "Interrupt handler should have been called."


@pytest.mark.asyncio
async def test_handle_interrupt_no_tasks():
    """Test that interrupt works when there are no tasks.

    The handler should not be called and the interrupt should be ignored.
    """
    _, bridge, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    # Add interrupt handler.
    call_count = 0

    def interrupt_handler(message):
        nonlocal call_count
        call_count += 1

    builder.interrupt_on(ControlEvent, interrupt_handler)
    interrupt_route = bridge.routes[ControlEvent][0]

    # Fire an interrupt without any active tasks.
    interrupt_message = Message(source="interrupt", event=ControlEvent())
    await interrupt_route.handle(interrupt_message)

    # No tasks should be active.
    assert len(handler._active_tasks) == 0

    # The interrupt handler should not be called.
    assert call_count == 0, "Interrupt handler should not be called."


@pytest.mark.asyncio
async def test_handle_interrupt_consecutive_calls():
    """
    Test that interrupt waits for existing interrupt task to complete.

    We take the opinionated stance that the second interrupt should wait for the first one to complete.
    This is to avoid race conditions where the first interrupt could be running a handler
    that is not expected to be interrupted.
    """
    _, _, builder, handler = get_route_data().as_tuple()

    call_count = 0

    async def slow_operation(message):
        await asyncio.sleep(0.5)
        return "completed"

    def slow_interrupt_handler1(message):
        time.sleep(0.2)
        nonlocal call_count
        call_count += 1

    def interrupt_handler2(message):
        nonlocal call_count
        assert call_count == 1, "Expected first interrupt handler to be called and handler completed."

    builder.map(slow_operation)

    # Fire a message.
    message = Message(source="test", event=DummyEvent())
    task_main = asyncio.create_task(handler.handle(message))

    await asyncio.sleep(0.05)  # Wait for the main task to start.
    assert len(handler._active_tasks) == 1, "Expected one active task."

    # Fire two interrupt messages.
    # Make tasks to ensure the first interrupt runs slightly before the second one.
    interrupt_message = Message(source="interrupt", event=ControlEvent())
    handler._interrupt(interrupt_message, slow_interrupt_handler1)

    await asyncio.sleep(0.01)

    interrupt_message = Message(source="interrupt", event=ControlEvent())
    handler._interrupt(interrupt_message, interrupt_handler2)

    result = await task_main
    await handler._task_cancel_all_tasks  # type: ignore

    assert result is None, "Expected main task to be cancelled."
    assert call_count == 1


# Concurrency stress tests


@pytest.mark.asyncio
async def test_handle_rapid_successive_calls():
    """Test handling many rapid successive calls to verify concurrency control"""
    _, _, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    # Set max concurrent tasks to a low number
    handler.route_config.max_concurrent_tasks = 3

    call_count = 0
    completed_calls = []

    async def track_calls(message, operations):
        nonlocal call_count
        call_count += 1
        call_id = call_count
        await asyncio.sleep(0.05)  # Simulate work
        completed_calls.append(call_id)
        return f"result_{call_id}"

    with patch.object(handler, "_process_operations", side_effect=track_calls):
        with patch.object(handler, "_clean_active_tasks_safe", new_callable=AsyncMock):
            with patch.object(handler, "_cancel_task_with_cleanup", new_callable=AsyncMock) as mock_cancel:
                # Fire 10 rapid successive calls
                messages = [Message(source=f"test{i}", event=DummyEvent()) for i in range(10)]
                tasks = [asyncio.create_task(handler.handle(msg)) for msg in messages]

                results = await asyncio.gather(*tasks)

                # All should complete successfully
                assert len(results) == 10
                assert all(result is not None for result in results)
                # Should have called cancel due to exceeding max_concurrent_tasks
                assert mock_cancel.call_count > 0
                # All calls should eventually complete
                assert len(completed_calls) == 10


@pytest.mark.asyncio
async def test_handle_interrupt_during_execution():
    """Test race conditions when interrupting tasks during execution"""
    _, _, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    interrupt_called = False
    execution_started = asyncio.Event()
    can_complete = asyncio.Event()

    async def long_running_operation(message, operations):
        execution_started.set()
        await can_complete.wait()  # Wait for permission to complete
        return "completed"

    def interrupt_handler(message):
        nonlocal interrupt_called
        interrupt_called = True

    # Add interrupt handler
    handler.route_config.interrupt_handlers[DummyEvent] = _EventHandlerDict(interrupt_handler)

    with patch.object(handler, "_process_operations", side_effect=long_running_operation):
        with patch.object(handler, "_clean_active_tasks_safe", new_callable=AsyncMock):
            # Start a long-running task
            message = Message(source="test", event=DummyEvent())
            task = asyncio.create_task(handler.handle(message))

            # Wait for execution to start
            await execution_started.wait()

            # Trigger interrupt while task is running
            interrupt_message = Message(source="interrupt", event=DummyEvent())
            handler._interrupt(interrupt_message)

            # Allow original task to complete
            can_complete.set()
            await task

            # Wait for the interrupt cleanup task to complete
            if handler._task_cancel_all_tasks:
                await handler._task_cancel_all_tasks

            # Task should be cancelled/interrupted
            assert len([t for t in handler._active_tasks if not t.done()]) == 0
            # Cleanup should handle the interruption gracefully


@pytest.mark.asyncio
async def test_handle_task_memory_leak_prevention():
    """Test that completed tasks are properly cleaned up to prevent memory leaks"""
    _, _, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    # Track task states
    task_states = []

    async def track_task_lifecycle(message, operations):
        # Simulate varying execution times
        await asyncio.sleep(0.01)
        return "completed"

    with patch.object(handler, "_process_operations", side_effect=track_task_lifecycle):
        # Execute multiple tasks sequentially
        for i in range(20):
            message = Message(source=f"test{i}", event=DummyEvent())
            await handler.handle(message)

            # Check that active_tasks doesn't grow unbounded
            active_count = len([t for t in handler._active_tasks if not t.done()])
            task_states.append(active_count)

        # Should not accumulate completed tasks
        final_active_tasks = len([t for t in handler._active_tasks if not t.done()])
        assert final_active_tasks == 0
        # Most intermediate states should have 0 or minimal active tasks
        assert max(task_states) <= 1


@pytest.mark.asyncio
async def test_concurrent_broadcasts():
    """Test multiple concurrent broadcast operations"""
    _, _, builder, handler = get_route_data().as_tuple()
    builder.map(lambda x: x)

    mock_bus = AsyncMock()
    handler.bridge.bus = mock_bus

    async def concurrent_broadcast_operation(data, broadcast_id):
        operations = [{"_fn_type": "broadcast", "event": None}]
        await handler._process_operations(f"{data}_{broadcast_id}", operations)

    # Run multiple concurrent broadcasts
    tasks = []
    for i in range(10):
        task = asyncio.create_task(concurrent_broadcast_operation("data", i))
        tasks.append(task)

    await asyncio.gather(*tasks)

    # Should have broadcast 10 times
    assert mock_bus.broadcast.call_count == 10
    # Each broadcast should contain unique data
    broadcast_events = [call[0][0].event for call in mock_bus.broadcast.call_args_list]
    expected_events = [f"data_{i}" for i in range(10)]
    assert sorted(broadcast_events) == sorted(expected_events)


# RouteHandler._process_operations() tests


@pytest.mark.asyncio
async def test_process_operations_empty_operations():
    """Test with empty operations list"""
    _, bridge, builder, handler = get_route_data().as_tuple()

    test_data = "test_input"
    result = await handler._process_operations(test_data, [])

    assert result == test_data


@pytest.mark.asyncio
async def test_process_operations_call_sync_function():
    """Test synchronous function call."""
    _, _, _, handler = get_route_data().as_tuple()

    def transform_data(data):
        return data + "_transformed"

    # Mock the bus to verify broadcast receives transformed data
    mock_bus = AsyncMock()
    handler.bridge.bus = mock_bus

    operations = [
        {"_fn_type": "map", "fn": transform_data},
        {"_fn_type": "broadcast", "event": None},
    ]
    result = await handler._process_operations("input", operations)

    assert result is None  # _process_operations returns None at end
    # Verify the broadcast received the transformed data
    mock_bus.broadcast.assert_called_once()
    call_args = mock_bus.broadcast.call_args[0][0]
    assert call_args.event == "input_transformed"


@pytest.mark.asyncio
async def test_process_operations_call_async_function():
    """Test asynchronous function call"""
    _, _, _, handler = get_route_data().as_tuple()

    async_transform = AsyncMock(return_value="input_async")

    # Mock the bus to verify broadcast receives transformed data
    mock_bus = AsyncMock()
    handler.bridge.bus = mock_bus

    operations = [
        {"_fn_type": "map", "fn": async_transform},
        {"_fn_type": "broadcast", "event": None},
    ]
    result = await handler._process_operations("input", operations)

    assert result is None  # _process_operations returns None at end
    async_transform.assert_called_once_with("input")
    # Verify the broadcast received the transformed data
    mock_bus.broadcast.assert_called_once()
    call_args = mock_bus.broadcast.call_args[0][0]
    assert call_args.event == "input_async"


@pytest.mark.asyncio
async def test_process_operations_multiple_calls():
    """Test chain of multiple call operations"""
    _, _, _, handler = get_route_data().as_tuple()

    add_suffix = Mock(return_value="input_suffix")
    add_prefix = Mock(return_value="prefix_input_suffix")

    # Mock the bus to verify broadcast receives final transformed data
    mock_bus = AsyncMock()
    handler.bridge.bus = mock_bus

    operations = [
        {"_fn_type": "map", "fn": add_suffix},
        {"_fn_type": "map", "fn": add_prefix},
        {"_fn_type": "broadcast", "event": None},
    ]
    result = await handler._process_operations("input", operations)

    assert result is None  # _process_operations returns None at end
    add_suffix.assert_called_once_with("input")
    add_prefix.assert_called_once_with("input_suffix")
    # Verify the broadcast received the final transformed data
    mock_bus.broadcast.assert_called_once()
    call_args = mock_bus.broadcast.call_args[0][0]
    assert call_args.event == "prefix_input_suffix"


@pytest.mark.asyncio
async def test_process_operations_filter_passes_sync():
    """Test synchronous filter that returns True"""
    _, _, _, handler = get_route_data().as_tuple()

    def allow_all(data):
        return True

    operations = [{"_fn_type": "filter", "fn": allow_all}]
    result = await handler._process_operations("input", operations)

    assert result is None  # _process_operations returns None at end


@pytest.mark.asyncio
async def test_process_operations_filter_fails_sync():
    """Test synchronous filter that returns False"""
    _, _, _, handler = get_route_data().as_tuple()

    def block_all(data):
        return False

    operations = [{"_fn_type": "filter", "fn": block_all}]
    result = await handler._process_operations("input", operations)

    assert result is None  # Filter failed, early termination


@pytest.mark.asyncio
async def test_process_operations_filter_passes_async():
    """Test asynchronous filter that returns True"""
    _, _, _, handler = get_route_data().as_tuple()

    async def async_allow(data):
        return True

    operations = [{"_fn_type": "filter", "fn": async_allow}]
    result = await handler._process_operations("input", operations)

    assert result is None  # _process_operations returns None at end


@pytest.mark.asyncio
async def test_process_operations_filter_fails_async():
    """Test asynchronous filter that returns False"""
    _, _, _, handler = get_route_data().as_tuple()

    async def async_block(data):
        return False

    operations = [{"_fn_type": "filter", "fn": async_block}]
    result = await handler._process_operations("input", operations)

    assert result is None  # Filter failed, early termination


@pytest.mark.asyncio
async def test_process_operations_stream_operation():
    """Test stream operation calls _handle_stream"""
    _, _, _, handler = get_route_data().as_tuple()

    def generator_fn(data):
        for i in range(10):
            yield data + f"_{i}"

    # Mock the bus to verify each streamed item gets broadcast
    mock_bus = AsyncMock()
    handler.bridge.bus = mock_bus

    operations = [
        {"_fn_type": "stream", "fn": generator_fn},
        {"_fn_type": "broadcast", "event": None},
    ]

    # Mock _process_operations for remaining ops to track how many times it's called
    with patch.object(handler, "_process_operations", wraps=handler._process_operations) as mock_process:
        result = await handler._process_operations("input", operations)

        assert result is None  # Stream consumed
        # Should be called 11 times: once for the initial call + 10 times for each yielded item
        assert mock_process.call_count == 11
        # Should broadcast each of the 10 yielded items
        assert mock_bus.broadcast.call_count == 10
        # Verify the first and last broadcast calls contain correct data
        first_call = mock_bus.broadcast.call_args_list[0][0][0]
        assert first_call.event == "input_0"
        last_call = mock_bus.broadcast.call_args_list[9][0][0]
        assert last_call.event == "input_9"


@pytest.mark.asyncio
async def test_process_operations_stream_async_generator():
    """Test stream operation with async generator function"""
    _, _, _, handler = get_route_data().as_tuple()

    async def async_generator_fn(data):
        for i in range(10):
            yield data + f"_{i}"

    # Mock the bus to verify each streamed item gets broadcast
    mock_bus = AsyncMock()
    handler.bridge.bus = mock_bus

    operations = [
        {"_fn_type": "stream", "fn": async_generator_fn},
        {"_fn_type": "broadcast", "event": None},
    ]

    # Mock _process_operations for remaining ops to track how many times it's called
    with patch.object(handler, "_process_operations", wraps=handler._process_operations) as mock_process:
        result = await handler._process_operations("input", operations)

        assert result is None  # Stream consumed
        # Should be called 11 times: once for the initial call + 10 times for each yielded item
        assert mock_process.call_count == 11
        # Should broadcast each of the 10 yielded items
        assert mock_bus.broadcast.call_count == 10
        # Verify the first and last broadcast calls contain correct data
        first_call = mock_bus.broadcast.call_args_list[0][0][0]
        assert first_call.event == "input_0"
        last_call = mock_bus.broadcast.call_args_list[9][0][0]
        assert last_call.event == "input_9"


@pytest.mark.asyncio
async def test_process_operations_stream_with_remaining_ops():
    """Test that remaining operations are passed to _handle_stream"""
    _, _, _, handler = get_route_data().as_tuple()

    def generator_fn(data):
        yield data

    remaining_op = {"_fn_type": "map", "fn": lambda x: x}
    operations = [{"_fn_type": "stream", "fn": generator_fn}, remaining_op]

    with patch.object(handler, "_handle_stream", new_callable=AsyncMock) as mock_handle:
        await handler._process_operations("input", operations)

        mock_handle.assert_called_once_with("input", generator_fn, [remaining_op])


@pytest.mark.asyncio
async def test_process_operations_broadcast_no_event_cls():
    """Test broadcast with event_cls = None"""
    _, _, _, handler = get_route_data().as_tuple()

    # Mock the bus
    mock_bus = AsyncMock()
    handler.bridge.bus = mock_bus

    operations = [{"_fn_type": "broadcast", "event": None}]
    result = await handler._process_operations("test_data", operations)

    assert result is None
    # Should broadcast the data as-is
    mock_bus.broadcast.assert_called_once()
    call_args = mock_bus.broadcast.call_args[0][0]  # First arg of first call
    assert call_args.source == handler.bridge.node_id
    assert call_args.event == "test_data"


@pytest.mark.asyncio
async def test_process_operations_broadcast_with_event_cls():
    """Test broadcast with specific event class"""
    _, _, _, handler = get_route_data().as_tuple()

    # Mock the bus
    mock_bus = AsyncMock()
    handler.bridge.bus = mock_bus

    # Create test data that can be used to construct NumberEvent
    test_data = {"value": 42}

    operations = [{"_fn_type": "broadcast", "event": NumberEvent}]
    result = await handler._process_operations(test_data, operations)

    assert result is None
    # Should construct NumberEvent and broadcast it
    mock_bus.broadcast.assert_called_once()
    call_args = mock_bus.broadcast.call_args[0][0]
    assert call_args.source == handler.bridge.node_id
    assert isinstance(call_args.event, NumberEvent)
    assert call_args.event.value == 42


@pytest.mark.asyncio
async def test_process_operations_broadcast_no_bus():
    """Test broadcast when bridge.bus is None"""
    _, _, _, handler = get_route_data().as_tuple()
    # No bus set
    handler.bridge.bus = None

    operations = [{"_fn_type": "broadcast", "event": None}]
    result = await handler._process_operations("test_data", operations)

    # Should complete without error
    assert result is None


@pytest.mark.asyncio
async def test_process_operations_broadcast_no_current_data():
    """Test broadcast when current_data is None"""
    _, _, _, handler = get_route_data().as_tuple()

    mock_bus = AsyncMock()
    handler.bridge.bus = mock_bus

    operations = [{"_fn_type": "broadcast", "event": None}]
    result = await handler._process_operations(None, operations)

    # Should not call broadcast when current_data is None
    mock_bus.broadcast.assert_not_called()
    assert result is None


@pytest.mark.asyncio
async def test_process_operations_broadcast_event_construction_error():
    """Test error handling when event construction fails"""
    _, _, _, handler = get_route_data().as_tuple()

    mock_bus = AsyncMock()
    handler.bridge.bus = mock_bus

    # Data that will fail NumberEvent construction (missing 'value' field)
    invalid_data = {"invalid_field": "value"}

    operations = [{"_fn_type": "broadcast", "event": NumberEvent}]

    with pytest.raises((ValidationError, KeyError)):  # NumberEvent construction should fail
        await handler._process_operations(invalid_data, operations)


@pytest.mark.asyncio
async def test_process_operations_call_then_filter_pass():
    """Test call operation followed by passing filter"""
    _, _, _, handler = get_route_data().as_tuple()

    def transform(data):
        return data + "_transformed"

    def allow_transformed(data):
        return "_transformed" in data

    operations = [
        {"_fn_type": "map", "fn": transform},
        {"_fn_type": "filter", "fn": allow_transformed},
    ]
    result = await handler._process_operations("input", operations)

    assert result is None  # Should complete successfully


@pytest.mark.asyncio
async def test_process_operations_call_then_filter_fail():
    """Test call operation followed by failing filter"""
    _, _, _, handler = get_route_data().as_tuple()

    def transform(data):
        return data + "_transformed"

    def block_transformed(data):
        return "_transformed" not in data

    operations = [
        {"_fn_type": "map", "fn": transform},
        {"_fn_type": "filter", "fn": block_transformed},
    ]
    result = await handler._process_operations("input", operations)

    assert result is None  # Filter failed, early termination


@pytest.mark.asyncio
async def test_process_operations_filter_then_broadcast():
    """Test filter followed by broadcast"""
    _, _, _, handler = get_route_data().as_tuple()

    mock_bus = AsyncMock()
    handler.bridge.bus = mock_bus

    def allow_all(data):
        return True

    operations = [
        {"_fn_type": "filter", "fn": allow_all},
        {"_fn_type": "broadcast", "event": None},
    ]
    result = await handler._process_operations("test_data", operations)

    assert result is None
    # Should broadcast since filter passed
    mock_bus.broadcast.assert_called_once()


@pytest.mark.asyncio
async def test_process_operations_call_function_throws():
    """Test exception handling in call operations"""
    _, _, _, handler = get_route_data().as_tuple()

    def failing_function(data):
        raise ValueError("Test error")

    operations = [{"_fn_type": "map", "fn": failing_function}]

    with pytest.raises(ValueError, match="Test error"):
        await handler._process_operations("input", operations)


@pytest.mark.asyncio
async def test_process_operations_unknown_fn_type():
    """Test behavior with unrecognized _fn_type"""
    _, _, _, handler = get_route_data().as_tuple()

    operations = [{"_fn_type": "unknown_type", "fn": lambda x: x}]
    result = await handler._process_operations("input", operations)

    # Should skip unknown operations and return None at end
    assert result is None


@pytest.mark.asyncio
async def test_process_operations_missing_fn():
    """Test operations with missing or None fn"""
    _, _, _, handler = get_route_data().as_tuple()

    operations = [{"_fn_type": "map"}]  # Missing 'fn'

    # Should handle gracefully - calling None will raise TypeError
    with pytest.raises(TypeError):
        await handler._process_operations("input", operations)
