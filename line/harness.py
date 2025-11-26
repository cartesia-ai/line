"""
ConversationHarness - WebSocket communication layer for agents
Handles input/output queues and event coordination
"""

import asyncio
from asyncio import QueueEmpty
import json
from typing import Any, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger
from pydantic import TypeAdapter

from line.events import (
    AgentSpeechSent,
    AgentStartedSpeaking,
    AgentStoppedSpeaking,
    DTMFInputEvent,
    UserStartedSpeaking,
    UserStoppedSpeaking,
    UserTranscriptionReceived,
    UserUnknownInputReceived,
)
from line.harness_types import (
    AgentSpeechInput,
    AgentStateInput,
    DTMFInput,
    DTMFOutput,
    EndCallOutput,
    ErrorOutput,
    InputMessage,
    LogEventOutput,
    LogMetricOutput,
    MessageOutput,
    OutputMessage,
    ToolCallOutput,
    TranscriptionInput,
    TransferOutput,
    UserStateInput,
)


class State:
    """User voice states."""

    SPEAKING = "speaking"
    IDLE = "idle"


class ConversationHarness:
    """
    Manages WebSocket communication, input/output queues, and coordination events
    for reasoning agents. Handles message parsing and event triggering.
    """

    def __init__(
        self,
        websocket: WebSocket,
        shutdown_event: asyncio.Event,
    ):
        """
        Initialize the conversation harness

        Args:
            websocket: FastAPI WebSocket connection
            shutdown_event: Event to signal shutdown
        """
        self.websocket = websocket

        # Use provided queues and events
        self.input_queue = asyncio.Queue()
        self.shutdown_event = shutdown_event

        # Task management
        self.input_task: Optional[asyncio.Task] = None

        # State tracking
        self.is_running = False

    async def start(self):
        """
        Start the harness tasks for input and output processing
        """
        if self.is_running:
            logger.warning("ConversationHarness already running")
            return

        self.is_running = True
        logger.debug("Starting ConversationHarness")

        # Start input and output tasks
        self.input_task = asyncio.create_task(self._input_processor())

        logger.debug("ConversationHarness started with input/output processors")

    async def _input_processor(self):
        """
        Continuously receive messages from WebSocket, parse them, and handle events
        """
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Receive message from WebSocket
                    message = await self.websocket.receive_json()
                    input = TypeAdapter(InputMessage).validate_python(message)
                    # Process the message and handle events
                    await self.input_queue.put(input)

                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected")
                    self.shutdown_event.set()
                    break
                except json.JSONDecodeError as e:
                    logger.exception(f"Failed to parse JSON message: {e}")
                    continue
                except Exception as e:
                    logger.exception(f"Error in input processor: {e}")
                    if not self.shutdown_event.is_set():
                        await asyncio.sleep(0.1)  # Brief pause before retry

        except asyncio.CancelledError:
            logger.info("Input processor cancelled")
        except Exception as e:
            logger.exception(f"Unexpected error in input processor: {e}")

    async def get(self) -> InputMessage:
        """Get a message from the input queue"""
        return await self.input_queue.get()

    async def _send(self, output: OutputMessage):
        try:
            if not self.shutdown_event.is_set():
                await self.websocket.send_json(output.model_dump())
        except Exception as e:
            logger.warning(f"Failed to send message via WebSocket: {e}")
            self.shutdown_event.set()

    async def end_call(self):
        """
        Send end_call message and signal shutdown
        """
        await self._send(EndCallOutput())
        logger.info("End call message sent")

    async def transfer_call(self, target_phone_number: str, timeout_s: int):
        """
        Send transfer_call message

        Args:
            target_phone_number: Optional target phone number for call transfer
        """
        await self._send(TransferOutput(target_phone_number=target_phone_number))
        logger.info(
            f"Transfer request sent. Waiting {timeout_s} seconds before"
            + f"gracefully shutting down the agent. {target_phone_number=}"
        )
        await asyncio.sleep(timeout_s)
        logger.info("Initiating shutdown...")
        self.shutdown_event.set()

    async def send_message(self, message: str):
        """Send a message via WebSocket with connection state checking"""
        logger.info(f'🤖 Agent said: "{message}"')
        await self._send(MessageOutput(content=message))

    async def send_error(self, error: str):
        """Send an error message via WebSocket with connection state checking"""
        await self._send(ErrorOutput(content=error))

    async def send_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_call_id: Optional[str] = None,
        result: Optional[str] = None,
    ):
        """Send a tool call result via WebSocket with connection state checking"""
        await self._send(
            ToolCallOutput(
                name=tool_name,
                arguments=tool_args,
                result=result,
                id=tool_call_id,
            )
        )

    async def log_event(self, event: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Send a log event via WebSocket

        Args:
            event: The event name/type being logged
            metadata: Optional metadata dictionary for the event
        """
        logger.debug(f"📊 Logging event: {event}" + (f" - {metadata}" if metadata else ""))
        await self._send(LogEventOutput(event=event, metadata=metadata))

    async def log_metric(self, name: str, value: Any):
        """
        Send a log metric via WebSocket

        Args:
            name: The metric name
            value: The metric value (can be any JSON-serializable type)
        """
        logger.debug(f"📈 Logging metric: {name}={value}")
        await self._send(LogMetricOutput(name=name, value=value))

    async def send_dtmf(self, button: str):
        """
        Send a DTMF event via WebSocket

        Args:
            button: The DTMF button to send
        """
        await self._send(DTMFOutput(button=button))

    async def cleanup(self):
        """
        Clean up resources and stop all tasks
        """
        logger.info("Cleaning up ConversationHarness")

        # Signal shutdown
        self.shutdown_event.set()
        self.is_running = False

        # Cancel tasks
        if self.input_task and not self.input_task.done():
            self.input_task.cancel()
            try:
                await self.input_task
            except asyncio.CancelledError:
                pass

        # Clear any remaining messages in queues
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
                self.input_queue.task_done()
            except QueueEmpty:
                break

        logger.info("ConversationHarness cleanup completed")

    def map_to_events(self, message: InputMessage) -> List[Any]:
        """Convert harness-specific message to bus events."""
        if isinstance(message, UserStateInput):
            if message.value == State.SPEAKING:
                logger.info("🎤 User started speaking")
                return [UserStartedSpeaking()]
            elif message.value == State.IDLE:
                logger.info("🔇 User stopped speaking")
                return [UserStoppedSpeaking()]
        elif isinstance(message, TranscriptionInput):
            logger.info(f'📝 User said: "{message.content}"')
            return [UserTranscriptionReceived(content=message.content)]
        elif isinstance(message, AgentStateInput):
            if message.value == State.SPEAKING:
                logger.info("🎤 Agent started speaking")
                return [AgentStartedSpeaking()]
            elif message.value == State.IDLE:
                logger.info("🔇 Agent stopped speaking")
                return [AgentStoppedSpeaking()]
        elif isinstance(message, AgentSpeechInput):
            logger.info(f'🗣️ Agent speech sent: "{message.content}"')
            return [AgentSpeechSent(content=message.content)]
        elif isinstance(message, DTMFInput):
            logger.info(f"🔔 DTMF received: {message.button}")
            return [DTMFInputEvent(button=message.button)]
        else:
            # Fallback for unknown types.
            logger.warning(f"Unknown message type: {type(message).__name__} ({message.model_dump_json()})")
            return [UserUnknownInputReceived(input_data=message.model_dump_json())]

        return []  # No events for unhandled states
