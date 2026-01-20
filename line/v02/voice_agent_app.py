"""
VoiceAgentApp - Simplified voice agent application that handles websocket communication directly.
"""

import asyncio
from datetime import datetime, timezone
import json
import os
import re
from typing import AsyncIterable, Awaitable, Callable, List, Optional
from urllib.parse import urlencode

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from loguru import logger
from pydantic import TypeAdapter
import uvicorn

from line.call_request import CallRequest, PreCallResult
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
from line.v02.agent import Agent, AgentSpec, EventFilter, TurnEnv
from line.v02.events import (
    AgentDTMFSent,
    AgentEndCall,
    AgentSendDTMF,
    AgentSendText,
    AgentTextSent,
    AgentToolCalled,
    AgentToolCalledInput,
    AgentToolReturned,
    AgentToolReturnedInput,
    AgentTransferCall,
    AgentTurnEnded,
    AgentTurnStarted,
    CallEnded,
    CallStarted,
    InputEvent,
    LogMessage,
    LogMetric,
    OutputEvent,
    SpecificAgentDTMFSent,
    SpecificAgentTextSent,
    SpecificAgentToolCalled,
    SpecificAgentToolReturned,
    SpecificAgentTurnEnded,
    SpecificAgentTurnStarted,
    SpecificCallEnded,
    SpecificCallStarted,
    SpecificInputEvent,
    SpecificUserDtmfSent,
    SpecificUserTextSent,
    SpecificUserTurnEnded,
    SpecificUserTurnStarted,
    UserDtmfSent,
    UserTextSent,
    UserTurnEnded,
    UserTurnStarted,
)


class UserState:
    """User voice states."""

    SPEAKING = "speaking"
    IDLE = "idle"


class AgentEnv:
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        self.loop = loop


load_dotenv()


class VoiceAgentApp:
    """
    VoiceAgentApp handles responding ot HTTP requests and managing websocket connections

    Uses ConversationRunner to manage the websocket loop for each connection.
    """

    def __init__(
        self,
        get_agent: Callable[[AgentEnv, CallRequest], Awaitable[AgentSpec]],
        pre_call_handler: Optional[Callable[[CallRequest], Awaitable[Optional[PreCallResult]]]] = None,
    ):
        """
        Initialize the VoiceAgentApp.

        Args:
            get_agent: Async function that creates a Node from AgentEnv and CallRequest.
            pre_call_handler: Optional async function to configure call settings before connection.
        """
        self.fastapi_app = FastAPI()
        self.get_agent = get_agent
        self.pre_call_handler = pre_call_handler
        self.ws_route = "/ws"

        self.fastapi_app.add_api_route("/chats", self.create_chat_session, methods=["POST"])
        self.fastapi_app.add_api_route("/status", self.get_status, methods=["GET"])
        self.fastapi_app.add_websocket_route(self.ws_route, self.websocket_endpoint)

    async def create_chat_session(self, request: Request) -> dict:
        """Create a new chat session and return the websocket URL."""
        body = await request.json()

        call_request = CallRequest(
            call_id=body.get("call_id", "unknown"),
            from_=body.get("from_", "unknown"),
            to=body.get("to", "unknown"),
            agent_call_id=body.get("agent_call_id", body.get("call_id", "unknown")),
            metadata=body.get("metadata", {}),
        )

        config = None
        if self.pre_call_handler:
            try:
                result = await self.pre_call_handler(call_request)
                if result is None:
                    raise HTTPException(status_code=403, detail="Call rejected")

                call_request.metadata.update(result.metadata)
                config = result.config

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error in pre_call_handler: {str(e)}")
                raise HTTPException(status_code=500, detail="Server error in call processing") from e

        url_params = {
            "call_id": call_request.call_id,
            "from": call_request.from_,
            "to": call_request.to,
            "agent_call_id": call_request.agent_call_id,
            "metadata": json.dumps(call_request.metadata),
        }

        query_string = urlencode(url_params)
        websocket_url = f"{self.ws_route}?{query_string}"

        response = {"websocket_url": websocket_url}
        if config:
            response["config"] = config
        return response

    async def get_status(self) -> dict:
        """Status endpoint that returns OK if the server is running."""
        logger.info("Health check endpoint called - voice agent is ready 🤖✅")
        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "cartesia-line",
        }

    async def websocket_endpoint(self, websocket: WebSocket):
        """Websocket endpoint that manages the complete call lifecycle."""
        await websocket.accept()
        logger.info("Client connected")

        query_params = dict(websocket.query_params)

        metadata = {}
        if "metadata" in query_params:
            try:
                metadata = json.loads(query_params["metadata"])
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Invalid metadata JSON: {query_params['metadata']}")
                metadata = {}

        call_request = CallRequest(
            call_id=query_params.get("call_id", "unknown"),
            from_=query_params.get("from", "unknown"),
            to=query_params.get("to", "unknown"),
            agent_call_id=query_params.get("agent_call_id", "unknown"),
            metadata=metadata,
        )

        runner: Optional[ConversationRunner] = None
        try:
            # Create the AgentEnv with the current event loop
            loop = asyncio.get_running_loop()
            env = AgentEnv(loop)
            agent_spec = await self.get_agent(env, call_request)

            # Create and run the conversation runner
            runner = ConversationRunner(websocket, agent_spec, env)
            await runner.run()

        except Exception as e:
            logger.exception(f"Error: {str(e)}")
            if runner:
                await runner.send_error("System has encountered an error, please try again later.")
        finally:
            logger.info("Websocket session ended")

    def run(self, host="0.0.0.0", port=None):
        """Run the voice agent server."""
        port = port or int(os.getenv("PORT", 8000))
        uvicorn.run(self.fastapi_app, host=host, port=port)


class ConversationRunner:
    """
    Manages the websocket loop for a single conversation.
    Converts websocket messages to v0.2 InputEvents, applies run/cancel filters,
    drives the agent async iterable, and serializes agent OutputEvents back to
    the websocket.
    """

    def __init__(self, websocket: WebSocket, agent_spec: AgentSpec, env: AgentEnv):
        """
        Initialize the ConversationRunner.

        Args:
            websocket: The WebSocket connection.
            agent_spec: Agent or (Agent, run_filter, cancel_filter).
            env: Environment passed to the agent.
        """
        self.websocket = websocket
        self.env = env
        self.shutdown_event = asyncio.Event()
        self.history: List[SpecificInputEvent] = []
        self.emitted_agent_text: str = (
            ""  # Buffer for all AgentSendText content (for whitespace interpolation)
        )

        self.agent_callable, self.run_filter, self.cancel_filter = self._prepare_agent(agent_spec)
        self.agent_task: Optional[asyncio.Task] = None

    ######### Initialization Methods #########

    def _prepare_agent(
        self, agent_spec: AgentSpec
    ) -> tuple[
        Callable[[TurnEnv, InputEvent], AsyncIterable[OutputEvent]],
        Callable[[InputEvent], bool],
        Callable[[InputEvent], bool],
    ]:
        """Extract agent callable and filters from agent_spec."""

        def default_run(ev: InputEvent) -> bool:
            return isinstance(ev, (CallStarted, UserTurnEnded, CallEnded))

        def default_cancel(ev: InputEvent) -> bool:
            return isinstance(ev, UserTurnStarted)

        agent_obj: Agent
        run_spec: EventFilter
        cancel_spec: EventFilter

        if isinstance(agent_spec, (list, tuple)) and len(agent_spec) == 3:
            agent_obj, run_spec, cancel_spec = agent_spec
        else:
            agent_obj = agent_spec
            run_spec = default_run
            cancel_spec = default_cancel

        run_filter = self._normalize_filter(run_spec)
        cancel_filter = self._normalize_filter(cancel_spec)

        def _agent_callable(turn_env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
            if hasattr(agent_obj, "process") and callable(agent_obj.process):
                return agent_obj.process(turn_env, event)  # type: ignore[return-value]
            if callable(agent_obj):
                return agent_obj(turn_env, event)  # type: ignore[return-value]
            raise TypeError("Agent must be callable or have a callable 'process' method.")

        return _agent_callable, run_filter, cancel_filter

    def _normalize_filter(self, filter_spec: EventFilter) -> Callable[[InputEvent], bool]:
        """Normalize EventFilter spec to a callable."""
        if callable(filter_spec):
            return filter_spec
        if isinstance(filter_spec, (list, tuple)):
            return lambda event: any(isinstance(event, cls) for cls in filter_spec)
        raise TypeError("EventFilter must be callable or list")

    ######### Run Loop Methods #########

    async def run(self):
        """
        Run the conversation loop.

        Processes incoming websocket messages until shutdown.
        """
        # Emit call_started to seed history/context
        start_event, self.history = self._wrap_with_history(self.history, SpecificCallStarted())
        await self._handle_event(TurnEnv(), start_event)

        while not self.shutdown_event.is_set():
            try:
                # Receive message from WebSocket
                message = await self.websocket.receive_json()
                input_msg = TypeAdapter(InputMessage).validate_python(message)

                # Map to events
                ev, self.history = self._map_input_event(self.history, input_msg)
                await self._handle_event(TurnEnv(), ev)

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected in loop")
                end_event, self.history = self._wrap_with_history(self.history, SpecificCallEnded())
                await self._handle_event(TurnEnv(), end_event)
                self.shutdown_event.set()
            except json.JSONDecodeError as e:
                logger.exception(f"Failed to parse JSON message: {e}")
            except Exception as e:
                await self.send_error(f"Error processing message: {e}")

        if self.agent_task:
            await self.agent_task

    async def _handle_event(self, turn_env: TurnEnv, event: InputEvent) -> None:
        """Apply run/cancel filters for a single event."""
        if self.run_filter(event):
            await self._start_agent_task(turn_env, event)
        elif self.cancel_filter(event):
            await self._cancel_agent_task()

    async def _start_agent_task(self, turn_env: TurnEnv, event: InputEvent) -> None:
        """Start the agent async iterable for the given event."""
        await self._cancel_agent_task()

        async def runner():
            try:
                async for output in self.agent_callable(turn_env, event):
                    # Buffer AgentSendText content for whitespace interpolation
                    if isinstance(output, AgentSendText):
                        self.emitted_agent_text += output.text
                    mapped = self._map_output_event(output)

                    if self.shutdown_event.is_set():
                        break
                    if mapped is None:
                        continue
                    await self.websocket.send_json(mapped.model_dump())
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"Agent iterable error: {exc}")
                await self.send_error(f"Unexpected error: {exc}")

        self.agent_task = asyncio.create_task(runner())

    async def _cancel_agent_task(self) -> None:
        """Cancel any running agent iterable task."""
        if self.agent_task and not self.agent_task.done():
            self.agent_task.cancel()
            try:
                await self.agent_task
            except asyncio.CancelledError:
                pass
        self.agent_task = None

    async def send_error(self, error: str):
        """Send an error message via WebSocket."""
        try:
            await self.websocket.send_json(ErrorOutput(content=error).model_dump())
        except Exception as e:
            logger.warning(f"Failed to send error via WebSocket: {e}")

    ######### Event Parsing Methods #########
    def _map_input_event(
        self, history: List[SpecificInputEvent], message: InputMessage
    ) -> tuple[InputEvent, List[SpecificInputEvent]]:
        """Pure mapping: history + harness message -> (InputEvent | None, updated history)."""
        if isinstance(message, UserStateInput):
            if message.value == UserState.SPEAKING:
                logger.info("🎤 User started speaking")
                return self._wrap_with_history(history, SpecificUserTurnStarted())
            if message.value == UserState.IDLE:
                logger.info("🔇 User stopped speaking")
                content = self._turn_content(
                    history,
                    SpecificUserTurnStarted,
                    (SpecificUserTextSent, SpecificUserDtmfSent),
                )
                return self._wrap_with_history(history, SpecificUserTurnEnded(content=content))

        elif isinstance(message, TranscriptionInput):
            logger.info(f'📝 User said: "{message.content}"')
            return self._wrap_with_history(history, SpecificUserTextSent(content=message.content))

        elif isinstance(message, AgentStateInput):
            if message.value == UserState.SPEAKING:
                logger.info("🎤 Agent started speaking")
                return self._wrap_with_history(history, SpecificAgentTurnStarted())
            if message.value == UserState.IDLE:
                logger.info("🔇 Agent stopped speaking")
                content = self._turn_content(
                    history,
                    SpecificAgentTurnStarted,
                    (
                        SpecificAgentTextSent,
                        SpecificAgentDTMFSent,
                        SpecificAgentToolCalled,
                        SpecificAgentToolReturned,
                    ),
                )
                return self._wrap_with_history(history, SpecificAgentTurnEnded(content=content))

        elif isinstance(message, AgentSpeechInput):
            logger.info(f'🗣️ Agent speech sent: "{message.content}"')
            return self._wrap_with_history(history, SpecificAgentTextSent(content=message.content))

        elif isinstance(message, DTMFInput):
            logger.info(f"🔔 DTMF received: {message.button}")
            return self._wrap_with_history(history, SpecificUserDtmfSent(button=message.button))

        raise ValueError(f"Unhandled input message type: {type(message).__name__}")

    def _turn_content(
        self,
        history: List[SpecificInputEvent],
        start_type: type,
        content_types: tuple[type, ...],
    ) -> List[SpecificInputEvent]:
        """Collect turn content since the most recent start_type event."""
        for idx in range(len(history) - 1, -1, -1):
            if isinstance(history[idx], start_type):
                return [ev for ev in history[idx + 1 :] if isinstance(ev, content_types)]
        return []

    def _wrap_with_history(
        self, history: List[SpecificInputEvent], specific_event: SpecificInputEvent
    ) -> tuple[InputEvent, List[SpecificInputEvent]]:
        """Create an InputEvent including history from a SpecificInputEvent.

        The raw history is updated with the new event, but the history passed to
        the InputEvent is processed to restore whitespace in SpecificAgentTextSent events.
        """
        raw_history = history + [specific_event]
        # Process history to restore whitespace before passing to agent
        processed_history = _get_processed_history(self.emitted_agent_text, history)
        base_data = specific_event.model_dump()

        if isinstance(specific_event, SpecificCallStarted):
            return CallStarted(history=processed_history, **base_data), raw_history
        if isinstance(specific_event, SpecificCallEnded):
            return CallEnded(history=processed_history, **base_data), raw_history
        if isinstance(specific_event, SpecificUserTurnStarted):
            return UserTurnStarted(history=processed_history, **base_data), raw_history
        if isinstance(specific_event, SpecificUserDtmfSent):
            return UserDtmfSent(history=processed_history, **base_data), raw_history
        if isinstance(specific_event, SpecificUserTextSent):
            return UserTextSent(history=processed_history, **base_data), raw_history
        if isinstance(specific_event, SpecificUserTurnEnded):
            return UserTurnEnded(history=processed_history, **base_data), raw_history
        if isinstance(specific_event, SpecificAgentTurnStarted):
            return AgentTurnStarted(history=processed_history, **base_data), raw_history
        if isinstance(specific_event, SpecificAgentTextSent):
            return AgentTextSent(history=processed_history, **base_data), raw_history
        if isinstance(specific_event, SpecificAgentDTMFSent):
            return AgentDTMFSent(history=processed_history, **base_data), raw_history
        if isinstance(specific_event, SpecificAgentToolCalled):
            return AgentToolCalledInput(history=processed_history, **base_data), raw_history
        if isinstance(specific_event, SpecificAgentToolReturned):
            return AgentToolReturnedInput(history=processed_history, **base_data), raw_history
        if isinstance(specific_event, SpecificAgentTurnEnded):
            return AgentTurnEnded(history=processed_history, **base_data), raw_history

        raise ValueError(f"Unknown event type: {type(specific_event).__name__}")

    def _map_output_event(self, event: OutputEvent) -> OutputMessage:
        """Convert OutputEvent to websocket OutputMessage."""
        if isinstance(event, AgentSendText):
            logger.info(f'🤖 Agent said: "{event.text}"')
            return MessageOutput(content=event.text)
        if isinstance(event, AgentSendDTMF):
            logger.info(f"🔢 DTMF output: {event.button}")
            return DTMFOutput(button=event.button)
        if isinstance(event, AgentEndCall):
            logger.info("📞 End call")
            return EndCallOutput()
        if isinstance(event, AgentTransferCall):
            logger.info(f"📱 Transfer to: {event.target_phone_number}")
            return TransferOutput(target_phone_number=event.target_phone_number)
        if isinstance(event, LogMetric):
            logger.debug(f"📈 Log metric: {event.name}={event.value}")
            return LogMetricOutput(name=event.name, value=event.value)
        if isinstance(event, LogMessage):
            logger.debug(f"🪵 Log message: {event.name} [{event.level}] {event.message}")
            return LogEventOutput(
                event=event.name,
                metadata={"level": event.level, "message": event.message, "metadata": event.metadata},
            )
        if isinstance(event, AgentToolCalled):
            logger.info(f"🔧 Tool called: {event.tool_name}({event.tool_args})")
            return ToolCallOutput(name=event.tool_name, arguments=event.tool_args)
        if isinstance(event, AgentToolReturned):
            logger.info(f"🔧 Tool returned: {event.tool_name}({event.tool_args}) -> {event.result}")
            result_str = str(event.result) if event.result is not None else None
            return ToolCallOutput(name=event.tool_name, arguments=event.tool_args, result=result_str)

        return ErrorOutput(content=f"Unhandled output event type: {type(event).__name__}")


# Regex to split text into words, whitespace, and punctuation
NORMAL_CHARACTERS_REGEX = r"(\s+|[^\w\s]+)"


def _get_processed_history(pending_text: str, history: List[SpecificInputEvent]) -> List[SpecificInputEvent]:
    """
    Process history to reinterpolate whitespace into SpecificAgentTextSent events.

    The TTS system strips whitespace when confirming what was spoken. This method
    uses the buffered AgentSendText content to restore proper whitespace formatting
    in the history passed to the agent's process method.

    Args:
        pending_text: Accumulated text from AgentSendText events (with whitespace)
        history: Raw history containing SpecificAgentTextSent with stripped whitespace

    Returns:
        Processed history with whitespace restored in SpecificAgentTextSent events
    """
    processed_events: List[SpecificInputEvent] = []
    for event in history:
        if isinstance(event, SpecificAgentTextSent):
            committed_text, pending_text = _parse_committed(pending_text, event.content)
            if committed_text:
                processed_events.append(SpecificAgentTextSent(content=committed_text))
            # If no committed_text (empty after strip), skip this event
        else:
            processed_events.append(event)

    return processed_events


def _parse_committed(pending_text: str, speech_text: str) -> tuple[str, str]:
    """
    Parse committed text by matching speech_text against pending_text to recover whitespace.

    The TTS system strips whitespace when confirming speech. This method matches the
    stripped speech_text against the original pending_text (with whitespace) to recover
    the properly formatted committed text.

    Args:
        pending_text: Accumulated text from AgentSendText events (with whitespace)
        speech_text: Confirmed speech from TTS (whitespace stripped)

    Returns:
        Tuple of (committed_text_with_whitespace, remaining_pending_text)
    """
    pending_parts = list(filter(lambda x: x != "", re.split(NORMAL_CHARACTERS_REGEX, pending_text)))
    speech_parts = list(filter(lambda x: x != "", re.split(NORMAL_CHARACTERS_REGEX, speech_text)))

    # If the pending text has no spaces (ex. non-latin languages), commit the entire speech text.
    if len([x for x in pending_parts if x.isspace()]) == 0:
        match_index = pending_text.find(speech_text)
        return speech_text, pending_text[match_index + len(speech_text) :]

    committed_parts: list[str] = []
    still_pending_parts: list[str] = []
    for pending_part in pending_parts:
        # If speech_text is empty, treat remaining pending parts as still pending.
        if not speech_parts:
            still_pending_parts.append(pending_part)
        # If the next pending text matches the start of what's been marked committed (as sent by TTS),
        # add it to committed and trim it from speech_parts.
        elif speech_parts[0].startswith(pending_part):
            speech_parts[0] = speech_parts[0][len(pending_part) :]
            committed_parts.append(pending_part)
            if len(speech_parts[0]) == 0:
                speech_parts.pop(0)
        # If the part is purely whitespace, add it directly to committed_parts.
        elif pending_part.isspace():
            committed_parts.append(pending_part)
        # Otherwise, this part isn't aligned with the committed speech
        # (possibly an interruption or TTS mismatch); skip it.
        else:
            pass

    committed_str = "".join(committed_parts).strip()
    return committed_str, "".join(still_pending_parts)
