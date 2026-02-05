"""
VoiceAgentApp - Simple harness that manages:
    1) HTTP endpoints to create chat sessions
    2) Websocket connections for each chat session

ConversationRunner - Manages the websocket loop for a single conversation,
    1) converting incoming websocket messages to InputEvents
    2) applying run/cancel filters
    2) calling agent#process as an async iterable
    3) serializing yield OutputEvents back to websocket
"""

import asyncio
from datetime import datetime, timezone
import json
import os
import re
import traceback
from typing import Any, AsyncIterable, Awaitable, Callable, Dict, List, Optional
from urllib.parse import urlencode

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
import uvicorn

from line._harness_types import (
    AgentSpeechInput,
    AgentStateInput,
    ConfigOutput,
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
    TTSConfig,
    UserStateInput,
)
from line.agent import Agent, AgentSpec, EventFilter, TurnEnv
from line.events import (
    AgentDtmfSent,
    AgentEndCall,
    AgentSendDtmf,
    AgentSendText,
    AgentTextSent,
    AgentToolCalled,
    AgentToolReturned,
    AgentTransferCall,
    AgentTurnEnded,
    AgentTurnStarted,
    AgentUpdateCall,
    CallEnded,
    CallStarted,
    InputEvent,
    LogMessage,
    LogMetric,
    OutputEvent,
    UserDtmfSent,
    UserTextSent,
    UserTurnEnded,
    UserTurnStarted,
)


# Call request types
class PreCallResult(BaseModel):
    """Result from pre_call_handler containing metadata and config."""

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata to include with the call")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration for the call")


class AgentConfig(BaseModel):
    """Agent information for the call."""

    system_prompt: Optional[str] = None  # System prompt to define the agent's role and behavior
    introduction: Optional[str] = None  # Introduction message for the agent to start the call with


class CallRequest(BaseModel):
    """Request body for the /chats endpoint."""

    call_id: str
    from_: str = Field(alias="from")  # Using from_ to avoid Python keyword conflict
    to: str
    agent_call_id: str  # Agent call ID for logging and correlation
    agent: AgentConfig
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        # Allow both field name (from_) and alias (from) for input
        populate_by_name=True
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
            agent=AgentConfig(**body.get("agent", {})),
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
            "agent": json.dumps(call_request.agent.model_dump()),
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

        agent_data = {}
        if "agent" in query_params:
            try:
                agent_data = json.loads(query_params["agent"])
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Invalid agent JSON: {query_params['agent']}")
                agent_data = {}

        call_request = CallRequest(
            call_id=query_params.get("call_id", "unknown"),
            from_=query_params.get("from", "unknown"),
            to=query_params.get("to", "unknown"),
            agent_call_id=query_params.get("agent_call_id", "unknown"),
            agent=AgentConfig(**agent_data),
            metadata=metadata,
        )

        runner: Optional[ConversationRunner] = None
        # Create the AgentEnv with the current event loop
        loop = asyncio.get_running_loop()
        env = AgentEnv(loop)
        try:
            agent_spec = await self.get_agent(env, call_request)
            runner = ConversationRunner(websocket, agent_spec, env)
        except Exception:
            error_msg = traceback.format_exc()
            error_string = f"Error in get_agent for {call_request.call_id}: {error_msg}"
            logger.error(error_string)
            await websocket.send_json(ErrorOutput(content=error_string).model_dump())
            await websocket.close()
            return

        # Create and run the conversation runner
        await runner.run()
        logger.info("Websocket session ended")

    def run(self, host="0.0.0.0", port=None):
        """Run the voice agent server."""
        port = port or int(os.getenv("PORT", 8000))
        uvicorn.run(self.fastapi_app, host=host, port=port)


class ConversationRunner:
    """
    Manages the websocket loop for a single conversation.
    Converts websocket messages to InputEvents, applies run/cancel filters,
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
        self.history: List[InputEvent] = []
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
        start_event, self.history = self._process_input_event(self.history, CallStarted())
        await self._handle_event(TurnEnv(), start_event)

        while not self.shutdown_event.is_set():
            try:
                # Receive message from WebSocket
                message = await self.websocket.receive_json()
                input_msg = TypeAdapter(InputMessage).validate_python(message)

                # Convert and process the input message
                event = self._convert_input_message(input_msg)
                ev, self.history = self._process_input_event(self.history, event)
                await self._handle_event(TurnEnv(), ev)

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected in loop")
                self.shutdown_event.set()
                end_event, self.history = self._process_input_event(self.history, CallEnded())
                await self._handle_event(TurnEnv(), end_event)
            except json.JSONDecodeError as e:
                # Don't send EndCall event, as that may trigger side effects
                # we accept the risk of incomplete call cleanup in this case,
                # since this is an exceptional case that we will fix at the
                # SDK level
                self.shutdown_event.set()
                logger.error(f"Failed to parse JSON message: {e}")
                await self.send_error(f"Failed to parse JSON message: {e}")
                await self.websocket.close()
            except Exception:
                # Most non-input processing messages are handled in the #runner loop
                # so this is almost certainly a message processing error.
                # Don't send EndCall event, as that may trigger side effects
                # we accept the risk of incomplete call cleanup in this case,
                # since this is an exceptional case that we will fix at the
                # SDK level
                self.shutdown_event.set()
                error_msg = traceback.format_exc()
                logger.error(f"Error in websocket loop (likely message processing): {error_msg}")
                await self.send_error(f"Error in websocket loop (likely message processing): {error_msg}")
                await self.websocket.close()

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
            except Exception:
                self.shutdown_event.set()
                error_msg = traceback.format_exc()
                logger.error(f"Error in agent.process: {error_msg}")
                await self.send_error(f"Error in agent.process: {error_msg}")
                await self.websocket.close()

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
    def _convert_input_message(self, message: InputMessage) -> InputEvent:
        """Convert an InputMessage to an InputEvent (with history=None)."""
        if isinstance(message, UserStateInput):
            if message.value == UserState.SPEAKING:
                return UserTurnStarted()
            elif message.value == UserState.IDLE:
                content = self._turn_content(
                    self.history,
                    UserTurnStarted,
                    (UserTextSent, UserDtmfSent),
                )
                return UserTurnEnded(content=content)

        elif isinstance(message, TranscriptionInput):
            return UserTextSent(content=message.content)

        elif isinstance(message, AgentStateInput):
            if message.value == UserState.SPEAKING:
                return AgentTurnStarted()
            elif message.value == UserState.IDLE:
                content = self._turn_content(
                    self.history,
                    AgentTurnStarted,
                    (AgentTextSent, AgentDtmfSent),
                )
                return AgentTurnEnded(content=content)

        elif isinstance(message, AgentSpeechInput):
            return AgentTextSent(content=message.content)

        elif isinstance(message, DTMFInput):
            return UserDtmfSent(button=message.button)

        raise ValueError(f"Unhandled input message type: {type(message).__name__}")

    def _turn_content(
        self,
        history: List[InputEvent],
        start_type: type,
        content_types: tuple[type, ...],
    ) -> List[InputEvent]:
        """Collect turn content since the most recent start_type event."""
        for idx in range(len(history) - 1, -1, -1):
            if isinstance(history[idx], start_type):
                return [ev for ev in history[idx + 1 :] if isinstance(ev, content_types)]
        return []

    def _process_input_event(
        self, history: List[InputEvent], raw_event: InputEvent
    ) -> tuple[InputEvent, List[InputEvent]]:
        """Create an InputEvent including history from an InputEvent (with history=None).

        The raw history is updated with the new event, but the history passed to
        the InputEvent is processed to restore whitespace in AgentTextSent events.
        """
        raw_history = history + [raw_event]
        # Process history to restore whitespace before passing to agent
        processed_history = _get_processed_history(self.emitted_agent_text, raw_history)
        processed_event = processed_history[-1]
        # Extract base data excluding history (we'll set it explicitly)
        base_data = {k: v for k, v in processed_event.model_dump().items() if k != "history"}

        event: InputEvent
        if isinstance(processed_event, CallStarted):
            event = CallStarted(history=processed_history, **base_data)
            logger.info("-> 📞 Call started")
        elif isinstance(processed_event, CallEnded):
            event = CallEnded(history=processed_history, **base_data)
            logger.info("-> 📞 Call ended")
        elif isinstance(processed_event, UserTurnStarted):
            event = UserTurnStarted(history=processed_history, **base_data)
            logger.info("-> 🧑🔊 User started speaking")
        elif isinstance(processed_event, UserDtmfSent):
            event = UserDtmfSent(history=processed_history, **base_data)
            logger.info(f"-> 🧑🔔 User DTMF received: {event.button}")
        elif isinstance(processed_event, UserTextSent):
            event = UserTextSent(history=processed_history, **base_data)
            logger.info(f'-> 🧑🗣️ User said: "{event.content}"')
        elif isinstance(processed_event, UserTurnEnded):
            event = UserTurnEnded(history=processed_history, **base_data)
            logger.info("-> 🧑🔇 User stopped speaking")
        elif isinstance(processed_event, AgentTurnStarted):
            event = AgentTurnStarted(history=processed_history, **base_data)
            logger.info("-> 🤖🔊 Agent started speaking")
        elif isinstance(processed_event, AgentTextSent):
            event = AgentTextSent(history=processed_history, **base_data)
            # special case: log the raw event content (without whitespace restoration)
            # otherwise we re-log the same text multiple times with the new stuff
            # concatenated
            logger.info(f'-> 🤖🗣️ Agent said: "{raw_event.content}"')
        elif isinstance(processed_event, AgentDtmfSent):
            event = AgentDtmfSent(history=processed_history, **base_data)
        elif isinstance(processed_event, AgentTurnEnded):
            event = AgentTurnEnded(history=processed_history, **base_data)
            logger.info("-> 🤖🔇 Agent stopped speaking")
        else:
            raise ValueError(f"Unknown event type: {type(processed_event).__name__}")

        return event, raw_history

    def _map_output_event(self, event: OutputEvent) -> OutputMessage:
        """Convert OutputEvent to websocket OutputMessage."""
        if isinstance(event, AgentSendText):
            logger.info(f'<- 🤖🗣️ Agent said: "{event.text}"')
            return MessageOutput(content=event.text)
        if isinstance(event, AgentSendDtmf):
            logger.info(f"<- 🤖🔔 Agent DTMF sent: {event.button}")
            return DTMFOutput(button=event.button)
        if isinstance(event, AgentEndCall):
            logger.info("<- 📞 End call")
            return EndCallOutput()
        if isinstance(event, AgentTransferCall):
            logger.info(f"<- 📱 Transfer to: {event.target_phone_number}")
            return TransferOutput(target_phone_number=event.target_phone_number)
        if isinstance(event, LogMetric):
            logger.debug(f"<- 📈 Log metric: {event.name}={event.value}")
            return LogMetricOutput(name=event.name, value=event.value)
        if isinstance(event, LogMessage):
            logger.debug(f"<- 🪵 Log message: {event.name} [{event.level}] {event.message}")
            return LogEventOutput(
                event=event.name,
                metadata={"level": event.level, "message": event.message, "metadata": event.metadata},
            )
        if isinstance(event, AgentToolCalled):
            logger.info(f"<- 🔧 Tool called: {event.tool_name}({event.tool_args})")
            return ToolCallOutput(name=event.tool_name, arguments=event.tool_args)
        if isinstance(event, AgentToolReturned):
            logger.info(f"<- 🔧 Tool returned: {event.tool_name}({event.tool_args}) -> {event.result}")
            result_str = str(event.result) if event.result is not None else None
            return ToolCallOutput(name=event.tool_name, arguments=event.tool_args, result=result_str)
        if isinstance(event, AgentUpdateCall):
            logger.info(
                f"<- ⚙️ Update call: voice_id={event.voice_id}, "
                f"pronunciation_dict_id={event.pronunciation_dict_id}"
            )
            return ConfigOutput(
                tts=TTSConfig(
                    voice_id=event.voice_id,
                    pronunciation_dict_id=event.pronunciation_dict_id,
                ),
            )

        return ErrorOutput(content=f"Unhandled output event type: {type(event).__name__}")


# Regex to split text into words, whitespace, and punctuation
NORMAL_CHARACTERS_REGEX = r"(\s+|[^\w\s]+)"


def _get_processed_history(pending_text: str, history: List[InputEvent]) -> List[InputEvent]:
    """
    Process history to reinterpolate whitespace into AgentTextSent events.

    The TTS system strips whitespace when confirming what was spoken. This method
    uses the buffered AgentSendText content to restore proper whitespace formatting
    in the history passed to the agent's process method.

    Args:
        pending_text: Accumulated text from AgentSendText events (with whitespace)
        history: Raw history containing AgentTextSent with stripped whitespace

    Returns:
        Processed history with whitespace restored in AgentTextSent events
    """
    processed_events: List[InputEvent] = []
    committed_text_buffer = ""
    for event in history:
        if isinstance(event, AgentTextSent):
            committed_text_buffer += event.content
        else:
            committed_text, committed_text_buffer, pending_text = _parse_committed(
                committed_text_buffer, pending_text
            )
            if committed_text:
                processed_events.append(AgentTextSent(content=committed_text))
            processed_events.append(event)

    committed_text, _, _ = _parse_committed(committed_text_buffer, pending_text)
    if committed_text:
        processed_events.append(AgentTextSent(content=committed_text))
    return processed_events


def _parse_committed(committed_buffer_text: str, pending_text: str) -> tuple[str, str, str]:
    """
    Parse committed text by matching speech_text against pending_text to recover whitespace.

    The TTS system strips whitespace when confirming speech. This method matches the
    stripped speech_text against the original pending_text (with whitespace) to recover
    the properly formatted committed text.

    Args:
        committed_buffer_text: Confirmed speech from TTS (whitespace stripped)
        pending_text: Accumulated text from AgentSendText events (with whitespace)

    Returns:
        Tuple of (committed_text_with_whitespace, remaining_pending_text)
    """
    committed_buffer_parts = list(
        filter(lambda x: x != "", re.split(NORMAL_CHARACTERS_REGEX, committed_buffer_text))
    )
    pending_parts = list(filter(lambda x: x != "", re.split(NORMAL_CHARACTERS_REGEX, pending_text)))

    # If the pending text has no spaces (ex. non-latin languages), commit the entire speech text.
    if len([x for x in pending_parts if x.isspace()]) == 0:
        match_index = pending_text.find(committed_buffer_text)
        return committed_buffer_text, "", pending_text[match_index + len(committed_buffer_text) :]

    committed_parts: list[str] = []
    still_pending_parts: list[str] = []
    for pending_part in pending_parts:
        # If speech_text is empty, treat remaining pending parts as still pending.
        if not committed_buffer_parts:
            still_pending_parts.append(pending_part)
        # If the next pending text matches the start of what's been marked committed (as sent by TTS),
        # add it to committed and trim it from committed_buffer_parts.
        elif committed_buffer_parts[0].startswith(pending_part):
            committed_buffer_parts[0] = committed_buffer_parts[0][len(pending_part) :]
            committed_parts.append(pending_part)
            if len(committed_buffer_parts[0]) == 0:
                committed_buffer_parts.pop(0)
        # If the part is purely whitespace, add it directly to committed_parts.
        elif pending_part.isspace():
            committed_parts.append(pending_part)
        # Otherwise, this part isn't aligned with the committed speech
        # (possibly an interruption or TTS mismatch); skip it.
        else:
            pass

    committed_str = "".join(committed_parts).strip()
    return committed_str, "".join(committed_buffer_parts), "".join(still_pending_parts)
