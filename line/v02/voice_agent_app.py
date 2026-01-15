"""
VoiceAgentApp - Simplified voice agent application that handles websocket communication directly.
"""

import asyncio
from datetime import datetime, timezone
import json
import os
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
from line.v02.agent import Agent, AgentSpec, EventFilter
from line.v02.events import (
    AgentDTMFSent,
    AgentEndCall,
    AgentSendDTMF,
    AgentSendText,
    AgentTextSent,
    AgentToolCalled,
    AgentToolReturned,
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
            "service": "mandolin-agent",
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

        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.exception(f"Error: {str(e)}")
            if runner:
                try:
                    await runner.send_error("System has encountered an error, please try again later.")
                    await runner.send_end_call()
                except:  # noqa: E722
                    pass
        finally:
            if runner:
                await runner.shutdown()
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

        self.agent_callable, self.run_filter, self.cancel_filter = self._prepare_agent(agent_spec)
        self.agent_task: Optional[asyncio.Task] = None

    ######### Initialization Methods #########

    def _prepare_agent(
        self, agent_spec: AgentSpec
    ) -> tuple[
        Callable[[InputEvent], AsyncIterable[OutputEvent]],
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

        def _agent_callable(event: InputEvent) -> AsyncIterable[OutputEvent]:
            if hasattr(agent_obj, "process") and callable(agent_obj.process):
                return agent_obj.process(self.env, event)  # type: ignore[return-value]
            if callable(agent_obj):
                return agent_obj(self.env, event)  # type: ignore[return-value]
            raise TypeError("Agent must be callable or have a callable 'process' method.")

        return _agent_callable, run_filter, cancel_filter

    def _normalize_filter(self, filter_spec: EventFilter) -> Callable[[InputEvent], bool]:
        """Normalize EventFilter spec to a callable."""
        if callable(filter_spec):
            return filter_spec  # type: ignore[return-value]
        if isinstance(filter_spec, (list, tuple)):

            def _fn(event: InputEvent) -> bool:
                return any(isinstance(event, cls) for cls in filter_spec)  # type: ignore[arg-type]

            return _fn
        raise TypeError("EventFilter must be callable or list")

    ######### Run Loop Methods #########

    async def run(self):
        """
        Run the conversation loop.

        Processes incoming websocket messages until shutdown.
        """
        # Emit call_started to seed history/context
        start_event, self.history = self._wrap_with_history(self.history, SpecificCallStarted())
        await self._handle_filters(start_event)

        while not self.shutdown_event.is_set():
            try:
                # Receive message from WebSocket
                message = await self.websocket.receive_json()
                input_msg = TypeAdapter(InputMessage).validate_python(message)

                # Map to events
                ev, self.history = self._map_input_event(self.history, input_msg)

                if ev is not None:
                    await self._handle_event(ev)

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected in loop")
                end_event, self.history = self._wrap_with_history(self.history, SpecificCallEnded())
                await self._handle_filters(end_event)
                self.shutdown_event.set()
                break
            except json.JSONDecodeError as e:
                logger.exception(f"Failed to parse JSON message: {e}")
                continue
            except Exception as e:
                logger.exception(f"Error in websocket loop: {e}")
                if not self.shutdown_event.is_set():
                    await asyncio.sleep(0.1)

        await self._cancel_agent_task()

    async def _handle_event(self, event: InputEvent) -> None:
        """Apply run/cancel filters for a single event."""
        if self.run_filter(event):
            await self._start_agent_task(event)
        elif self.cancel_filter(event):
            await self._cancel_agent_task()

    async def _start_agent_task(self, event: InputEvent) -> None:
        """Start the agent async iterable for the given event."""
        await self._cancel_agent_task()

        async def runner():
            try:
                async for output in self.agent_callable(event):
                    mapped = self._map_output_event(output)
                    if mapped is None:
                        continue

                    is_terminal = isinstance(output, (AgentEndCall, AgentTransferCall))
                    if is_terminal:
                        self.shutdown_event.set()

                    if not self.shutdown_event.is_set() or is_terminal:
                        await self.websocket.send_json(mapped.model_dump())
                    else:
                        break
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"Agent iterable error: {exc}")
                self.shutdown_event.set()

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

    async def shutdown(self) -> None:
        """Shutdown helper for websocket_endpoint cleanup."""
        await self._cancel_agent_task()
        self.shutdown_event.set()

    ######### Event Parsing Methods #########

    def _map_input_event(
        self, history: List[SpecificInputEvent], message: InputMessage
    ) -> tuple[Optional[InputEvent], List[SpecificInputEvent]]:
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

        else:
            logger.warning(f"Unknown message type: {type(message).__name__} ({message.model_dump_json()})")
            # Unknown messages are ignored for v0.2 events

        return None, history

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
        """Create an InputEvent including history from a SpecificInputEvent."""
        history = history + [specific_event]
        base_data = specific_event.model_dump()

        if isinstance(specific_event, SpecificCallStarted):
            return CallStarted(history=history, **base_data), history
        if isinstance(specific_event, SpecificCallEnded):
            return CallEnded(history=history, **base_data), history
        if isinstance(specific_event, SpecificUserTurnStarted):
            return UserTurnStarted(history=history, **base_data), history
        if isinstance(specific_event, SpecificUserDtmfSent):
            return UserDtmfSent(history=history, **base_data), history
        if isinstance(specific_event, SpecificUserTextSent):
            return UserTextSent(history=history, **base_data), history
        if isinstance(specific_event, SpecificUserTurnEnded):
            return UserTurnEnded(history=history, **base_data), history
        if isinstance(specific_event, SpecificAgentTurnStarted):
            return AgentTurnStarted(history=history, **base_data), history
        if isinstance(specific_event, SpecificAgentTextSent):
            return AgentTextSent(history=history, **base_data), history
        if isinstance(specific_event, SpecificAgentDTMFSent):
            return AgentDTMFSent(history=history, **base_data), history
        if isinstance(specific_event, SpecificAgentToolCalled):
            return AgentToolCalled(history=history, **base_data), history
        if isinstance(specific_event, SpecificAgentToolReturned):
            return AgentToolReturned(history=history, **base_data), history
        if isinstance(specific_event, SpecificAgentTurnEnded):
            return AgentTurnEnded(history=history, **base_data), history

        raise ValueError(f"Unhandled specific event type: {type(specific_event).__name__}")

    def _map_output_event(self, event: OutputEvent) -> Optional[OutputMessage]:
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

        logger.warning(f"Unknown event type: {type(event)}")
        return None

    async def send_error(self, error: str):
        """Send an error message via WebSocket."""
        try:
            await self.websocket.send_json(ErrorOutput(content=error).model_dump())
        except Exception as e:
            logger.warning(f"Failed to send error via WebSocket: {e}")
