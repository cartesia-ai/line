"""
VoiceAgentApp - Simplified voice agent application that handles websocket communication directly.
"""

import asyncio
from datetime import datetime, timezone
import json
import os
from typing import Any, Awaitable, Callable, List, Optional
from urllib.parse import urlencode

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from loguru import logger
from pydantic import TypeAdapter
import uvicorn

from lib.node import Node, AgentOutputEvent
from line.call_request import CallRequest, PreCallResult
from line.events import (
    AgentError,
    AgentResponse,
    AgentSpeechSent,
    AgentStartedSpeaking,
    AgentStoppedSpeaking,
    DTMFInputEvent,
    DTMFOutputEvent,
    EndCall,
    EventInstance,
    LogMetric,
    ToolResult,
    TransferCall,
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
    LogMetricOutput,
    MessageOutput,
    OutputMessage,
    ToolCallOutput,
    TranscriptionInput,
    TransferOutput,
    UserStateInput,
)
from line.nodes.conversation_context import ConversationContext


class UserState:
    """User voice states."""
    SPEAKING = "speaking"
    IDLE = "idle"


class CartesiaEnv:
    """
    Environment context for Cartesia voice agent operations.
    
    Provides access to the asyncio event loop for scheduling async operations
    from synchronous contexts (e.g., RxPY callbacks).
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        """
        Initialize the CartesiaEnv.

        Args:
            loop: The asyncio event loop to use for async operations.
        """
        self._loop = loop

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Get the asyncio event loop."""
        return self._loop


load_dotenv()
class VoiceAgentApp:
    """
    VoiceAgentApp handles HTTP and websocket communication for voice agents.
    
    Uses ConversationRunner to manage the websocket loop for each connection.
    """

    def __init__(
        self,
        get_agent: Callable[["CartesiaEnv", CallRequest], Awaitable[Node]],
        pre_call_handler: Optional[Callable[[CallRequest], Awaitable[Optional[PreCallResult]]]] = None,
    ):
        """
        Initialize the VoiceAgentApp.

        Args:
            get_agent: Async function that creates a Node from CartesiaEnv and CallRequest.
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
            # Create the CartesiaEnv with the current event loop
            loop = asyncio.get_running_loop()
            env = CartesiaEnv(loop)
            agent = await self.get_agent(env, call_request)

            # Create and run the conversation runner
            runner = ConversationRunner(websocket, agent)
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
                runner.shutdown_event.set()
            logger.info("Websocket session ended")

    def run(self, host="0.0.0.0", port=None):
        """Run the voice agent server."""
        port = port or int(os.getenv("PORT", 8000))
        uvicorn.run(self.fastapi_app, host=host, port=port)

class ConversationRunner:
    """
    Manages the websocket loop for a single conversation.
    
    Aggregates incoming events into ConversationContext, passes them to the Node,
    and serializes Node output events back to the websocket.
    """

    def __init__(self, websocket: WebSocket, agent: Node):
        """
        Initialize the ConversationRunner.

        Args:
            websocket: The WebSocket connection.
            agent: The Node to process conversation events.
        """
        self.websocket = websocket
        self.agent = agent
        self.shutdown_event = asyncio.Event()
        self.events: List[EventInstance] = []

    async def run(self):
        """
        Run the conversation loop.
        
        Subscribes to agent output, then processes incoming websocket messages
        until shutdown.
        """
        # Subscribe to agent output events
        self.agent.out.subscribe(
            on_next=self._on_agent_output,
            on_error=self._on_agent_error,
            on_completed=self._on_agent_complete,
        )

        while not self.shutdown_event.is_set():
            try:
                # Receive message from WebSocket
                message = await self.websocket.receive_json()
                input_msg = TypeAdapter(InputMessage).validate_python(message)

                # Map to events
                new_events = self._map_to_events(input_msg)
                self.events.extend(new_events)

                # Build ConversationContext and pass to agent
                context = ConversationContext(
                    events=self.events.copy(),
                    system_prompt="",  # Agent manages its own system prompt
                )

                # Pass context to agent's on_next
                self.agent.on_next(context)

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected in loop")
                self.shutdown_event.set()
                break
            except json.JSONDecodeError as e:
                logger.exception(f"Failed to parse JSON message: {e}")
                continue
            except Exception as e:
                logger.exception(f"Error in websocket loop: {e}")
                if not self.shutdown_event.is_set():
                    await asyncio.sleep(0.1)

    def _map_to_events(self, message: InputMessage) -> List[EventInstance]:
        """Convert websocket input message to conversation events."""
        if isinstance(message, UserStateInput):
            if message.value == UserState.SPEAKING:
                logger.info("🎤 User started speaking")
                return [UserStartedSpeaking()]
            elif message.value == UserState.IDLE:
                logger.info("🔇 User stopped speaking")
                return [UserStoppedSpeaking()]
        elif isinstance(message, TranscriptionInput):
            logger.info(f'📝 User said: "{message.content}"')
            return [UserTranscriptionReceived(content=message.content)]
        elif isinstance(message, AgentStateInput):
            if message.value == UserState.SPEAKING:
                logger.info("🎤 Agent started speaking")
                return [AgentStartedSpeaking()]
            elif message.value == UserState.IDLE:
                logger.info("🔇 Agent stopped speaking")
                return [AgentStoppedSpeaking()]
        elif isinstance(message, AgentSpeechInput):
            logger.info(f'🗣️ Agent speech sent: "{message.content}"')
            return [AgentSpeechSent(content=message.content)]
        elif isinstance(message, DTMFInput):
            logger.info(f"🔔 DTMF received: {message.button}")
            return [DTMFInputEvent(button=message.button)]
        else:
            logger.warning(f"Unknown message type: {type(message).__name__} ({message.model_dump_json()})")
            return [UserUnknownInputReceived(input_data=message.model_dump_json())]

        return []  # No events for unhandled states

    def _on_agent_output(self, event: AgentOutputEvent):
        """Handle agent output events by sending them to the websocket."""
        asyncio.create_task(self._send_event(event))

    def _on_agent_error(self, error: Exception):
        """Handle agent errors."""
        logger.error(f"Agent error: {error}")
        self.shutdown_event.set()

    def _on_agent_complete(self):
        """Handle agent completion."""
        logger.info("Agent completed")
        self.shutdown_event.set()

    async def _send_event(self, event: AgentOutputEvent):
        """
        Serialize and send an agent output event to the websocket.

        Converts AgentOutputEvent types from line.events to the appropriate
        websocket output format (harness_types).

        Args:
            event: The agent output event to send.
        """
        if self.shutdown_event.is_set():
            return

        try:
            # Convert event to websocket output format and log
            output: OutputMessage
            if isinstance(event, AgentResponse):
                logger.info(f'🤖 Agent said: "{event.content}"')
                output = MessageOutput(content=event.content)
            elif isinstance(event, DTMFOutputEvent):
                logger.info(f"🔢 DTMF output: {event.button}")
                output = DTMFOutput(button=event.button)
            elif isinstance(event, EndCall):
                logger.info("📞 End call")
                self.shutdown_event.set()
                output = EndCallOutput()
            elif isinstance(event, AgentError):
                logger.warning(f"⚠️ Agent error: {event.content}")
                output = ErrorOutput(content=event.content)
            elif isinstance(event, ToolResult):
                logger.info(f"🔧 Tool result: {event.tool_name}({event.tool_args})")
                output = ToolCallOutput(name=event.tool_name, arguments=event.tool_args)
            elif isinstance(event, TransferCall):
                logger.info(f"📱 Transfer to: {event.target_phone_number}")
                self.shutdown_event.set()
                output = TransferOutput(target_phone_number=event.target_phone_number)
            elif isinstance(event, LogMetric):
                logger.debug(f"📈 Log metric: {event.name}={event.value}")
                output = LogMetricOutput(name=event.name, value=event.value)
            else:
                logger.warning(f"Unknown event type: {type(event)}")
                return

            await self.websocket.send_json(output.model_dump())
        except Exception as e:
            logger.warning(f"Failed to send event via WebSocket: {e}")
            self.shutdown_event.set()

    async def send_error(self, error: str):
        """Send an error message via WebSocket."""
        try:
            await self.websocket.send_json(ErrorOutput(content=error).model_dump())
        except Exception as e:
            logger.warning(f"Failed to send error via WebSocket: {e}")

    async def send_end_call(self):
        """Send end_call message via WebSocket."""
        try:
            await self.websocket.send_json(EndCallOutput().model_dump())
        except Exception as e:
            logger.warning(f"Failed to send end_call via WebSocket: {e}")

