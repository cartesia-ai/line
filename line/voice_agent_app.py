from datetime import datetime, timezone
import json
import os
from typing import Awaitable, Callable, Optional
from urllib.parse import urlencode

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from loguru import logger
import uvicorn

from line.call_request import AgentConfig, CallRequest, PreCallResult
from line.voice_agent_system import VoiceAgentSystem

# Load environment variables from .env file
load_dotenv()


class VoiceAgentApp:
    """
    VoiceAgentApp (name tbd) abstracts away the HTTP and websocket handling,
    which should be invisible to developers, because this transport may change
    in the future (eg to WebRTC).
    """

    def __init__(
        self,
        call_handler,
        pre_call_handler: Optional[Callable[[CallRequest], Awaitable[Optional[PreCallResult]]]] = None,
    ):
        self.fastapi_app = FastAPI()
        self.call_handler = call_handler
        self.pre_call_handler = pre_call_handler
        self.ws_route = "/ws"

        self.fastapi_app.add_api_route("/chats", self.create_chat_session, methods=["POST"])
        self.fastapi_app.add_api_route("/status", self.get_status, methods=["GET"])
        self.fastapi_app.add_websocket_route(self.ws_route, self.websocket_endpoint)

    async def create_chat_session(self, request: Request) -> dict:
        """Create a new chat session and return the websocket URL."""
        # Parse JSON body
        body = await request.json()

        # Create initial CallRequest
        call_request = CallRequest(
            call_id=body.get("call_id", "unknown"),
            from_=body.get("from_", "unknown"),
            to=body.get("to", "unknown"),
            agent_call_id=body.get("agent_call_id", body.get("call_id", "unknown")),
            agent=AgentConfig(
                system_prompt=body.get("agent", {}).get("system_prompt", ""),
                introduction=body.get("agent", {}).get("introduction", ""),
            ),
            metadata=body.get("metadata", {}),
        )

        # Run pre-call handler if provided
        config = None
        if self.pre_call_handler:
            try:
                result = await self.pre_call_handler(call_request)
                if result is None:
                    raise HTTPException(status_code=403, detail="Call rejected")

                # Update call_request metadata with result
                call_request.metadata.update(result.metadata)
                config = result.config

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error in pre_call_handler: {str(e)}")
                raise HTTPException(status_code=500, detail="Server error in call processing") from e

        # Create URL parameters from processed call_request
        url_params = {
            "call_id": call_request.call_id,
            "from": call_request.from_,
            "to": call_request.to,
            "agent": json.dumps(call_request.agent.model_dump()),  # JSON encode agent
            "agent_call_id": call_request.agent_call_id,
            "metadata": json.dumps(call_request.metadata),  # JSON encode metadata
        }

        # Build websocket URL with parameters
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

        # Parse query parameters from WebSocket URL
        query_params = dict(websocket.query_params)

        # Parse metadata JSON
        metadata = {}
        if "metadata" in query_params:
            try:
                metadata = json.loads(query_params["metadata"])
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Invalid metadata JSON: {query_params['metadata']}")
                metadata = {}

        # Parse agent JSON
        agent_data = {}
        if "agent" in query_params:
            try:
                agent_data = json.loads(query_params["agent"])
            except (json.JSONDecodeError, TypeError):
                logger.error(f"Invalid agent JSON: {query_params['agent']}")
                agent_data = {}

        # Create CallRequest from URL parameters
        call_request = CallRequest(
            call_id=query_params.get("call_id", "unknown"),
            from_=query_params.get("from", "unknown"),
            to=query_params.get("to", "unknown"),
            agent_call_id=query_params.get("agent_call_id", "unknown"),
            agent=AgentConfig(
                system_prompt=agent_data.get("system_prompt", ""),
                introduction=agent_data.get("introduction", ""),
            ),
            metadata=metadata,
        )

        system = VoiceAgentSystem(websocket)

        try:
            # Handler configures nodes and bridges, then starts system
            await self.call_handler(system, call_request)
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.exception(f"Error: {str(e)}")
            try:
                await system.harness.send_error("System has encountered an error, please try again later.")
                await system.harness.end_call()
            except:  # noqa: E722
                pass
        finally:
            await system.cleanup()

    def run(self, host="0.0.0.0", port=None):
        """Run the voice agent server."""
        port = port or int(os.getenv("PORT", 8000))
        uvicorn.run(self.fastapi_app, host=host, port=port)
