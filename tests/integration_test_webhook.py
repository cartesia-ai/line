"""
Integration tests for webhook_tool over real HTTP.

Starts a local HTTP server, constructs webhook_tools pointing at it, calls
tool.func() directly (no LLM), and asserts the server received the correct
method, headers, body, query params, and URL path.

Usage:
    uv run python -m pytest tests/integration_test_webhook.py -v
"""

import json
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List
from urllib.parse import parse_qs
from unittest.mock import MagicMock

import pytest

from line.llm_agent.tools.system import webhook_tool

pytestmark = [pytest.mark.anyio, pytest.mark.parametrize("anyio_backend", ["asyncio"])]


# =============================================================================
# Shared test server
# =============================================================================
#
# Special paths control server behavior:
#   /error/{status_code}  — returns that HTTP status with an error body
#   /slow/{seconds}       — sleeps before responding (for timeout tests)
#   /large/{num_chars}    — returns a response body of that size
#   /auth-required/...    — returns 401 unless Authorization is "Bearer valid-token"
#   anything else         — returns 201 with a JSON receipt


class _RequestLog:
    """Collects requests received by the test server."""

    def __init__(self):
        self.requests: List[Dict[str, Any]] = []

    def record(self, method: str, path: str, headers: dict, body: Any, query: str):
        self.requests.append({
            "method": method,
            "path": path,
            "headers": dict(headers),
            "body": body,
            "query": query,
        })

    @property
    def last(self) -> Dict[str, Any]:
        return self.requests[-1]

    def clear(self):
        self.requests.clear()


_log = _RequestLog()


def _make_handler(log: _RequestLog):
    class Handler(BaseHTTPRequestHandler):
        def _handle(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b""
            body = json.loads(raw) if raw else None
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            path = self.path.split("?", 1)[0]

            log.record(self.command, path, self.headers, body, query)

            # /auth-required/* — check token
            if path.startswith("/auth-required"):
                auth = self.headers.get("Authorization", "")
                if auth != "Bearer valid-token":
                    self._respond(401, {"error": "unauthorized", "detail": "bad token"})
                    return

            # /error/{code} — return that status
            if path.startswith("/error/"):
                code = int(path.split("/error/")[1].split("/")[0])
                self._respond(code, {"error": f"server returned {code}"})
                return

            # /slow/{seconds} — delay
            if path.startswith("/slow/"):
                delay = float(path.split("/slow/")[1].split("/")[0])
                time.sleep(delay)

            # /large/{n} — return n chars of padding
            if path.startswith("/large/"):
                n = int(path.split("/large/")[1].split("/")[0])
                self._respond(200, {"data": "x" * n})
                return

            # Default — 201 with receipt
            self._respond(201, {
                "ticket_id": "TKT-00001",
                "status": "created",
                "received_body": body,
            })

        def _respond(self, status: int, body: Any):
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        do_GET = _handle
        do_POST = _handle
        do_PUT = _handle
        do_PATCH = _handle
        do_DELETE = _handle

        def log_message(self, *args):
            pass

    return Handler


@pytest.fixture(scope="module")
def server():
    """Start a local HTTP server on a random port for the test module."""
    srv = HTTPServer(("127.0.0.1", 0), _make_handler(_log))
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


@pytest.fixture(autouse=True)
def clear_log():
    _log.clear()


@pytest.fixture
def ctx():
    return MagicMock()


# =============================================================================
# Tests
# =============================================================================


async def test_post_with_body_and_constants(server, ctx, anyio_backend, monkeypatch):
    """POST request sends LLM args + constant_value fields in the JSON body."""
    monkeypatch.setenv("TEST_API_KEY", "secret-123")

    tool = webhook_tool(
        name="create_ticket",
        description="Create a ticket.",
        url=f"{server}/api/tickets",
        method="POST",
        body_schema={
            "type": "object",
            "required": ["subject"],
            "properties": {
                "subject": {"type": "string", "description": "Summary."},
                "source": {"type": "string", "constant_value": "voice_agent"},
            },
        },
        auth={"Authorization": "Bearer ${TEST_API_KEY}"},
        is_background=False,
    )

    result = json.loads(await tool.func(ctx, subject="Printer is broken"))

    assert result["ok"] is True
    assert result["status"] == 201
    assert "TKT-00001" in result["body"]

    req = _log.last
    assert req["method"] == "POST"
    assert req["path"] == "/api/tickets"
    assert req["headers"]["Authorization"] == "Bearer secret-123"
    assert req["body"]["subject"] == "Printer is broken"
    assert req["body"]["source"] == "voice_agent"


async def test_nested_constant_values(server, ctx, anyio_backend):
    """Nested constant_value fields are deep-merged into the request body."""
    tool = webhook_tool(
        name="t",
        description="d",
        url=f"{server}/api/tickets",
        method="POST",
        body_schema={
            "type": "object",
            "required": ["subject"],
            "properties": {
                "subject": {"type": "string"},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "channel": {"type": "string", "constant_value": "phone"},
                        "version": {"type": "integer", "constant_value": 2},
                    },
                },
            },
        },
        is_background=False,
    )

    await tool.func(ctx, subject="Test")

    body = _log.last["body"]
    assert body["subject"] == "Test"
    assert body["metadata"]["channel"] == "phone"
    assert body["metadata"]["version"] == 2


async def test_nested_constant_values_without_object_type(server, ctx, anyio_backend):
    """Object-like schemas still hide and inject constants when type is omitted."""
    tool = webhook_tool(
        name="t",
        description="d",
        url=f"{server}/api/tickets",
        method="POST",
        body_schema={
            "type": "object",
            "required": ["subject"],
            "properties": {
                "subject": {"type": "string"},
                "metadata": {
                    "properties": {
                        "channel": {"type": "string", "constant_value": "phone"},
                        "version": {"type": "integer", "constant_value": 2},
                    },
                },
            },
        },
        is_background=False,
    )

    assert "metadata" not in tool.parameters

    await tool.func(ctx, subject="Test")

    body = _log.last["body"]
    assert body["subject"] == "Test"
    assert body["metadata"]["channel"] == "phone"
    assert body["metadata"]["version"] == 2


async def test_freeform_object_without_properties_remains_visible(server, ctx, anyio_backend):
    """Bare object fields remain valid LLM-visible parameters."""
    tool = webhook_tool(
        name="t",
        description="d",
        url=f"{server}/api/tickets",
        method="POST",
        body_schema={
            "type": "object",
            "properties": {
                "metadata": {"type": "object"},
            },
        },
        is_background=False,
    )

    assert tool.parameters["metadata"].type_annotation is dict

    await tool.func(ctx, metadata={"note": "freeform"})

    assert _log.last["body"]["metadata"]["note"] == "freeform"


async def test_required_mixed_nested_object_remains_required(server, ctx, anyio_backend):
    """Required nested objects stay required when they mix visible and constant children."""
    tool = webhook_tool(
        name="t",
        description="d",
        url=f"{server}/api/tickets",
        method="POST",
        body_schema={
            "type": "object",
            "required": ["ticket"],
            "properties": {
                "ticket": {
                    "type": "object",
                    "required": ["subject"],
                    "properties": {
                        "subject": {"type": "string"},
                        "source": {"type": "string", "constant_value": "voice_agent"},
                    },
                },
            },
        },
        is_background=False,
    )

    assert tool.parameters["ticket"].required is True


async def test_url_template_params(server, ctx, anyio_backend):
    """URL template variables are interpolated and percent-encoded."""
    tool = webhook_tool(
        name="t",
        description="d",
        url=f"{server}/orgs/{{org_id}}/tickets",
        method="POST",
        body_schema={
            "type": "object",
            "required": ["subject"],
            "properties": {"subject": {"type": "string"}},
        },
        is_background=False,
    )

    await tool.func(ctx, org_id="acme corp", subject="Issue")

    req = _log.last
    # spaces should be percent-encoded
    assert req["path"] == "/orgs/acme%20corp/tickets"
    assert req["body"]["subject"] == "Issue"


@pytest.mark.parametrize(
    "tool_kwargs",
    [
        {"subject": "Issue"},
        {"org_id": None, "subject": "Issue"},
    ],
)
async def test_missing_url_template_param_fails_before_request(
    server, ctx, anyio_backend, tool_kwargs
):
    """Missing or null URL template variables fail without sending a request."""
    tool = webhook_tool(
        name="t",
        description="d",
        url=f"{server}/orgs/{{org_id}}/tickets",
        method="POST",
        body_schema={
            "type": "object",
            "required": ["subject"],
            "properties": {"subject": {"type": "string"}},
        },
        is_background=False,
    )

    result = json.loads(await tool.func(ctx, **tool_kwargs))

    assert result["ok"] is False
    assert result["status"] is None
    assert "Missing required URL path parameter(s): org_id" in result["error"]
    assert _log.requests == []


async def test_query_params(server, ctx, anyio_backend):
    """Query parameters including booleans are sent in the URL query string."""
    tool = webhook_tool(
        name="t",
        description="d",
        url=f"{server}/search",
        method="GET",
        query_params_schema={
            "type": "object",
            "required": ["q"],
            "properties": {
                "q": {"type": "string"},
                "limit": {"type": "integer"},
                "verbose": {"type": "boolean"},
            },
        },
        is_background=False,
    )

    await tool.func(ctx, q="hello world", limit=5, verbose=True)

    req = _log.last
    assert req["method"] == "GET"
    assert req["body"] is None  # GET with no body_schema sends no body
    assert "q=hello" in req["query"]
    assert "limit=5" in req["query"]
    assert "verbose=true" in req["query"]


async def test_query_params_with_constants(server, ctx, anyio_backend):
    """constant_value query params are hidden from the LLM and still sent."""
    tool = webhook_tool(
        name="t",
        description="d",
        url=f"{server}/search",
        method="GET",
        query_params_schema={
            "type": "object",
            "required": ["q", "api_key"],
            "properties": {
                "q": {"type": "string"},
                "api_key": {"type": "string", "constant_value": "public-token"},
                "include_archived": {"type": "boolean", "constant_value": False},
            },
        },
        is_background=False,
    )

    assert "api_key" not in tool.parameters
    assert "include_archived" not in tool.parameters

    await tool.func(ctx, q="hello world")

    req = _log.last
    assert req["method"] == "GET"
    assert req["body"] is None
    parsed_query = parse_qs(req["query"])
    assert parsed_query["q"] == ["hello world"]
    assert parsed_query["api_key"] == ["public-token"]
    assert parsed_query["include_archived"] == ["false"]


async def test_custom_headers(server, ctx, anyio_backend):
    """Static headers are sent alongside the request."""
    tool = webhook_tool(
        name="t",
        description="d",
        url=f"{server}/api",
        method="POST",
        body_schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
        },
        headers={"X-Custom-Header": "custom-value", "X-Request-Id": "req-42"},
        is_background=False,
    )

    await tool.func(ctx, x="test")

    hdrs = _log.last["headers"]
    assert hdrs["X-Custom-Header"] == "custom-value"
    assert hdrs["X-Request-Id"] == "req-42"


async def test_put_method(server, ctx, anyio_backend):
    """Non-POST methods work correctly."""
    tool = webhook_tool(
        name="t",
        description="d",
        url=f"{server}/api/items/{{item_id}}",
        method="PUT",
        body_schema={
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        },
        is_background=False,
    )

    await tool.func(ctx, item_id="42", name="Updated Widget")

    req = _log.last
    assert req["method"] == "PUT"
    assert req["path"] == "/api/items/42"
    assert req["body"]["name"] == "Updated Widget"


async def test_delete_method(server, ctx, anyio_backend):
    """DELETE method with URL template works."""
    tool = webhook_tool(
        name="t",
        description="d",
        url=f"{server}/api/items/{{item_id}}",
        method="DELETE",
        is_background=False,
    )

    result = json.loads(await tool.func(ctx, item_id="99"))

    assert result["ok"] is True
    req = _log.last
    assert req["method"] == "DELETE"
    assert req["path"] == "/api/items/99"


async def test_response_truncation(server, ctx, anyio_backend):
    """Large response bodies are truncated at 4096 chars."""
    tool = webhook_tool(
        name="t",
        description="d",
        url=f"{server}/large/6000",
        method="GET",
        is_background=False,
    )

    result = json.loads(await tool.func(ctx))

    assert result["ok"] is True
    assert result["status"] == 200
    assert result["body"].endswith("... (truncated)")
    assert len(result["body"]) < 4200


async def test_small_response_not_truncated(server, ctx, anyio_backend):
    """Responses under 4096 chars are returned in full without truncation."""
    tool = webhook_tool(
        name="t",
        description="d",
        url=f"{server}/api",
        method="POST",
        body_schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
        },
        is_background=False,
    )

    result = json.loads(await tool.func(ctx, x="test"))

    assert result["ok"] is True
    assert result["status"] == 201
    assert "truncated" not in result["body"]
    assert "TKT-00001" in result["body"]


async def test_server_error_status(server, ctx, anyio_backend):
    """Non-2xx responses return ok=false with the status code and error body."""
    tool = webhook_tool(
        name="t",
        description="d",
        url=f"{server}/error/500",
        method="POST",
        body_schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
        },
        is_background=False,
    )

    result = json.loads(await tool.func(ctx, x="test"))

    assert result["ok"] is False
    assert result["status"] == 500
    assert "server returned 500" in result["error"]


async def test_not_found_status(server, ctx, anyio_backend):
    """404 responses return ok=false with status 404."""
    tool = webhook_tool(
        name="t",
        description="d",
        url=f"{server}/error/404",
        method="GET",
        is_background=False,
    )

    result = json.loads(await tool.func(ctx))

    assert result["ok"] is False
    assert result["status"] == 404


async def test_auth_rejected(server, ctx, anyio_backend, monkeypatch):
    """Server returns 401 when auth token is wrong."""
    monkeypatch.setenv("BAD_KEY", "wrong-token")

    tool = webhook_tool(
        name="t",
        description="d",
        url=f"{server}/auth-required/tickets",
        method="POST",
        body_schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
        },
        auth={"Authorization": "Bearer ${BAD_KEY}"},
        is_background=False,
    )

    result = json.loads(await tool.func(ctx, x="test"))

    assert result["ok"] is False
    assert result["status"] == 401
    assert "unauthorized" in result["error"]


async def test_auth_accepted(server, ctx, anyio_backend, monkeypatch):
    """Server returns 201 when auth token is correct."""
    monkeypatch.setenv("GOOD_KEY", "valid-token")

    tool = webhook_tool(
        name="t",
        description="d",
        url=f"{server}/auth-required/tickets",
        method="POST",
        body_schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
        },
        auth={"Authorization": "Bearer ${GOOD_KEY}"},
        is_background=False,
    )

    result = json.loads(await tool.func(ctx, x="test"))

    assert result["ok"] is True
    assert result["status"] == 201
    assert "TKT-00001" in result["body"]


async def test_connection_refused(ctx, anyio_backend):
    """Connection refused returns ok=false with a descriptive error."""
    tool = webhook_tool(
        name="t",
        description="d",
        url="http://127.0.0.1:1",  # port 1 — nothing is listening
        method="POST",
        body_schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
        },
        is_background=False,
    )

    result = json.loads(await tool.func(ctx, x="test"))

    assert result["ok"] is False
    assert result["status"] is None
    assert "Request failed" in result["error"]


async def test_timeout(server, ctx, anyio_backend):
    """Request that exceeds the timeout returns ok=false with timeout error."""
    tool = webhook_tool(
        name="t",
        description="d",
        url=f"{server}/slow/5",  # server sleeps 5s
        method="GET",
        timeout=0.3,  # but we only wait 0.3s
        is_background=False,
    )

    result = json.loads(await tool.func(ctx))

    assert result["ok"] is False
    assert result["status"] is None
    assert "timed out" in result["error"]


async def test_combined_url_body_query_auth_headers(server, ctx, anyio_backend, monkeypatch):
    """All features combined in a single request."""
    monkeypatch.setenv("MY_TOKEN", "tok-abc")

    tool = webhook_tool(
        name="full_test",
        description="d",
        url=f"{server}/v2/{{tenant}}/tickets",
        method="PATCH",
        body_schema={
            "type": "object",
            "required": ["status"],
            "properties": {
                "status": {"type": "string"},
                "internal": {"type": "boolean", "constant_value": True},
                "meta": {
                    "type": "object",
                    "properties": {
                        "updated_by": {"type": "string", "constant_value": "system"},
                    },
                },
            },
        },
        query_params_schema={
            "type": "object",
            "required": ["notify"],
            "properties": {"notify": {"type": "boolean"}},
        },
        auth={"Authorization": "Bearer ${MY_TOKEN}"},
        headers={"X-Trace": "trace-123"},
        is_background=False,
    )

    result = json.loads(await tool.func(ctx, tenant="acme", status="resolved", notify=True))

    assert result["ok"] is True
    assert result["status"] == 201

    req = _log.last
    assert req["method"] == "PATCH"
    assert req["path"] == "/v2/acme/tickets"
    assert req["headers"]["Authorization"] == "Bearer tok-abc"
    assert req["headers"]["X-Trace"] == "trace-123"
    assert req["body"]["status"] == "resolved"
    assert req["body"]["internal"] is True
    assert req["body"]["meta"]["updated_by"] == "system"
    assert "notify=true" in req["query"]
