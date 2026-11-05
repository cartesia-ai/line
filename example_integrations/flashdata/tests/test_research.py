"""Exercise the real MCP protocol and Line tool binding with an in-memory transport."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from client import ALLOWED_TOOLS, ENDPOINT, FAILURE, QUERY_TOOLS, FlashDataClient
import httpx
from main import FlashDataTools, get_agent
import pytest

REQUESTS = json.loads(Path(__file__).with_name("requests.json").read_text())


class MCPFixture:
    def __init__(self, *, failure: str | None = None, text_result: bool = False):
        self.calls: list[dict[str, Any]] = []
        self.failure = failure
        self.text_result = text_result

    def handle(self, request: httpx.Request) -> httpx.Response:
        assert str(request.url) == ENDPOINT
        assert request.headers["X-API-Key"] == "test-only-key"
        if request.method != "POST":
            return httpx.Response(405)
        payload = json.loads(request.content)
        if "id" not in payload:
            return httpx.Response(202)
        method = payload["method"]
        if method == "initialize":
            result = {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif method == "tools/list":
            names = ALLOWED_TOOLS | {"youtube_download", "submit_batch", "future_write"}
            result = {
                "tools": [
                    {"name": name, "description": "Fixture tool", "inputSchema": {"type": "object"}}
                    for name in sorted(names)
                ]
            }
        else:
            assert method == "tools/call"
            self.calls.append(payload["params"])
            if self.failure == "redirect":
                return httpx.Response(307, headers={"Location": "https://untrusted.invalid/mcp"})
            if self.failure == "timeout":
                raise httpx.ReadTimeout("private-url-and-storage-secret", request=request)
            if self.failure == "tool":
                result = {"isError": True, "content": [{"type": "text", "text": "private-secret"}]}
            else:
                name = payload["params"]["name"]
                body = {"source": name, "results": [{"title": "Fixture only", "url": "https://example.com"}]}
                if name == "get_job":
                    body = {
                        "job": {"id": "job_fixture", "source": "youtube_download", "status": "completed"},
                        "output": {"signed_url": "private-url-and-storage-secret"},
                    }
                if name == "get_result":
                    body = {"result_id": "result_fixture", "next_offset": 12, "chunk": "fixture-part"}
                result = {"content": [{"type": "text", "text": json.dumps(body)}]}
                if not self.text_result:
                    result["structuredContent"] = body
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

    def client(self) -> FlashDataClient:
        return FlashDataClient("test-only-key", transport=httpx.MockTransport(self.handle))


async def test_discovery_excludes_storage_and_batch_tools():
    fixture = MCPFixture()
    result = await fixture.client().catalog()
    assert {tool["name"] for tool in result["tools"]} == ALLOWED_TOOLS
    assert len(QUERY_TOOLS) == 14
    assert not fixture.calls


@pytest.mark.parametrize("name", sorted(QUERY_TOOLS))
async def test_query_binding_submits_once(name):
    fixture = MCPFixture()
    tool = FlashDataTools(fixture.client()).flashdata_call
    assert tool.is_background
    arguments = REQUESTS[name]
    result = json.loads(await tool.func(None, name, json.dumps(arguments)))
    assert result["source"] == name
    assert len(fixture.calls) == 1
    assert fixture.calls[0]["arguments"] == arguments


@pytest.mark.parametrize("failure", ["redirect", "timeout", "tool"])
async def test_failed_submissions_are_not_replayed_or_exposed(failure):
    fixture = MCPFixture(failure=failure)
    result = await fixture.client().call("google_search", {"query": "fixture"})
    assert result == {"error": FAILURE}
    assert len(fixture.calls) == 1


@pytest.mark.parametrize("name", ["youtube_download", "submit_batch", "future_write"])
async def test_unsupported_tool_never_reaches_transport(name):
    fixture = MCPFixture()
    assert "error" in await fixture.client().call(name, {})
    assert not fixture.calls


@pytest.mark.parametrize("arguments", ["[]", "invalid-json"])
async def test_invalid_arguments_do_not_submit(arguments):
    fixture = MCPFixture()
    result = await FlashDataTools(fixture.client()).flashdata_call.func(None, "google_search", arguments)
    assert "error" in json.loads(result)
    assert not fixture.calls


async def test_download_job_does_not_enter_voice_history():
    result = await MCPFixture().client().call("get_job", {"job_id": "job_fixture"})
    assert result["job"] == {"id": "job_fixture", "status": "completed"}
    assert "private" not in json.dumps(result).replace("private delivery manifest", "")
    assert "output" not in result


async def test_text_result_preserves_chunk_recovery_metadata():
    result = await MCPFixture(text_result=True).client().call("get_result", {"result_id": "result_fixture"})
    assert result["next_offset"] == 12
    assert result["chunk"] == "fixture-part"


async def test_agent_constructs_offline_with_real_sdk(monkeypatch):
    monkeypatch.setenv("FLASHDATA_API_KEY", "test-only-key")
    monkeypatch.setenv("LLM_API_KEY", "test-only-model-key")
    monkeypatch.setenv("LINE_MODEL", "openai/gpt-4o-mini")
    agent = await get_agent(SimpleNamespace(), SimpleNamespace())
    assert {tool.name for tool in agent._tools} == {"flashdata_catalog", "flashdata_call", "end_call"}


async def test_missing_configuration_fails_before_a_session(monkeypatch):
    for name in ["FLASHDATA_API_KEY", "LLM_API_KEY", "LINE_MODEL"]:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="Set FLASHDATA_API_KEY"):
        await get_agent(SimpleNamespace(), SimpleNamespace())
