"""Runtime tests for the Parallel example integration."""

from contextlib import asynccontextmanager
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from line.agent import AgentEnv, TurnEnv
from line.llm_agent.tools.utils import FunctionTool, ToolEnv
from line.voice_agent_app import AgentConfig, CallRequest

MODULE_PATH = Path(__file__).parents[1] / "example_integrations" / "parallel" / "main.py"
SPEC = importlib.util.spec_from_file_location("parallel_example", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
main = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(main)


class FakeSession:
    """Records the exact hosted MCP tool call."""

    calls = []
    result = None

    def __init__(self, read, write):
        del read, write

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def initialize(self):
        return None

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self.result


@asynccontextmanager
async def fake_transport(url):
    assert url == main.PARALLEL_MCP_URL
    yield object(), object(), None


@pytest.fixture(autouse=True)
def patch_mcp(monkeypatch):
    FakeSession.calls = []
    monkeypatch.setattr(main, "ClientSession", FakeSession)
    monkeypatch.setattr(main, "streamablehttp_client", fake_transport)


@pytest.mark.asyncio
async def test_voice_agent_registers_and_invokes_schema_faithful_search(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "line.llm_agent.llm_agent._get_model_config",
        lambda *args, **kwargs: SimpleNamespace(
            backend="http", supports_reasoning_effort=False, default_reasoning_effort=None
        ),
    )
    FakeSession.result = SimpleNamespace(
        structuredContent={"results": [{"title": str(index)} for index in range(8)]},
        content=[SimpleNamespace(text="duplicate mirrored payload")],
    )
    request = CallRequest(
        call_id="call",
        from_="caller",
        to="agent",
        agent_call_id="agent-call",
        agent=AgentConfig(),
    )

    agent = await main.get_agent(AgentEnv(), request)
    search_tool = next(tool for tool in agent._tools if tool.name == "web_search")
    assert isinstance(search_tool, FunctionTool)
    result = await search_tool.func(
        ToolEnv(TurnEnv(AgentEnv())),
        objective="Find current launch details",
        search_queries=["Parallel latest launch"],
    )

    assert FakeSession.calls == [
        (
            "web_search",
            {
                "objective": "Find current launch details",
                "search_queries": ["Parallel latest launch"],
            },
        )
    ]
    assert "duplicate mirrored payload" not in result
    assert '"title": "4"' in result
    assert '"title": "5"' not in result


@pytest.mark.asyncio
async def test_search_uses_text_only_when_structured_content_absent():
    FakeSession.result = SimpleNamespace(
        structuredContent=None,
        content=[SimpleNamespace(text="first"), SimpleNamespace(text="second")],
    )

    result = await main.web_search.func(
        ToolEnv(TurnEnv(AgentEnv())), objective="objective", search_queries=["query"]
    )

    assert result == "first\nsecond"
    assert FakeSession.calls[0][1] == {"objective": "objective", "search_queries": ["query"]}
