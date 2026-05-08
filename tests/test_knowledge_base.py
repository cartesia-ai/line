"""Tests for KnowledgeBase client and environment wiring."""

import asyncio
import json
from typing import Dict, Optional
from urllib.parse import urlparse

import pytest

from line.agent import AgentEnv, TurnEnv
from line.knowledge_base import KnowledgeBase, KnowledgeBaseError
from line.llm_agent.tools.utils import ToolEnv

pytestmark = [pytest.mark.anyio, pytest.mark.parametrize("anyio_backend", ["asyncio"])]


class FakeKBEndpoint:
    """Mock the aiohttp boundary for the bifrost KB query endpoint."""

    def __init__(self, results: Optional[list] = None, status: int = 200, body: Optional[str] = None):
        self.results = results if results is not None else []
        self.status = status
        self.body = body
        self.raise_on_get: Optional[BaseException] = None
        self.last_url: Optional[str] = None
        self.last_path: Optional[str] = None
        self.last_query: Optional[Dict[str, str]] = None
        self.last_auth: Optional[str] = None
        self.session_timeout_s: Optional[float] = None

    def client_session_factory(self):
        endpoint = self

        class FakeClientSession:
            def __init__(self, timeout):
                endpoint.session_timeout_s = timeout.total

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def get(self, url, params=None, headers=None):
                if endpoint.raise_on_get is not None:
                    raise endpoint.raise_on_get
                endpoint.last_url = url
                endpoint.last_path = urlparse(url).path
                endpoint.last_query = dict(params or {})
                endpoint.last_auth = (headers or {}).get("Authorization")
                return FakeKBResponse(endpoint)

        return FakeClientSession


class FakeKBResponse:
    def __init__(self, endpoint: FakeKBEndpoint):
        self._endpoint = endpoint
        self.status = endpoint.status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self) -> str:
        endpoint = self._endpoint
        if endpoint.body is not None:
            return endpoint.body
        return json.dumps({"results": endpoint.results})


@pytest.fixture
def kb_endpoint(monkeypatch):
    fake = FakeKBEndpoint()
    monkeypatch.setattr(
        "line.knowledge_base.aiohttp.ClientSession",
        fake.client_session_factory(),
    )
    return fake


# =============================================================================
# KnowledgeBase client
# =============================================================================


async def test_query_sends_bearer_token_and_returns_results(kb_endpoint, anyio_backend):
    fake = kb_endpoint
    fake.results = [{"content": "first chunk"}, {"content": "second chunk"}]

    kb = KnowledgeBase(
        agent_id="agent-123",
        agent_token="tok-abc",
        base_url="https://kb.example.test",
    )
    results = await kb.query("hello world")

    assert results == [{"content": "first chunk"}, {"content": "second chunk"}]
    assert fake.last_path == "/agents/agent-123/documents/query"
    assert fake.last_query == {"query": "hello world", "top_k": "5"}
    assert fake.last_auth == "Bearer tok-abc"


async def test_query_passes_through_unknown_fields(kb_endpoint, anyio_backend):
    # If bifrost adds fields (e.g. score, document_id, metadata), the client
    # forwards them untouched so callers can use them without an SDK bump.
    fake = kb_endpoint
    fake.results = [
        {"content": "x", "score": 0.91, "document_id": "doc-1"},
    ]

    kb = KnowledgeBase(
        agent_id="agent-123",
        agent_token="tok-abc",
        base_url="https://kb.example.test",
    )
    results = await kb.query("q")

    assert results == [{"content": "x", "score": 0.91, "document_id": "doc-1"}]


async def test_query_serializes_filters_as_json(kb_endpoint, anyio_backend):
    fake = kb_endpoint
    fake.results = []

    kb = KnowledgeBase(
        agent_id="agent-123",
        agent_token="tok-abc",
        base_url="https://kb.example.test",
    )
    await kb.query("q", filters={"category": "billing"}, top_k=3)

    assert fake.last_query is not None
    assert json.loads(fake.last_query["filters"]) == {"category": "billing"}
    assert fake.last_query["top_k"] == "3"


async def test_query_empty_results_returns_empty_list(kb_endpoint, anyio_backend):
    fake = kb_endpoint
    fake.results = []

    kb = KnowledgeBase(
        agent_id="agent-123",
        agent_token="tok-abc",
        base_url="https://kb.example.test",
    )
    results = await kb.query("anything")

    assert results == []


async def test_query_raises_on_non_200(kb_endpoint, anyio_backend):
    fake = kb_endpoint
    fake.status = 401
    fake.body = "unauthorized"

    kb = KnowledgeBase(
        agent_id="agent-123",
        agent_token="bad-token",
        base_url="https://kb.example.test",
    )

    with pytest.raises(KnowledgeBaseError, match="401"):
        await kb.query("q")


async def test_query_without_credentials_raises(anyio_backend):
    kb = KnowledgeBase(agent_id=None, agent_token=None, base_url="http://localhost")
    with pytest.raises(KnowledgeBaseError, match="not available"):
        await kb.query("q")


async def test_query_per_call_timeout_overrides_client_default(kb_endpoint, anyio_backend):
    # Per-query timeout takes precedence over the client-configured default.
    # Timeouts are wrapped as KnowledgeBaseError at the client boundary so
    # the tool layer's single-exception catch handles them gracefully.
    fake = kb_endpoint
    fake.raise_on_get = asyncio.TimeoutError()

    kb = KnowledgeBase(
        agent_id="agent-123",
        agent_token="tok-abc",
        base_url="https://kb.example.test",
        timeout_s=5.0,
    )
    with pytest.raises(KnowledgeBaseError, match="timed out"):
        await kb.query("q", timeout_s=0.05)

    assert fake.session_timeout_s == 0.05


# =============================================================================
# TurnEnv / ToolEnv wiring
# =============================================================================


def test_agent_env_knowledge_base_threads_creds(anyio_backend):
    agent_env = AgentEnv(agent_id="a1", agent_token="t1", base_url="http://example.com")
    kb = agent_env.knowledge_base()

    assert isinstance(kb, KnowledgeBase)
    assert kb.agent_id == "a1"
    assert kb.agent_token == "t1"
    assert kb.base_url == "http://example.com"


def test_turn_env_delegates_to_agent_env(anyio_backend):
    agent_env = AgentEnv(agent_id="a1", agent_token="t1")
    env = TurnEnv(agent_env=agent_env)

    kb = env.knowledge_base()
    assert isinstance(kb, KnowledgeBase)
    assert kb.agent_id == "a1"
    assert kb.agent_token == "t1"


def test_tool_env_delegates_to_turn_env(anyio_backend):
    agent_env = AgentEnv(agent_id="a1", agent_token="t1")
    ctx = ToolEnv(turn_env=TurnEnv(agent_env=agent_env))

    kb = ctx.knowledge_base()
    assert isinstance(kb, KnowledgeBase)
    assert kb.agent_id == "a1"


def test_api_base_url_threads_from_start_message(anyio_backend):
    # Inferno forwards api_base_url on the start message; it should land on
    # the KnowledgeBase client untouched, taking precedence over the env
    # var and the hardcoded prod fallback.
    from line.voice_agent_app import _call_request_from_start_data

    call_request = _call_request_from_start_data(
        {
            "type": "start",
            "call_id": "c1",
            "agent": {},
            "agent_token": "tok",
            "api_base_url": "https://api.staging.example",
        }
    )
    assert call_request.api_base_url == "https://api.staging.example"

    agent_env = AgentEnv(
        agent_id="a1",
        agent_token=call_request.agent_token,
        base_url=call_request.api_base_url,
    )
    kb = agent_env.knowledge_base()
    assert kb.base_url == "https://api.staging.example"


def test_agent_id_threads_from_start_message_agent_id_field(anyio_backend):
    # Inferno's AgentConfig carries `id` on the start message; it should land
    # on CallRequest.agent.id verbatim.
    from line.voice_agent_app import _call_request_from_start_data

    call_request = _call_request_from_start_data(
        {
            "type": "start",
            "call_id": "c1",
            "agent": {"id": "agent-from-config"},
            "agent_token": "tok",
            "api_base_url": "https://api.staging.example",
        }
    )
    assert call_request.agent.id == "agent-from-config"
