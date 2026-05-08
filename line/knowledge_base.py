"""Knowledge base client for querying agent-scoped documents via the Cartesia API."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger

DEFAULT_BASE_URL = "https://api.cartesia.ai"
DEFAULT_TOP_K = 5
DEFAULT_TIMEOUT_S = 3.0
LOG_TRUNCATION = 500
# Threshold above which we surface a warning so a developer who accidentally
# set an unreasonably long timeout (e.g. ``60.0``) can spot the misconfiguration
# from the logs rather than wondering why the call appears to hang.
LONG_TIMEOUT_WARN_S = 10.0


class KnowledgeBaseError(RuntimeError):
    """Raised when a knowledge base query fails."""


def _warn_if_long_timeout(timeout_s: Optional[float], *, source: str) -> None:
    if timeout_s is not None and timeout_s > LONG_TIMEOUT_WARN_S:
        logger.warning(
            "{} timeout_s={}s exceeds {}s; long timeouts can stall the call "
            "while a slow knowledge base query is in flight.",
            source,
            timeout_s,
            LONG_TIMEOUT_WARN_S,
        )


class KnowledgeBase:
    """Client for the agent knowledge base query endpoint.

    Calls `GET /agents/{agent_id}/documents/query` with the agent-scoped
    JWT minted by the Cartesia API at session start.
    """

    def __init__(
        self,
        agent_id: Optional[str],
        agent_token: Optional[str],
        base_url: Optional[str] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self.agent_id = agent_id
        self.agent_token = agent_token
        self.base_url = (base_url or os.getenv("CARTESIA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s
        _warn_if_long_timeout(timeout_s, source="KnowledgeBase")

    async def query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = DEFAULT_TOP_K,
        timeout_s: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Run a query against the agent's knowledge base.

        Returns the raw result objects from the API, each currently shaped as
        ``{"content": str}``. Pass-through is intentional so new fields the
        API adds (scores, metadata, document IDs, …) flow to callers without
        a SDK change. Callers are responsible for extracting fields they
        care about and formatting for downstream use (e.g. LLM consumption).

        ``timeout_s`` overrides the per-call timeout for this query; falls
        back to the client's configured timeout when None.
        """
        if not self.agent_id or not self.agent_token:
            raise KnowledgeBaseError(
                "Knowledge base is not available in this session "
                "(missing agent_id or agent_token from start message)."
            )

        params: Dict[str, str] = {"query": query, "top_k": str(top_k)}
        if filters:
            params["filters"] = json.dumps(filters)

        url = f"{self.base_url}/agents/{self.agent_id}/documents/query"
        headers = {"Authorization": f"Bearer {self.agent_token}"}

        effective_timeout = timeout_s if timeout_s is not None else self.timeout_s
        timeout = aiohttp.ClientTimeout(total=effective_timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params, headers=headers) as resp:
                    body = await resp.text()
                    logger.info("knowledge_base query: url={} params={} status={} body={}", url, params, resp.status, body[:LOG_TRUNCATION])
                    if resp.status != 200:
                        logger.warning(
                            "knowledge_base query failed: status={} body={}",
                            resp.status,
                            body[:LOG_TRUNCATION],
                        )
                        raise KnowledgeBaseError(
                            f"Knowledge base query failed with status {resp.status}: {body[:LOG_TRUNCATION]}"
                        )
                    try:
                        payload = json.loads(body)
                        logger.debug("knowledge_base query payload: {}", payload)
                    except json.JSONDecodeError as e:
                        raise KnowledgeBaseError(f"Invalid JSON from knowledge base: {e}") from e
        except asyncio.TimeoutError as e:
            raise KnowledgeBaseError(f"Knowledge base query timed out after {effective_timeout}s") from e
        except aiohttp.ClientError as e:
            raise KnowledgeBaseError(f"Knowledge base query transport error: {e}") from e

        return payload.get("results") or []
