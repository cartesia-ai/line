"""A disposable first completion; it never executes agent code or changes history."""

import asyncio
from copy import deepcopy
from time import perf_counter
from typing import Any, AsyncIterator

from line.llm_agent.config import LlmConfig
from line.llm_agent.provider import LlmProvider, Message, StreamChunk
from line.llm_agent.schema_converter import tools_to_litellm
from line.llm_agent.tools.utils import FunctionTool


def _request_key(
    messages: list[Message], tools: list[FunctionTool], config: LlmConfig, kwargs: dict[str, Any]
) -> tuple:
    return deepcopy((messages, tools_to_litellm(tools, strict=config.strict_tool_schemas), config, kwargs))


class PreparedResponse:
    """Own a bounded stream buffer until a correlated final accepts or discards it.

    Only the provider reader runs speculatively. After confirmation, LlmAgent's
    ordinary generation loop consumes the buffered and subsequent chunks. It
    validates the entire request again, including history and tool availability.
    """

    def __init__(
        self,
        event_id: str,
        provider: LlmProvider,
        messages: list[Message],
        tools: list[FunctionTool],
        config: LlmConfig,
        kwargs: dict[str, Any],
    ) -> None:
        self.event_id = event_id
        self.started_at = perf_counter()
        self.first_chunk_ms: float | None = None
        self.first_text_ms: float | None = None
        self._key = _request_key(messages, tools, config, kwargs)
        self._queue: asyncio.Queue[StreamChunk | None] = asyncio.Queue(maxsize=64)
        self._aborted = False
        self._claimed = False
        self._error: Exception | None = None
        self.task = asyncio.create_task(self._read(provider, messages, tools, config, kwargs))

    async def _read(
        self,
        provider: LlmProvider,
        messages: list[Message],
        tools: list[FunctionTool],
        config: LlmConfig,
        kwargs: dict[str, Any],
    ) -> None:
        try:
            iterator = provider.chat(messages, tools or None, config=config, **kwargs).__aiter__()
            try:
                async for chunk in iterator:
                    if self._aborted:
                        return
                    elapsed_ms = (perf_counter() - self.started_at) * 1000
                    if self.first_chunk_ms is None and (chunk.text or chunk.tool_calls):
                        self.first_chunk_ms = elapsed_ms
                    if self.first_text_ms is None and chunk.text:
                        self.first_text_ms = elapsed_ms
                    await self._queue.put(deepcopy(chunk))
            finally:
                close = getattr(iterator, "aclose", None)
                if close is not None:
                    await close()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._error = exc
        finally:
            if not self._aborted:
                await self._queue.put(None)

    def claim(
        self, messages: list[Message], tools: list[FunctionTool], config: LlmConfig, kwargs: dict[str, Any]
    ) -> bool:
        """Accept only an unchanged first request; a failed prediction falls back."""
        if (
            self._aborted
            or self._claimed
            or self._error
            or self._key != _request_key(messages, tools, config, kwargs)
        ):
            self.abort()
            return False
        self._claimed = True
        return True

    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        if not self._claimed or self._aborted:
            raise RuntimeError("Prepared response has not been accepted")
        try:
            while (chunk := await self._queue.get()) is not None:
                yield chunk
            if self._error:
                raise self._error
        finally:
            self.abort()

    def abort(self) -> None:
        """Detach immediately; provider cleanup is owned by the tracked reader task."""
        self._aborted = True
        if not self.task.done():
            self.task.cancel()
