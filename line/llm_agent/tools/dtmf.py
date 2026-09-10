"""Collect keypad input before invoking a tool callback."""

import asyncio
import inspect
import math
from typing import Any, AsyncIterator, Awaitable, Callable

from line._dtmf import _BUTTONS
from line.events import AgentSendText
from line.llm_agent.tools.utils import FunctionTool, ToolEnv


def dtmf_tool(
    *,
    prompt: str,
    inter_digit_timeout: float = 2.0,
    finish_on_key: str = "#",
    first_digit_timeout: float = 15.0,
    max_digits: int = 64,
    callback_timeout: float = 30.0,
) -> Callable[[Callable[[ToolEnv, str], Awaitable[Any]]], FunctionTool]:
    """Decorate an async ``(ctx: ToolEnv, digits: str)`` callback as a keypad tool.

    The LLM calls this tool without arguments. The SDK arms collection before
    speaking ``prompt`` and supplies the collected digits to the callback. Input
    ends on ``finish_on_key`` (excluded from the value) or after
    ``inter_digit_timeout`` seconds without another digit. The first digit has
    a separate timeout, measured from when the prompt is sent.

    Speech cannot cancel or restart the agent during collection or the callback.
    A speech turn that overlaps this step is consumed through its end. Hangup
    still cancels the operation. Only one collection can be active per call.

    Empty input returns ``{"status": "no_input"}``; exceeding ``max_digits``
    returns ``{"status": "too_many_digits"}``. Neither invokes the callback.
    Callback timeout returns ``{"status": "callback_timeout"}``. Otherwise the
    callback's return value becomes the tool result for the LLM.

    Requires a transport that delivers DTMF events to VoiceAgentApp. This helper
    does not redact recordings, transcripts, or logs.
    """
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if len(finish_on_key) != 1 or finish_on_key not in _BUTTONS:
        raise ValueError("finish_on_key must be one of 0-9, *, #")
    for name, value in (
        ("inter_digit_timeout", inter_digit_timeout),
        ("first_digit_timeout", first_digit_timeout),
        ("callback_timeout", callback_timeout),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if isinstance(max_digits, bool) or not isinstance(max_digits, int) or max_digits <= 0:
        raise ValueError("max_digits must be a positive integer")

    def decorate(func: Callable[[ToolEnv, str], Awaitable[Any]]) -> FunctionTool:
        parameters = list(inspect.signature(func).parameters.values())
        if (
            not inspect.iscoroutinefunction(func)
            or len(parameters) != 2
            or parameters[0].name not in ("ctx", "context")
            or parameters[1].name != "digits"
            or any(p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in parameters)
        ):
            raise TypeError("dtmf_tool requires an async callback with signature (ctx: ToolEnv, digits: str)")

        async def collect(ctx: ToolEnv) -> AsyncIterator[Any]:
            dtmf = ctx.turn_env.agent_env._dtmf
            with dtmf.collect(inter_digit_timeout, first_digit_timeout, finish_on_key, max_digits) as capture:
                yield AgentSendText(text=prompt)
                status = await capture.result
                if status != "complete":
                    yield {"status": status}
                else:
                    try:
                        result = await asyncio.wait_for(func(ctx, capture.digits), timeout=callback_timeout)
                    except asyncio.TimeoutError:
                        result = {"status": "callback_timeout"}
                    except Exception:
                        # Preserve normal tool error handling, after overlapping
                        # speech ends so its response can be heard.
                        await dtmf.wait_for_speech_end()
                        raise
                    # Keep the guard until LlmAgent has recorded the result and
                    # consumed this generator, including while output is sent.
                    yield result
                # Let an overlapping "done" finish before generating a spoken
                # response; the audio transport may drop replies during speech.
                await dtmf.wait_for_speech_end()

        description = (func.__doc__ or "").strip()
        description += (
            " Collect keypad input from the caller and process it. "
            "This tool speaks its own prompt; call it without asking for digits first."
        )
        return FunctionTool(name=func.__name__, description=description.strip(), func=collect, parameters={})

    return decorate
