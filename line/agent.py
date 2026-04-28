"""Core types for Agents"""

from __future__ import annotations

import asyncio
from typing import AsyncIterable, Callable, Optional, Protocol, Sequence, Union

from line.events import InputEvent, OutputEvent


# Equivalent to a constructor type in TS where a class provides a process method.
class AgentClass(Protocol):
    def process(self, env: "TurnEnv", event: InputEvent) -> AsyncIterable[OutputEvent]: ...


# A callable agent: (env, event) -> AsyncIterable[OutputEvent]
AgentCallable = Callable[["TurnEnv", InputEvent], AsyncIterable[OutputEvent]]

# Agent can be either a callable or a class implementing process().
Agent = Union[AgentCallable, AgentClass]

# EventFilter matches either a callable predicate or a list/tuple of event types.
EventFilter = Union[Callable[[InputEvent], bool], Sequence[type[InputEvent]]]

# get_agent may return just the Agent or Agent plus run/cancel filters.
AgentSpec = Union[Agent, tuple[Agent, EventFilter, EventFilter]]


class AgentEnv:
    """Per-call environment created by the harness once per websocket connection."""

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        self.loop = loop


class TurnEnv:
    """Per-turn environment passed to agents and tools."""

    def __init__(self, agent_env: AgentEnv):
        self.agent_env = agent_env


def call_agent(agent: Agent, turn_env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
    """Call an agent, handling both AgentClass and AgentCallable."""
    if hasattr(agent, "process"):
        return agent.process(turn_env, event)  # type: ignore[union-attr]
    else:
        return agent(turn_env, event)  # type: ignore[return-value]
