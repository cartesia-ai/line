"""Shared agent typing for v0.2 voice agents."""

from __future__ import annotations

from typing import AsyncIterable, Callable, Optional, Protocol, Sequence, Tuple, Union

from line.v02.events import InputEvent, OutputEvent

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
AgentSpec = Union[Agent, Tuple[Agent, Optional[EventFilter], Optional[EventFilter]]]

# Forward declaration to avoid circular typing at runtime.
class AgentEnv:
    pass

class TurnEnv:
    pass
