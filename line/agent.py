"""Core types for Agents"""

from __future__ import annotations

import asyncio
from typing import AsyncIterable, Callable, Optional, Protocol, Sequence, Union

from line.events import InputEvent, OutputEvent
from line.knowledge_base import KnowledgeBase


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
    """Per-call environment created by the harness once per websocket connection.

    Owns the long-lived, call-scoped data: the asyncio loop and the
    agent-scoped credentials needed to call back into the Cartesia API on
    behalf of the agent (e.g. for knowledge base queries).
    """

    def __init__(
        self,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        agent_id: Optional[str] = None,
        agent_token: Optional[str] = None,
        base_url: Optional[str] = None,
        zdr: bool = False,
    ):
        self.loop = loop
        self.agent_id = agent_id
        self.agent_token = agent_token
        self.base_url = base_url
        self.zdr = zdr

    def knowledge_base(self) -> KnowledgeBase:
        """Return a KnowledgeBase client scoped to the calling agent."""
        return KnowledgeBase(
            agent_id=self.agent_id,
            agent_token=self.agent_token,
            base_url=self.base_url,
        )


class TurnEnv:
    """Per-turn environment passed to agents and tools.

    Holds a reference to the call's `AgentEnv` and delegates call-scoped
    operations (like knowledge base lookups) to it.
    """

    def __init__(self, agent_env: AgentEnv):
        self.agent_env = agent_env

    def knowledge_base(self) -> KnowledgeBase:
        return self.agent_env.knowledge_base()


def call_agent(agent: Agent, turn_env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
    """Call an agent, handling both AgentClass and AgentCallable."""
    if hasattr(agent, "process"):
        return agent.process(turn_env, event)  # type: ignore[union-attr]
    else:
        return agent(turn_env, event)  # type: ignore[return-value]
