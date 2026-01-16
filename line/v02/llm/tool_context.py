"""Tool execution context passed to tool functions."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from line.v02.agent import TurnEnv
    from line.v02.llm.config import LlmConfig


@dataclass
class ToolContext:
    """Context passed to tool functions during execution."""

    conversation_history: List[Any] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    turn_env: Optional["TurnEnv"] = None
    config: Optional["LlmConfig"] = None


@dataclass
class ToolResult:
    """Result from executing a tool (internal use)."""

    tool_call_id: str
    tool_name: str
    result: Any = None
    error: Optional[str] = None
    events: List[Any] = field(default_factory=list)
    handoff_target: Any = None

    @property
    def success(self) -> bool:
        return self.error is None
