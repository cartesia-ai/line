from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class PreCallResult(BaseModel):
    """Result from pre_call_handler containing metadata and config."""

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata to include with the call")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration for the call")


class AgentConfig(BaseModel):
    """Agent information for the call."""

    system_prompt: Optional[str] = None  # System prompt to define the agent's role and behavior
    introduction: Optional[str] = None  # Introduction message for the agent to start the call with


class CallRequest(BaseModel):
    """Request body for the /chats endpoint."""

    call_id: str
    from_: str = Field(alias="from")  # Using from_ to avoid Python keyword conflict
    to: str
    agent_call_id: str  # Agent call ID for logging and correlation
    agent: AgentConfig
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        # Allow both field name (from_) and alias (from) for input
        populate_by_name=True
    )
