"""Pydantic models for the persistent chat co-pilot (§21)."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

VALID_INTENTS = {"question", "modify", "abort", "request_artifact", "navigate"}
VALID_CATEGORIES = {
    "eda", "preprocessing", "target", "model_selection", "threshold",
    "fairness", "drift", "deliverables", "general", "request_plot",
}


class ChatIntent(BaseModel):
    intent: str  # question | modify | abort | request_artifact | navigate
    confidence: float  # 0-1
    category: str  # which pipeline area is affected
    structured_payload: dict[str, Any]  # what specifically to change
    needs_confirmation: bool
    reasoning: str

    @classmethod
    def question_fallback(cls) -> "ChatIntent":
        """Safe default when Haiku classification fails."""
        return cls(
            intent="question",
            confidence=0.0,
            category="general",
            structured_payload={},
            needs_confirmation=False,
            reasoning="intent classification failed - defaulting to question",
        )


class StrategyDiff(BaseModel):
    field_path: str  # dot-path to changed field, e.g. "preprocessing.columns.age.imputation_strategy"
    before: Any
    after: Any
    summary: str
    run_id: str


class PendingIntent(BaseModel):
    """A modify intent queued during an expensive pipeline step (§2 interrupt semantics)."""

    intent: ChatIntent
    queued_at: datetime = Field(default_factory=datetime.utcnow)
    step_at_queue_time: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.model_dump(),
            "queued_at": self.queued_at.isoformat(),
            "step_at_queue_time": self.step_at_queue_time,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PendingIntent":
        return cls(
            intent=ChatIntent.model_validate(d["intent"]),
            queued_at=datetime.fromisoformat(d["queued_at"]),
            step_at_queue_time=d["step_at_queue_time"],
        )


class UserDirective(BaseModel):
    """A verbatim human override recorded in chat and replayed to a producing agent.

    The clinician can override ANY AI decision by chatting (§2, §21). Rather than
    parsing the instruction into a single rigid field edit, the verbatim text is
    stored and injected into the producing agent's prompt when its step re-runs,
    so the AI regenerates its decision honouring the human's instruction.

    `columns_to_drop` is a structured hint (from intent classification) used only
    as a deterministic backstop: any column the human unambiguously asked to drop
    is guaranteed dropped even if the agent under-complies. It never replaces the
    verbatim instruction as the primary signal.
    """

    category: str  # decision category this directive overrides (e.g. "preprocessing")
    instruction: str  # the user's verbatim chat message
    columns_to_drop: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "instruction": self.instruction,
            "columns_to_drop": list(self.columns_to_drop),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UserDirective":
        created = d.get("created_at")
        return cls(
            category=d["category"],
            instruction=d["instruction"],
            columns_to_drop=list(d.get("columns_to_drop") or []),
            created_at=datetime.fromisoformat(created) if created else datetime.utcnow(),
        )


class ChatMessageCreate(BaseModel):
    content: str
