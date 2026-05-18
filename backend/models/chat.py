"""Pydantic models for the persistent chat co-pilot (§21)."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

VALID_INTENTS = {"question", "modify", "abort", "request_artifact", "navigate"}
VALID_CATEGORIES = {
    "eda", "preprocessing", "model_selection", "threshold",
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


class ChatMessageCreate(BaseModel):
    content: str
