"""AuditableDecision - typed model for AI decisions that cross the audit boundary (§21, §24)."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AuditableDecision(BaseModel):
    """A single AI-generated or user-overridden decision recorded in the audit log.

    Every strategy change, model choice, threshold, and user override must be
    wrapped in this model before being passed to audit.append(). This ensures
    the decision carries structured metadata (actor, category, reasoning) in
    addition to the payload value.

    JSON schema emitted to the model:
    {
      "run_id": "<uuid>",
      "actor": "ai" | "user" | "system",
      "category": "<pipeline area>",
      "action": "<event name>",
      "payload": { ... },
      "reason": "<human-readable justification>",
      "overrides": null | { "field": "<path>", "before": ..., "after": ... }
    }
    """

    run_id: str
    actor: str  # "ai" | "user" | "system"
    category: str
    action: str
    payload: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None

    # Populated when this decision replaces a prior AI decision
    overrides: "DecisionOverride | None" = None

    decided_at: datetime = Field(default_factory=datetime.utcnow)


class DecisionOverride(BaseModel):
    """Records the before/after values when a user overrides an AI decision."""

    field_path: str
    before: Any
    after: Any
    override_reason: str | None = None
