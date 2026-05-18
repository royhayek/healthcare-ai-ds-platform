"""Pydantic models for EDA agent output (§9).

EdaReport is stored as JSONB in runs.eda_report and consumed by:
- The chat agent (context block)
- The preprocessing agent (recommendations feed)
- The insight agent (final report context)
- The frontend checkpoint card
"""

from typing import Any

from pydantic import BaseModel, field_validator, model_validator


class QualityIssue(BaseModel):
    column: str | None = None
    issue: str = ""
    severity: str = "medium"  # low | medium | high; default if the model omits
    recommendation: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalise_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        # severity: the model sometimes uses "impact", "level", "priority", "risk"
        if "severity" not in data:
            for alt in ("impact", "level", "priority", "risk", "criticality"):
                if alt in data:
                    data["severity"] = data[alt]
                    break
        # recommendation: the model sometimes uses "fix", "action", "suggestion", "mitigation"
        if "recommendation" not in data:
            for alt in ("fix", "action", "suggestion", "mitigation", "action_item", "resolution"):
                if alt in data:
                    data["recommendation"] = data[alt]
                    break
        # issue: the model sometimes uses "description", "problem", "detail"
        if "issue" not in data:
            for alt in ("description", "problem", "detail", "details", "notes"):
                if alt in data:
                    data["issue"] = data[alt]
                    break
        return data


class PreprocessingRecommendation(BaseModel):
    column: str | None = None
    strategy: str = ""
    reason: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalise_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        # column: the model sometimes uses "columns" (plural) for multi-column recs
        if "column" not in data and "columns" in data:
            cols = data["columns"]
            data["column"] = ", ".join(cols) if isinstance(cols, list) else str(cols)
        # strategy: the model sometimes uses "action", "approach", "method", "recommendation"
        if "strategy" not in data:
            for alt in ("action", "approach", "method", "recommendation", "technique", "encoding"):
                if alt in data:
                    data["strategy"] = data[alt]
                    break
        # reason: the model sometimes uses "description", "rationale", "justification", "notes"
        if "reason" not in data:
            for alt in ("description", "rationale", "justification", "notes", "explanation", "note"):
                if alt in data:
                    data["reason"] = data[alt]
                    break
        return data


class EdaReport(BaseModel):
    overview: str
    target_analysis: dict[str, Any]
    quality_issues: list[QualityIssue]
    correlations: dict[str, Any]
    preprocessing_recommendations: list[PreprocessingRecommendation]
    model_recommendation: str
    summary: str

    @field_validator("model_recommendation", mode="before")
    @classmethod
    def _coerce_model_recommendation(cls, v: Any) -> str:
        # the model sometimes returns {"model_type": "GradientBoosting", "justification": "..."}
        if isinstance(v, dict):
            return str(v.get("model_type", v.get("model", v.get("type", v.get("name", "gradient_boosting")))))
        return str(v) if v is not None else "gradient_boosting"

    @classmethod
    def safe_fallback(cls, error_detail: str = "") -> "EdaReport":
        """Return a minimal valid report for pipeline continuity on parse failure."""
        # Truncate error to avoid dumping a multi-line Pydantic trace into the UI
        short = (error_detail[:120] + "…") if len(error_detail) > 120 else error_detail
        return cls(
            overview=f"EDA completed with parse error{': ' + short if short else ''}.",
            target_analysis={},
            quality_issues=[],
            correlations={},
            preprocessing_recommendations=[],
            model_recommendation="gradient_boosting",
            summary="EDA parse failed - review raw output in audit log.",
        )
