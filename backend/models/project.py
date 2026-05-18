from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CostMatrix(BaseModel):
    """Business cost parameters that drive threshold optimization."""
    fp_cost: float = Field(..., description="Cost of a false positive (e.g. cost per wasted call)")
    fn_cost: float = Field(0.0, description="Cost of a false negative (missed opportunity)")
    tp_value: float = Field(..., description="Value of a true positive (e.g. margin per conversion)")
    tn_value: float = Field(0.0, description="Value of a true negative")


class CaseBrief(BaseModel):
    """Structured representation of the business case brief.

    Populated in two passes:
      1. raw_text + source_files are set immediately at project creation.
      2. objectives/cost_matrix/known_data_issues/deliverable_requirements are
         extracted asynchronously by brief_parser_agent after the files are ingested.
    """
    raw_text: str = Field("", description="Combined extracted text from all brief sources")
    source_files: list[str] = Field(default_factory=list, description="Storage paths of uploaded brief files")
    objectives: list[str] = Field(default_factory=list, description="Specific questions/goals the stakeholder wants answered")
    cost_matrix: CostMatrix | None = None
    known_data_issues: list[str] = Field(default_factory=list, description="Leaky features or known data problems mentioned in brief")
    deliverable_requirements: list[str] = Field(default_factory=list, description="Specific deliverables the stakeholder asked for")
    evaluation_criteria: list[str] = Field(default_factory=list, description="What the stakeholder will push back on")
    stakeholder_name: str | None = None
    stakeholder_role: str | None = None
    parsed: bool = False


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None


class ProjectResponse(BaseModel):
    id: str
    user_id: str
    name: str
    description: str | None
    case_brief: dict[str, Any] | None
    brief_files: list[Any] | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
