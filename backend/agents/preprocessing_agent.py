"""Preprocessing strategy agent - per-column decisions (§10).

Model: Sonnet - needs reasoning quality to avoid bad preprocessing
decisions that would silently corrupt the pipeline.

Expected JSON schema from the model:
{
  "columns": {
    "<col_name>": {
      "action": "keep | drop",
      "impute_strategy": "mean | median | most_frequent | constant | none",
      "encode_strategy": "onehot | ordinal | binary | none",
      "scale_strategy": "standard | minmax | robust | none",
      "dtype_hint": "numeric | categorical",
      "reason": "brief justification"
    },
    ...
  },
  "drop_high_correlation": ["col_a", "col_b"],
  "notes": "optional 1-2 sentence summary of key decisions"
}
"""

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.base import call_claude, extract_json
from backend.core import audit
from backend.core.config import settings
from backend.core.events import ProgressEmitter
from backend.models.strategy import ColumnPreprocessingStrategy, PreprocessingStrategy

logger = logging.getLogger(__name__)

_SYSTEM = """You are a senior ML engineer designing a preprocessing pipeline.
You receive a compressed dataset profile and EDA report.
Return ONLY a valid JSON object matching the schema documented in the caller.
Do not include any text outside the JSON object. Do not use markdown fences.

Rules:
- Set action=drop for: ID columns, near-duplicate derived columns, columns with >50% nulls
- For numeric columns: choose impute_strategy (prefer median for skewed), scale_strategy
- For categorical columns: choose encode_strategy (onehot for <20 categories, ordinal for >=20)
- For high-cardinality categoricals (>100 unique values): consider dropping or target encoding
- Columns flagged as leakage risk by EDA should be dropped
- Every decision must have a reason field"""

_PROMPT_TEMPLATE = """Design the preprocessing strategy for this dataset.

Dataset profile (compressed):
{profile_json}

EDA findings:
{eda_json}

Target column: {target_column}
Task type: {task_type}

Return a JSON object with a "columns" dict (keyed by column name) containing
the preprocessing strategy for each column. Also include "drop_high_correlation"
(list of column names to drop due to correlation) and "notes"."""


async def run_preprocessing_agent(
    session: AsyncSession,
    run_id: str,
    compressed_profile: dict[str, Any],
    eda_report: dict[str, Any],
    target_column: str,
    task_type: str,
    emitter: ProgressEmitter,
) -> PreprocessingStrategy:
    """Run the preprocessing strategy agent.

    Returns a PreprocessingStrategy. Falls back to sane defaults on parse failure.
    Commits one audit event (success or failure).
    """
    import json as _json

    await emitter.emit_async("preprocessing", "Designing preprocessing strategy…", 42)

    prompt = _PROMPT_TEMPLATE.format(
        profile_json=_json.dumps(compressed_profile, indent=2),
        eda_json=_json.dumps(eda_report, indent=2),
        target_column=target_column,
        task_type=task_type,
    )

    # call_claude can raise (retries exhausted, transport error). This call sits
    # before the per-column strategy build, so an unguarded exception here would
    # fail the entire run instead of degrading to the safe fallback. Catch it.
    try:
        raw_text = await call_claude(
            messages=[{"role": "user", "content": prompt}],
            model=settings.CLAUDE_SONNET_MODEL,
            system=_SYSTEM,
            max_tokens=16384,
        )
    except Exception as exc:
        logger.error("Preprocessing agent: model call failed for run %s: %s", run_id, exc)
        raw_text = ""

    # Persist raw output for debugging parse failures (mirrors the EDA agent).
    try:
        import os
        debug_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"preprocessing_raw_{run_id}.txt")
        with open(debug_path, "w", encoding="utf-8") as _f:
            _f.write(raw_text)
    except Exception as _exc:
        logger.warning("Could not write preprocessing debug file: %s", _exc)

    parsed = extract_json(raw_text)
    parse_failed = not parsed

    if parse_failed:
        logger.error("Preprocessing agent: JSON parse failure for run %s (raw len=%d)", run_id, len(raw_text))
        strategy = _safe_fallback(compressed_profile, target_column, task_type)
    else:
        try:
            strategy = _build_strategy(parsed, target_column, task_type)
        except Exception as exc:
            logger.error("Preprocessing agent: strategy build failed for run %s: %s", run_id, exc)
            strategy = _safe_fallback(compressed_profile, target_column, task_type)
            parse_failed = True

    action = "preprocessing_parse_failure" if parse_failed else "preprocessing_strategy_complete"
    await audit.append(
        session,
        run_id=run_id,
        actor="ai",
        category="preprocessing",
        action=action,
        payload={
            "model": settings.CLAUDE_SONNET_MODEL,
            "n_columns": len(strategy.columns),
            "dropped_columns": [c for c, s in strategy.columns.items() if s.action == "drop"],
            "parse_failed": parse_failed,
        },
        reason=strategy.notes or "preprocessing strategy generated",
    )
    await session.commit()

    await emitter.emit_async("preprocessing", "Preprocessing strategy ready", 45)
    return strategy


def _build_strategy(
    parsed: dict[str, Any],
    target_column: str,
    task_type: str,
) -> PreprocessingStrategy:
    columns: dict[str, ColumnPreprocessingStrategy] = {}
    for col_name, col_data in parsed.get("columns", {}).items():
        columns[col_name] = ColumnPreprocessingStrategy(**{
            k: v for k, v in col_data.items()
            if k in ColumnPreprocessingStrategy.model_fields
        })

    return PreprocessingStrategy(
        columns=columns,
        target_column=target_column,
        task_type=task_type,
        drop_high_correlation=parsed.get("drop_high_correlation", []),
        notes=parsed.get("notes"),
    )


def _safe_fallback(
    profile: dict[str, Any],
    target_column: str,
    task_type: str,
) -> PreprocessingStrategy:
    """Sane defaults when model output fails to parse."""
    columns: dict[str, ColumnPreprocessingStrategy] = {}

    for col_info in profile.get("columns", []):
        col_name = col_info.get("name", "")
        if not col_name or col_name == target_column:
            continue

        dtype = col_info.get("dtype", "")
        null_pct = col_info.get("null_pct", 0.0)

        if null_pct > 0.5:
            columns[col_name] = ColumnPreprocessingStrategy(
                action="drop", reason=">50% nulls - auto-dropped"
            )
            continue

        if "int" in dtype or "float" in dtype:
            columns[col_name] = ColumnPreprocessingStrategy(
                action="keep",
                dtype_hint="numeric",
                impute_strategy="median",
                scale_strategy="standard",
                reason="fallback: numeric default",
            )
        else:
            columns[col_name] = ColumnPreprocessingStrategy(
                action="keep",
                dtype_hint="categorical",
                impute_strategy="most_frequent",
                encode_strategy="onehot",
                reason="fallback: categorical default",
            )

    return PreprocessingStrategy(
        columns=columns,
        target_column=target_column,
        task_type=task_type,
        notes="Safe fallback - model JSON parse failed",
    )
