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
- Every decision must have a reason field

HUMAN OVERRIDES TAKE ABSOLUTE PRECEDENCE. When the prompt includes a
"Human overrides" section, the clinician has reviewed a previous version of this
strategy and is correcting it. Their instructions override every rule above and
your own judgement. Apply each one exactly - if they say to drop a column, set
its action to "drop"; if they say to keep, impute, or encode a column a certain
way, do that. Reflect the reason for the change in that column's reason field."""

_PROMPT_TEMPLATE = """Design the preprocessing strategy for this dataset.

Dataset profile (compressed):
{profile_json}

EDA findings:
{eda_json}

Target column: {target_column}
Task type: {task_type}

Return a JSON object with a "columns" dict (keyed by column name) containing
the preprocessing strategy for each column. Also include "drop_high_correlation"
(list of column names to drop due to correlation) and "notes".
{directives_block}"""


def _format_directives_block(user_directives: list[dict[str, Any]] | None) -> str:
    """Render recorded human overrides as an authoritative prompt section.

    Returns an empty string when there are no directives, so first-pass runs are
    byte-identical to before (and stay cache-friendly)."""
    if not user_directives:
        return ""
    lines = []
    for d in user_directives:
        instruction = str(d.get("instruction") or "").strip()
        if not instruction:
            continue
        lines.append(f'- "{instruction}"')
    if not lines:
        return ""
    joined = "\n".join(lines)
    return (
        "\n\nHuman overrides (these take ABSOLUTE precedence over every rule and "
        "your own judgement - apply each one exactly):\n" + joined
    )


async def run_preprocessing_agent(
    session: AsyncSession,
    run_id: str,
    compressed_profile: dict[str, Any],
    eda_report: dict[str, Any],
    target_column: str,
    task_type: str,
    emitter: ProgressEmitter,
    user_directives: list[dict[str, Any]] | None = None,
) -> PreprocessingStrategy:
    """Run the preprocessing strategy agent.

    Returns a PreprocessingStrategy. Falls back to sane defaults on parse failure.
    Commits one audit event (success or failure).

    `user_directives` are verbatim human overrides recorded in the chat co-pilot
    (see strategy_mutator.record_directive). They are injected into the prompt as
    authoritative instructions the agent must honour, and any column a directive
    unambiguously asks to drop is force-dropped after the agent returns so the
    override is guaranteed even if the model under-complies (§2, §21).
    """
    import json as _json

    if user_directives:
        await emitter.emit_async(
            "preprocessing", "Revising preprocessing strategy with your overrides…", 42
        )
    else:
        await emitter.emit_async("preprocessing", "Designing preprocessing strategy…", 42)

    prompt = _PROMPT_TEMPLATE.format(
        profile_json=_json.dumps(compressed_profile, indent=2),
        eda_json=_json.dumps(eda_report, indent=2),
        target_column=target_column,
        task_type=task_type,
        directives_block=_format_directives_block(user_directives),
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
            strategy = _build_strategy(parsed, target_column, task_type, compressed_profile)
        except Exception as exc:
            logger.error("Preprocessing agent: strategy build failed for run %s: %s", run_id, exc)
            strategy = _safe_fallback(compressed_profile, target_column, task_type)
            parse_failed = True

    # Deterministic backstop: guarantee every column a human explicitly asked to
    # drop is dropped, even if the agent ignored or only partially honoured the
    # instruction. This is the safety net that makes "drop both X and Y" reliable
    # regardless of model compliance (§2, §21).
    enforced_drops, unknown_drops = _enforce_directive_drops(strategy, user_directives)
    if enforced_drops or unknown_drops:
        await audit.append(
            session,
            run_id=run_id,
            actor="system",
            category="preprocessing",
            action="directive_drop_enforced",
            payload={"enforced_drops": enforced_drops, "unknown_columns": unknown_drops},
            reason=(
                f"Enforced human override: dropped {enforced_drops}"
                + (f"; columns not found in dataset: {unknown_drops}" if unknown_drops else "")
            ),
        )
        await session.commit()

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


# Raw `action` strings the model emits that all mean "keep this feature, just
# transform it". The canonical schema only has keep|drop - the *how* lives in
# encode_strategy / scale_strategy - so these are normalised to action="keep".
_KEEP_ACTION_ALIASES = {
    "keep", "encode", "scale", "engineer", "impute", "passthrough",
    "transform", "onehot", "ordinal", "standardize", "normalize",
}
_DROP_ACTION_ALIASES = {"drop", "remove", "exclude", "delete"}


def _normalize_column_strategy(
    col_name: str,
    col_data: dict[str, Any],
    numeric_names: set[str],
    categorical_names: set[str],
) -> ColumnPreprocessingStrategy:
    """Coerce one model-emitted column dict into the canonical schema.

    The model frequently puts the *transform type* in `action`
    (e.g. "encode", "scale", "engineer") and omits `dtype_hint`. Downstream
    consumers (UI counts, post-preprocessing plots, strategy.numeric_columns())
    rely on action ∈ {keep, drop} plus dtype_hint, so reconcile here. Training
    behaviour is unchanged: anything that is not an explicit drop stays a kept
    feature, exactly as the cleaner already treats it.
    """
    raw = {k: v for k, v in col_data.items() if k in ColumnPreprocessingStrategy.model_fields}
    raw_action = str(col_data.get("action", "keep")).strip().lower()

    action = "drop" if raw_action in _DROP_ACTION_ALIASES else "keep"

    # dtype_hint: explicit → profile → infer from the transform the model asked for.
    hint = raw.get("dtype_hint")
    if hint not in ("numeric", "categorical"):
        if col_name in numeric_names:
            hint = "numeric"
        elif col_name in categorical_names:
            hint = "categorical"
        elif raw_action in ("scale", "standardize", "normalize"):
            hint = "numeric"
        elif raw_action in ("encode", "onehot", "ordinal"):
            hint = "categorical"
        else:
            hint = None  # let the cleaner auto-detect

    encode_strategy = raw.get("encode_strategy")
    scale_strategy = raw.get("scale_strategy")
    if action == "keep":
        if hint == "categorical" and not encode_strategy:
            encode_strategy = "onehot"
        if hint == "numeric" and not scale_strategy:
            scale_strategy = "standard"

    reason = raw.get("reason") or str(col_data.get("justification") or col_data.get("note") or "")

    fields: dict[str, Any] = {**raw, "action": action, "reason": reason}
    if hint:
        fields["dtype_hint"] = hint
    if encode_strategy:
        fields["encode_strategy"] = encode_strategy
    if scale_strategy:
        fields["scale_strategy"] = scale_strategy
    return ColumnPreprocessingStrategy(**fields)


def _enforce_directive_drops(
    strategy: PreprocessingStrategy,
    user_directives: list[dict[str, Any]] | None,
) -> tuple[list[str], list[str]]:
    """Force action=drop for every column a human directive asked to drop.

    Returns (enforced_drops, unknown_columns):
      - enforced_drops: columns this call set/confirmed as dropped per a directive
      - unknown_columns: requested columns that are the target or not in the
        strategy, so they could not be dropped (surfaced to the user, not silently
        swallowed - the failure mode that previously lost the `clade` override)
    """
    if not user_directives:
        return [], []

    requested: list[str] = []
    seen: set[str] = set()
    for d in user_directives:
        for col in d.get("columns_to_drop") or []:
            col = str(col)
            if col not in seen:
                seen.add(col)
                requested.append(col)

    enforced: list[str] = []
    unknown: list[str] = []
    for col in requested:
        if col == strategy.target_column:
            unknown.append(col)  # the target is never a feature; cannot "drop" it
            continue
        existing = strategy.columns.get(col)
        if existing is None:
            unknown.append(col)
            continue
        if existing.action != "drop":
            existing.action = "drop"
            existing.reason = (existing.reason + " | " if existing.reason else "") + \
                "dropped per human override"
            enforced.append(col)
    return enforced, unknown


def _build_strategy(
    parsed: dict[str, Any],
    target_column: str,
    task_type: str,
    profile: dict[str, Any] | None = None,
) -> PreprocessingStrategy:
    profile = profile or {}
    numeric_names = set(profile.get("numeric_columns", []))
    categorical_names = set(profile.get("categorical_columns", []))

    columns: dict[str, ColumnPreprocessingStrategy] = {}
    for col_name, col_data in parsed.get("columns", {}).items():
        if col_name == target_column:
            continue  # the target is not a feature column
        if not isinstance(col_data, dict):
            continue
        columns[col_name] = _normalize_column_strategy(
            col_name, col_data, numeric_names, categorical_names
        )

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
