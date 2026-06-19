"""JSON-safety helpers for values headed into Postgres JSONB / float columns.

Postgres JSONB rejects the JSON-invalid tokens ``NaN``, ``Infinity`` and
``-Infinity`` that Python's ``json.dumps`` emits by default, and asyncpg cannot
bind numpy scalar types. Both situations arise constantly with pandas-derived
prediction payloads (missing CSV cells become ``NaN``), so every dict or float
crossing into the DB must be passed through these helpers first.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def json_safe(value: Any) -> Any:
    """Recursively coerce a value into something Postgres JSONB accepts.

    Converts numpy scalars to native Python types and maps non-finite floats
    (NaN/Inf) and pandas missing sentinels to ``None``.
    """
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def safe_float(value: Any) -> float | None:
    """Coerce a value to a finite float, or ``None`` for NaN/Inf/missing."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None
