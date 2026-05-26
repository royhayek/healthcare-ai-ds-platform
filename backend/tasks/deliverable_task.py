"""Deliverable generation Celery task (§4, §23).

Orchestrates all eight generators, uploads to storage, inserts Deliverable
ORM rows, and bundles into a downloadable ZIP. Called as the final step of
the analysis pipeline AFTER verify_chain() passes.

Eight deliverables:
  1. executive_summary    - PDF      (the model + Jinja + weasyprint)
  2. technical_report     - PDF      (Jinja + weasyprint + SHAP chart)
  3. model_card           - MD + PDF (the model + Jinja + weasyprint) = 2 files
  4. data_quality_report  - PDF      (Jinja + weasyprint)
  5. predictions          - XLSX     (openpyxl, per-row SHAP, conditional fmt)
  6. audit_log            - CSV + JSON (append-only chain) = 2 files
  7. repro_manifest       - YAML
  8. risk_register        - MD       (the model)
  + bundle               - ZIP      (all of the above)
"""

from __future__ import annotations

import asyncio
import io
import logging
import zipfile
from datetime import datetime, timezone
from typing import Any

from backend.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="tasks.generate_deliverables",
    max_retries=1,
    soft_time_limit=600,
)
def generate_deliverables_task(self, run_id: str) -> dict[str, Any]:
    """Celery entry point - bridges sync Celery into async generation."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(_async_generate_deliverables(run_id))


async def _async_generate_deliverables(run_id: str) -> dict[str, Any]:
    from sqlalchemy import select

    from backend.core import audit
    from backend.core.database import Dataset, Deliverable, Run, async_session_factory
    from backend.core.events import ProgressEmitter
    from backend.core.storage import storage
    from backend.deliverables.base import GeneratedDeliverable, run_summary_context

    async with async_session_factory() as session:
        run = await session.get(Run, run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        emitter = ProgressEmitter(run_id)

        # ── Hard gate: verify audit chain before generating anything ──────────
        await emitter.emit_async("deliverables_start", "Verifying audit chain…", 96)
        chain_ok = await audit.verify_chain(session, run_id)
        if not chain_ok:
            run.status = "failed"
            run.error_message = "Audit chain verification failed - deliverables not generated"
            session.add(run)
            await session.commit()
            raise RuntimeError(f"Audit chain verification failed for run {run_id}")

        await audit.append(
            session=session,
            run_id=run_id,
            actor="system",
            category="deliverables",
            action="chain_verified",
            payload={"run_id": run_id},
            reason="Audit chain valid - proceeding to deliverable generation",
        )
        await session.flush()

        # ── Load training dataset and project ─────────────────────────────────
        from backend.core.database import Project

        dataset: Dataset | None = None
        if run.training_dataset_id:
            dataset = await session.get(Dataset, run.training_dataset_id)

        project: Project | None = await session.get(Project, run.project_id)
        case_brief = project.case_brief if project else None

        ctx = run_summary_context(run, dataset, case_brief)

        # ── Run generators ────────────────────────────────────────────────────
        await emitter.emit_async("deliverables_generating", "Generating deliverables…", 97)

        all_deliverables: list[GeneratedDeliverable] = []

        from backend.deliverables.audit_log_export import generate_audit_log_export
        from backend.deliverables.data_quality_report import generate_data_quality_report
        from backend.deliverables.executive_summary import generate_executive_summary
        from backend.deliverables.model_card import generate_model_card
        from backend.deliverables.predictions_artifact import generate_predictions_artifact
        from backend.deliverables.repro_manifest import generate_repro_manifest
        from backend.deliverables.risk_register import generate_risk_register
        from backend.deliverables.technical_report import generate_technical_report

        # Determine which optional generators to run based on the case brief.
        # audit_log and repro_manifest are always generated (governance/reproducibility).
        optional_generators = _select_generators(case_brief)

        coros = []
        generator_names = []

        # Always-on
        coros += [
            generate_audit_log_export(run, session, ctx),
            generate_repro_manifest(run, dataset, session, ctx),
        ]
        generator_names += ["audit_log", "repro_manifest"]

        # Brief-scoped optional generators
        generator_map = {
            "executive_summary": lambda: generate_executive_summary(run, dataset, session, ctx),
            "technical_report": lambda: generate_technical_report(run, dataset, session, ctx),
            "model_card": lambda: generate_model_card(run, dataset, session, ctx),
            "data_quality_report": lambda: generate_data_quality_report(run, dataset, session, ctx),
            "predictions": lambda: generate_predictions_artifact(run, dataset, session, ctx),
            "risk_register": lambda: generate_risk_register(run, session, ctx),
        }
        for gname in optional_generators:
            coros.append(generator_map[gname]())
            generator_names.append(gname)

        results = await asyncio.gather(*coros, return_exceptions=True)
        for name, result in zip(generator_names, results):
            if isinstance(result, Exception):
                logger.error("Generator %s failed: %s", name, result)
                # Non-fatal: record failure but keep going for the other deliverables
                await audit.append(
                    session=session,
                    run_id=run_id,
                    actor="system",
                    category="deliverables",
                    action="generator_failed",
                    payload={"generator": name, "error": str(result)},
                    reason=f"Generator {name} raised an exception",
                )
                await session.flush()
            elif isinstance(result, list):
                all_deliverables.extend(result)
            else:
                all_deliverables.append(result)

        # ── Upload each deliverable to storage ────────────────────────────────
        await emitter.emit_async("deliverables_upload", "Uploading deliverables…", 98)

        for d in all_deliverables:
            try:
                await storage.upload(d.storage_path, d.content)
            except Exception as exc:
                logger.error("Failed to upload %s: %s", d.storage_path, exc)

        # ── Insert Deliverable ORM rows ───────────────────────────────────────
        for d in all_deliverables:
            orm_row = Deliverable(
                run_id=run_id,
                name=d.name,
                format=d.fmt,
                storage_path=d.storage_path,
                checksum_sha256=d.checksum_sha256,
                generator_version=d.generator_version,
                inputs_used=d.inputs_used,
                audience=d.audience,
                generated_at=d.generated_at,
            )
            session.add(orm_row)

        await session.flush()

        # ── Build ZIP bundle ──────────────────────────────────────────────────
        await emitter.emit_async("deliverables_bundle", "Building ZIP bundle…", 99)

        zip_bytes = _build_zip(all_deliverables)
        zip_path = f"runs/{run_id}/deliverables/bundle.zip"
        try:
            await storage.upload(zip_path, zip_bytes)
        except Exception as exc:
            logger.error("Failed to upload ZIP bundle: %s", exc)

        import hashlib
        zip_sha = hashlib.sha256(zip_bytes).hexdigest()

        zip_row = Deliverable(
            run_id=run_id,
            name="bundle",
            format="zip",
            storage_path=zip_path,
            checksum_sha256=zip_sha,
            generator_version="1.0.0",
            inputs_used=[d.name for d in all_deliverables],
            audience="all",
            generated_at=datetime.now(timezone.utc),
        )
        session.add(zip_row)

        await audit.append(
            session=session,
            run_id=run_id,
            actor="system",
            category="deliverables",
            action="bundle_created",
            payload={
                "deliverable_count": len(all_deliverables),
                "bundle_path": zip_path,
                "bundle_sha256": zip_sha,
            },
            reason="All deliverables generated and bundled",
        )

        # ── Mark run completed ────────────────────────────────────────────────
        run.status = "completed"
        run.current_step = "completed"
        run.progress = 100
        run.completed_at = datetime.now(timezone.utc)
        session.add(run)
        await session.commit()

        await emitter.emit_async("done", "Run complete - deliverables ready", 100)

        return {
            "run_id": run_id,
            "deliverable_count": len(all_deliverables) + 1,  # +1 for bundle
            "bundle_path": zip_path,
        }


@celery_app.task(
    bind=True,
    name="tasks.generate_notebook_export",
    max_retries=1,
    soft_time_limit=120,
)
def generate_notebook_export_task(self, run_id: str) -> dict[str, Any]:
    """Generate a .ipynb notebook for the run and persist it as a Deliverable."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(_async_generate_notebook(run_id))


async def _async_generate_notebook(run_id: str) -> dict[str, Any]:
    from sqlalchemy import select

    from backend.core.database import Dataset, Deliverable, Run, async_session_factory
    from backend.core.storage import storage
    from backend.deliverables.base import run_summary_context
    from backend.deliverables.notebook_export import generate_notebook

    async with async_session_factory() as session:
        run = await session.get(Run, run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        dataset: Dataset | None = None
        if run.training_dataset_id:
            dataset = await session.get(Dataset, run.training_dataset_id)

        ctx = run_summary_context(run, dataset)
        deliverable = await generate_notebook(run, dataset, session, ctx)

        await storage.upload(deliverable.storage_path, deliverable.content)

        orm_row = Deliverable(
            run_id=run_id,
            name=deliverable.name,
            format=deliverable.fmt,
            storage_path=deliverable.storage_path,
            checksum_sha256=deliverable.checksum_sha256,
            generator_version=deliverable.generator_version,
            inputs_used=deliverable.inputs_used,
            audience=deliverable.audience,
            generated_at=deliverable.generated_at,
        )
        session.add(orm_row)
        await session.commit()

        return {
            "run_id": run_id,
            "storage_path": deliverable.storage_path,
            "checksum_sha256": deliverable.checksum_sha256,
        }


_ALL_OPTIONAL_GENERATORS = [
    "executive_summary",
    "technical_report",
    "model_card",
    "data_quality_report",
    "predictions",
    "risk_register",
]

# Keyword phrases in deliverable_requirements that map to generator names
_REQUIREMENT_KEYWORDS: list[tuple[list[str], str]] = [
    (["executive summary", "exec summary", "executive report"], "executive_summary"),
    (["technical report", "methodology", "technical doc"], "technical_report"),
    (["model card"], "model_card"),
    (["data quality", "data report"], "data_quality_report"),
    (["prediction", "predictions", "scoring", "scored", "excel", "xlsx"], "predictions"),
    (["risk register", "risk report"], "risk_register"),
]


def _select_generators(case_brief: dict[str, Any] | None) -> list[str]:
    """Return the list of optional generator names to run.

    If the case brief has explicit deliverable_requirements, map them to
    generator names. Fall back to all optional generators when no brief or
    no requirements are specified.
    """
    if not case_brief or not case_brief.get("deliverable_requirements"):
        return _ALL_OPTIONAL_GENERATORS

    reqs = [r.lower() for r in case_brief["deliverable_requirements"]]
    selected: list[str] = []
    for keywords, gen_name in _REQUIREMENT_KEYWORDS:
        if any(kw in req for req in reqs for kw in keywords):
            selected.append(gen_name)

    # If nothing mapped (requirements are vague), fall back to all
    return selected if selected else _ALL_OPTIONAL_GENERATORS


def _build_zip(deliverables: list[Any]) -> bytes:
    """Bundle all deliverable files into a single ZIP archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for d in deliverables:
            filename = d.storage_path.split("/deliverables/", 1)[-1]
            zf.writestr(filename, d.content)
    return buf.getvalue()
