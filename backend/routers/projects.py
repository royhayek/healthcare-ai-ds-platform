import asyncio
import hashlib
import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.auth import get_current_user
from backend.core.database import Dataset, Deliverable, Project, Run, get_db
from backend.core.storage import storage
from backend.models.project import ProjectResponse
from backend.utils.brief_extractor import ACCEPTED_EXTENSIONS, extract_text_from_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["projects"])

_MAX_BRIEF_FILE_SIZE = 20 * 1024 * 1024  # 20 MB per file


@router.post("", status_code=status.HTTP_201_CREATED, response_model=ProjectResponse)
async def create_project(
    background_tasks: BackgroundTasks,
    name: str = Form(..., min_length=1, max_length=255),
    description: str | None = Form(None),
    brief_text: str | None = Form(None),
    brief_files: list[UploadFile] = File(default=[]),
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ProjectResponse:
    # ── Validate file extensions ──────────────────────────────────────────────
    for f in brief_files:
        fname = (f.filename or "").lower()
        if not any(fname.endswith(ext) for ext in ACCEPTED_EXTENSIONS):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unsupported file type: {f.filename}. Accepted: PDF, DOCX, TXT, MD",
            )

    # ── Create project row first to get an ID ─────────────────────────────────
    project = Project(
        user_id=user_id,
        name=name.strip(),
        description=description.strip() if description else None,
        case_brief=None,
        brief_files=None,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)

    project_id = project.id

    # ── Read + validate file sizes ─────────────────────────────────────────────
    file_contents: list[tuple[str, bytes]] = []
    for upload in brief_files:
        content = await upload.read()
        if len(content) > _MAX_BRIEF_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"{upload.filename} exceeds the 20 MB limit",
            )
        file_contents.append((upload.filename or "upload", content))

    # ── Store files and extract text (blocking IO → thread pool) ──────────────
    stored_paths: list[str] = []
    extracted_texts: list[str] = []

    for filename, content in file_contents:
        sha = hashlib.sha256(content).hexdigest()[:16]
        storage_path = f"projects/{project_id}/brief_files/{sha}_{filename}"
        try:
            await storage.upload(storage_path, content)
            stored_paths.append(storage_path)
        except Exception as exc:
            logger.warning("Failed to store brief file %s: %s", filename, exc)

        text = await asyncio.to_thread(extract_text_from_bytes, filename, content)
        if text.strip():
            extracted_texts.append(f"[Source: {filename}]\n{text.strip()}")

    if brief_text and brief_text.strip():
        extracted_texts.insert(0, brief_text.strip())

    raw_text = "\n\n---\n\n".join(extracted_texts)

    # ── Build initial case_brief (unparsed) ───────────────────────────────────
    initial_brief: dict[str, Any] = {
        "raw_text": raw_text,
        "source_files": stored_paths,
        "objectives": [],
        "cost_matrix": None,
        "known_data_issues": [],
        "deliverable_requirements": [],
        "evaluation_criteria": [],
        "stakeholder_name": None,
        "stakeholder_role": None,
        "parsed": False,
    }

    project.case_brief = initial_brief
    project.brief_files = [
        {"filename": fname, "storage_path": path}
        for (fname, _), path in zip(file_contents, stored_paths)
    ]
    db.add(project)
    await db.commit()
    await db.refresh(project)

    # ── Trigger async parsing in background (only if we have text to parse) ───
    if raw_text.strip():
        background_tasks.add_task(_parse_brief_background, project_id, raw_text)

    return ProjectResponse.model_validate(project)


async def _parse_brief_background(project_id: str, raw_text: str) -> None:
    """Parse the raw brief text with the model and update the project record."""
    from backend.agents.brief_parser_agent import parse_case_brief
    from backend.core.database import async_session_factory

    try:
        parsed = await parse_case_brief(raw_text)
    except Exception as exc:
        logger.error("Brief parsing failed for project %s: %s", project_id, exc)
        return

    async with async_session_factory() as session:
        project = await session.get(Project, project_id)
        if project is None:
            return
        existing = dict(project.case_brief or {})
        existing.update(parsed)
        project.case_brief = existing
        session.add(project)
        await session.commit()
        logger.info("Case brief parsed and updated for project %s", project_id)


@router.get("", response_model=list[ProjectResponse])
async def list_projects(
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[ProjectResponse]:
    result = await db.execute(
        select(Project).where(Project.user_id == user_id).order_by(Project.created_at.desc())
    )
    return [ProjectResponse.model_validate(p) for p in result.scalars()]


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ProjectResponse:
    result = await db.execute(
        select(Project).where(Project.id == project_id, Project.user_id == user_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
        )
    return ProjectResponse.model_validate(project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a project and all associated data (datasets, runs, deliverables, audit log).

    Data retention / right-to-erasure endpoint. Storage objects are deleted
    best-effort - failures are logged but do not prevent the DB deletion so
    the operation is always idempotent from the client's perspective.

    DB cascade (ondelete=CASCADE) handles: Dataset, Run, AuditEvent,
    ChatMessage, Deliverable, Prediction rows automatically.

    The audit_events table carries an append-only trigger that rejects every
    DELETE - so the cascade into it would abort the whole transaction. To purge
    an entire project legitimately we set the transaction-local flag
    `app.allow_audit_purge` immediately before the delete; the trigger honors
    that flag for DELETE only, and `SET LOCAL` scopes it to this transaction so
    no ordinary write can ever tamper with the trail. (Spec §24)
    """
    result = await db.execute(
        select(Project).where(Project.id == project_id, Project.user_id == user_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
        )

    # ── Collect all storage paths before deleting DB rows ─────────────────────
    storage_paths: list[str] = []

    # Brief files attached to the project
    for brief_file in project.brief_files or []:
        if isinstance(brief_file, dict) and brief_file.get("storage_path"):
            storage_paths.append(brief_file["storage_path"])

    # Dataset files
    datasets_result = await db.execute(
        select(Dataset).where(Dataset.project_id == project_id)
    )
    for dataset in datasets_result.scalars():
        if dataset.storage_path:
            storage_paths.append(dataset.storage_path)

    # Deliverable files
    runs_result = await db.execute(
        select(Run).where(Run.project_id == project_id)
    )
    run_ids = [r.id for r in runs_result.scalars()]

    if run_ids:
        deliverables_result = await db.execute(
            select(Deliverable).where(Deliverable.run_id.in_(run_ids))
        )
        for deliverable in deliverables_result.scalars():
            if deliverable.storage_path:
                storage_paths.append(deliverable.storage_path)

    # ── Delete storage objects (best-effort) ──────────────────────────────────
    for path in storage_paths:
        try:
            await storage.delete(path)
        except Exception as exc:
            logger.warning("Storage delete failed for %s: %s", path, exc)

    # ── Delete the project row (DB cascade handles all child rows) ─────────────
    # Authorize the cascade to pass the append-only trigger on audit_events.
    # SET LOCAL keeps this scoped to the current transaction only.
    await db.execute(text("SET LOCAL \"app.allow_audit_purge\" = 'on'"))
    await db.delete(project)
    await db.commit()
    logger.info(
        "Project %s deleted by user %s - %d storage objects purged",
        project_id, user_id, len(storage_paths),
    )
