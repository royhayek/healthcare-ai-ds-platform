# Why Postgres, not SQLite:
# The §24 schema uses gen_random_uuid(), JSONB, TIMESTAMPTZ, and an
# append-only trigger on audit_events. SQLite supports none of these natively.
# Local Postgres runs via docker-compose.yml. Supabase replaces it in production
# by swapping DATABASE_URL - no code changes needed.
#
# Why user_id is Text (not UUID FK to profiles):
# In dev mode, auth.py returns the literal string from the X-User-Id header
# (e.g. "dev-user-1"). Supabase Auth uses UUID sub-claims. Rather than
# maintaining two ORM schemas, we store user_id as Text with no FK constraint
# here and add the Supabase FK via Alembic migration when real auth lands (Step 2+).

from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import TIMESTAMP
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func, text

from backend.core.config import settings


class Base(DeclarativeBase):
    type_annotation_map = {
        dict[str, Any]: JSONB,
        list[Any]: JSONB,
    }


engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session


# ── ORM models (one per §24 table) ────────────────────────────────────────────


class Profile(Base):
    __tablename__ = "profiles"

    # id is Text in local dev (stores the auth stub's user identifier).
    # Becomes UUID FK to Supabase auth.users via migration when real auth lands.
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    email: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str | None] = mapped_column(Text)
    plan: Mapped[str] = mapped_column(String(20), server_default="free")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now()
    )


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(
        PGUUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")
    )
    user_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    case_brief: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    brief_files: Mapped[list[Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(
        PGUUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")
    )
    project_id: Mapped[str] = mapped_column(
        PGUUID(as_uuid=False), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="training"
    )  # training | inference | holdout | reference | comparison
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(Text, nullable=False)
    schema_hash: Mapped[str] = mapped_column(Text, nullable=False)
    row_count: Mapped[int | None] = mapped_column(Integer)
    col_count: Mapped[int | None] = mapped_column(Integer)
    target_column: Mapped[str | None] = mapped_column(Text)
    task_type: Mapped[str | None] = mapped_column(
        String(30)
    )  # binary_classification | multiclass | regression
    time_range_start: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    time_range_end: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    profile: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now()
    )


class DatasetJoin(Base):
    __tablename__ = "dataset_joins"

    id: Mapped[str] = mapped_column(
        PGUUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")
    )
    run_id: Mapped[str | None] = mapped_column(PGUUID(as_uuid=False))
    left_dataset_id: Mapped[str] = mapped_column(
        PGUUID(as_uuid=False), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    right_dataset_id: Mapped[str] = mapped_column(
        PGUUID(as_uuid=False), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    join_type: Mapped[str] = mapped_column(
        String(10), nullable=False
    )  # inner | left | right | outer
    join_keys: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    rows_before: Mapped[int | None] = mapped_column(Integer)
    rows_after: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now()
    )


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(
        PGUUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")
    )
    project_id: Mapped[str] = mapped_column(
        PGUUID(as_uuid=False), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    training_dataset_id: Mapped[str | None] = mapped_column(
        PGUUID(as_uuid=False), ForeignKey("datasets.id", ondelete="SET NULL")
    )
    holdout_dataset_id: Mapped[str | None] = mapped_column(
        PGUUID(as_uuid=False), ForeignKey("datasets.id", ondelete="SET NULL")
    )
    job_id: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), server_default="queued")
    current_step: Mapped[str | None] = mapped_column(Text)
    completed_steps: Mapped[list[Any]] = mapped_column(JSONB, server_default="[]")
    pending_steps: Mapped[list[Any]] = mapped_column(JSONB, server_default="[]")
    progress: Mapped[int] = mapped_column(Integer, server_default="0")

    # Pipeline outputs - stored as JSONB so the schema survives new analysis steps
    eda_report: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    # Target-level hygiene (drop unlabelled rows, binary collapse) applied before
    # profiling + training. Derived from the brief, overridable via chat (§7, §10).
    target_strategy: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    preprocessing_strategy: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    model_selection: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    model_comparison: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    stat_tests: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    best_model_name: Mapped[str | None] = mapped_column(Text)
    best_model_score: Mapped[float | None] = mapped_column(Float)
    tuning_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    tuning_result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    threshold_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    threshold_result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    final_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    calibration_report: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    shap_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    similarity_index_built: Mapped[bool] = mapped_column(Boolean, server_default="false")
    drift_report: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    fairness_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    fairness_report: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    insight_report: Mapped[str | None] = mapped_column(Text)
    final_audit_hash: Mapped[str | None] = mapped_column(Text)

    # Artifact storage paths
    model_storage_path: Mapped[str | None] = mapped_column(Text)
    preprocessor_storage_path: Mapped[str | None] = mapped_column(Text)
    calibrator_storage_path: Mapped[str | None] = mapped_column(Text)
    faiss_index_path: Mapped[str | None] = mapped_column(Text)

    # Reproducibility fields (§4.7)
    seeds: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    library_versions: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    claude_models_used: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    # Queued modify intents during expensive steps (§2 interrupt semantics)
    pending_intents: Mapped[list[Any]] = mapped_column(JSONB, server_default="[]")

    # Verbatim human overrides recorded in the chat co-pilot, keyed by decision
    # category (e.g. "preprocessing"). When the producing step re-runs, these are
    # injected into the agent prompt as authoritative instructions the AI MUST
    # honour, so a clinician can override ANY AI decision by chatting (§2, §21).
    user_directives: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")

    eval_plots: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    created_by: Mapped[str | None] = mapped_column(Text)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    # Append-only - no UPDATE or DELETE. Enforced at DB level by trigger (see
    # init_db) and at application level by audit.append() in core/audit.py (Step 2).
    id: Mapped[str] = mapped_column(
        PGUUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")
    )
    run_id: Mapped[str] = mapped_column(
        PGUUID(as_uuid=False), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    actor: Mapped[str] = mapped_column(String(10), nullable=False)  # ai | user | system
    category: Mapped[str] = mapped_column(Text, nullable=False)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    prev_hash: Mapped[str] = mapped_column(Text, nullable=False)
    self_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    __table_args__ = (
        UniqueConstraint("run_id", "seq", name="uq_audit_run_seq"),
        Index("idx_audit_events_run_seq", "run_id", "seq"),
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(
        PGUUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")
    )
    run_id: Mapped[str] = mapped_column(
        PGUUID(as_uuid=False), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(String(10), nullable=False)  # user | assistant | system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    strategy_diff: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    artifact_ref: Mapped[str | None] = mapped_column(Text)
    tokens_in: Mapped[int | None] = mapped_column(Integer)
    tokens_out: Mapped[int | None] = mapped_column(Integer)
    model: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now()
    )

    __table_args__ = (Index("idx_chat_messages_run_created", "run_id", "created_at"),)


class Deliverable(Base):
    __tablename__ = "deliverables"

    id: Mapped[str] = mapped_column(
        PGUUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")
    )
    run_id: Mapped[str] = mapped_column(
        PGUUID(as_uuid=False), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    format: Mapped[str] = mapped_column(
        String(10), nullable=False
    )  # pdf | xlsx | md | csv | json | yaml | ipynb | zip
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    generator_version: Mapped[str] = mapped_column(Text, nullable=False)
    inputs_used: Mapped[list[Any] | None] = mapped_column(JSONB)
    audience: Mapped[str | None] = mapped_column(Text)
    generated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now()
    )
    superseded_by: Mapped[str | None] = mapped_column(
        PGUUID(as_uuid=False), ForeignKey("deliverables.id", ondelete="SET NULL")
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(
        PGUUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")
    )
    run_id: Mapped[str] = mapped_column(
        PGUUID(as_uuid=False), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    inference_dataset_id: Mapped[str | None] = mapped_column(
        PGUUID(as_uuid=False), ForeignKey("datasets.id", ondelete="SET NULL")
    )
    input_data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    prediction: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    probability: Mapped[float | None] = mapped_column(Float)
    similarity_score: Mapped[float | None] = mapped_column(Float)
    confidence_band: Mapped[str | None] = mapped_column(String(10))  # high | medium | low
    shap_values: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    threshold_used: Mapped[float | None] = mapped_column(Float)
    risk_flag: Mapped[bool] = mapped_column(Boolean, server_default="false")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now()
    )


# ── Database initialisation ────────────────────────────────────────────────────


async def init_db() -> None:
    """Create all tables and install the append-only trigger on audit_events.

    Safe to call multiple times - create_all uses IF NOT EXISTS and the
    trigger creation is wrapped in a DO block that checks for existence first.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # append-only enforcement on audit_events (§24)
        #
        # UPDATE is never permitted. DELETE is rejected too, EXCEPT during an
        # authorized full-project purge (right-to-erasure), which sets the
        # transaction-local flag `app.allow_audit_purge = 'on'` immediately
        # before the cascading project delete. The flag is SET LOCAL, so it is
        # scoped to that single transaction and cannot leak to ordinary writes.
        # This keeps the audit trail tamper-proof for all normal operations
        # while allowing a deliberate, owner-initiated deletion of an entire
        # project together with its sealed trail. (Spec §24, project-delete)
        await conn.execute(text("""
            CREATE OR REPLACE FUNCTION reject_audit_modifications()
            RETURNS TRIGGER AS $$
            BEGIN
                IF TG_OP = 'DELETE'
                   AND current_setting('app.allow_audit_purge', true) = 'on' THEN
                    RETURN OLD;
                END IF;
                RAISE EXCEPTION
                    'audit_events is append-only - % not permitted', TG_OP;
            END;
            $$ LANGUAGE plpgsql
        """))

        await conn.execute(text("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger
                    WHERE tgname = 'no_update_audit'
                ) THEN
                    CREATE TRIGGER no_update_audit
                        BEFORE UPDATE ON audit_events
                        FOR EACH ROW EXECUTE FUNCTION reject_audit_modifications();
                END IF;
            END $$
        """))

        await conn.execute(text("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger
                    WHERE tgname = 'no_delete_audit'
                ) THEN
                    CREATE TRIGGER no_delete_audit
                        BEFORE DELETE ON audit_events
                        FOR EACH ROW EXECUTE FUNCTION reject_audit_modifications();
                END IF;
            END $$
        """))

        # Column migrations - idempotent ADD COLUMN IF NOT EXISTS blocks
        # Add each new column here when the ORM model gains a column that
        # won't be created by create_all (which only creates missing tables).
        await conn.execute(text("""
            ALTER TABLE runs
                ADD COLUMN IF NOT EXISTS pending_intents JSONB NOT NULL DEFAULT '[]'::jsonb
        """))

        await conn.execute(text("""
            ALTER TABLE runs
                ADD COLUMN IF NOT EXISTS user_directives JSONB NOT NULL DEFAULT '{}'::jsonb
        """))

        await conn.execute(text("""
            ALTER TABLE runs
                ADD COLUMN IF NOT EXISTS target_strategy JSONB
        """))

        await conn.execute(text("""
            ALTER TABLE projects
                ADD COLUMN IF NOT EXISTS case_brief JSONB
        """))

        await conn.execute(text("""
            ALTER TABLE projects
                ADD COLUMN IF NOT EXISTS brief_files JSONB
        """))
