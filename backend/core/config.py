from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the .env file relative to this file so the path is correct regardless
# of the working directory the process is launched from.
_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ───────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://aids:aids@localhost:5432/aids"

    # ── Redis ──────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── File storage ──────────────────────────────────────────────────────────
    STORAGE_ROOT: str = "./data"

    # ── CORS ───────────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # ── Supabase ──────────────────────────────────────────────────────────────
    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    SUPABASE_STORAGE_BUCKET: str = "datasets"

    # ── Auth ───────────────────────────────────────────────────────────────────
    # Production verifies Supabase access tokens (ES256) against the project JWKS
    # at {SUPABASE_URL}/auth/v1/.well-known/jwks.json — see backend/core/auth.py.
    # No shared HS256 secret is used.
    DEV_MODE: bool = True

    # ── Anthropic API ──────────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = ""

    # Model name strings - per project rule 6: never use aliases such as
    # `claude-sonnet-latest`. The reproducibility manifest (§4.7) records the
    # exact string that produced each agent's output. Changing these requires a
    # corresponding note in DECISIONS.md.
    CLAUDE_SONNET_MODEL: str = "claude-sonnet-4-6"
    CLAUDE_OPUS_MODEL: str = "claude-opus-4-7"
    CLAUDE_HAIKU_MODEL: str = "claude-haiku-4-5-20251001"


settings = Settings()
