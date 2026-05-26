from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.database import get_db, init_db
from backend.routers import analysis, audit, chat, checkpoints, datasets, deliverables, joins, plots, predict, projects, results


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await init_db()
    yield


app = FastAPI(
    title="AI Data Science Co-Pilot",
    description="Senior-grade AI co-pilot for working data scientists and ML engineers.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router)
app.include_router(datasets.router)
app.include_router(analysis.router)
app.include_router(chat.router)
app.include_router(deliverables.router)
app.include_router(predict.router)
app.include_router(joins.router)
app.include_router(plots.router)
app.include_router(audit.router)
app.include_router(checkpoints.router)
app.include_router(results.router)


@app.get("/health", tags=["meta"])
async def health(db: AsyncSession = Depends(get_db)) -> dict[str, str]:
    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "unreachable"
    return {"status": "ok", "database": db_status}
