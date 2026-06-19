# Healthcare AI Data-Science Platform

An explainable, domain-generic AI co-pilot for data scientists and ML engineers working
on healthcare problems. A user uploads one or more real datasets, and the platform runs
the rigorous work a senior data scientist would do: EDA, data-integrity checks, reasoned
preprocessing, model selection with stability checks, calibration, business cost-matrix
threshold optimization, SHAP explanations, similarity scoring, drift detection, and
fairness analysis.

Every decision the pipeline makes is shown, justified, and overridable through a chat
panel that stays open on the side of the screen and can change pipeline decisions
mid-flight. Each run produces eight professional deliverables, not just a model file.

The platform is domain-generic (it works for any tabular medical analysis) and is
demonstrated end-to-end on a real hantavirus genome dataset.

---

## Highlights

- **Always-open chat co-pilot** that acts as a steering wheel, not a help bot: typing
  "use class_weight instead of SMOTE" changes the strategy, not just the explanation.
- **Built-in data-integrity guardrails** that catch the traps which silently inflate
  accuracy: proxy/lookup-table leakage, unlabeled-target placeholders, and per-entity
  leakage via optional group-aware cross-validation.
- **Stability-first reporting**: every model score is mean +/- std across multiple seeds
  and folds, never a single cross-validation number.
- **Business-aware thresholds**: classification thresholds are tuned against a cost matrix
  (a missed dangerous case is far worse than a false alarm), not a naive 0.5 cutoff.
- **Out-of-distribution abstention**: inputs unlike the training set are flagged for manual
  review instead of receiving a confident, unreliable prediction.
- **Eight deliverables per run**: executive summary, technical report, model card, data
  quality report, predictions spreadsheet with per-row SHAP, hash-chained audit log,
  reproducibility manifest, and risk register.
- **Healthcare layer**: clinical terminology, PHI detection and redaction, clinical
  reference-range checks, equity dashboards, and clinical language in all reports.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14 (React, App Router), TypeScript, Tailwind CSS, Zustand |
| Backend API | FastAPI (Python 3.11+) |
| Background jobs | Celery task queue + Redis broker |
| Machine learning | scikit-learn, XGBoost / Gradient Boosting, SHAP, fairlearn |
| LLM reasoning agents | Anthropic API (Haiku for fast classification, Sonnet for reasoning and chat, Opus for report writing) |
| Database and auth | PostgreSQL (Supabase-compatible) |
| Reports | WeasyPrint (PDFs), openpyxl (Excel) |
| Local infra | Docker / docker-compose |

---

## Repository structure

```
.
├── backend/                 # FastAPI + Celery service
│   ├── main.py
│   ├── routers/             # projects, datasets, analysis, chat, deliverables, audit
│   ├── tasks/               # Celery tasks (analysis, fairness, drift, deliverables)
│   ├── agents/              # LLM-powered reasoning agents
│   ├── ml/                  # pure-Python ML modules (profiler, trainer, calibration,
│   │                        #   threshold_optimizer, leakage_detector, fairness, ...)
│   ├── deliverables/        # the eight document generators + Jinja templates
│   ├── models/              # Pydantic + ORM models
│   ├── core/                # config, storage, database, events, audit, auth
│   └── tests/               # unit + integration tests with real-business fixtures
├── frontend/                # Next.js 14 app (dashboard + persistent chat panel)
│   ├── app/                 # App Router routes
│   ├── components/          # chat, checkpoints, results, deliverables, analysis
│   ├── lib/                 # API client + helpers
│   └── store/               # Zustand stores
├── scripts/                 # standalone demo + utility scripts
│   ├── hantavirus_pipeline.py
│   └── simulate_run.py
├── datasets/                # input datasets (the hantavirus demo dataset lives here)
├── docker-compose.yml       # Postgres + Redis for local development
└── README.md
```

Generated at runtime and not tracked by git: `data/` (local file storage for uploaded
datasets, projects, and run artifacts) and `pipeline_plots/` (output of the demo script,
see below).

---

## Getting started

### Prerequisites

- Python 3.11+ — on macOS, use a non-Anaconda interpreter (e.g. the python.org
  build or Homebrew Python). Anaconda ships its own `glib`/`cairo`/`harfbuzz`
  that conflict with the system PDF libraries and segfault WeasyPrint.
- Node.js 18+
- Docker (for Postgres and Redis)
- An Anthropic API key (the reasoning agents call the Anthropic API)
- **macOS only — WeasyPrint native libraries** (used by the PDF deliverable
  generators). Install once with Homebrew:

  ```bash
  brew install pango cairo gdk-pixbuf libffi
  ```

  Because these live under `$(brew --prefix)/lib` (e.g. `/opt/homebrew/lib` on
  Apple Silicon), which Python's loader does not search by default, any process
  that renders PDFs — the API, the Celery worker, and the test suite — must be
  started with `DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix)/lib"` set. The
  commands below and `backend/run_tests.sh` already do this.

### 1. Start local infrastructure

```bash
docker-compose up -d        # Postgres on :5433, Redis on :6380
```

### 2. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # macOS: use a non-Anaconda python (see Prerequisites)
pip install -e ".[dev]"                             # installs the app + test dependencies

# Configure environment: copy the example and fill in values
cp .env.example .env
```

Required backend environment variables (see `backend/.env.example`):

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Async Postgres URL, e.g. `postgresql+asyncpg://aids:aids@localhost:5433/aids` |
| `REDIS_URL` | Redis broker URL, e.g. `redis://localhost:6380/0` |
| `STORAGE_ROOT` | Local file-storage root (default `./data`) |
| `ANTHROPIC_API_KEY` | API key for the reasoning agents |
| `CORS_ORIGINS` | Allowed frontend origins, e.g. `["http://localhost:3000"]` |
| `DEV_MODE` | `true` for local development (relaxed auth) |
| `SUPABASE_*` | Optional, only when using Supabase storage/auth instead of local |

Then run the worker and API in two terminals. The app uses absolute imports
(`backend.*`), so both commands use the `backend.` package prefix and can be run
from anywhere once `pip install -e ".[dev]"` has registered the package. The API
must listen on **port 8001** — that is the port the frontend proxy expects (see
`frontend/next.config.mjs`).

```bash
# Terminal 1 — Celery worker
celery -A backend.tasks.celery_app worker --loglevel=info

# Terminal 2 — FastAPI
uvicorn backend.main:app --reload --port 8001
```

> **macOS:** prefix both commands with `DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix)/lib"`
> so WeasyPrint (PDF deliverables) can load its native libraries — e.g.
> `DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix)/lib" uvicorn backend.main:app --reload --port 8001`.

### 3. Frontend

```bash
cd frontend
npm install                 # also installs the test toolchain (vitest, Playwright, Testing Library)

npm run dev                 # http://localhost:3000
```

Environment variables (all have working local defaults in `next.config.mjs`, so
no `.env.local` is required for local development):

| Variable | Purpose | Default |
|---|---|---|
| `FASTAPI_URL` | Backend URL the server-side proxy targets. Must match the API port. | `http://127.0.0.1:8001` |
| `NEXT_PUBLIC_DEV_MODE` | When `true`, `AuthGuard` skips the Supabase session check (mirrors the backend `DEV_MODE` auth stub used by the proxy's hard-coded dev user). Set to `false` in production to restore the real session gate. | `true` |
| `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Supabase project credentials — only needed when `NEXT_PUBLIC_DEV_MODE=false`. | unset |

---

## The demo: hantavirus pathogenicity classifier

The headline demonstration is a standalone, end-to-end run on a real dataset of 2,096
hantavirus isolates from a public scientific database.

```bash
python scripts/hantavirus_pipeline.py
```

- Input: `datasets/hantavirus_genome.csv`
- Output: plots written to `pipeline_plots/` (regenerated on each run)

### What is being predicted

From an isolate's tabular metadata, predict `pathogenicity_class` (high / moderate / low),
a multiclass classification problem. The business framing: missing a dangerous isolate
(calling it safe) can cost roughly 40x more than a false alarm, so the decision threshold
is tuned against that cost rather than a naive 50/50 cutoff.

### The real story: catching the accuracy trap

A naive run on this dataset reports about 100% accuracy, and that number is a trap. The
platform's job is to catch it automatically, and it does:

1. **Proxy (lookup-table) leakage.** Columns such as `clinical_syndrome` and `clade` are
   near-perfect copies of the answer. A model handed these memorizes a lookup table instead
   of learning. Both are flagged as high-severity leakage and removed.
2. **Unlabeled target values.** The `unknown` value (~39% of rows) is a missing-label
   placeholder, not a danger level. Those rows are excluded from supervised training.
3. **Optimistic evaluation.** Optional group-aware cross-validation prevents records of the
   same entity from leaking across train/test.

These checks are generic. They fire for any dataset (a `diagnosis_code -> disease` proxy, a
`discharge_unit -> mortality` proxy, an `unknown`/`pending` target value), not because
anything knows this is hantavirus.

### Honest results, after the guardrails

Using only legitimate metadata features (leakage removed, `unknown` excluded, 3 classes):

| Setup | Accuracy | Balanced acc | What it measures |
|---|---|---|---|
| Most-common-class baseline | 60.4% | - | The floor to beat |
| Naive run (leakage + `unknown` left in) | ~100% | - | Inflated: memorization, not skill |
| Platform run, known virus families | 85.6% +/- 3.0% | 88.6% | Honest, real signal in the metadata |
| Platform run, group-aware check | 85.4% +/- 1.2% | 87.7% | Confirms it is not per-entity leakage |
| Genuinely novel virus family (held out) | ~43% | ~48% | Below baseline: novel danger cannot be extrapolated |

**Conclusion:** on a new isolate of a known family the platform predicts pathogenicity at
about 85% accuracy from legitimate metadata alone, well above the 60% baseline and
leakage-free. On a genuinely novel family - the headline outbreak scenario - no model can
reliably extrapolate the danger level, so the correct behaviour is to abstain and flag for
manual review rather than to guess. That abstention is the safe, honest behaviour for a
biosafety tool, and it is exactly the rigor a medical AI tool must have.

> Note on the dataset: `hantavirus_genome.csv` is assembled from public scientific records
> and includes provenance/annotation metadata columns. It is provided for demonstration of
> the pipeline only.

---

## Testing

### Backend

```bash
cd backend
./run_tests.sh               # unit tests (309)
./run_tests.sh -m integration   # full pipeline on real-business fixtures (needs Postgres + Redis)
./run_tests.sh -k profiler   # any extra args are forwarded to pytest
```

`run_tests.sh` pins the project venv and exports `DYLD_FALLBACK_LIBRARY_PATH` so
WeasyPrint can load its native libraries on macOS. On Linux you can also just run
`pytest` directly after `pip install -e ".[dev]"`.

### Frontend

```bash
cd frontend
npm run lint
npm run typecheck
npm test                     # vitest — unit + component tests (jsdom)
npx playwright install chromium   # one-time: download the browser binary
npm run e2e                  # playwright — end-to-end happy path
```

`npm run e2e` needs the full stack running: start Postgres + Redis
(`docker-compose up -d`) and the backend API on port 8001, then run it with
`NEXT_PUBLIC_DEV_MODE=true` (the default) so the dashboard is reachable without a
Supabase session. Playwright starts the Next.js dev server itself. The unit
suite (`npm test`) has no such dependencies and runs in isolation.

Test fixtures live in `backend/tests/fixtures/` and use real-business-flavored datasets
(telco churn, credit default, lead scoring, claims triage, housing) chosen to exercise
imbalanced, cost-sensitive, calibration-sensitive, multiclass, and regression paths.

---

## Roadmap

The data-integrity guardrails are built and wired in; the platform works end-to-end today.
Remaining items are mostly production and deployment work:

- UI control to pick the group column for group-aware cross-validation (capability exists
  in the trainer).
- Surface out-of-distribution abstention more prominently as an explicit "manual review"
  badge in the results UI.
- Supabase row-level security policies for per-project data isolation.
- API rate limiting and production CORS configuration.
- Cloud deployment configuration (production Docker, hosting).
- Large-dataset support (chunked / out-of-core processing) for datasets over 200 MB.

---

## License

Add a license of your choice before publishing.
