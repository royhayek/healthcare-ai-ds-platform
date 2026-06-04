/** Typed API wrappers - all calls route through /api/proxy. Never fetch FASTAPI_URL directly. */

import type { AuditEvent, AuditVerifyResult, Dataset, DatasetPlot, DatasetPreview, DeliverableItem, JoinKeyCandidate, JoinRecord, JoinSuggestResponse, PredictRequest, PredictResponse, PredictionListItem, Project, Run } from "@/lib/types"

const BASE = "/api/proxy"

async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}/${path}`)
  if (!res.ok) throw new Error(`GET /${path} failed: ${res.status}`)
  return res.json() as Promise<T>
}

async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => String(res.status))
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

async function apiDelete(path: string): Promise<void> {
  const res = await fetch(`${BASE}/${path}`, { method: "DELETE" })
  if (!res.ok && res.status !== 204) {
    const detail = await res.text().catch(() => String(res.status))
    throw new Error(detail || `DELETE /${path} failed: ${res.status}`)
  }
}

// Projects
export const getProjects = () => apiGet<Project[]>("projects")
export const getProject = (id: string) => apiGet<Project>(`projects/${id}`)
export const createProject = (formData: FormData) =>
  fetch("/api/proxy/projects", { method: "POST", body: formData }).then((r) => r.json() as Promise<Project>)
export const deleteProject = (id: string) => apiDelete(`projects/${id}`)

// Datasets
export const getDatasets = (projectId: string) =>
  apiGet<Dataset[]>(`projects/${projectId}/datasets`)

export async function uploadDataset(
  projectId: string,
  file: File,
  role: string,
  targetColumn?: string,
): Promise<Dataset> {
  const form = new FormData()
  form.append("file", file)
  form.append("role", role)
  if (targetColumn) form.append("target_column", targetColumn)

  const res = await fetch(`${BASE}/projects/${projectId}/datasets`, {
    method: "POST",
    body: form,
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => String(res.status))
    throw new Error(detail)
  }
  return res.json() as Promise<Dataset>
}

export async function updateDatasetRole(
  projectId: string,
  datasetId: string,
  role: string,
): Promise<Dataset> {
  const res = await fetch(`${BASE}/projects/${projectId}/datasets/${datasetId}/role`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role }),
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => String(res.status))
    throw new Error(detail)
  }
  return res.json() as Promise<Dataset>
}

export async function resumeRun(runId: string): Promise<Run> {
  const res = await fetch(`${BASE}/runs/${runId}/resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => String(res.status))
    throw new Error(detail)
  }
  return res.json() as Promise<Run>
}

export async function updateDatasetTargetColumn(
  projectId: string,
  datasetId: string,
  targetColumn: string,
): Promise<Dataset> {
  const res = await fetch(`${BASE}/projects/${projectId}/datasets/${datasetId}/target-column`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target_column: targetColumn }),
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => String(res.status))
    throw new Error(detail)
  }
  return res.json() as Promise<Dataset>
}

// Dataset preview & plots
export const getDatasetPreview = (projectId: string, datasetId: string, rows = 20) =>
  apiGet<DatasetPreview>(`projects/${projectId}/datasets/${datasetId}/preview?rows=${rows}`)

export const getDatasetPlots = (projectId: string, datasetId: string) =>
  apiGet<DatasetPlot[]>(`projects/${projectId}/datasets/${datasetId}/plots`)

export const getDatasetPlot = (projectId: string, datasetId: string, plotId: string) =>
  apiGet<DatasetPlot & { image_b64: string }>(`projects/${projectId}/datasets/${datasetId}/plots/${plotId}`)

export const getComparisonPlots = (projectId: string, datasetId: string, referenceDatasetId: string) =>
  apiGet<DatasetPlot[]>(`projects/${projectId}/datasets/${datasetId}/plots/vs/${referenceDatasetId}`)

// Run-level plots (pipeline stages)
export const getRunPlots = (runId: string, stage?: string) =>
  apiGet<DatasetPlot[]>(`runs/${runId}/plots${stage ? `?stage=${stage}` : ""}`)

export const getRunPlot = (runId: string, plotId: string) =>
  apiGet<DatasetPlot & { image_b64: string }>(`runs/${runId}/plots/${plotId}`)

export const triggerRunPlots = (runId: string, stage: string) =>
  apiPost<{ status: string; stage: string }>(`runs/${runId}/plots`, { stage })

// Joins
export const suggestJoinKeys = (projectId: string, leftId: string, rightId: string) =>
  apiPost<JoinSuggestResponse>(`projects/${projectId}/joins/suggest`, {
    left_dataset_id: leftId,
    right_dataset_id: rightId,
  })

export const createJoin = (
  projectId: string,
  payload: {
    left_dataset_id: string
    right_dataset_id: string
    join_type: string
    join_keys: string[]
    result_filename?: string
  },
) => apiPost<Dataset>(`projects/${projectId}/joins`, payload)

export const listJoins = (projectId: string) =>
  apiGet<JoinRecord[]>(`projects/${projectId}/joins`)

// Runs
export const getRuns = (projectId: string) =>
  apiGet<Run[]>(`projects/${projectId}/runs`)

export const getRun = (runId: string) => apiGet<Run>(`runs/${runId}`)

export const createRun = (
  projectId: string,
  payload: { training_dataset_id: string; holdout_dataset_id?: string },
) => apiPost<Run>(`projects/${projectId}/runs`, payload)

// Deliverables
export const getDeliverables = (runId: string) =>
  apiGet<DeliverableItem[]>(`runs/${runId}/deliverables`)

export async function downloadDeliverable(runId: string, name: string): Promise<Blob> {
  const res = await fetch(`${BASE}/runs/${runId}/deliverables/${name}/download`)
  if (!res.ok) throw new Error(`Download failed: ${res.status}`)
  return res.blob()
}

export const regenerateDeliverables = (runId: string) =>
  apiPost<{ message: string; task_id: string }>(`runs/${runId}/deliverables/regenerate`, {})

export const requestNotebookExport = (runId: string) =>
  apiPost<{ message: string; task_id: string }>(`runs/${runId}/deliverables/notebook`, {})

// Predict
export const predictSingle = (runId: string, inputData: Record<string, unknown>) =>
  apiPost<PredictResponse>(`runs/${runId}/predict`, { input_data: inputData } satisfies PredictRequest)

export const predictBatch = (runId: string, inferenceDatasetId: string) =>
  apiPost<{ job_id: string; run_id: string; inference_dataset_id: string; n_rows: number; status: string }>(
    `runs/${runId}/predict/batch`,
    { inference_dataset_id: inferenceDatasetId },
  )

export const getPredictions = (runId: string, limit = 50, offset = 0) =>
  apiGet<PredictionListItem[]>(`runs/${runId}/predictions?limit=${limit}&offset=${offset}`)

// Audit
export const getAuditLog = (runId: string, limit = 100, offset = 0) =>
  apiGet<AuditEvent[]>(`runs/${runId}/audit?limit=${limit}&offset=${offset}`)

export const verifyAuditChain = (runId: string) =>
  apiGet<AuditVerifyResult>(`runs/${runId}/audit/verify`)

// SWR fetcher (generic)
export const fetcher = <T>(url: string): Promise<T> =>
  fetch(url).then((r) => {
    if (!r.ok) throw new Error(`${r.status}`)
    return r.json() as Promise<T>
  })
