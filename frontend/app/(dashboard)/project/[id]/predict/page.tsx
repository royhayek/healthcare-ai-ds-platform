"use client"

import { useEffect, useState } from "react"
import { useParams, useSearchParams } from "next/navigation"
import useSWR from "swr"
import Link from "next/link"
import { ArrowLeft } from "lucide-react"
import { fetcher, getDatasets, getPredictionCount, getPredictions, predictBatch, predictSingle } from "@/lib/api"
import type { Dataset, PredictResponse, PredictionListItem, Run } from "@/lib/types"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

export default function PredictPage() {
  const { id: projectId } = useParams<{ id: string }>()
  const searchParams = useSearchParams()
  const runIdParam = searchParams.get("run_id")

  const { data: runs } = useSWR<Run[]>(
    `/api/proxy/projects/${projectId}/runs`,
    fetcher,
  )

  const completedRuns = runs?.filter((r) => r.status === "completed") ?? []
  const [selectedRunId, setSelectedRunId] = useState<string>(runIdParam ?? "")
  const activeRunId = selectedRunId || completedRuns[completedRuns.length - 1]?.id

  const { data: run } = useSWR<Run>(
    activeRunId ? `/api/proxy/runs/${activeRunId}` : null,
    fetcher,
  )

  const columns = run?.preprocessing_strategy?.columns ?? {}
  const targetColumn = run?.preprocessing_strategy?.target_column
  // Feature columns = everything the model trains on: any column not dropped and
  // not the target. This includes "encode" columns (categoricals), which the
  // model needs as inputs — mirrors backend PreprocessingStrategy.feature_columns().
  const keptColumns = Object.entries(columns).filter(
    ([col, s]) => s.action !== "drop" && s.action !== "target" && col !== targetColumn,
  )

  const [inputValues, setInputValues] = useState<Record<string, string>>({})
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const { data: history, mutate: refreshHistory } = useSWR<PredictionListItem[]>(
    activeRunId ? `predictions-${activeRunId}` : null,
    () => (activeRunId ? getPredictions(activeRunId, 20) : Promise.resolve([])),
  )

  // Batch prediction state
  const { data: allDatasets } = useSWR<Dataset[]>(
    projectId ? `/api/proxy/projects/${projectId}/datasets` : null,
    fetcher,
  )
  const inferenceDatasets = allDatasets?.filter((d) => d.role === "inference") ?? []

  const [batchDatasetId, setBatchDatasetId] = useState<string>("")
  const [batchJob, setBatchJob] = useState<{ job_id: string; n_rows: number; inference_dataset_id: string } | null>(null)
  const [batchError, setBatchError] = useState<string | null>(null)
  const [batchLoading, setBatchLoading] = useState(false)
  const [batchDone, setBatchDone] = useState(false)

  // Poll predictions count when a batch job is in-flight
  const { data: batchPoll } = useSWR<{ run_id: string; count: number }>(
    batchJob && !batchDone && activeRunId ? `predictions-batch-${activeRunId}` : null,
    () => getPredictionCount(activeRunId!),
    { refreshInterval: 3000 },
  )

  useEffect(() => {
    if (!batchJob || batchDone) return
    if (batchPoll && batchPoll.count >= batchJob.n_rows) {
      setBatchDone(true)
      void refreshHistory()
    }
  }, [batchPoll, batchJob, batchDone, refreshHistory])

  async function handleBatchPredict() {
    if (!activeRunId || !batchDatasetId) return
    setBatchError(null)
    setBatchDone(false)
    setBatchJob(null)
    setBatchLoading(true)
    try {
      const res = await predictBatch(activeRunId, batchDatasetId)
      setBatchJob({ job_id: res.job_id, n_rows: res.n_rows, inference_dataset_id: res.inference_dataset_id })
    } catch (e) {
      setBatchError(e instanceof Error ? e.message : "Batch prediction failed")
    } finally {
      setBatchLoading(false)
    }
  }

  async function handlePredict() {
    if (!activeRunId) return
    setError(null)
    setLoading(true)
    try {
      const input: Record<string, unknown> = {}
      for (const [col, val] of Object.entries(inputValues)) {
        const num = Number(val)
        input[col] = !isNaN(num) && val.trim() !== "" ? num : val
      }
      const res = await predictSingle(activeRunId, input)
      setResult(res)
      await refreshHistory()
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-8 max-w-4xl mx-auto space-y-6">
      <div>
        <Link
          href={`/project/${projectId}/results${activeRunId ? `?run_id=${activeRunId}` : ""}`}
          className="inline-flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 mb-3 transition-colors"
        >
          <ArrowLeft className="w-3 h-3" />
          Back to results
        </Link>
        <h1 className="text-xl font-semibold text-zinc-100">Interactive prediction</h1>
      </div>

      {/* Run selector */}
      {completedRuns.length > 1 && (
        <div className="flex items-center gap-2 text-sm">
          <span className="text-zinc-500">Run:</span>
          <select
            className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-zinc-200 text-xs font-mono"
            value={activeRunId ?? ""}
            onChange={(e) => setSelectedRunId(e.target.value)}
          >
            {completedRuns.map((r) => (
              <option key={r.id} value={r.id}>
                {r.id.slice(0, 8)}… ({new Date(r.created_at).toLocaleDateString()})
              </option>
            ))}
          </select>
        </div>
      )}

      {!activeRunId && (
        <p className="text-zinc-500">No completed run found. Start an analysis first.</p>
      )}

      {activeRunId && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Input form */}
          <Card>
            <CardHeader>
              <CardTitle>Input features</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {keptColumns.length === 0 && (
                <p className="text-sm text-zinc-500">
                  Preprocessing strategy not available yet. Enter values when the run completes.
                </p>
              )}

              {keptColumns.map(([col, strategy]) => (
                <div key={col} className="flex flex-col gap-1">
                  <label className="text-xs text-zinc-400 font-mono">
                    {col}
                    <span className="ml-2 text-zinc-600">
                      ({strategy.dtype_hint ?? "?"})
                    </span>
                  </label>
                  <input
                    type="text"
                    className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-zinc-200 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-blue-500"
                    placeholder={strategy.dtype_hint === "numeric" ? "0.0" : "value"}
                    value={inputValues[col] ?? ""}
                    onChange={(e) =>
                      setInputValues((prev) => ({ ...prev, [col]: e.target.value }))
                    }
                  />
                </div>
              ))}

              {error && (
                <p className="text-sm text-red-400 bg-red-950/30 border border-red-900 rounded p-2">
                  {error}
                </p>
              )}

              <Button
                onClick={handlePredict}
                disabled={loading || keptColumns.length === 0}
                className="w-full"
              >
                {loading ? "Running…" : "Predict"}
              </Button>
            </CardContent>
          </Card>

          {/* Result */}
          <div className="space-y-4">
            {result && <PredictionResult result={result} />}
            {!result && (
              <div className="flex items-center justify-center h-40 rounded-lg border border-dashed border-zinc-700 text-zinc-600 text-sm">
                Fill in features and click Predict
              </div>
            )}
          </div>
        </div>
      )}

      {/* Batch prediction */}
      {activeRunId && (
        <Card>
          <CardHeader>
            <CardTitle>Batch prediction</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {inferenceDatasets.length === 0 ? (
              <p className="text-sm text-zinc-500">
                No inference-role datasets found.{" "}
                <Link
                  href={`/project/${projectId}/datasets`}
                  className="text-blue-400 underline"
                >
                  Upload a dataset
                </Link>{" "}
                with role <span className="font-mono text-zinc-400">inference</span> first.
              </p>
            ) : (
              <div className="flex flex-wrap items-end gap-3">
                <div className="flex flex-col gap-1">
                  <label className="text-xs text-zinc-500">Inference dataset</label>
                  <select
                    className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-zinc-200 text-xs font-mono"
                    value={batchDatasetId}
                    onChange={(e) => {
                      setBatchDatasetId(e.target.value)
                      setBatchJob(null)
                      setBatchDone(false)
                      setBatchError(null)
                    }}
                  >
                    <option value="">- select -</option>
                    {inferenceDatasets.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.filename}
                        {d.row_count != null ? ` (${d.row_count.toLocaleString()} rows)` : ""}
                      </option>
                    ))}
                  </select>
                </div>
                <Button
                  onClick={handleBatchPredict}
                  disabled={batchLoading || !batchDatasetId || (!!batchJob && !batchDone)}
                >
                  {batchLoading ? "Queuing…" : "Run batch prediction"}
                </Button>
              </div>
            )}

            {batchError && (
              <p className="text-sm text-red-400 bg-red-950/30 border border-red-900 rounded p-2">
                {batchError}
              </p>
            )}

            {batchJob && !batchDone && (
              <div className="rounded-lg border border-zinc-700 bg-zinc-800/40 px-4 py-3 space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-zinc-400">
                    Processing {batchJob.n_rows.toLocaleString()} rows…
                  </span>
                  <Badge variant="warning">running</Badge>
                </div>
                {batchPoll && batchPoll.count > 0 && (
                  <div className="w-full bg-zinc-700 rounded-full h-1.5">
                    <div
                      className="bg-blue-500 h-1.5 rounded-full transition-all"
                      style={{ width: `${Math.min(100, (batchPoll.count / batchJob.n_rows) * 100).toFixed(1)}%` }}
                    />
                  </div>
                )}
                <p className="text-[11px] text-zinc-600">
                  {batchPoll?.count ?? 0} / {batchJob.n_rows} predictions written
                </p>
              </div>
            )}

            {batchDone && batchJob && (
              <div className="rounded-lg border border-emerald-900/60 bg-emerald-950/20 px-4 py-3 space-y-2">
                <p className="text-sm text-emerald-300">
                  Batch complete - {batchJob.n_rows.toLocaleString()} predictions ready.
                </p>
                <div className="flex flex-wrap gap-2">
                  {(["xlsx", "csv", "parquet"] as const).map((fmt) => (
                    <a
                      key={fmt}
                      href={`/api/proxy/runs/${activeRunId}/predict/batch/${batchJob.inference_dataset_id}/download?format=${fmt}`}
                      download
                      className="text-xs px-3 py-1 rounded border border-emerald-800 bg-emerald-900/30 text-emerald-300 hover:bg-emerald-900/50 transition-colors font-mono"
                    >
                      ↓ {fmt.toUpperCase()}
                    </a>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Prediction history */}
      {history && history.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recent predictions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-zinc-500 border-b border-zinc-800 text-left">
                    <th className="pb-1 pr-3 font-medium">Prediction</th>
                    <th className="pb-1 pr-3 font-medium">Probability</th>
                    <th className="pb-1 pr-3 font-medium">Confidence</th>
                    <th className="pb-1 pr-3 font-medium">Similarity</th>
                    <th className="pb-1 pr-3 font-medium">Threshold</th>
                    <th className="pb-1 font-medium">Risk</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800">
                  {history.map((p) => (
                    <tr key={p.id}>
                      <td className="py-1.5 pr-3 font-mono text-zinc-300">
                        {String(p.prediction.value ?? "-")}
                      </td>
                      <td className="py-1.5 pr-3 font-mono">
                        {p.probability != null ? p.probability.toFixed(4) : "-"}
                      </td>
                      <td className="py-1.5 pr-3">
                        <ConfidenceBadge band={p.confidence_band} />
                      </td>
                      <td className="py-1.5 pr-3 font-mono text-zinc-400">
                        {p.similarity_score != null ? p.similarity_score.toFixed(3) : "-"}
                      </td>
                      <td className="py-1.5 pr-3 font-mono text-zinc-400">
                        {p.threshold_used != null ? p.threshold_used.toFixed(3) : "-"}
                      </td>
                      <td className="py-1.5">
                        {p.risk_flag && <Badge variant="error" className="text-[10px]">risk</Badge>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function PredictionResult({ result }: { result: PredictResponse }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Prediction</span>
          <ConfidenceBadge band={result.confidence_band} />
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <Tile label="Result" value={String(result.prediction)} large />
          {result.probability != null && (
            <Tile label="Probability" value={result.probability.toFixed(4)} large />
          )}
          <Tile label="Threshold" value={result.threshold_used.toFixed(3)} />
          {result.similarity_score != null && (
            <Tile
              label="Training similarity"
              value={`${(result.similarity_score * 100).toFixed(1)}%`}
              warn={result.similarity_score < 0.3}
            />
          )}
        </div>

        {result.shap_drivers.length > 0 && (
          <div>
            <p className="text-[11px] text-zinc-500 mb-1">Top drivers (pushing positive)</p>
            <div className="flex flex-wrap gap-1">
              {result.shap_drivers.map((f) => (
                <span key={f} className="text-xs bg-green-900/30 border border-green-800 text-green-300 rounded px-1.5 py-0.5 font-mono">
                  {f}
                </span>
              ))}
            </div>
          </div>
        )}

        {result.shap_dampeners.length > 0 && (
          <div>
            <p className="text-[11px] text-zinc-500 mb-1">Top dampeners (pushing negative)</p>
            <div className="flex flex-wrap gap-1">
              {result.shap_dampeners.map((f) => (
                <span key={f} className="text-xs bg-red-900/30 border border-red-900 text-red-300 rounded px-1.5 py-0.5 font-mono">
                  {f}
                </span>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function Tile({
  label,
  value,
  large,
  warn,
}: {
  label: string
  value: string
  large?: boolean
  warn?: boolean
}) {
  return (
    <div className="bg-zinc-800/50 rounded-lg p-3 space-y-1">
      <p className="text-[11px] text-zinc-500">{label}</p>
      <p
        className={`font-mono font-semibold ${large ? "text-xl" : "text-base"} ${
          warn ? "text-amber-400" : "text-zinc-200"
        }`}
      >
        {value}
      </p>
    </div>
  )
}

function ConfidenceBadge({ band }: { band: string | null }) {
  if (!band) return null
  const variant = band === "high" ? "success" : band === "medium" ? "warning" : "error"
  return <Badge variant={variant}>{band}</Badge>
}
