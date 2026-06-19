"use client"

import { useState } from "react"
import { ChevronDown, ChevronRight, Eye, BarChart2, Trash2 } from "lucide-react"
import useSWR from "swr"
import { Badge } from "@/components/ui/badge"
import { DatasetPlotGrid } from "@/components/datasets/DatasetPlotGrid"
import { deleteDataset, fetcher, updateDatasetTargetColumn } from "@/lib/api"
import type { Dataset, DatasetPreview } from "@/lib/types"
import { cn } from "@/lib/cn"

const ROLE_VARIANT: Record<Dataset["role"], "default" | "success" | "warning" | "info" | "outline" | "error"> = {
  training: "success",
  holdout: "error",
  inference: "info",
  reference: "default",
  comparison: "warning",
}

type Tab = "preview" | "plots"

interface Props {
  dataset: Dataset
  projectId: string
  onDatasetUpdated?: (updated: Dataset) => void
  onDeleted?: (datasetId: string) => void
}

function fmtBytes(n: number | null) {
  if (n == null) return "-"
  if (n < 1024) return `${n} B`
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 ** 2).toFixed(1)} MB`
}

function PreviewTable({ projectId, datasetId }: { projectId: string; datasetId: string }) {
  const { data, error, isLoading } = useSWR<DatasetPreview>(
    `/api/proxy/projects/${projectId}/datasets/${datasetId}/preview?rows=20`,
    fetcher,
  )

  if (isLoading) return <p className="text-xs text-zinc-500 py-3">Loading preview…</p>
  if (error || !data) return <p className="text-xs text-red-400 py-3">Could not load preview.</p>

  return (
    <div className="space-y-2">
      <p className="text-xs text-zinc-500">
        Showing first {data.rows.length} of {data.total_rows.toLocaleString()} rows
      </p>
      <div className="overflow-x-auto rounded-lg border border-zinc-800">
        <table className="text-xs w-full">
          <thead>
            <tr className="bg-zinc-900 border-b border-zinc-800">
              {data.columns.map((col) => (
                <th
                  key={col}
                  className="px-3 py-2 text-left font-medium text-zinc-400 whitespace-nowrap"
                >
                  <div>{col}</div>
                  <div className="text-zinc-600 font-normal">{data.dtypes[col]}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row, i) => (
              <tr
                key={i}
                className="border-b border-zinc-800/60 last:border-0 hover:bg-zinc-800/30"
              >
                {data.columns.map((col) => (
                  <td key={col} className="px-3 py-1.5 text-zinc-300 whitespace-nowrap max-w-[160px] truncate">
                    {row[col] == null || row[col] === "" ? (
                      <span className="text-zinc-600 italic">null</span>
                    ) : (
                      String(row[col])
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function TargetColumnSelector({
  dataset,
  projectId,
  onUpdated,
}: {
  dataset: Dataset
  projectId: string
  onUpdated: (updated: Dataset) => void
}) {
  const { data: preview } = useSWR<DatasetPreview>(
    `/api/proxy/projects/${projectId}/datasets/${dataset.id}/preview?rows=1`,
    fetcher,
  )
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const columns = preview?.columns ?? []

  async function handleChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const col = e.target.value
    if (!col) return
    setSaving(true)
    setError(null)
    try {
      const updated = await updateDatasetTargetColumn(projectId, dataset.id, col)
      onUpdated(updated)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to set target column")
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-amber-950/30 border-t border-amber-800/40">
      <span className="text-xs text-amber-400 font-medium shrink-0">Target column required</span>
      <select
        className="flex-1 min-w-0 text-xs bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-zinc-200 focus:outline-none focus:border-zinc-500 disabled:opacity-50"
        defaultValue=""
        onChange={handleChange}
        disabled={saving || columns.length === 0}
      >
        <option value="" disabled>
          {columns.length === 0 ? "Loading columns…" : "Select target column"}
        </option>
        {columns.map((col) => (
          <option key={col} value={col}>
            {col}
          </option>
        ))}
      </select>
      {error && <span className="text-xs text-red-400 shrink-0">{error}</span>}
    </div>
  )
}

export function DatasetCard({ dataset: initialDataset, projectId, onDatasetUpdated, onDeleted }: Props) {
  const [dataset, setDataset] = useState(initialDataset)
  function handleDatasetUpdated(updated: Dataset) {
    setDataset(updated)
    onDatasetUpdated?.(updated)
  }
  const [open, setOpen] = useState(false)
  const [tab, setTab] = useState<Tab>("preview")
  const [deleting, setDeleting] = useState(false)

  async function handleDelete(e: React.MouseEvent) {
    e.stopPropagation()
    if (
      !window.confirm(
        `Delete "${dataset.filename}"? This removes the file and cannot be undone.`,
      )
    )
      return
    setDeleting(true)
    try {
      await deleteDataset(projectId, dataset.id)
      onDeleted?.(dataset.id)
      // On success the parent drops this card from its list; no state reset needed.
    } catch (err) {
      window.alert(err instanceof Error ? err.message : "Failed to delete dataset")
      setDeleting(false)
    }
  }

  return (
    <div className="rounded-lg border border-zinc-800 overflow-hidden">
      {/* Target column warning for training datasets */}
      {dataset.role === "training" && !dataset.target_column && (
        <TargetColumnSelector dataset={dataset} projectId={projectId} onUpdated={handleDatasetUpdated} />
      )}
      {/* Header row */}
      <div className="flex items-center bg-zinc-900/60 hover:bg-zinc-800/60 transition-colors">
        <button
          className="flex flex-1 min-w-0 items-center gap-3 px-4 py-3 text-left"
          onClick={() => setOpen((v) => !v)}
        >
          {open ? (
            <ChevronDown className="w-3.5 h-3.5 text-zinc-500 shrink-0" />
          ) : (
            <ChevronRight className="w-3.5 h-3.5 text-zinc-500 shrink-0" />
          )}

          <span className="font-mono text-sm text-zinc-200 flex-1 truncate">{dataset.filename}</span>

          <Badge variant={ROLE_VARIANT[dataset.role] ?? "default"} className="shrink-0">
            {dataset.role}
          </Badge>

          {dataset.task_type && (
            <span className="text-xs text-zinc-500 shrink-0">{dataset.task_type}</span>
          )}

          <span className="text-xs text-zinc-500 shrink-0">
            {dataset.row_count != null ? `${dataset.row_count.toLocaleString()} rows` : "-"}
            {" × "}
            {dataset.col_count != null ? `${dataset.col_count} cols` : "-"}
          </span>

          <span className="text-xs text-zinc-600 shrink-0">{fmtBytes(dataset.file_size_bytes)}</span>

          {dataset.target_column && (
            <span className="text-xs text-zinc-600 shrink-0">
              target: <span className="text-zinc-400">{dataset.target_column}</span>
            </span>
          )}
        </button>

        <button
          onClick={handleDelete}
          disabled={deleting}
          title="Delete dataset"
          aria-label={`Delete ${dataset.filename}`}
          data-testid={`delete-dataset-${dataset.id}`}
          className="shrink-0 px-3 py-3 text-zinc-500 hover:text-red-400 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Expanded panel */}
      {open && (
        <div className="border-t border-zinc-800 bg-zinc-950/40">
          {/* Tabs */}
          <div className="flex gap-0 border-b border-zinc-800">
            <button
              onClick={() => setTab("preview")}
              className={cn(
                "flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium transition-colors",
                tab === "preview"
                  ? "text-zinc-100 border-b-2 border-blue-500"
                  : "text-zinc-500 hover:text-zinc-300",
              )}
            >
              <Eye className="w-3 h-3" />
              Preview
            </button>
            <button
              onClick={() => setTab("plots")}
              className={cn(
                "flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium transition-colors",
                tab === "plots"
                  ? "text-zinc-100 border-b-2 border-blue-500"
                  : "text-zinc-500 hover:text-zinc-300",
              )}
            >
              <BarChart2 className="w-3 h-3" />
              EDA Plots
            </button>
          </div>

          {/* Tab content */}
          <div className="p-4">
            {tab === "preview" && (
              <PreviewTable projectId={projectId} datasetId={dataset.id} />
            )}
            {tab === "plots" && (
              <DatasetPlotGrid projectId={projectId} datasetId={dataset.id} />
            )}
          </div>
        </div>
      )}
    </div>
  )
}
