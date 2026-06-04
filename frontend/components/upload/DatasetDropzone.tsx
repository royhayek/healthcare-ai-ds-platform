"use client"

import { useCallback, useState } from "react"
import { useDropzone } from "react-dropzone"
import { cn } from "@/lib/cn"
import { uploadDataset } from "@/lib/api"
import type { Dataset } from "@/lib/types"
import { PhiWarningBanner, type PhiColumn } from "./PhiWarningBanner"

interface Props {
  projectId: string
  onUploaded: (dataset: Dataset) => void
}

const ACCEPTED = { "text/csv": [".csv"], "application/octet-stream": [".parquet"] }

function extractPhiColumns(dataset: Dataset): PhiColumn[] {
  const profile = dataset.profile as Record<string, unknown> | null
  if (!profile) return []
  const raw = profile["phi_columns"]
  if (!Array.isArray(raw)) return []
  return raw
    .filter(
      (p): p is { column: string; confidence: string } =>
        typeof p === "object" &&
        p !== null &&
        typeof (p as Record<string, unknown>)["column"] === "string" &&
        ["high", "medium"].includes(String((p as Record<string, unknown>)["confidence"])),
    )
    .map((p) => ({
      column: p.column,
      confidence: p.confidence as PhiColumn["confidence"],
    }))
}

export default function DatasetDropzone({ projectId, onUploaded }: Props) {
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Datasets awaiting PHI acknowledgment before being forwarded to onUploaded
  const [pendingDatasets, setPendingDatasets] = useState<
    { dataset: Dataset; phiCols: PhiColumn[] }[]
  >([])

  const onDrop = useCallback(
    async (accepted: File[]) => {
      if (accepted.length === 0) return
      setError(null)
      setUploading(true)
      try {
        for (const file of accepted) {
          const dataset = await uploadDataset(projectId, file, "training")
          const phiCols = extractPhiColumns(dataset)
          if (phiCols.length > 0) {
            // Queue for acknowledgment - do NOT call onUploaded yet
            setPendingDatasets((prev) => [...prev, { dataset, phiCols }])
          } else {
            onUploaded(dataset)
          }
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Upload failed")
      } finally {
        setUploading(false)
      }
    },
    [projectId, onUploaded],
  )

  function handleAcknowledge(dataset: Dataset) {
    setPendingDatasets((prev) => prev.filter((p) => p.dataset.id !== dataset.id))
    onUploaded(dataset)
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    disabled: uploading || pendingDatasets.length > 0,
    multiple: true,
  })

  return (
    <div className="space-y-3">
      {/* PHI banners - one per uploaded dataset awaiting acknowledgment */}
      {pendingDatasets.map(({ dataset, phiCols }) => (
        <PhiWarningBanner
          key={dataset.id}
          phiColumns={phiCols}
          onAcknowledge={() => handleAcknowledge(dataset)}
        />
      ))}

      {pendingDatasets.length === 0 && (
        <div
          {...getRootProps()}
          className={cn(
            "flex flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-10 text-center transition-colors cursor-pointer",
            isDragActive
              ? "border-neutral-400 bg-neutral-800/60"
              : "border-neutral-700 bg-neutral-900 hover:border-neutral-500 hover:bg-neutral-800/40",
            uploading && "opacity-50 cursor-not-allowed",
          )}
        >
          <input {...getInputProps()} data-testid="dataset-file-input" />
          {uploading ? (
            <p className="text-sm text-neutral-400">Uploading…</p>
          ) : isDragActive ? (
            <p className="text-sm text-neutral-300">Drop files here</p>
          ) : (
            <>
              <p className="text-sm text-neutral-400">
                Drag & drop CSV or Parquet files, or{" "}
                <span className="text-neutral-200 underline underline-offset-2">click to browse</span>
              </p>
              <p className="mt-1 text-xs text-neutral-600">Uploaded as role: training</p>
            </>
          )}
        </div>
      )}

      {error && <p className="text-xs text-red-400">{error}</p>}
    </div>
  )
}
