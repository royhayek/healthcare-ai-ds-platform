"use client"

/** Dedicated upload page - standalone route for adding a dataset to a project.
 *
 * Spec §6 names this page at project/[id]/upload. The existing datasets page
 * embeds upload in a dropzone, but this page provides a full-screen workflow
 * with schema preview and target column selection before committing the upload.
 */

import { useState, useCallback } from "react"
import { useRouter, useParams } from "next/navigation"
import Link from "next/link"
import { ArrowLeft, UploadCloud } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ColumnPreview } from "@/components/upload/ColumnPreview"
import { uploadDataset } from "@/lib/api"

const ROLES = ["training", "inference", "holdout", "reference", "comparison"] as const
type DatasetRole = (typeof ROLES)[number]

const ROLE_DESCRIPTIONS: Record<DatasetRole, string> = {
  training: "Labelled data the model learns from",
  inference: "Unlabelled data the model will predict on",
  holdout: "Sealed evaluation set - never touched during training",
  reference: "Side table to enrich training data via join",
  comparison: "Second snapshot for drift detection",
}

interface ColumnInfo {
  name: string
  dtype: string
  missing_pct: number
  unique: number
  sample_values: (string | number)[]
}

function parseLocalSchema(file: File): Promise<ColumnInfo[] | null> {
  // Parse first 200 rows client-side for instant preview (no server round-trip).
  return new Promise((resolve) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string
        const lines = text.split("\n").filter(Boolean)
        if (lines.length < 2) { resolve(null); return }
        const header = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""))
        const rows = lines.slice(1, 201).map((l) =>
          l.split(",").map((v) => v.trim().replace(/^"|"$/g, ""))
        )
        const cols: ColumnInfo[] = header.map((name, i) => {
          const vals = rows.map((r) => r[i] ?? "").filter(Boolean)
          const numeric = vals.filter((v) => !isNaN(Number(v)))
          const dtype = numeric.length / Math.max(vals.length, 1) > 0.8 ? "float64" : "object"
          const missing = rows.filter((r) => !r[i] || r[i] === "").length
          const unique = new Set(vals).size
          return {
            name,
            dtype,
            missing_pct: (missing / rows.length) * 100,
            unique,
            sample_values: [...new Set(vals)].slice(0, 4),
          }
        })
        resolve(cols)
      } catch {
        resolve(null)
      }
    }
    reader.readAsText(file.slice(0, 64 * 1024))  // first 64 KB only
  })
}

export default function UploadPage() {
  const params = useParams<{ id: string }>()
  const projectId = params.id
  const router = useRouter()

  const [file, setFile] = useState<File | null>(null)
  const [role, setRole] = useState<DatasetRole>("training")
  const [targetColumn, setTargetColumn] = useState<string | null>(null)
  const [columns, setColumns] = useState<ColumnInfo[] | null>(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFileDrop = useCallback(async (dropped: File) => {
    setFile(dropped)
    setTargetColumn(null)
    setError(null)
    const schema = await parseLocalSchema(dropped)
    setColumns(schema)
  }, [])

  async function handleUpload() {
    if (!file) return
    setUploading(true)
    setError(null)
    try {
      await uploadDataset(projectId, file, role, targetColumn ?? undefined)
      router.push(`/project/${projectId}/datasets`)
    } catch (e) {
      setError(String(e))
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto px-4 py-8 space-y-6">
      {/* Back nav */}
      <Link
        href={`/project/${projectId}/datasets`}
        className="inline-flex items-center gap-1.5 text-sm text-zinc-400 hover:text-zinc-200 transition-colors"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to datasets
      </Link>

      <div>
        <h1 className="text-xl font-semibold text-zinc-100">Upload Dataset</h1>
        <p className="text-sm text-zinc-500 mt-1">
          Add a new dataset to this project. Choose the role that determines how
          the pipeline will use it.
        </p>
      </div>

      {/* Role selector */}
      <div>
        <div className="text-xs text-zinc-500 mb-2 font-medium uppercase tracking-wide">Dataset role</div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {ROLES.map((r) => (
            <button
              key={r}
              type="button"
              onClick={() => setRole(r)}
              className={[
                "rounded-lg border px-3 py-2 text-left transition-colors",
                role === r
                  ? "border-blue-600 bg-blue-900/20 text-blue-300"
                  : "border-zinc-800 bg-zinc-900 text-zinc-400 hover:border-zinc-600",
              ].join(" ")}
            >
              <div className="text-sm font-medium capitalize mb-0.5">{r}</div>
              <div className="text-xs text-zinc-600 leading-snug">{ROLE_DESCRIPTIONS[r]}</div>
            </button>
          ))}
        </div>
      </div>

      {/* File picker */}
      <label className="flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-zinc-700 bg-zinc-900 px-6 py-10 cursor-pointer hover:border-zinc-500 transition-colors">
        <UploadCloud className="h-8 w-8 text-zinc-600" />
        <span className="text-sm text-zinc-400">Drop a CSV or Parquet file, or click to browse</span>
        <input
          type="file"
          accept=".csv,.parquet"
          className="sr-only"
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) handleFileDrop(f)
          }}
        />
        {file && <span className="text-xs text-emerald-400 mt-1">{file.name}</span>}
      </label>

      {/* Schema preview */}
      {file && columns && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="text-xs font-medium text-zinc-400">
              {file.name} - {(file.size / 1024).toFixed(0)} KB
            </div>
            {targetColumn && role === "training" && (
              <Badge variant="success" className="text-xs">target: {targetColumn}</Badge>
            )}
          </div>
          <ColumnPreview
            columns={columns}
            totalRows={0}
            selectedTarget={targetColumn ?? undefined}
            onSelectTarget={role === "training" ? setTargetColumn : undefined}
          />
          {role === "training" && !targetColumn && (
            <p className="text-xs text-amber-400">
              Select a target column by clicking a row above.
            </p>
          )}
        </div>
      )}

      {error && (
        <div className="rounded-md border border-red-800/50 bg-red-900/20 px-3 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* Upload button */}
      {file && (
        <div className="flex gap-3">
          <Button
            onClick={handleUpload}
            disabled={uploading || (role === "training" && !targetColumn)}
            className="gap-2"
          >
            <UploadCloud className="h-4 w-4" />
            {uploading ? "Uploading…" : "Upload dataset"}
          </Button>
          <Button variant="ghost" onClick={() => { setFile(null); setColumns(null) }}>
            Clear
          </Button>
        </div>
      )}
    </div>
  )
}
