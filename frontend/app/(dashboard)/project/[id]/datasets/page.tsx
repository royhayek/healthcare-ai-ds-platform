"use client"

import { useCallback, useState } from "react"
import Link from "next/link"
import useSWR from "swr"
import { fetcher, updateDatasetRole } from "@/lib/api"
import type { Dataset } from "@/lib/types"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { DatasetCard } from "@/components/datasets/DatasetCard"
import { JoinWizard } from "@/components/datasets/JoinWizard"
import { GitMerge } from "lucide-react"

const ROLE_LABELS: Record<string, string> = {
  training: "Training",
  inference: "Inference",
  holdout: "Holdout",
  reference: "Reference",
  comparison: "Comparison",
}

const ROLE_DESCRIPTIONS: Record<string, string> = {
  training: "Labelled data the model learns from",
  inference: "Unlabelled data the model will predict on",
  holdout: "Sealed evaluation set - never touched during training",
  reference: "Side table to enrich training data via join",
  comparison: "Second snapshot for drift detection",
}

const ROLE_BADGE: Record<string, "info" | "warning" | "error" | "success" | "outline"> = {
  training: "success",
  inference: "info",
  holdout: "error",
  reference: "outline",
  comparison: "warning",
}

const ROLES = ["training", "inference", "holdout", "reference", "comparison"] as const

export default function DatasetsPage({ params }: { params: { id: string } }) {
  const { id: projectId } = params

  const { data: datasets, mutate } = useSWR<Dataset[]>(
    `/api/proxy/projects/${projectId}/datasets`,
    fetcher,
  )

  const [error, setError] = useState<string | null>(null)
  const [showJoin, setShowJoin] = useState(false)

  async function handleRoleChange(datasetId: string, role: string) {
    setError(null)
    try {
      await updateDatasetRole(projectId, datasetId, role)
      mutate()
    } catch (e) {
      setError(String(e))
    }
  }

  const handleJoined = useCallback(
    (_result: Dataset) => {
      mutate()
      setShowJoin(false)
    },
    [mutate],
  )

  const byRole = ROLES.reduce(
    (acc, role) => {
      acc[role] = datasets?.filter((d) => d.role === role) ?? []
      return acc
    },
    {} as Record<string, Dataset[]>,
  )

  const canJoin = (datasets?.length ?? 0) >= 2

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Datasets</h1>
          <p className="text-sm text-zinc-500 mt-1">
            Manage dataset roles. Click any dataset to preview its data and EDA plots.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {canJoin && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowJoin((v) => !v)}
              className="gap-1.5 text-xs"
            >
              <GitMerge className="w-3.5 h-3.5" />
              {showJoin ? "Cancel join" : "Join datasets"}
            </Button>
          )}
          <Link href={`/project/${projectId}`}>
            <Button variant="ghost" size="sm">← Project</Button>
          </Link>
        </div>
      </div>

      {error && (
        <div className="rounded-md border border-red-700/50 bg-red-900/20 px-3 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* Role legend */}
      <div className="grid grid-cols-5 gap-2">
        {ROLES.map((role) => (
          <div key={role} className="rounded-lg border border-zinc-800 p-3">
            <Badge variant={ROLE_BADGE[role] ?? "outline"} className="mb-1.5">
              {ROLE_LABELS[role]}
            </Badge>
            <p className="text-xs text-zinc-500">{ROLE_DESCRIPTIONS[role]}</p>
            <p className="text-xs text-zinc-600 mt-1">{byRole[role].length} file(s)</p>
          </div>
        ))}
      </div>

      {/* Join wizard */}
      {showJoin && canJoin && datasets && (
        <JoinWizard
          projectId={projectId}
          datasets={datasets}
          onJoined={handleJoined}
          onCancel={() => setShowJoin(false)}
        />
      )}

      {/* Dataset cards with inline preview & plots */}
      <div className="space-y-2">
        {!datasets && (
          <p className="text-sm text-zinc-500">Loading…</p>
        )}
        {datasets?.length === 0 && (
          <p className="text-sm text-zinc-500">No datasets yet.</p>
        )}
        {datasets?.map((d) => (
          <div key={d.id} className="space-y-1">
            <DatasetCard dataset={d} projectId={projectId} />
            <div className="flex items-center gap-3 pl-4">
              <span className="text-xs text-zinc-600">Role:</span>
              <select
                value={d.role}
                onChange={(e) => handleRoleChange(d.id, e.target.value)}
                className="bg-zinc-800 border border-zinc-700 text-zinc-200 text-xs rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                {ROLES.map((r) => (
                  <option key={r} value={r}>{ROLE_LABELS[r]}</option>
                ))}
              </select>
            </div>
          </div>
        ))}
      </div>

      {/* Holdout notice */}
      {byRole.holdout.length > 0 && (
        <div className="rounded-md border border-amber-700/50 bg-amber-900/10 px-3 py-3 text-sm">
          <p className="font-medium text-amber-300">
            🔒 {byRole.holdout.length} holdout dataset(s) are sealed
          </p>
          <p className="text-amber-400/80 mt-1 text-xs">
            Holdout data is never used during training, CV, or tuning. It will be opened exactly
            once at the end of the pipeline for a final unbiased evaluation.
          </p>
        </div>
      )}
    </div>
  )
}
