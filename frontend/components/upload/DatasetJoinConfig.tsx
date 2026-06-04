"use client"

import { useState } from "react"
import type { Dataset } from "@/lib/types"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

type JoinType = "inner" | "left" | "right" | "outer"

const JOIN_DESCRIPTIONS: Record<JoinType, string> = {
  inner: "Keep only rows with a match in both datasets",
  left: "Keep all rows from the left dataset",
  right: "Keep all rows from the right dataset",
  outer: "Keep all rows from both datasets",
}

interface JoinConfig {
  leftDatasetId: string
  rightDatasetId: string
  joinType: JoinType
  joinKeys: string[]
}

interface DatasetJoinConfigProps {
  datasets: Dataset[]
  onConfirm: (config: JoinConfig) => void
  onCancel?: () => void
}

export function DatasetJoinConfig({ datasets, onConfirm, onCancel }: DatasetJoinConfigProps) {
  const trainingDs = datasets.filter((d) => d.role === "training")
  const referenceDs = datasets.filter((d) => d.role === "reference")

  const [leftId, setLeftId] = useState(trainingDs[0]?.id ?? "")
  const [rightId, setRightId] = useState(referenceDs[0]?.id ?? "")
  const [joinType, setJoinType] = useState<JoinType>("left")
  const [joinKeyInput, setJoinKeyInput] = useState("")
  const [joinKeys, setJoinKeys] = useState<string[]>([])

  function addKey() {
    const key = joinKeyInput.trim()
    if (key && !joinKeys.includes(key)) {
      setJoinKeys((prev) => [...prev, key])
      setJoinKeyInput("")
    }
  }

  function removeKey(key: string) {
    setJoinKeys((prev) => prev.filter((k) => k !== key))
  }

  function handleConfirm() {
    if (!leftId || !rightId || joinKeys.length === 0) return
    onConfirm({ leftDatasetId: leftId, rightDatasetId: rightId, joinType, joinKeys })
  }

  const canConfirm = Boolean(leftId && rightId && joinKeys.length > 0 && leftId !== rightId)

  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Configure dataset join</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Dataset selection */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-zinc-400 mb-1.5">Left dataset</label>
            <select
              value={leftId}
              onChange={(e) => setLeftId(e.target.value)}
              className="w-full bg-zinc-800 border border-zinc-700 text-zinc-200 text-sm rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="">Select…</option>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.filename} ({d.role})
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-zinc-400 mb-1.5">Right dataset</label>
            <select
              value={rightId}
              onChange={(e) => setRightId(e.target.value)}
              className="w-full bg-zinc-800 border border-zinc-700 text-zinc-200 text-sm rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="">Select…</option>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.filename} ({d.role})
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Join type */}
        <div>
          <label className="block text-xs font-medium text-zinc-400 mb-1.5">Join type</label>
          <div className="grid grid-cols-4 gap-2">
            {(["inner", "left", "right", "outer"] as JoinType[]).map((jt) => (
              <button
                key={jt}
                onClick={() => setJoinType(jt)}
                className={[
                  "rounded border px-3 py-2 text-sm transition-colors text-left",
                  joinType === jt
                    ? "border-blue-500 bg-blue-500/10 text-blue-300"
                    : "border-zinc-700 text-zinc-400 hover:border-zinc-600",
                ].join(" ")}
              >
                <div className="font-medium uppercase text-xs">{jt}</div>
                <div className="text-xs mt-0.5 leading-tight">{JOIN_DESCRIPTIONS[jt]}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Join keys */}
        <div>
          <label className="block text-xs font-medium text-zinc-400 mb-1.5">Join keys</label>
          <div className="flex gap-2">
            <input
              type="text"
              value={joinKeyInput}
              onChange={(e) => setJoinKeyInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addKey()}
              placeholder="Column name (e.g. customer_id)"
              className="flex-1 bg-zinc-800 border border-zinc-700 text-zinc-200 text-sm rounded px-3 py-1.5 placeholder-zinc-600 focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
            <Button variant="outline" size="sm" onClick={addKey}>
              Add
            </Button>
          </div>
          {joinKeys.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mt-2">
              {joinKeys.map((k) => (
                <Badge key={k} variant="info" className="cursor-pointer" onClick={() => removeKey(k)}>
                  {k} ✕
                </Badge>
              ))}
            </div>
          )}
          <p className="text-xs text-zinc-600 mt-1.5">
            Columns must exist in both datasets with the same name. Press Enter or Add.
          </p>
        </div>

        {/* Actions */}
        <div className="flex gap-2 pt-2">
          <Button
            onClick={handleConfirm}
            disabled={!canConfirm}
            className="flex-1"
          >
            Apply join
          </Button>
          {onCancel && (
            <Button variant="ghost" onClick={onCancel}>
              Cancel
            </Button>
          )}
        </div>

        {leftId === rightId && leftId && (
          <p className="text-xs text-red-400">Left and right datasets must be different.</p>
        )}
      </CardContent>
    </Card>
  )
}
