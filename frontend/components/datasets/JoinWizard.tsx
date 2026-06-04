"use client"

import { useState } from "react"
import { createJoin, suggestJoinKeys } from "@/lib/api"
import type { Dataset, JoinKeyCandidate, JoinSuggestResponse } from "@/lib/types"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/cn"

type JoinType = "inner" | "left" | "right" | "outer"

const JOIN_LABELS: Record<JoinType, { title: string; desc: string }> = {
  left:  { title: "LEFT",  desc: "Keep all left rows" },
  inner: { title: "INNER", desc: "Only matching rows" },
  right: { title: "RIGHT", desc: "Keep all right rows" },
  outer: { title: "FULL",  desc: "Keep all rows" },
}

interface Props {
  projectId: string
  datasets: Dataset[]
  onJoined: (result: Dataset) => void
  onCancel?: () => void
}

type Step = "select" | "suggest" | "configure" | "executing"

export function JoinWizard({ projectId, datasets, onJoined, onCancel }: Props) {
  const [step, setStep] = useState<Step>("select")
  const [leftId, setLeftId] = useState(datasets[0]?.id ?? "")
  const [rightId, setRightId] = useState(datasets[1]?.id ?? "")
  const [suggestion, setSuggestion] = useState<JoinSuggestResponse | null>(null)
  const [joinType, setJoinType] = useState<JoinType>("left")
  const [selectedKeys, setSelectedKeys] = useState<string[]>([])
  const [manualKey, setManualKey] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const leftDs = datasets.find((d) => d.id === leftId)
  const rightDs = datasets.find((d) => d.id === rightId)

  async function handleSuggest() {
    if (!leftId || !rightId || leftId === rightId) return
    setError(null)
    setLoading(true)
    try {
      const result = await suggestJoinKeys(projectId, leftId, rightId)
      setSuggestion(result)
      setJoinType(result.recommended_join_type as JoinType)
      const recommended = result.candidates.filter((c) => c.recommended).map((c) => c.column)
      setSelectedKeys(recommended.length > 0 ? [recommended[0]] : [])
      setStep("suggest")
    } catch (e) {
      setError(e instanceof Error ? e.message : "Suggest failed")
    } finally {
      setLoading(false)
    }
  }

  async function handleExecute() {
    if (selectedKeys.length === 0) return
    setError(null)
    setStep("executing")
    setLoading(true)
    try {
      const result = await createJoin(projectId, {
        left_dataset_id: leftId,
        right_dataset_id: rightId,
        join_type: joinType,
        join_keys: selectedKeys,
      })
      onJoined(result)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Join failed")
      setStep("suggest")
    } finally {
      setLoading(false)
    }
  }

  function toggleKey(col: string) {
    setSelectedKeys((prev) =>
      prev.includes(col) ? prev.filter((k) => k !== col) : [...prev, col],
    )
  }

  function addManualKey() {
    const key = manualKey.trim()
    if (key && !selectedKeys.includes(key)) {
      setSelectedKeys((prev) => [...prev, key])
      setManualKey("")
    }
  }

  return (
    <div className="rounded-xl border border-zinc-700 bg-zinc-900 divide-y divide-zinc-800">
      <div className="px-4 py-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-zinc-200">Join datasets</h3>
        {onCancel && (
          <button onClick={onCancel} className="text-zinc-500 hover:text-zinc-300 text-sm">
            ✕
          </button>
        )}
      </div>

      <div className="p-4 space-y-4">
        {/* Dataset selection */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium text-zinc-400 mb-1">Left (primary)</label>
            <select
              value={leftId}
              onChange={(e) => { setLeftId(e.target.value); setStep("select"); setSuggestion(null) }}
              className="w-full bg-zinc-800 border border-zinc-700 text-zinc-200 text-sm rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500"
              disabled={step === "executing"}
            >
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.filename} ({d.role})
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-zinc-400 mb-1">Right (lookup)</label>
            <select
              value={rightId}
              onChange={(e) => { setRightId(e.target.value); setStep("select"); setSuggestion(null) }}
              className="w-full bg-zinc-800 border border-zinc-700 text-zinc-200 text-sm rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500"
              disabled={step === "executing"}
            >
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.filename} ({d.role})
                </option>
              ))}
            </select>
          </div>
        </div>

        {leftId === rightId && leftId && (
          <p className="text-xs text-red-400">Select two different datasets.</p>
        )}

        {/* Step 1: detect join keys */}
        {step === "select" && (
          <Button
            onClick={handleSuggest}
            disabled={loading || !leftId || !rightId || leftId === rightId}
            size="sm"
            className="w-full"
          >
            {loading ? "Detecting join keys…" : "Detect join keys →"}
          </Button>
        )}

        {/* Step 2: show suggestions */}
        {(step === "suggest" || step === "configure" || step === "executing") && suggestion && (
          <>
            {/* Info note */}
            <div className="rounded-md bg-blue-950/30 border border-blue-800/40 px-3 py-2 text-xs text-blue-300">
              {suggestion.note}
            </div>

            {/* Join key candidates */}
            <div>
              <label className="block text-xs font-medium text-zinc-400 mb-2">
                Join key candidates - select one or more
              </label>
              <div className="space-y-1.5">
                {suggestion.candidates.map((c) => (
                  <button
                    key={c.column}
                    onClick={() => toggleKey(c.column)}
                    disabled={step === "executing"}
                    className={cn(
                      "w-full flex items-center gap-3 rounded-lg border px-3 py-2.5 text-left text-sm transition-colors",
                      selectedKeys.includes(c.column)
                        ? "border-blue-500 bg-blue-500/10"
                        : "border-zinc-700 hover:border-zinc-600",
                    )}
                  >
                    <div className={cn(
                      "w-4 h-4 rounded border flex items-center justify-center shrink-0",
                      selectedKeys.includes(c.column)
                        ? "border-blue-500 bg-blue-500"
                        : "border-zinc-600",
                    )}>
                      {selectedKeys.includes(c.column) && (
                        <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 10 10">
                          <path d="M2 5l2.5 2.5L8 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      )}
                    </div>
                    <span className="font-mono text-zinc-200 flex-1">{c.column}</span>
                    <span className="text-xs text-zinc-500">
                      {(c.overlap_pct * 100).toFixed(0)}% overlap
                    </span>
                    <span className="text-xs text-zinc-600">
                      {c.left_unique.toLocaleString()} / {c.right_unique.toLocaleString()} unique
                    </span>
                    {c.recommended && (
                      <Badge variant="success" className="text-xs">recommended</Badge>
                    )}
                  </button>
                ))}
              </div>

              {/* Manual key input */}
              <div className="flex gap-2 mt-2">
                <input
                  type="text"
                  value={manualKey}
                  onChange={(e) => setManualKey(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && addManualKey()}
                  placeholder="Or type a column name manually"
                  disabled={step === "executing"}
                  className="flex-1 bg-zinc-800 border border-zinc-700 text-zinc-200 text-xs rounded px-3 py-1.5 placeholder-zinc-600 focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
                <Button variant="outline" size="sm" onClick={addManualKey} disabled={step === "executing"}>
                  Add
                </Button>
              </div>
            </div>

            {/* Join type */}
            <div>
              <label className="block text-xs font-medium text-zinc-400 mb-2">Join type</label>
              <div className="grid grid-cols-4 gap-2">
                {(["inner", "left", "right", "outer"] as JoinType[]).map((jt) => (
                  <button
                    key={jt}
                    onClick={() => setJoinType(jt)}
                    disabled={step === "executing"}
                    className={cn(
                      "rounded border px-3 py-2 text-xs text-left transition-colors",
                      joinType === jt
                        ? "border-blue-500 bg-blue-500/10 text-blue-300"
                        : "border-zinc-700 text-zinc-400 hover:border-zinc-600",
                    )}
                  >
                    <div className="font-semibold">{JOIN_LABELS[jt].title}</div>
                    <div className="mt-0.5 leading-tight text-zinc-500">{JOIN_LABELS[jt].desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Selected keys summary */}
            {selectedKeys.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {selectedKeys.map((k) => (
                  <Badge
                    key={k}
                    variant="info"
                    className={cn("cursor-pointer", step !== "executing" && "hover:opacity-70")}
                    onClick={() => step !== "executing" && toggleKey(k)}
                  >
                    {k} {step !== "executing" && "✕"}
                  </Badge>
                ))}
              </div>
            )}

            <Button
              onClick={handleExecute}
              disabled={loading || selectedKeys.length === 0 || step === "executing"}
              className="w-full"
            >
              {step === "executing"
                ? "Materializing joined dataset…"
                : `Execute ${joinType} join on [${selectedKeys.join(", ")}] →`}
            </Button>
          </>
        )}

        {error && (
          <p className="text-xs text-red-400">{error}</p>
        )}
      </div>
    </div>
  )
}
