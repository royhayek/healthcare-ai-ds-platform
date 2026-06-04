"use client"

/**
 * DecisionCard - a single pipeline decision with bidirectional chat binding.
 *
 * Subscribes to useStrategyStore.pendingDiffs. When the chat agent applies an
 * override whose field_path matches this card's fieldPath prop, the card
 * re-renders with the ⚡ "Updated via chat" indicator (project hard rule 3,
 * spec §11 bidirectional binding requirement).
 *
 * Buttons:
 *   Accept    - acknowledges the decision, calls onAccept()
 *   Override  - opens an inline form for the user to type a replacement value
 *   Ask       - opens the chat panel pre-filled with context about this decision
 */

import { useState, useMemo } from "react"
import { Zap } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { MarkdownBody } from "@/components/ui/MarkdownBody"
import { useStrategyStore } from "@/store/strategyStore"

interface DecisionCardProps {
  /** Dot-delimited path matching StrategyDiff.field_path, e.g. "preprocessing.columns.age.scale_strategy" */
  fieldPath: string
  label: string
  value: string | number | boolean | null
  reason?: string
  severity?: "info" | "warn" | "critical"
  accepted?: boolean
  onAccept?: () => void
  /** Called with the user's override value string */
  onOverride?: (value: string) => void
  /** Called when the user clicks "Ask via chat" */
  onAskChat?: (context: string) => void
}

export function DecisionCard({
  fieldPath,
  label,
  value,
  reason,
  severity = "info",
  accepted = false,
  onAccept,
  onOverride,
  onAskChat,
}: DecisionCardProps) {
  const [overriding, setOverriding] = useState(false)
  const [overrideInput, setOverrideInput] = useState("")

  // Bidirectional binding: check if this field has a pending chat override
  const pendingDiffs = useStrategyStore((s) => s.pendingDiffs)
  const chatOverride = useMemo(
    () => pendingDiffs.find((d) => d.field_path === fieldPath),
    [pendingDiffs, fieldPath],
  )

  const displayValue = chatOverride ? chatOverride.after : value
  const wasUpdatedViaChat = chatOverride !== undefined

  const borderColor = accepted
    ? "border-emerald-900/60"
    : wasUpdatedViaChat
      ? "border-indigo-600/60"
      : severity === "critical"
        ? "border-red-900/60"
        : severity === "warn"
          ? "border-yellow-900/50"
          : "border-neutral-800"

  const bgColor = accepted
    ? "bg-emerald-950/20"
    : wasUpdatedViaChat
      ? "bg-indigo-950/20"
      : "bg-neutral-900/50"

  return (
    <div className={`rounded-lg border ${borderColor} ${bgColor} px-4 py-3 space-y-2`}>
      {/* Header row */}
      <div className="space-y-1.5">
        <div className="flex items-center gap-2 min-w-0 flex-wrap">
          <span className="text-xs font-medium text-neutral-300">{label}</span>
          {wasUpdatedViaChat && (
            <span className="inline-flex items-center gap-0.5 text-[10px] text-indigo-400 font-medium shrink-0">
              <Zap className="w-2.5 h-2.5" />
              Updated via chat
            </span>
          )}
          {accepted && (
            <Badge variant="success" className="text-[10px] px-1.5 py-0">accepted</Badge>
          )}
        </div>
        <code className="block text-xs text-neutral-200 font-mono bg-neutral-800 px-1.5 py-0.5 rounded break-words whitespace-pre-wrap">
          {String(displayValue ?? "-")}
        </code>
      </div>

      {/* Chat diff detail */}
      {wasUpdatedViaChat && (
        <div className="text-[11px] text-indigo-300/70 bg-indigo-950/30 rounded px-2 py-1">
          {chatOverride.summary}
        </div>
      )}

      {/* Reason */}
      {reason && <MarkdownBody>{reason}</MarkdownBody>}

      {/* Override inline form */}
      {overriding && (
        <div className="flex gap-1.5">
          <input
            className="flex-1 text-xs bg-neutral-800 border border-neutral-700 rounded px-2 py-1 text-neutral-200 focus:outline-none focus:border-indigo-500"
            placeholder="Enter new value…"
            value={overrideInput}
            onChange={(e) => setOverrideInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                onOverride?.(overrideInput)
                setOverriding(false)
                setOverrideInput("")
              }
              if (e.key === "Escape") {
                setOverriding(false)
                setOverrideInput("")
              }
            }}
            autoFocus
          />
          <Button
            size="sm"
            variant="default"
            className="h-6 text-[10px] px-2"
            onClick={() => {
              onOverride?.(overrideInput)
              setOverriding(false)
              setOverrideInput("")
            }}
          >
            Apply
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="h-6 text-[10px] px-2"
            onClick={() => { setOverriding(false); setOverrideInput("") }}
          >
            Cancel
          </Button>
        </div>
      )}

      {/* Action buttons - hidden when accepted */}
      {!accepted && !overriding && (
        <div className="flex items-center gap-1.5 pt-0.5">
          {onAccept && (
            <Button
              size="sm"
              variant="outline"
              className="h-6 text-[10px] px-2 border-emerald-900 text-emerald-400 hover:bg-emerald-950/40"
              onClick={onAccept}
            >
              Accept
            </Button>
          )}
          {onOverride && (
            <Button
              size="sm"
              variant="ghost"
              className="h-6 text-[10px] px-2 text-neutral-400 hover:text-neutral-200"
              onClick={() => setOverriding(true)}
            >
              Override
            </Button>
          )}
          {onAskChat && (
            <Button
              size="sm"
              variant="ghost"
              className="h-6 text-[10px] px-2 text-indigo-400 hover:text-indigo-300"
              onClick={() =>
                onAskChat(
                  `Explain the ${label} decision: current value is "${String(displayValue)}". ${reason ?? ""}`,
                )
              }
            >
              Ask co-pilot
            </Button>
          )}
        </div>
      )}
    </div>
  )
}
