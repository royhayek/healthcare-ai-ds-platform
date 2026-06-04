"use client"

import { useState } from "react"
import { TERM } from "@/lib/terminology"

export interface PhiColumn {
  column: string
  confidence: "low" | "medium" | "high"
}

interface Props {
  phiColumns: PhiColumn[]
  /** Called when the clinician has read and dismissed the warning. */
  onAcknowledge: () => void
}

const CONFIDENCE_LABEL: Record<string, string> = {
  high: "High confidence",
  medium: "Medium confidence",
  low: "Low confidence",
}

const CONFIDENCE_CLASS: Record<string, string> = {
  high: "text-red-300",
  medium: "text-amber-300",
  low: "text-zinc-400",
}

/**
 * Banner shown after dataset upload when the profiler detects likely PHI columns.
 * The user must explicitly acknowledge before the analysis proceeds.
 * Acknowledgement is recorded as an audit event server-side.
 */
export function PhiWarningBanner({ phiColumns, onAcknowledge }: Props) {
  const [acknowledged, setAcknowledged] = useState(false)

  if (phiColumns.length === 0) return null

  function handleAcknowledge() {
    setAcknowledged(true)
    onAcknowledge()
  }

  return (
    <div
      role="alert"
      aria-live="assertive"
      className="rounded-lg border border-amber-600/50 bg-amber-950/30 p-4 space-y-3"
    >
      {/* Header */}
      <div className="flex items-start gap-3">
        <span className="text-amber-400 text-lg mt-0.5" aria-hidden>⚠</span>
        <div>
          <h3 className="font-semibold text-amber-300 text-sm">{TERM.phi_warning_title}</h3>
          <p className="text-xs text-amber-200/80 mt-0.5">{TERM.phi_warning_body}</p>
        </div>
      </div>

      {/* Column list */}
      <div className="rounded border border-amber-700/30 bg-zinc-900/60 overflow-hidden">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-amber-700/20">
              <th className="text-left px-3 py-2 font-medium text-zinc-400">Column</th>
              <th className="text-right px-3 py-2 font-medium text-zinc-400">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {phiColumns.map((col) => (
              <tr key={col.column} className="border-b border-zinc-800/40 last:border-0">
                <td className="px-3 py-1.5 font-mono text-zinc-200">{col.column}</td>
                <td className={`px-3 py-1.5 text-right font-medium ${CONFIDENCE_CLASS[col.confidence]}`}>
                  {CONFIDENCE_LABEL[col.confidence]}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* What happens next */}
      <p className="text-xs text-zinc-400">
        These columns will be <span className="text-amber-300 font-medium">excluded from all AI analysis</span>.
        Their statistical summaries will not be sent to the model. This exclusion is recorded in the audit log.
      </p>

      {/* Acknowledge button */}
      {!acknowledged ? (
        <button
          type="button"
          onClick={handleAcknowledge}
          className="rounded bg-amber-600 hover:bg-amber-500 active:bg-amber-700 px-4 py-1.5 text-xs font-medium text-white transition-colors"
        >
          I understand - proceed without PHI columns
        </button>
      ) : (
        <p className="text-xs text-emerald-400 font-medium">
          ✓ Acknowledged. PHI columns excluded from analysis.
        </p>
      )}
    </div>
  )
}
