"use client"

import { TERM, RISK_TIER_CLASSES, riskTier } from "@/lib/terminology"

export interface PatientRiskResult {
  /** Calibrated probability [0,1]. */
  probability: number
  /** Binary prediction at optimal threshold. */
  prediction: 0 | 1
  /** The threshold used to produce the binary prediction. */
  threshold_used: number
  /** Top SHAP contributors that raised the risk (feature: value pairs). */
  shap_drivers: string[]
  /** Top SHAP contributors that lowered the risk. */
  shap_dampeners: string[]
  /** Similarity to training cohort [0,1]. */
  similarity_score: number | null
  /** "high" | "medium" | "low" confidence band from similarity. */
  confidence_band: "high" | "medium" | "low"
  /** Set to true when similarity < low-confidence threshold. */
  risk_flag: boolean
  /** Opaque row identifier - never the patient name. */
  row_id?: string | number
}

interface Props {
  result: PatientRiskResult
  /** Optional: the clinical outcome name shown in labels (e.g. "30-day readmission"). */
  outcomeName?: string
  /** Optional: called when clinician clicks "Flag as incorrect". Triggers audit event. */
  onOverride?: (rowId: string | number | undefined, newPrediction: 0 | 1) => void
}

const BAND_CLASSES: Record<string, string> = {
  high: "text-emerald-400",
  medium: "text-amber-400",
  low: "text-red-400",
}

const BAND_LABEL: Record<string, string> = {
  high: "High confidence",
  medium: "Medium confidence",
  low: "Low confidence - review recommended",
}

/**
 * Displays a single patient's risk prediction with clinical framing.
 * Designed to be rendered in a results table or a per-patient drill-down view.
 *
 * Shows: calibrated probability, risk tier badge, top SHAP drivers/dampeners,
 * similarity-to-cohort confidence band, and a clinician override button.
 */
export function PatientRiskCard({ result, outcomeName = "outcome", onOverride }: Props) {
  const tier = riskTier(result.probability)
  const tierStyle = RISK_TIER_CLASSES[tier]

  const pct = (result.probability * 100).toFixed(1)
  const flipped: 0 | 1 = result.prediction === 1 ? 0 : 1

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 space-y-4">
      {/* Risk score header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs text-zinc-500 uppercase tracking-wide">{TERM.prediction}</p>
          <p className="text-3xl font-mono font-bold text-zinc-100 mt-0.5">{pct}%</p>
          <p className="text-xs text-zinc-500 mt-1">
            predicted risk of {outcomeName}
          </p>
        </div>

        <div className="flex flex-col items-end gap-2">
          {/* Risk tier badge */}
          <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold ${tierStyle.badge}`}>
            {tierStyle.label}
          </span>

          {/* Confidence band */}
          <span className={`text-xs ${BAND_CLASSES[result.confidence_band]}`}>
            {BAND_LABEL[result.confidence_band]}
          </span>
        </div>
      </div>

      {/* Low-similarity flag */}
      {result.risk_flag && (
        <div className="rounded border border-amber-700/50 bg-amber-950/30 px-3 py-2 text-xs text-amber-300">
          ⚠ This patient's profile is dissimilar to the training cohort - interpret with caution.
          {result.similarity_score != null && (
            <span className="ml-1 font-mono">
              (similarity: {(result.similarity_score * 100).toFixed(0)}%)
            </span>
          )}
        </div>
      )}

      {/* SHAP drivers */}
      {(result.shap_drivers.length > 0 || result.shap_dampeners.length > 0) && (
        <div className="grid grid-cols-2 gap-3">
          {result.shap_drivers.length > 0 && (
            <div className="space-y-1.5">
              <p className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">
                Factors increasing risk
              </p>
              <ul className="space-y-1">
                {result.shap_drivers.slice(0, 3).map((d, i) => (
                  <li key={i} className="flex items-center gap-1.5 text-xs">
                    <span className="text-red-400" aria-hidden>▲</span>
                    <span className="text-zinc-300">{d}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.shap_dampeners.length > 0 && (
            <div className="space-y-1.5">
              <p className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">
                Factors reducing risk
              </p>
              <ul className="space-y-1">
                {result.shap_dampeners.slice(0, 3).map((d, i) => (
                  <li key={i} className="flex items-center gap-1.5 text-xs">
                    <span className="text-emerald-400" aria-hidden>▼</span>
                    <span className="text-zinc-300">{d}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Threshold context */}
      <p className="text-[11px] text-zinc-600">
        Predicted <strong className="text-zinc-400">{result.prediction === 1 ? "positive" : "negative"}</strong> at
        threshold {result.threshold_used.toFixed(3)} (clinical cost-optimised).
      </p>

      {/* Clinician override */}
      {onOverride && (
        <button
          type="button"
          onClick={() => onOverride(result.row_id, flipped)}
          className="text-[11px] text-zinc-500 hover:text-amber-400 underline underline-offset-2 transition-colors"
        >
          Clinician disagrees with this prediction - record override
        </button>
      )}
    </div>
  )
}
