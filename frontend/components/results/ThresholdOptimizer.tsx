"use client"

import type { ThresholdResult } from "@/lib/types"
import { TERM } from "@/lib/terminology"

interface Props {
  result: ThresholdResult
  /** Optional: the clinical outcome name shown in cost context labels. */
  outcomeName?: string
}

// Clinical labels for the metric table keys
const CLINICAL_METRIC_LABELS: Record<string, string> = {
  sensitivity: "Sensitivity (detection rate)",
  specificity: "Specificity",
  precision: "Precision (PPV)",
  f1: "F1 score",
  tp: "True positives (detected cases)",
  fp: "False positives (unnecessary referrals)",
  fn: "False negatives (missed cases)",
  tn: "True negatives",
  tpr: "True positive rate",
  fpr: "False positive rate",
}

export function ThresholdOptimizer({ result, outcomeName = "outcome" }: Props) {
  if (!result) return null

  const curve: Array<{ threshold: number; cost: number }> =
    (result as unknown as { cost_curve?: Array<{ threshold: number; cost: number }> }).cost_curve ?? []

  const improvement = result.improvement_pct
  const fnRate = result.metric_at_optimal["fn"] ?? result.metric_at_optimal["fnr"] ?? null
  const sensitivity = result.metric_at_optimal["sensitivity"] ?? result.metric_at_optimal["tpr"] ?? null

  return (
    <div className="space-y-4">
      {/* Clinical context header */}
      <div className="rounded border border-zinc-800 bg-zinc-900/50 px-3 py-2 text-xs text-zinc-400 space-y-1">
        <p>
          <span className="text-zinc-300 font-medium">{TERM.threshold}:</span>{" "}
          optimised using the clinical cost matrix (FN cost weighted higher than FP for a missed-case-sensitive context).
          This is the operating threshold for {outcomeName} risk stratification.
        </p>
        <p className="text-zinc-500">
          {TERM.fn_cost_clinical} &bull; {TERM.fp_cost_clinical}
        </p>
      </div>

      {/* Key metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <Metric label="Clinical threshold" value={result.optimal_threshold.toFixed(3)} highlight />
        <Metric label="Clinical cost (optimised)" value={result.cost_at_optimal.toFixed(2)} />
        <Metric label="Clinical cost (default 0.5)" value={result.cost_at_default.toFixed(2)} />
        <Metric
          label="Cost reduction"
          value={`${improvement > 0 ? "+" : ""}${improvement.toFixed(1)}%`}
          positive={improvement > 0}
        />
      </div>

      {/* Clinical safety callouts */}
      {(fnRate != null || sensitivity != null) && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {sensitivity != null && (
            <div className="rounded border border-emerald-800/40 bg-emerald-950/20 px-3 py-2">
              <p className="text-[11px] text-zinc-500 mb-0.5">Detection rate at this threshold</p>
              <p className="text-xl font-mono font-semibold text-emerald-400">
                {(sensitivity * 100).toFixed(1)}%
              </p>
              <p className="text-[11px] text-zinc-500 mt-0.5">
                of patients with {outcomeName} will be flagged
              </p>
            </div>
          )}
          {fnRate != null && (
            <div className="rounded border border-amber-800/40 bg-amber-950/20 px-3 py-2">
              <p className="text-[11px] text-zinc-500 mb-0.5">Missed cases at this threshold</p>
              <p className="text-xl font-mono font-semibold text-amber-400">
                {typeof fnRate === "number" && fnRate < 1
                  ? `${(fnRate * 100).toFixed(1)}%`
                  : String(fnRate)}
              </p>
              <p className="text-[11px] text-zinc-500 mt-0.5">
                patients with {outcomeName} not flagged - review clinically
              </p>
            </div>
          )}
        </div>
      )}

      {/* Per-metric table at optimal threshold */}
      {Object.keys(result.metric_at_optimal).length > 0 && (
        <div className="overflow-x-auto">
          <p className="text-[11px] text-zinc-500 mb-1.5">All metrics at clinical threshold</p>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-zinc-500 border-b border-zinc-800">
                {Object.keys(result.metric_at_optimal).map((k) => (
                  <th key={k} className="pb-1 pr-4 font-medium text-right first:text-left">
                    {CLINICAL_METRIC_LABELS[k.toLowerCase()] ?? k}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                {Object.values(result.metric_at_optimal).map((v, i) => (
                  <td key={i} className="py-1 pr-4 font-mono text-zinc-300 text-right first:text-left">
                    {typeof v === "number" ? v.toFixed(4) : String(v)}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {/* Mini cost curve sparkline */}
      {curve.length > 0 && (
        <CostCurveSparkline curve={curve} optimal={result.optimal_threshold} />
      )}

      {result.note && (
        <p className="text-xs text-zinc-500">{result.note}</p>
      )}
    </div>
  )
}

function Metric({
  label,
  value,
  highlight,
  positive,
}: {
  label: string
  value: string
  highlight?: boolean
  positive?: boolean
}) {
  return (
    <div className="bg-zinc-800/50 rounded-lg p-3 space-y-1">
      <p className="text-[11px] text-zinc-500">{label}</p>
      <p
        className={`text-lg font-mono font-semibold ${
          highlight ? "text-blue-400" : positive === true ? "text-green-400" : positive === false ? "text-red-400" : "text-zinc-200"
        }`}
      >
        {value}
      </p>
    </div>
  )
}

function CostCurveSparkline({
  curve,
  optimal,
}: {
  curve: Array<{ threshold: number; cost: number }>
  optimal: number
}) {
  const W = 400
  const H = 80
  const pad = 8

  const costs = curve.map((c) => c.cost)
  const minC = Math.min(...costs)
  const maxC = Math.max(...costs)
  const rangeC = maxC - minC || 1

  const points = curve.map((c, i) => {
    const x = pad + ((i / (curve.length - 1)) * (W - 2 * pad))
    const y = pad + ((1 - (c.cost - minC) / rangeC) * (H - 2 * pad))
    return `${x},${y}`
  })

  const optIdx = curve.reduce(
    (best, c, i) => (Math.abs(c.threshold - optimal) < Math.abs(curve[best].threshold - optimal) ? i : best),
    0,
  )
  const optX = pad + ((optIdx / (curve.length - 1)) * (W - 2 * pad))

  return (
    <div className="space-y-1">
      <p className="text-[11px] text-zinc-500">Cost curve (lower is better)</p>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full h-20 rounded bg-zinc-900"
        aria-label="Cost curve sparkline"
      >
        <polyline
          points={points.join(" ")}
          fill="none"
          stroke="#3b82f6"
          strokeWidth="1.5"
        />
        <line
          x1={optX}
          y1={pad}
          x2={optX}
          y2={H - pad}
          stroke="#10b981"
          strokeWidth="1"
          strokeDasharray="3 2"
        />
        <text x={optX + 3} y={pad + 10} fill="#10b981" fontSize="9" fontFamily="monospace">
          {optimal.toFixed(2)}
        </text>
      </svg>
    </div>
  )
}
