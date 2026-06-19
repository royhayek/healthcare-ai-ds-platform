"use client"

import type {
  RocCurveMultiSeries,
  PrCurveMultiSeries,
  CalibrationCurveMultiSeries,
} from "@/lib/types"
import { RocCurve } from "./RocCurve"
import { PrCurve } from "./PrCurve"
import { CalibrationCurve } from "./CalibrationCurve"

/**
 * Multiclass one-vs-rest curves rendered as small multiples — one chart per
 * class — reusing the binary chart components. This is the standard way to read
 * per-class discrimination/calibration in a multiclass model.
 */

function ClassLabel({ label }: { label: string }) {
  return (
    <p className="text-xs font-mono text-zinc-400 mb-1">
      <span className="text-zinc-500">class</span> {label}{" "}
      <span className="text-zinc-600">(one-vs-rest)</span>
    </p>
  )
}

export function MultiRocCurves({ series }: { series: RocCurveMultiSeries[] }) {
  if (!series?.length) return null
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {series.map((s) => (
        <div key={s.label}>
          <ClassLabel label={s.label} />
          <RocCurve data={{ fpr: s.fpr, tpr: s.tpr }} auc={s.auc} />
        </div>
      ))}
    </div>
  )
}

export function MultiPrCurves({ series }: { series: PrCurveMultiSeries[] }) {
  if (!series?.length) return null
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {series.map((s) => (
        <div key={s.label}>
          <ClassLabel label={s.label} />
          <PrCurve data={{ precision: s.precision, recall: s.recall, ap: s.ap }} />
        </div>
      ))}
    </div>
  )
}

export function MultiCalibrationCurves({
  series,
}: {
  series: CalibrationCurveMultiSeries[]
}) {
  if (!series?.length) return null
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {series.map((s) => (
        <div key={s.label}>
          <ClassLabel label={s.label} />
          <CalibrationCurve data={{ prob_true: s.prob_true, prob_pred: s.prob_pred }} />
        </div>
      ))}
    </div>
  )
}
