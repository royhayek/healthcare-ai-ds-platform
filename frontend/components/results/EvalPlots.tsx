"use client"

import type { Run } from "@/lib/types"
import { RocCurve } from "@/components/results/RocCurve"
import { PrCurve } from "@/components/results/PrCurve"
import { BinaryConfusionMatrix, MultiConfusionMatrix } from "@/components/results/ConfusionMatrixPlot"
import { MultiRocCurves, MultiPrCurves, MultiCalibrationCurves } from "@/components/results/MultiClassCurves"
import { ScoreDistribution } from "@/components/results/ScoreDistribution"
import { CalibrationCurve } from "@/components/results/CalibrationCurve"
import { PredictedVsActual, ResidualsPlot } from "@/components/results/RegressionPlots"

/**
 * Renders every available evaluation chart from run.eval_plots (ROC/PR,
 * confusion matrix, calibration, score distribution, multiclass one-vs-rest,
 * regression diagnostics). Shared between the final-review checkpoint and the
 * results dashboard so the two never drift. Each group renders only when its
 * data is present, so binary / multiclass / regression runs all work.
 */
export function EvalPlots({ run }: { run: Run }) {
  const plots = run.eval_plots
  if (!plots) return null

  const threshold = run.threshold_result?.optimal_threshold
  const auc = run.final_metrics?.auc

  return (
    <div className="space-y-4">
      {(plots.roc_curve || plots.pr_curve) && (
        <PlotCard title="Classifier curves">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {plots.roc_curve && auc != null && <RocCurve data={plots.roc_curve} auc={auc} />}
            {plots.pr_curve && <PrCurve data={plots.pr_curve} />}
          </div>
        </PlotCard>
      )}

      {plots.confusion_matrix && (
        <PlotCard title="Confusion matrix">
          <BinaryConfusionMatrix data={plots.confusion_matrix} />
        </PlotCard>
      )}
      {plots.confusion_matrix_multi && (
        <PlotCard title="Confusion matrix">
          <MultiConfusionMatrix data={plots.confusion_matrix_multi} />
        </PlotCard>
      )}

      {plots.roc_curve_multi && plots.roc_curve_multi.length > 0 && (
        <PlotCard title="ROC curves (one-vs-rest)">
          <MultiRocCurves series={plots.roc_curve_multi} />
        </PlotCard>
      )}
      {plots.pr_curve_multi && plots.pr_curve_multi.length > 0 && (
        <PlotCard title="Precision-recall curves (one-vs-rest)">
          <MultiPrCurves series={plots.pr_curve_multi} />
        </PlotCard>
      )}
      {plots.calibration_curve_multi && plots.calibration_curve_multi.length > 0 && (
        <PlotCard title="Calibration curves (one-vs-rest)">
          <MultiCalibrationCurves series={plots.calibration_curve_multi} />
        </PlotCard>
      )}

      {(plots.score_distribution || plots.calibration_curve) && (
        <PlotCard title="Score distribution & calibration">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {plots.score_distribution && (
              <ScoreDistribution data={plots.score_distribution} threshold={threshold} />
            )}
            {plots.calibration_curve && <CalibrationCurve data={plots.calibration_curve} />}
          </div>
        </PlotCard>
      )}

      {(plots.predicted_vs_actual || plots.residuals) && (
        <PlotCard title="Regression diagnostics">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {plots.predicted_vs_actual && <PredictedVsActual data={plots.predicted_vs_actual} />}
            {plots.residuals && <ResidualsPlot data={plots.residuals} />}
          </div>
        </PlotCard>
      )}
    </div>
  )
}

function PlotCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 px-4 py-3">
      <h4 className="text-xs font-semibold text-neutral-400 mb-3">{title}</h4>
      {children}
    </div>
  )
}
