"use client"

import { useEffect, useState } from "react"
import { useParams, useSearchParams } from "next/navigation"
import useSWR from "swr"
import Link from "next/link"
import { fetcher } from "@/lib/api"
import type { Run } from "@/lib/types"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ModelComparison } from "@/components/results/ModelComparison"
import { ShapPlot } from "@/components/results/ShapPlot"
import { ThresholdOptimizer } from "@/components/results/ThresholdOptimizer"
import { ConfusionMatrix } from "@/components/results/ConfusionMatrix"
import { InsightReport } from "@/components/results/InsightReport"
import { DriftReportPanel } from "@/components/results/DriftReport"
import { RunPlotGrid } from "@/components/checkpoints/RunPlotGrid"
import { FairnessReportPanel } from "@/components/results/FairnessReport"
import { RocCurve } from "@/components/results/RocCurve"
import { PrCurve } from "@/components/results/PrCurve"
import { BinaryConfusionMatrix, MultiConfusionMatrix } from "@/components/results/ConfusionMatrixPlot"
import { ScoreDistribution } from "@/components/results/ScoreDistribution"
import { CalibrationCurve } from "@/components/results/CalibrationCurve"
import { PredictedVsActual, ResidualsPlot } from "@/components/results/RegressionPlots"

export default function ResultsPage() {
  const { id: projectId } = useParams<{ id: string }>()
  const searchParams = useSearchParams()
  const runIdParam = searchParams.get("run_id")

  const [activeRunId, setActiveRunId] = useState<string | null>(runIdParam)

  const { data: runsList } = useSWR<Run[]>(
    !runIdParam ? `/api/proxy/projects/${projectId}/runs` : null,
    fetcher,
  )

  useEffect(() => {
    if (!runIdParam && runsList) {
      const completed = runsList.filter((r) => r.status === "completed")
      if (completed.length > 0) {
        setActiveRunId(completed[completed.length - 1].id)
      }
    }
  }, [runsList, runIdParam])

  const { data: run, isLoading } = useSWR<Run>(
    activeRunId ? `/api/proxy/runs/${activeRunId}` : null,
    fetcher,
  )

  if (!activeRunId || isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-zinc-500">
        {!activeRunId ? "No completed run found. Start an analysis first." : "Loading results…"}
      </div>
    )
  }

  if (!run) return null

  if (run.status !== "completed") {
    return (
      <div className="flex items-center justify-center h-64 text-zinc-500">
        Run is not completed (status: {run.status}).{" "}
        <Link href={`/project/${projectId}/analysis/${run.id}`} className="text-blue-400 ml-1 underline">
          View progress
        </Link>
      </div>
    )
  }

  const taskType = (run.model_selection as Record<string, unknown> | null)?.task_type as string | undefined ?? "binary_classification"
  const isClassification = taskType !== "regression"

  return (
    <div className="max-w-5xl mx-auto px-4 py-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-zinc-100">Results</h1>
          <p className="text-sm text-zinc-500 mt-0.5">
            Run <code className="font-mono text-zinc-400">{run.id.slice(0, 8)}…</code>
            {run.completed_at && (
              <> &mdash; completed {new Date(run.completed_at).toLocaleString()}</>
            )}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href={`/project/${projectId}/predict?run_id=${run.id}`}>Predict</Link>
          </Button>
          <Button variant="outline" asChild>
            <Link href={`/project/${projectId}/audit?run_id=${run.id}`}>Audit log</Link>
          </Button>
          <Button variant="ghost" asChild>
            <Link href={`/project/${projectId}/deliverables?run_id=${run.id}`}>Deliverables</Link>
          </Button>
        </div>
      </div>

      {/* Best model summary */}
      {run.best_model_name && (
        <div className="flex flex-wrap items-center gap-4 p-4 rounded-lg bg-zinc-800/50 border border-zinc-700">
          <div>
            <p className="text-xs text-zinc-500">Best model</p>
            <p className="text-lg font-mono font-semibold text-zinc-100">{run.best_model_name}</p>
          </div>
          {run.best_model_score != null && (
            <div>
              <p className="text-xs text-zinc-500">Score</p>
              <p className="text-lg font-mono font-semibold text-blue-400">
                {run.best_model_score.toFixed(4)}
              </p>
            </div>
          )}
          {run.threshold_result?.optimal_threshold != null && (
            <div>
              <p className="text-xs text-zinc-500">Decision threshold</p>
              <p className="text-lg font-mono font-semibold text-green-400">
                {run.threshold_result.optimal_threshold.toFixed(3)}
              </p>
            </div>
          )}
          <Badge variant="success" className="ml-auto">completed</Badge>
        </div>
      )}

      {/* Final metrics */}
      {run.final_metrics && Object.keys(run.final_metrics).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Final test-set metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <ConfusionMatrix
              metrics={run.final_metrics as Record<string, number>}
              taskType={taskType}
            />
          </CardContent>
        </Card>
      )}

      {/* Evaluation plots */}
      {run.eval_plots && (() => {
        const plots = run.eval_plots
        const threshold = run.threshold_result?.optimal_threshold

        return (
          <>
            {/* ROC + PR curves side by side */}
            {(plots.roc_curve || plots.pr_curve) && (
              <Card>
                <CardHeader>
                  <CardTitle>Classifier curves</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {plots.roc_curve && run.final_metrics?.auc != null && (
                      <RocCurve data={plots.roc_curve} auc={run.final_metrics.auc} />
                    )}
                    {plots.pr_curve && (
                      <PrCurve data={plots.pr_curve} />
                    )}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Confusion matrix */}
            {plots.confusion_matrix && (
              <Card>
                <CardHeader>
                  <CardTitle>Confusion matrix</CardTitle>
                </CardHeader>
                <CardContent>
                  <BinaryConfusionMatrix data={plots.confusion_matrix} />
                </CardContent>
              </Card>
            )}
            {plots.confusion_matrix_multi && (
              <Card>
                <CardHeader>
                  <CardTitle>Confusion matrix</CardTitle>
                </CardHeader>
                <CardContent>
                  <MultiConfusionMatrix data={plots.confusion_matrix_multi} />
                </CardContent>
              </Card>
            )}

            {/* Score distribution + calibration side by side */}
            {(plots.score_distribution || plots.calibration_curve) && (
              <Card>
                <CardHeader>
                  <CardTitle>Score distribution &amp; calibration</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {plots.score_distribution && (
                      <ScoreDistribution data={plots.score_distribution} threshold={threshold} />
                    )}
                    {plots.calibration_curve && (
                      <CalibrationCurve data={plots.calibration_curve} />
                    )}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Regression plots */}
            {(plots.predicted_vs_actual || plots.residuals) && (
              <Card>
                <CardHeader>
                  <CardTitle>Regression diagnostics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {plots.predicted_vs_actual && (
                      <PredictedVsActual data={plots.predicted_vs_actual} />
                    )}
                    {plots.residuals && (
                      <ResidualsPlot data={plots.residuals} />
                    )}
                  </div>
                </CardContent>
              </Card>
            )}
          </>
        )
      })()}

      {/* Model comparison */}
      {run.model_comparison && run.model_comparison.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Model comparison (stability runs)</CardTitle>
          </CardHeader>
          <CardContent>
            <ModelComparison
              results={run.model_comparison}
              bestModelName={run.best_model_name}
              primaryMetric={(run.model_selection as Record<string, unknown> | null)?.primary_metric as string | undefined}
              statTests={run.stat_tests}
            />
          </CardContent>
        </Card>
      )}

      {/* Threshold optimization */}
      {isClassification && run.threshold_result && (
        <Card>
          <CardHeader>
            <CardTitle>Business-cost threshold optimization</CardTitle>
          </CardHeader>
          <CardContent>
            <ThresholdOptimizer result={run.threshold_result} />
          </CardContent>
        </Card>
      )}

      {/* SHAP feature importance */}
      {run.shap_summary && (
        <Card>
          <CardHeader>
            <CardTitle>Feature importance (SHAP)</CardTitle>
          </CardHeader>
          <CardContent>
            <ShapPlot summary={run.shap_summary} />
          </CardContent>
        </Card>
      )}

      {/* Drift - text summary + visual comparison plots (D7) */}
      {run.drift_report && (
        <Card>
          <CardHeader>
            <CardTitle>Data drift</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <DriftReportPanel report={run.drift_report} />
            {activeRunId && (
              <div>
                <p className="text-xs font-semibold text-neutral-500 uppercase tracking-wider mb-2">
                  Distribution comparison plots
                </p>
                <RunPlotGrid runId={activeRunId} stage="drift" priorityOnly />
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Fairness */}
      {run.fairness_report && (
        <Card>
          <CardHeader>
            <CardTitle>Bias &amp; fairness</CardTitle>
          </CardHeader>
          <CardContent>
            <FairnessReportPanel report={run.fairness_report} />
          </CardContent>
        </Card>
      )}

      {/* AI insight report */}
      {run.insight_report && (
        <Card>
          <CardHeader>
            <CardTitle>AI insight report</CardTitle>
          </CardHeader>
          <CardContent>
            <InsightReport report={run.insight_report} />
          </CardContent>
        </Card>
      )}
    </div>
  )
}
