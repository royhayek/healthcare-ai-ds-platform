"use client"

import { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import useSWR from "swr"
import Link from "next/link"
import {
  ArrowLeft,
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  ChevronRight,
  BarChart2,
  FileText,
  Zap,
  Shield,
  AlertTriangle,
  TrendingUp,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { RunPlotGrid } from "@/components/checkpoints/RunPlotGrid"
import { fetcher, resumeRun } from "@/lib/api"
import { useChatStore } from "@/store/chatStore"
import { useJobStore } from "@/store/jobStore"
import type {
  Run,
  StabilityResult,
  SHAPSummary,
  ThresholdResult,
  CalibrationReport,
  DriftReport,
  FairnessReport,
} from "@/lib/types"

const STATUS_POLL_MS = 2000

// ── Step order determines feed ordering ───────────────────────────────────────

const PIPELINE_STEPS = [
  "load",
  "profile",
  "eda",
  "preprocessing",
  "model_selection",
  "training",
  "tuning",
  "calibration",
  "threshold",
  "shap",
  "similarity",
  "drift",
  "fairness",
  "holdout",
  "insight",
  "deliverables",
] as const

const STEP_LABELS: Record<string, string> = {
  load: "Loading dataset",
  profile: "Profiling columns",
  eda: "Exploratory data analysis",
  checkpoint_1_eda: "Checkpoint 1 - EDA review",
  preprocessing: "Preprocessing strategy",
  checkpoint_2_preprocessing: "Checkpoint 2 - Preprocessing review",
  model_selection: "Model selection",
  checkpoint_3_model_selection: "Checkpoint 3 - Model selection review",
  training: "Training candidates (3 seeds × 5 folds)",
  checkpoint_4_training: "Checkpoint 4 - Training review",
  tuning: "Hyperparameter tuning (Optuna)",
  calibration: "Probability calibration",
  threshold: "Threshold optimisation",
  shap: "SHAP explanations",
  similarity: "Similarity index",
  drift: "Drift detection",
  fairness: "Fairness audit",
  holdout: "Holdout evaluation",
  insight: "Insight report (the model)",
  checkpoint_5_final: "Checkpoint 5 - Final review",
  deliverables: "Generating deliverables",
}

// ── Derived narrative events from Run fields ───────────────────────────────────

interface NarrativeEvent {
  key: string
  label: string
  status: "done" | "running" | "pending" | "skipped"
  lines: string[]
  severity?: "ok" | "warn" | "error"
}

function fmt(n: number, decimals = 3) {
  return n.toFixed(decimals)
}

function deriveEvents(run: Run): NarrativeEvent[] {
  const events: NarrativeEvent[] = []
  const step = run.current_step ?? ""
  const stepIdx = PIPELINE_STEPS.indexOf(step as typeof PIPELINE_STEPS[number])

  function isDone(s: string) {
    const i = PIPELINE_STEPS.indexOf(s as typeof PIPELINE_STEPS[number])
    if (i === -1) return run.status === "completed"
    if (run.status === "completed") return true
    return i < stepIdx
  }

  function isRunning(s: string) {
    return run.current_step === s && run.status === "running"
  }

  // 1. Load + Profile
  const profileDone = isDone("profile") || !!run.eda_report
  const profileLines: string[] = []
  if (run.eda_report) {
    const r = run.eda_report as Record<string, unknown>
    const overview = r.overview
    if (typeof overview === "string" && overview) profileLines.push(overview.slice(0, 200))
    const targetAnalysis = r.target_analysis as Record<string, unknown> | undefined
    if (targetAnalysis) {
      const tt = targetAnalysis.task_type as string | undefined
      const tc = targetAnalysis.target_column as string | undefined
      if (tc && tt) profileLines.push(`Target: "${tc}" - ${tt.replace(/_/g, " ")}`)
      const dist = targetAnalysis.class_distribution as Record<string, number> | undefined
      if (dist) {
        const entries = Object.entries(dist)
          .map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`)
          .join(", ")
        profileLines.push(`Class distribution: ${entries}`)
      }
      const imbalance = targetAnalysis.class_imbalance_ratio as number | undefined
      if (imbalance && imbalance > 2)
        profileLines.push(`⚠ Class imbalance ratio ${fmt(imbalance, 2)}× - imbalance handling recommended`)
    }
  }
  events.push({
    key: "profile",
    label: "Dataset loaded & profiled",
    status: profileDone ? "done" : isRunning("load") || isRunning("profile") ? "running" : "pending",
    lines: profileLines,
  })

  // 2. EDA
  const edaDone = isDone("eda") || !!run.eda_report
  const edaLines: string[] = []
  if (run.eda_report) {
    const r = run.eda_report as Record<string, unknown>
    const issues = r.quality_issues as Array<Record<string, unknown>> | undefined
    if (issues?.length) {
      edaLines.push(`${issues.length} data quality issue${issues.length > 1 ? "s" : ""} found:`)
      issues.slice(0, 4).forEach((q) => {
        const col = q.column as string | undefined
        const desc = q.description as string | undefined
        if (col && desc) edaLines.push(`  • ${col}: ${desc}`)
      })
    }
    const corrRaw = r.correlations
    if (typeof corrRaw === "string" && corrRaw) edaLines.push(corrRaw.slice(0, 180))
    const summary = r.summary
    if (typeof summary === "string" && summary) edaLines.push(summary.slice(0, 240))
  }
  events.push({
    key: "eda",
    label: "EDA - findings from the model",
    status: edaDone ? "done" : isRunning("eda") ? "running" : "pending",
    lines: edaLines,
  })

  // 3. Preprocessing
  const prepDone = isDone("preprocessing") || !!run.preprocessing_strategy
  const prepLines: string[] = []
  if (run.preprocessing_strategy) {
    const s = run.preprocessing_strategy
    const kept = Object.entries(s.columns).filter(([, c]) => c.action === "keep").length
    const dropped = Object.entries(s.columns).filter(([, c]) => c.action === "drop")
    prepLines.push(`${kept} columns kept, ${dropped.length} dropped`)
    if (dropped.length) {
      dropped.slice(0, 3).forEach(([col, c]) => {
        prepLines.push(`  • Dropped "${col}": ${c.reason}`)
      })
    }
    if (s.drop_high_correlation?.length) {
      prepLines.push(`High-correlation drops: ${s.drop_high_correlation.join(", ")}`)
    }
    const imputationCols = Object.entries(s.columns)
      .filter(([, c]) => c.action === "keep" && c.impute_strategy)
      .slice(0, 3)
    if (imputationCols.length) {
      imputationCols.forEach(([col, c]) => {
        prepLines.push(`  • "${col}": ${c.impute_strategy} imputation - ${c.reason}`)
      })
    }
    if (s.notes) prepLines.push(s.notes)
  }
  events.push({
    key: "preprocessing",
    label: "Preprocessing decisions",
    status: prepDone ? "done" : isRunning("preprocessing") ? "running" : "pending",
    lines: prepLines,
  })

  // 4. Model Selection
  const selDone = isDone("model_selection") || !!run.model_selection
  const selLines: string[] = []
  if (run.model_selection) {
    const s = run.model_selection
    selLines.push(`Candidates: ${s.candidates.join(", ")}`)
    selLines.push(`Primary metric: ${s.primary_metric}`)
    if (s.excluded?.length) {
      s.excluded.slice(0, 3).forEach((e) => {
        selLines.push(`  • Excluded ${e.name}: ${e.reason}`)
      })
    }
    if (s.reasoning) selLines.push(s.reasoning.slice(0, 240))
  }
  events.push({
    key: "model_selection",
    label: "Model candidates selected",
    status: selDone ? "done" : isRunning("model_selection") ? "running" : "pending",
    lines: selLines,
  })

  // 5. Training
  const trainDone = isDone("training") || !!run.model_comparison
  const trainLines: string[] = []
  if (run.model_comparison?.length) {
    trainLines.push("Stability results (3 seeds × 5 folds):")
    ;(run.model_comparison as StabilityResult[])
      .sort((a, b) => b.mean - a.mean)
      .slice(0, 5)
      .forEach((r) => {
        const overfit = r.overfit_gap > 0.15 ? " ⚠ overfit" : ""
        const winner = r.model_name === run.best_model_name ? " ← winner" : ""
        trainLines.push(
          `  • ${r.model_name}: ${fmt(r.mean)} ± ${fmt(r.std, 4)}${overfit}${winner}`,
        )
      })
    if (run.stat_tests) {
      const st = run.stat_tests as Record<string, unknown>
      const pval = st.p_value as number | undefined
      const ma = st.model_a as string | undefined
      const mb = st.model_b as string | undefined
      if (pval !== undefined && ma && mb)
        trainLines.push(
          `Stat test (${ma} vs ${mb}): p=${fmt(pval, 4)} - ${pval < 0.05 ? "significant difference" : "no significant difference"}`,
        )
    }
  }
  events.push({
    key: "training",
    label: "Candidate training complete",
    status: trainDone ? "done" : isRunning("training") ? "running" : "pending",
    lines: trainLines,
  })

  // 6. Tuning
  const tuneDone = isDone("tuning") || !!run.tuning_result
  const tuneLines: string[] = []
  if (run.tuning_result) {
    const t = run.tuning_result as Record<string, unknown>
    const nTrials = t.n_trials as number | undefined
    const bestScore = t.best_score as number | undefined
    const improvement = t.improvement_over_baseline as number | undefined
    const bestParams = t.best_params as Record<string, unknown> | undefined
    if (nTrials) tuneLines.push(`Optuna: ${nTrials} trials`)
    if (bestScore !== undefined) tuneLines.push(`Best ${t.metric ?? "score"}: ${fmt(bestScore)}`)
    if (improvement !== undefined)
      tuneLines.push(`Improvement over CV baseline: +${fmt(improvement, 4)}`)
    if (bestParams) {
      const paramStr = Object.entries(bestParams)
        .slice(0, 4)
        .map(([k, v]) => `${k}=${v}`)
        .join(", ")
      tuneLines.push(`Best params: ${paramStr}`)
    }
  }
  events.push({
    key: "tuning",
    label: "Hyperparameter tuning",
    status: tuneDone ? "done" : isRunning("tuning") ? "running" : "pending",
    lines: tuneLines,
  })

  // 7. Calibration
  const calDone = isDone("calibration") || !!run.calibration_report
  const calLines: string[] = []
  if (run.calibration_report) {
    const c = run.calibration_report as CalibrationReport
    calLines.push(`Method: ${c.method}`)
    calLines.push(`Brier score: ${fmt(c.brier_before)} → ${fmt(c.brier_after)} (${c.improvement_pct > 0 ? "+" : ""}${fmt(c.improvement_pct, 1)}%)`)
    calLines.push(`ECE: ${fmt(c.ece_before, 4)} → ${fmt(c.ece_after, 4)}`)
  }
  events.push({
    key: "calibration",
    label: "Probability calibration",
    status: calDone ? "done" : isRunning("calibration") ? "running" : "pending",
    lines: calLines,
  })

  // 8. Threshold
  const thrDone = isDone("threshold") || !!run.threshold_result
  const thrLines: string[] = []
  if (run.threshold_result) {
    const t = run.threshold_result as ThresholdResult
    thrLines.push(`Default threshold 0.50 → optimal ${fmt(t.optimal_threshold, 3)}`)
    if (t.cost_at_default && t.cost_at_optimal)
      thrLines.push(
        `Cost: ${fmt(t.cost_at_default)} at 0.5 → ${fmt(t.cost_at_optimal)} at ${fmt(t.optimal_threshold, 3)} (${fmt(t.improvement_pct, 1)}% saving)`,
      )
    if (t.note) thrLines.push(t.note)
  }
  events.push({
    key: "threshold",
    label: "Threshold optimisation",
    status: thrDone ? "done" : isRunning("threshold") ? "running" : "pending",
    lines: thrLines,
  })

  // 9. SHAP
  const shapDone = isDone("shap") || !!run.shap_summary
  const shapLines: string[] = []
  if (run.shap_summary) {
    const s = run.shap_summary as SHAPSummary
    shapLines.push(`Explainer: ${s.explainer_type} on ${s.n_samples} samples`)
    shapLines.push("Top predictors:")
    s.top_k_features.slice(0, 5).forEach((f, i) => {
      const val = s.mean_abs_shap[i]
      shapLines.push(`  ${i + 1}. ${f}${val !== undefined ? ` (mean |SHAP| = ${fmt(val, 4)})` : ""}`)
    })
  }
  events.push({
    key: "shap",
    label: "SHAP feature explanations",
    status: shapDone ? "done" : isRunning("shap") ? "running" : "pending",
    lines: shapLines,
  })

  // 10. Drift
  const driftDone = isDone("drift") || !!run.drift_report
  const driftLines: string[] = []
  let driftSeverity: NarrativeEvent["severity"] = "ok"
  if (run.drift_report) {
    const d = run.drift_report as DriftReport
    driftSeverity = d.overall_severity === "significant" ? "warn" : "ok"
    driftLines.push(`Overall severity: ${d.overall_severity} (aggregate PSI = ${fmt(d.aggregate_psi, 4)})`)
    driftLines.push(`${d.n_features_drifted} / ${d.features.length} features show drift`)
    if (d.significant_features?.length)
      driftLines.push(`Most drifted: ${d.significant_features.slice(0, 5).join(", ")}`)
    if (d.warning) driftLines.push(`⚠ ${d.warning}`)
  } else if (driftDone) {
    driftLines.push("No comparison dataset - drift check skipped")
  }
  events.push({
    key: "drift",
    label: "Drift detection",
    status: run.drift_report ? "done" : driftDone ? "skipped" : isRunning("drift") ? "running" : "pending",
    lines: driftLines,
    severity: driftSeverity,
  })

  // 11. Fairness
  const fairnessDone = isDone("fairness") || !!run.fairness_report
  const fairnessLines: string[] = []
  let fairnessSeverity: NarrativeEvent["severity"] = "ok"
  if (run.fairness_report) {
    const f = run.fairness_report as FairnessReport
    const sev = f.overall_severity
    fairnessSeverity = sev === "severe" || sev === "moderate" ? "warn" : "ok"
    fairnessLines.push(`Overall severity: ${sev}`)
    f.attributes.slice(0, 3).forEach((a) => {
      fairnessLines.push(`  • ${a.attribute}: DP diff = ${fmt(a.demographic_parity_diff, 4)} (${a.severity})`)
    })
    if (f.blocks_deliverables)
      fairnessLines.push("⚠ Severe disparity detected - deliverables blocked until acknowledged")
  } else if (fairnessDone) {
    fairnessLines.push("No protected attributes configured - fairness audit skipped")
  }
  events.push({
    key: "fairness",
    label: "Fairness audit",
    status: run.fairness_report ? "done" : fairnessDone ? "skipped" : isRunning("fairness") ? "running" : "pending",
    lines: fairnessLines,
    severity: fairnessSeverity,
  })

  // 12. Insight
  const insightDone = isDone("insight") || !!run.insight_report
  const insightLines: string[] = []
  if (run.insight_report) {
    // Show first 400 chars of the insight report
    const excerpt = run.insight_report.replace(/#+\s/g, "").slice(0, 400)
    insightLines.push(excerpt + (run.insight_report.length > 400 ? "…" : ""))
  }
  events.push({
    key: "insight",
    label: "Insight report - the model",
    status: insightDone ? "done" : isRunning("insight") ? "running" : "pending",
    lines: insightLines,
  })

  return events
}

// ── Components ─────────────────────────────────────────────────────────────────

function StatusIcon({ status }: { status: Run["status"] }) {
  if (status === "completed") return <CheckCircle2 className="w-4 h-4 text-emerald-400" />
  if (status === "failed") return <XCircle className="w-4 h-4 text-red-400" />
  if (status === "running") return <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
  return <Clock className="w-4 h-4 text-neutral-500" />
}

function ProgressBar({ value }: { value: number }) {
  return (
    <div className="w-full bg-neutral-800 rounded-full h-1.5 overflow-hidden">
      <div
        className="h-full bg-emerald-500 rounded-full transition-all duration-500"
        style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
      />
    </div>
  )
}

function EventIcon({ status, severity }: { status: NarrativeEvent["status"]; severity?: NarrativeEvent["severity"] }) {
  if (status === "running") return <Loader2 className="w-3.5 h-3.5 text-blue-400 animate-spin flex-shrink-0 mt-0.5" />
  if (status === "pending") return <div className="w-3.5 h-3.5 rounded-full border border-neutral-700 flex-shrink-0 mt-0.5" />
  if (status === "skipped") return <div className="w-3.5 h-3.5 rounded-full border border-neutral-700 bg-neutral-800 flex-shrink-0 mt-0.5" />
  if (severity === "warn") return <AlertTriangle className="w-3.5 h-3.5 text-yellow-400 flex-shrink-0 mt-0.5" />
  if (severity === "error") return <XCircle className="w-3.5 h-3.5 text-red-400 flex-shrink-0 mt-0.5" />
  return <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0 mt-0.5" />
}

function NarrativeStep({
  event,
  extra,
}: {
  event: NarrativeEvent
  extra?: React.ReactNode
}) {
  const dimmed = event.status === "pending"
  const [expanded, setExpanded] = useState(false)

  // Collapse long detail blocks behind a "See more" toggle. Triggers on either
  // many lines or a lot of text overall (some sections emit one very long line,
  // e.g. preprocessing notes), so both shapes get clamped uniformly.
  const totalChars = event.lines.reduce((sum, l) => sum + l.length, 0)
  const isLong = event.lines.length > 6 || totalChars > 480
  const showLines = event.lines.length > 0 && event.status !== "pending"

  return (
    <div className={`flex gap-3 ${dimmed ? "opacity-30" : ""}`}>
      <div className="flex flex-col items-center">
        <EventIcon status={event.status} severity={event.severity} />
        <div className="w-px flex-1 bg-neutral-800 mt-1" />
      </div>
      <div className="pb-4 flex-1 min-w-0">
        <p className={`text-xs font-medium ${event.status === "running" ? "text-blue-300" : event.status === "pending" ? "text-neutral-600" : "text-neutral-200"}`}>
          {event.label}
        </p>
        {showLines && (
          <>
            <div
              className={`mt-1 space-y-0.5 relative ${
                isLong && !expanded ? "max-h-36 overflow-hidden" : ""
              }`}
            >
              {event.lines.map((line, i) => (
                <p key={i} className="text-xs text-neutral-500 leading-relaxed">{line}</p>
              ))}
              {isLong && !expanded && (
                <div className="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-neutral-950 to-transparent" />
              )}
            </div>
            {isLong && (
              <button
                type="button"
                onClick={() => setExpanded((v) => !v)}
                className="mt-1 inline-flex items-center gap-0.5 text-xs font-medium text-blue-400 hover:text-blue-300 transition-colors"
              >
                {expanded ? "See less" : "See more"}
                <ChevronRight className={`h-3 w-3 transition-transform ${expanded ? "-rotate-90" : "rotate-90"}`} />
              </button>
            )}
          </>
        )}
        {extra && event.status === "done" && (
          <div className="mt-3">{extra}</div>
        )}
      </div>
    </div>
  )
}

function MetricPill({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className={`rounded-lg px-3 py-2 border ${highlight ? "border-emerald-800/60 bg-emerald-950/30" : "border-neutral-800 bg-neutral-900"}`}>
      <p className="text-xs text-neutral-500 mb-0.5">{label}</p>
      <p className={`text-sm font-semibold tabular-nums ${highlight ? "text-emerald-300" : "text-neutral-100"}`}>{value}</p>
    </div>
  )
}

function CompletedSummary({ run, projectId, runId }: { run: Run; projectId: string; runId: string }) {
  const m = run.final_metrics
  const shap = run.shap_summary as SHAPSummary | null
  const threshold = (run.threshold_result as ThresholdResult | null)?.optimal_threshold

  const metricPills: Array<{ label: string; value: string; highlight?: boolean }> = []
  if (m) {
    if (m.auc !== undefined) metricPills.push({ label: "AUC (test)", value: m.auc.toFixed(4), highlight: true })
    if (m.f1 !== undefined) metricPills.push({ label: "F1 (test)", value: m.f1.toFixed(4) })
    if (m.accuracy !== undefined) metricPills.push({ label: "Accuracy", value: `${(m.accuracy * 100).toFixed(1)}%` })
    if (m.macro_auc !== undefined) metricPills.push({ label: "Macro AUC", value: m.macro_auc.toFixed(4), highlight: true })
    if (m.macro_f1 !== undefined) metricPills.push({ label: "Macro F1", value: m.macro_f1.toFixed(4) })
    if (m.rmse !== undefined) metricPills.push({ label: "RMSE (test)", value: m.rmse.toFixed(4), highlight: true })
    if (m.mae !== undefined) metricPills.push({ label: "MAE", value: m.mae.toFixed(4) })
    if (m.r2 !== undefined) metricPills.push({ label: "R²", value: m.r2.toFixed(4) })
  }
  if (threshold !== undefined) metricPills.push({ label: "Decision threshold", value: threshold.toFixed(3) })
  if (run.best_model_name) metricPills.push({ label: "Model", value: run.best_model_name })

  return (
    <div className="space-y-5">
      {/* Final metrics row */}
      {metricPills.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {metricPills.map((p) => (
            <MetricPill key={p.label} {...p} />
          ))}
        </div>
      )}

      {/* Top SHAP features */}
      {shap && shap.top_k_features.length > 0 && (
        <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-3">
          <p className="text-xs font-medium text-neutral-400 mb-2 flex items-center gap-1.5">
            <TrendingUp className="w-3.5 h-3.5" />
            Top predictors (SHAP)
          </p>
          <div className="space-y-1.5">
            {shap.top_k_features.slice(0, 5).map((f, i) => {
              const maxVal = shap.mean_abs_shap[0] || 1
              const val = shap.mean_abs_shap[i] ?? 0
              const pct = Math.round((val / maxVal) * 100)
              return (
                <div key={f} className="flex items-center gap-2">
                  <span className="text-xs text-neutral-500 w-4 text-right tabular-nums">{i + 1}.</span>
                  <span className="text-xs text-neutral-300 w-36 truncate" title={f}>{f}</span>
                  <div className="flex-1 bg-neutral-800 rounded-full h-1.5">
                    <div
                      className="h-full bg-indigo-500 rounded-full"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="text-xs text-neutral-600 tabular-nums w-10 text-right">
                    {val.toFixed(4)}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Insight excerpt */}
      {run.insight_report && (
        <div className="rounded-lg border border-indigo-900/50 bg-indigo-950/20 p-3">
          <p className="text-xs font-medium text-indigo-300 mb-1.5 flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5" />
            Key insight
          </p>
          <p className="text-xs text-indigo-200/80 leading-relaxed">
            {run.insight_report.replace(/#+\s/g, "").slice(0, 500)}
            {run.insight_report.length > 500 ? "…" : ""}
          </p>
        </div>
      )}

      {/* Navigation tiles */}
      <div className="grid grid-cols-2 gap-2">
        <Link href={`/project/${projectId}/results?run=${runId}`}>
          <div className="rounded-lg border border-neutral-800 bg-neutral-900 hover:border-neutral-600 hover:bg-neutral-800 p-3 flex items-center gap-2.5 transition-colors cursor-pointer">
            <BarChart2 className="w-4 h-4 text-indigo-400 flex-shrink-0" />
            <div>
              <p className="text-xs font-medium text-neutral-200">Results dashboard</p>
              <p className="text-xs text-neutral-600">Metrics, SHAP, threshold, drift</p>
            </div>
          </div>
        </Link>
        <Link href={`/project/${projectId}/deliverables?run=${runId}`}>
          <div className="rounded-lg border border-neutral-800 bg-neutral-900 hover:border-neutral-600 hover:bg-neutral-800 p-3 flex items-center gap-2.5 transition-colors cursor-pointer">
            <FileText className="w-4 h-4 text-emerald-400 flex-shrink-0" />
            <div>
              <p className="text-xs font-medium text-neutral-200">Deliverables</p>
              <p className="text-xs text-neutral-600">8 documents ready to download</p>
            </div>
          </div>
        </Link>
        <Link href={`/project/${projectId}/predict?run=${runId}`}>
          <div className="rounded-lg border border-neutral-800 bg-neutral-900 hover:border-neutral-600 hover:bg-neutral-800 p-3 flex items-center gap-2.5 transition-colors cursor-pointer">
            <Zap className="w-4 h-4 text-yellow-400 flex-shrink-0" />
            <div>
              <p className="text-xs font-medium text-neutral-200">Predict</p>
              <p className="text-xs text-neutral-600">Single-row or batch inference</p>
            </div>
          </div>
        </Link>
        <Link href={`/project/${projectId}/audit?run=${runId}`}>
          <div className="rounded-lg border border-neutral-800 bg-neutral-900 hover:border-neutral-600 hover:bg-neutral-800 p-3 flex items-center gap-2.5 transition-colors cursor-pointer">
            <Shield className="w-4 h-4 text-neutral-400 flex-shrink-0" />
            <div>
              <p className="text-xs font-medium text-neutral-200">Audit log</p>
              <p className="text-xs text-neutral-600">Every decision, hash-chained</p>
            </div>
          </div>
        </Link>
      </div>
    </div>
  )
}

// ── Inline EDA plot section (shown inside the EDA narrative step) ──────────────

function InlineEDAPlots({ runId }: { runId: string }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-neutral-800/40 transition-colors"
      >
        <span className="text-xs font-medium text-neutral-400 flex items-center gap-1.5">
          <BarChart2 className="w-3 h-3" />
          EDA Plots
        </span>
        <span className="text-xs text-neutral-600">{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div className="px-3 pb-3 pt-1 border-t border-neutral-800">
          <RunPlotGrid runId={runId} stage="eda" priorityOnly />
        </div>
      )}
    </div>
  )
}

// ── Page ───────────────────────────────────────────────────────────────────────

export default function AnalysisPage() {
  const { id: projectId, run_id: runId } = useParams<{ id: string; run_id: string }>()
  const setRunId = useChatStore((s) => s.setRunId)
  const { setRunId: setJobRunId, setRun, clearJob } = useJobStore()
  const [resuming, setResuming] = useState(false)
  const [resumeError, setResumeError] = useState<string | null>(null)

  useEffect(() => {
    setRunId(runId)
    setJobRunId(runId)
    return () => clearJob()
  }, [runId, setRunId, setJobRunId, clearJob])

  const { data: run, mutate: mutateRun } = useSWR<Run>(
    runId ? `/api/proxy/runs/${runId}` : null,
    fetcher,
    {
      refreshInterval: (data) =>
        data?.status === "completed" || data?.status === "failed" ? 0 : STATUS_POLL_MS,
      revalidateOnFocus: false,
      onSuccess: (data) => setRun(data ?? null),
    },
  )

  async function handleResume() {
    setResumeError(null)
    setResuming(true)
    try {
      await resumeRun(runId)
      mutateRun()
    } catch (e) {
      setResumeError(e instanceof Error ? e.message : "Failed to resume run")
    } finally {
      setResuming(false)
    }
  }

  const isTerminal = run?.status === "completed" || run?.status === "failed"
  const events = run ? deriveEvents(run) : []

  return (
    <div className="p-8 max-w-2xl mx-auto space-y-6">
      {/* Breadcrumb */}
      <Link
        href={`/project/${projectId}`}
        className="inline-flex items-center gap-1 text-xs text-neutral-500 hover:text-neutral-300 transition-colors"
      >
        <ArrowLeft className="w-3 h-3" />
        Back to project
      </Link>

      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold text-neutral-100">Analysis run</h1>
          <p className="text-xs text-neutral-600 font-mono mt-0.5">{runId}</p>
        </div>
        {run && (
          <div className="flex items-center gap-1.5">
            <StatusIcon status={run.status} />
            <span className="text-sm text-neutral-300 capitalize">
              {run.status.replace(/_/g, " ")}
            </span>
          </div>
        )}
      </div>

      {/* Progress bar (only while running) */}
      {run && !isTerminal && (
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs text-neutral-500">
            <span>
              {run.current_step
                ? (STEP_LABELS[run.current_step] ?? run.current_step)
                : "Initialising"}
            </span>
            <span>{run.progress}%</span>
          </div>
          <ProgressBar value={run.progress} />
        </div>
      )}

      {/* Checkpoint CTA */}
      {run?.status === "awaiting_checkpoint" && run.current_step && (
        <div className="rounded-lg border border-indigo-900/60 bg-indigo-950/20 px-4 py-3 flex items-center justify-between gap-4">
          <div>
            <p className="text-sm text-indigo-200 font-medium">
              {STEP_LABELS[run.current_step] ?? run.current_step}
            </p>
            <p className="text-xs text-indigo-400/70 mt-0.5">
              The pipeline is paused. Review the AI's decisions and confirm or override before continuing.
            </p>
          </div>
          <Link href={`/project/${projectId}/analysis/${runId}/checkpoint/${run.current_step}`}>
            <Button
              size="sm"
              className="bg-indigo-700 hover:bg-indigo-600 text-white shrink-0"
            >
              Review decisions
              <ChevronRight className="w-3.5 h-3.5 ml-1" />
            </Button>
          </Link>
        </div>
      )}

      {/* Error state */}
      {run?.status === "failed" && (
        <div className="rounded-lg border border-red-900/60 bg-red-950/20 px-4 py-3 space-y-3">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <p className="text-sm text-red-300 font-medium">Pipeline failed</p>
              {run.error_message && (
                <p className="text-xs text-red-400/80 mt-1 font-mono break-words">{run.error_message}</p>
              )}
            </div>
            <Button
              size="sm"
              onClick={handleResume}
              disabled={resuming}
              className="bg-red-800 hover:bg-red-700 text-white shrink-0"
            >
              {resuming ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                  Resuming…
                </>
              ) : (
                "Retry failed step"
              )}
            </Button>
          </div>
          {resumeError && (
            <p className="text-xs text-red-400 font-mono">{resumeError}</p>
          )}
        </div>
      )}

      {/* Narrative event feed */}
      {!run ? (
        <div className="text-xs text-neutral-600">Connecting…</div>
      ) : (
        <div className="pt-2">
          {events.map((event) => (
            <NarrativeStep
              key={event.key}
              event={event}
              extra={
                event.key === "eda" ? (
                  <InlineEDAPlots runId={runId} />
                ) : undefined
              }
            />
          ))}
        </div>
      )}

      {/* Completed summary */}
      {run?.status === "completed" && (
        <div className="border-t border-neutral-800 pt-6">
          <p className="text-sm font-medium text-neutral-200 mb-4">Run complete</p>
          <CompletedSummary run={run} projectId={projectId} runId={runId} />
        </div>
      )}

      {/* Active hint */}
      {run && !isTerminal && run.status === "running" && (
        <p className="text-xs text-neutral-600">
          The co-pilot on the right is active. Ask what was found in the EDA, or direct
          the pipeline - e.g. "use class_weight instead of SMOTE".
        </p>
      )}
    </div>
  )
}
