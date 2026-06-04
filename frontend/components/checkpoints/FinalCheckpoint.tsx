"use client"

import type { Run } from "@/lib/types"
import { InsightReport } from "@/components/results/InsightReport"
import { ThresholdOptimizer } from "@/components/results/ThresholdOptimizer"
import { EquityReport } from "@/components/results/EquityReport"

export function FinalCheckpoint({ run }: { run: Run }) {
  const metrics = run.final_metrics
  const threshold = run.threshold_result
  const cal = run.calibration_report
  const shap = run.shap_summary
  const insight = run.insight_report
  const fairness = run.fairness_report

  // Derive a human-readable outcome name from the target column or task type
  const outcomeName =
    run.preprocessing_strategy?.target_column?.replace(/_/g, " ") ??
    (run.preprocessing_strategy?.task_type === "binary_classification" ? "outcome" : "diagnosis")

  return (
    <div className="space-y-6">
      {/* Final metrics */}
      {metrics && (
        <Section title="Test Set Metrics (sealed set, opened once)">
          <div className={`grid gap-2 ${Object.keys(metrics).length <= 3 ? "grid-cols-3" : "grid-cols-2 sm:grid-cols-4"}`}>
            {Object.entries(metrics).map(([k, v]) => (
              <StatBox
                key={k}
                label={k.toUpperCase()}
                value={typeof v === "number" ? v.toFixed(4) : String(v)}
                highlight={k === "auc" || k === "macro_auc"}
              />
            ))}
          </div>
        </Section>
      )}

      {/* Threshold optimization - clinical framing */}
      {threshold && (
        <Section title="Clinical Threshold Optimization">
          <ThresholdOptimizer result={threshold} outcomeName={outcomeName} />
        </Section>
      )}

      {/* Calibration */}
      {cal && (
        <Section title="Probability Calibration">
          <div className="grid grid-cols-2 gap-2">
            <StatBox label={`Brier (before → after)`} value={`${cal.brier_before.toFixed(4)} → ${cal.brier_after.toFixed(4)}`} />
            <StatBox label={`ECE (before → after)`} value={`${cal.ece_before.toFixed(4)} → ${cal.ece_after.toFixed(4)}`} />
          </div>
          <div className="mt-2 text-xs text-neutral-500">
            Method: {cal.method} · Improvement: {cal.improvement_pct.toFixed(1)}%
          </div>
        </Section>
      )}

      {/* SHAP top features */}
      {shap?.top_k_features && shap.top_k_features.length > 0 && (
        <Section title="Top Predictors (SHAP)">
          <div className="space-y-1">
            {shap.top_k_features.slice(0, 8).map((feat, i) => {
              const idx = shap.feature_names.indexOf(feat)
              const val = idx >= 0 ? shap.mean_abs_shap[idx] : null
              const maxVal = shap.mean_abs_shap[0] ?? 1
              const pct = val != null ? (val / maxVal) * 100 : 0

              return (
                <div key={feat} className="flex items-center gap-2">
                  <span className="text-[10px] text-neutral-500 w-4 text-right">{i + 1}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2 mb-0.5">
                      <span className="text-xs font-mono text-neutral-300 truncate">{feat}</span>
                      {val != null && (
                        <span className="text-[10px] text-neutral-500 font-mono shrink-0">
                          {val.toFixed(4)}
                        </span>
                      )}
                    </div>
                    <div className="h-1 bg-neutral-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-indigo-500 rounded-full"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
          <p className="text-[10px] text-neutral-600 mt-1">
            {shap.explainer_type}Explainer · {shap.n_samples} test samples
          </p>
        </Section>
      )}

      {/* Population equity */}
      {fairness && (
        <Section title="Population Equity Analysis">
          <EquityReport report={fairness} />
        </Section>
      )}

      {/* Insight report */}
      {insight && (
        <Section title="Insight Report">
          <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 px-4 py-4 overflow-auto max-h-96">
            <InsightReport report={insight} />
          </div>
        </Section>
      )}
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold text-neutral-400 uppercase tracking-wider">{title}</h3>
      {children}
    </div>
  )
}

function StatBox({
  label,
  value,
  highlight = false,
}: {
  label: string
  value: string
  highlight?: boolean
}) {
  return (
    <div
      className={`rounded-lg border px-3 py-2 space-y-0.5 ${
        highlight
          ? "border-emerald-900/60 bg-emerald-950/20"
          : "border-neutral-800 bg-neutral-900/50"
      }`}
    >
      <span className="text-[10px] text-neutral-500">{label}</span>
      <p className={`text-sm font-mono ${highlight ? "text-emerald-300" : "text-neutral-200"}`}>
        {value}
      </p>
    </div>
  )
}
