"use client"

import { DecisionCard } from "./DecisionCard"
import { RunPlotGrid } from "./RunPlotGrid"
import { MarkdownBody } from "@/components/ui/MarkdownBody"
import type { Run } from "@/lib/types"

interface EDAReport {
  overview?: string
  target_analysis?: {
    column?: string
    task_type?: string
    class_balance?: Record<string, unknown>
    notes?: string
  }
  quality_issues?: Array<{
    column?: string
    issue?: string
    severity?: "low" | "medium" | "high"
    recommendation?: string
  }>
  correlations?: {
    high_pairs?: Array<{ col_a: string; col_b: string; correlation: number }>
    leakage_risk?: Array<{ column: string; reason: string }>
  }
  model_recommendation?: string
  summary?: string
}

export function EdaCheckpoint({ run, runId }: { run: Run; runId: string }) {
  const eda = run.eda_report as EDAReport | null
  if (!eda) return <EmptyState />

  const highIssues = eda.quality_issues?.filter((i) => i.severity === "high") ?? []
  const mediumIssues = eda.quality_issues?.filter((i) => i.severity === "medium") ?? []
  const leakageRisks = eda.correlations?.leakage_risk ?? []

  return (
    <div className="space-y-6">
      {/* Overview */}
      <Section title="Dataset Overview">
        {eda.overview
          ? <MarkdownBody>{eda.overview}</MarkdownBody>
          : <p className="text-xs text-neutral-500">No overview available.</p>}
      </Section>

      {/* Target analysis */}
      {eda.target_analysis && (
        <Section title="Target Column">
          <div className="grid grid-cols-2 gap-2">
            <StatBox label="Column" value={eda.target_analysis.column ?? "-"} />
            <StatBox label="Task type" value={eda.target_analysis.task_type ?? "-"} />
          </div>
          {eda.target_analysis.notes && (
            <MarkdownBody className="mt-1">{eda.target_analysis.notes}</MarkdownBody>
          )}
        </Section>
      )}

      {/* Model recommendation */}
      {eda.model_recommendation && (
        <Section title="Initial Model Recommendation">
          <DecisionCard
            fieldPath="eda.model_recommendation"
            label="Recommended model"
            value={eda.model_recommendation}
            reason="Based on dataset size, task type, and feature characteristics."
            severity="info"
          />
        </Section>
      )}

      {/* Leakage risks */}
      {leakageRisks.length > 0 && (
        <Section title="Leakage Risks">
          {leakageRisks.map((risk) => (
            <DecisionCard
              key={risk.column}
              fieldPath={`eda.leakage_risk.${risk.column}`}
              label={risk.column}
              value="LEAKAGE RISK"
              reason={risk.reason}
              severity="critical"
            />
          ))}
        </Section>
      )}

      {/* Quality issues */}
      {(highIssues.length > 0 || mediumIssues.length > 0) && (
        <Section title={`Quality Issues (${highIssues.length} high, ${mediumIssues.length} medium)`}>
          {[...highIssues, ...mediumIssues].map((issue, i) => (
            <DecisionCard
              key={i}
              fieldPath={`eda.quality.${issue.column ?? i}`}
              label={issue.column ?? "Dataset-level"}
              value={issue.issue ?? "unknown issue"}
              reason={issue.recommendation}
              severity={issue.severity === "high" ? "critical" : "warn"}
            />
          ))}
        </Section>
      )}

      {/* Summary */}
      {eda.summary && (
        <div className="rounded-lg border border-blue-900/40 bg-blue-950/20 px-4 py-3">
          <MarkdownBody className="text-blue-100">{eda.summary}</MarkdownBody>
        </div>
      )}

      {/* EDA plots - rendered by the pipeline after profiling */}
      <Section title="EDA Plots">
        <p className="text-xs text-zinc-500 mb-3">
          Box plots flag outliers (red dots = values beyond 1.5×IQR). Class distribution shows imbalance.
          Click any tile to expand. Priority plots shown first.
        </p>
        <RunPlotGrid runId={runId} stage="eda" priorityOnly />
      </Section>
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

function StatBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900/50 px-3 py-2 space-y-0.5">
      <span className="text-[10px] text-neutral-500">{label}</span>
      <p className="text-sm font-mono text-neutral-200">{value}</p>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 px-6 py-10 text-center">
      <p className="text-sm text-neutral-500">EDA report not yet available.</p>
    </div>
  )
}
