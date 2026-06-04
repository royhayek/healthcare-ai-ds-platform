"use client"

/** EDA findings component - renders quality issues, correlations, and recommendations
 * from the run's eda_report field (§9). */

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ExpandableSection } from "@/components/shared/ExpandableSection"

interface QualityIssue {
  column: string | null
  issue: string
  severity: "high" | "medium" | "low"
  recommendation: string
}

interface EDAReportData {
  summary: string
  model_recommendation: string
  task_type: string
  quality_issues: QualityIssue[]
  correlations?: {
    leakage_risk?: Array<{ column: string; reason: string; correlation: number }>
    high_correlation_pairs?: Array<{ col1: string; col2: string; correlation: number }>
  }
  numeric_features?: string[]
  categorical_features?: string[]
  target_analysis?: Record<string, unknown>
}

const SEVERITY_VARIANT: Record<string, "error" | "warning" | "outline"> = {
  high: "error",
  medium: "warning",
  low: "outline",
}

function IssueRow({ issue }: { issue: QualityIssue }) {
  return (
    <div className="flex items-start gap-3 py-2.5 border-b border-zinc-800 last:border-0">
      <Badge variant={SEVERITY_VARIANT[issue.severity]} className="mt-0.5 shrink-0 text-xs capitalize">
        {issue.severity}
      </Badge>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          {issue.column && (
            <span className="font-mono text-xs text-zinc-400">{issue.column}</span>
          )}
          <span className="text-sm text-zinc-300">{issue.issue}</span>
        </div>
        <p className="text-xs text-zinc-500 mt-0.5">{issue.recommendation}</p>
      </div>
    </div>
  )
}

interface EdaReportProps {
  report: EDAReportData
}

export function EdaReportPanel({ report }: EdaReportProps) {
  const highIssues = report.quality_issues.filter((q) => q.severity === "high")
  const otherIssues = report.quality_issues.filter((q) => q.severity !== "high")
  const leakageRisks = report.correlations?.leakage_risk ?? []
  const highCorrPairs = report.correlations?.high_correlation_pairs ?? []

  return (
    <div className="space-y-4">
      {/* Summary */}
      <Card className="bg-zinc-900 border-zinc-800">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base font-medium">EDA Summary</CardTitle>
            <div className="flex gap-2">
              <Badge variant="outline" className="text-xs">{report.task_type}</Badge>
              <Badge variant="default" className="text-xs">rec: {report.model_recommendation}</Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-zinc-300 leading-relaxed">{report.summary}</p>
        </CardContent>
      </Card>

      {/* High-severity issues */}
      {highIssues.length > 0 && (
        <Card className="bg-zinc-900 border-red-900/40">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-red-400">
              High-Severity Issues ({highIssues.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            {highIssues.map((q, i) => <IssueRow key={i} issue={q} />)}
          </CardContent>
        </Card>
      )}

      {/* Other issues */}
      {otherIssues.length > 0 && (
        <ExpandableSection
          label="Other Quality Issues"
          badgeCount={otherIssues.length}
          defaultOpen={highIssues.length === 0}
        >
          {otherIssues.map((q, i) => <IssueRow key={i} issue={q} />)}
        </ExpandableSection>
      )}

      {/* Leakage risks */}
      {leakageRisks.length > 0 && (
        <Card className="bg-zinc-900 border-amber-800/40">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-amber-400">
              Leakage Risks ({leakageRisks.length})
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {leakageRisks.map((r, i) => (
              <div key={i} className="flex items-center gap-3 text-sm">
                <span className="font-mono text-zinc-300">{r.column}</span>
                <span className="text-zinc-500 text-xs">r={r.correlation.toFixed(3)}</span>
                <span className="text-zinc-400 text-xs">{r.reason}</span>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* High-correlation pairs */}
      {highCorrPairs.length > 0 && (
        <ExpandableSection label="High-Correlation Pairs" badgeCount={highCorrPairs.length}>
          <div className="space-y-1.5">
            {highCorrPairs.slice(0, 20).map((p, i) => (
              <div key={i} className="flex items-center gap-3 text-sm">
                <span className="font-mono text-zinc-300">{p.col1}</span>
                <span className="text-zinc-600">↔</span>
                <span className="font-mono text-zinc-300">{p.col2}</span>
                <span className="font-mono text-xs text-amber-400 ml-auto">
                  r={p.correlation.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </ExpandableSection>
      )}
    </div>
  )
}
