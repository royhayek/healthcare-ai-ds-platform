"use client"

/** In-browser executive summary preview - renders key headline numbers and
 * the AI-generated narrative without downloading the PDF. */

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { MetricCard } from "@/components/shared/MetricCard"
import { ExpandableSection } from "@/components/shared/ExpandableSection"
import { MarkdownBody } from "@/components/ui/MarkdownBody"

interface ExecutiveSummaryPreviewProps {
  projectName: string
  modelName: string | null
  taskType: string | null
  finalMetrics: Record<string, number> | null
  insightReport: string | null
  topFeatures: string[]
  thresholdUsed: number
  driftSeverity: string | null
  fairnessSeverity: string | null
  completedAt: string | null
}

const SEVERITY_VARIANT: Record<string, "success" | "warning" | "error" | "outline"> = {
  stable: "success",
  none: "success",
  mild: "warning",
  moderate: "warning",
  significant: "error",
  severe: "error",
}

export function ExecutiveSummaryPreview({
  projectName,
  modelName,
  taskType,
  finalMetrics,
  insightReport,
  topFeatures,
  thresholdUsed,
  driftSeverity,
  fairnessSeverity,
  completedAt,
}: ExecutiveSummaryPreviewProps) {
  const metrics = finalMetrics ?? {}
  const primaryMetric = metrics.auc ?? metrics.macro_auc ?? metrics.r2

  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-medium">Executive Summary Preview</CardTitle>
          <div className="flex gap-1.5">
            {taskType && <Badge variant="outline" className="text-xs">{taskType}</Badge>}
            {completedAt && (
              <span className="text-xs text-zinc-600">
                {new Date(completedAt).toLocaleDateString()}
              </span>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Headline banner */}
        <div className="rounded-lg bg-zinc-800/60 px-4 py-3">
          <div className="text-xs text-zinc-500 mb-0.5">{projectName}</div>
          <div className="flex items-baseline gap-3">
            <span className="font-mono text-lg font-semibold text-zinc-100">
              {modelName ?? "-"}
            </span>
            {primaryMetric != null && (
              <span className="font-mono text-2xl font-bold text-emerald-400">
                {primaryMetric.toFixed(4)}
              </span>
            )}
          </div>
          <div className="text-xs text-zinc-500 mt-0.5">
            threshold {thresholdUsed.toFixed(3)}
            {driftSeverity && (
              <>
                {" · "}
                <Badge variant={SEVERITY_VARIANT[driftSeverity] ?? "outline"} className="text-xs">
                  drift: {driftSeverity}
                </Badge>
              </>
            )}
            {fairnessSeverity && fairnessSeverity !== "none" && (
              <>
                {" · "}
                <Badge variant={SEVERITY_VARIANT[fairnessSeverity] ?? "outline"} className="text-xs">
                  fairness: {fairnessSeverity}
                </Badge>
              </>
            )}
          </div>
        </div>

        {/* Top metrics */}
        {Object.keys(metrics).length > 0 && (
          <div className="grid grid-cols-3 gap-2">
            {Object.entries(metrics).slice(0, 3).map(([k, v]) => (
              <MetricCard key={k} label={k.replace(/_/g, " ")} value={v} />
            ))}
          </div>
        )}

        {/* Key drivers */}
        {topFeatures.length > 0 && (
          <div>
            <div className="text-xs text-zinc-500 mb-2">Top predictors</div>
            <div className="flex flex-wrap gap-1.5">
              {topFeatures.slice(0, 6).map((f, i) => (
                <span key={i} className="rounded-md bg-zinc-800 px-2 py-0.5 font-mono text-xs text-zinc-300">
                  {i + 1}. {f}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Insight narrative */}
        {insightReport && (
          <ExpandableSection label="AI Insight Narrative" defaultOpen>
            <div className="prose prose-sm prose-invert max-w-none">
              <MarkdownBody>{insightReport}</MarkdownBody>
            </div>
          </ExpandableSection>
        )}
      </CardContent>
    </Card>
  )
}
