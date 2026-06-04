"use client"

import type { AttributeFairnessResult, FairnessReport, GroupMetrics } from "@/lib/types"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

const SEVERITY_BADGE: Record<string, "success" | "warning" | "error" | "outline"> = {
  none: "success",
  mild: "warning",
  moderate: "warning",
  severe: "error",
}

const SEVERITY_LABEL: Record<string, string> = {
  none: "None",
  mild: "Mild",
  moderate: "Moderate",
  severe: "Severe",
}

function SeverityBadge({ severity }: { severity: string }) {
  const variant = SEVERITY_BADGE[severity] ?? "outline"
  return <Badge variant={variant}>{SEVERITY_LABEL[severity] ?? severity}</Badge>
}

function GroupTable({ groups }: { groups: GroupMetrics[] }) {
  return (
    <table className="w-full text-sm mt-2">
      <thead>
        <tr className="border-b border-zinc-800">
          <th className="text-left pb-1.5 font-medium text-zinc-400">Group</th>
          <th className="text-right pb-1.5 font-medium text-zinc-400">N</th>
          <th className="text-right pb-1.5 font-medium text-zinc-400">Selection rate</th>
          <th className="text-right pb-1.5 font-medium text-zinc-400">TPR</th>
          <th className="text-right pb-1.5 font-medium text-zinc-400">FPR</th>
        </tr>
      </thead>
      <tbody>
        {groups.map((g) => (
          <tr key={g.group} className="border-b border-zinc-800/60">
            <td className="py-1.5 font-mono text-xs text-zinc-200">{g.group}</td>
            <td className="py-1.5 text-right font-mono text-xs text-zinc-400">{g.n_samples}</td>
            <td className="py-1.5 text-right font-mono text-xs">{(g.selection_rate * 100).toFixed(1)}%</td>
            <td className="py-1.5 text-right font-mono text-xs">
              {g.true_positive_rate != null ? `${(g.true_positive_rate * 100).toFixed(1)}%` : "-"}
            </td>
            <td className="py-1.5 text-right font-mono text-xs">
              {g.false_positive_rate != null ? `${(g.false_positive_rate * 100).toFixed(1)}%` : "-"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function AttributeBlock({ result }: { result: AttributeFairnessResult }) {
  return (
    <div className="rounded-lg border border-zinc-800 p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <span className="font-mono text-sm text-zinc-200">{result.attribute}</span>
          <div className="text-xs text-zinc-500 mt-0.5">
            Demographic parity diff:{" "}
            <span className="font-mono">{result.demographic_parity_diff.toFixed(4)}</span>
            {result.equalized_odds_diff != null && (
              <>
                {" "}
                · Equalized odds diff:{" "}
                <span className="font-mono">{result.equalized_odds_diff.toFixed(4)}</span>
              </>
            )}
          </div>
        </div>
        <SeverityBadge severity={result.severity} />
      </div>

      {result.note && (
        <p className="text-xs text-amber-400 mb-3">{result.note}</p>
      )}

      <GroupTable groups={result.by_group} />
    </div>
  )
}

interface FairnessReportProps {
  report: FairnessReport
}

export function FairnessReportPanel({ report }: FairnessReportProps) {
  const allAttrs = [...report.attributes, ...report.intersectional]

  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-medium">Fairness Analysis</CardTitle>
          <SeverityBadge severity={report.overall_severity} />
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Block warning */}
        {report.blocks_deliverables && !report.acknowledged && (
          <div className="rounded-md border border-red-700/60 bg-red-900/20 px-3 py-2 text-sm text-red-300">
            ⛔ Severe fairness disparity detected. Deliverable generation is blocked until you
            acknowledge this in the chat panel.
          </div>
        )}

        {report.requires_acknowledgment && report.acknowledged && (
          <div className="rounded-md border border-amber-700/50 bg-amber-900/20 px-3 py-2 text-sm text-amber-300">
            ⚠ Disparity acknowledged and recorded in the audit log.
          </div>
        )}

        {/* Per-attribute results */}
        {report.attributes.length > 0 && (
          <div className="space-y-3">
            <h4 className="text-xs font-medium uppercase tracking-wide text-zinc-500">
              Per-attribute results
            </h4>
            {report.attributes.map((r) => (
              <AttributeBlock key={r.attribute} result={r} />
            ))}
          </div>
        )}

        {/* Intersectional */}
        {report.intersectional.length > 0 && (
          <div className="space-y-3">
            <h4 className="text-xs font-medium uppercase tracking-wide text-zinc-500">
              Intersectional analysis
            </h4>
            {report.intersectional.map((r) => (
              <AttributeBlock key={r.attribute} result={r} />
            ))}
          </div>
        )}

        {report.overall_severity === "none" && (
          <p className="text-sm text-emerald-400">
            ✓ No significant fairness disparities detected across protected attributes.
          </p>
        )}

        {allAttrs.length === 0 && (
          <p className="text-sm text-zinc-500">
            No protected attributes configured. Set them in the chat to enable fairness
            analysis.
          </p>
        )}
      </CardContent>
    </Card>
  )
}
