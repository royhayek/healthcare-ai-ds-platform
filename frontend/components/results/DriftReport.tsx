"use client"

import type { DriftReport, FeatureDriftResult } from "@/lib/types"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

const SEVERITY_BADGE: Record<string, "success" | "warning" | "error"> = {
  stable: "success",
  mild: "warning",
  significant: "error",
}

const SEVERITY_LABEL: Record<string, string> = {
  stable: "Stable",
  mild: "Mild drift",
  significant: "Significant drift",
}

function SeverityBadge({ severity }: { severity: string }) {
  const variant = SEVERITY_BADGE[severity] ?? "outline"
  return <Badge variant={variant}>{SEVERITY_LABEL[severity] ?? severity}</Badge>
}

function FeatureRow({ feature }: { feature: FeatureDriftResult }) {
  return (
    <tr className="border-b border-zinc-800 hover:bg-zinc-800/40 transition-colors">
      <td className="py-2 pr-4 font-mono text-sm text-zinc-200">{feature.feature}</td>
      <td className="py-2 pr-4">
        <span className="text-xs text-zinc-500 uppercase">{feature.type}</span>
      </td>
      <td className="py-2 pr-4 text-right font-mono text-sm">
        {feature.psi != null ? feature.psi.toFixed(4) : "-"}
      </td>
      <td className="py-2 pr-4 text-right font-mono text-sm">
        {feature.ks_statistic != null
          ? `${feature.ks_statistic.toFixed(3)} (p=${(feature.ks_p_value ?? 1).toFixed(3)})`
          : feature.js_divergence != null
          ? feature.js_divergence.toFixed(4)
          : "-"}
      </td>
      <td className="py-2">
        <SeverityBadge severity={feature.severity} />
      </td>
    </tr>
  )
}

interface DriftReportProps {
  report: DriftReport
  comparisonLabel?: string
}

export function DriftReportPanel({ report, comparisonLabel }: DriftReportProps) {
  const topFeatures = report.features.slice(0, 20)

  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-medium">
            Drift Analysis{comparisonLabel ? ` - vs. ${comparisonLabel}` : ""}
          </CardTitle>
          <SeverityBadge severity={report.overall_severity} />
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Summary row */}
        <div className="grid grid-cols-3 gap-4 rounded-lg bg-zinc-800/50 p-3 text-sm">
          <div>
            <div className="text-zinc-500 text-xs mb-1">Aggregate PSI</div>
            <div className="font-mono font-medium">{report.aggregate_psi.toFixed(4)}</div>
            <div className="text-zinc-600 text-xs mt-0.5">threshold: 0.25 = retrain</div>
          </div>
          <div>
            <div className="text-zinc-500 text-xs mb-1">Features drifted</div>
            <div className="font-mono font-medium">{report.n_features_drifted}</div>
            <div className="text-zinc-600 text-xs mt-0.5">of {report.features.length} analyzed</div>
          </div>
          <div>
            <div className="text-zinc-500 text-xs mb-1">Dataset sizes</div>
            <div className="font-mono font-medium">
              {report.n_train_rows.toLocaleString()} → {report.n_new_rows.toLocaleString()}
            </div>
            <div className="text-zinc-600 text-xs mt-0.5">train → new</div>
          </div>
        </div>

        {/* Warning */}
        {report.warning && (
          <div className="rounded-md border border-amber-700/50 bg-amber-900/20 px-3 py-2 text-sm text-amber-300">
            ⚠ {report.warning}
          </div>
        )}

        {/* Feature table */}
        {topFeatures.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-zinc-700">
                  <th className="text-left pb-2 font-medium text-zinc-400">Feature</th>
                  <th className="text-left pb-2 font-medium text-zinc-400">Type</th>
                  <th className="text-right pb-2 font-medium text-zinc-400">PSI</th>
                  <th className="text-right pb-2 font-medium text-zinc-400">KS / JS</th>
                  <th className="text-left pb-2 pl-2 font-medium text-zinc-400">Severity</th>
                </tr>
              </thead>
              <tbody>
                {topFeatures.map((f) => (
                  <FeatureRow key={f.feature} feature={f} />
                ))}
              </tbody>
            </table>
            {report.features.length > 20 && (
              <p className="mt-2 text-xs text-zinc-600">
                Showing top 20 of {report.features.length} features
              </p>
            )}
          </div>
        )}

        {report.overall_severity === "stable" && (
          <p className="text-sm text-emerald-400">
            ✓ Predictions on this dataset can be trusted at current performance.
          </p>
        )}
      </CardContent>
    </Card>
  )
}
