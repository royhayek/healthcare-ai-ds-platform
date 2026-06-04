"use client"

import type { FairnessReport, AttributeFairnessResult, GroupMetrics } from "@/lib/types"
import { TERM, SEVERITY_LABEL } from "@/lib/terminology"

// ── Sub-components ─────────────────────────────────────────────────────────────

function SeverityBadge({ severity }: { severity: string }) {
  const classes: Record<string, string> = {
    none: "bg-emerald-900/40 text-emerald-300 border border-emerald-700/50",
    mild: "bg-amber-900/40 text-amber-300 border border-amber-700/50",
    moderate: "bg-orange-900/40 text-orange-300 border border-orange-700/50",
    severe: "bg-red-900/40 text-red-300 border border-red-700/60",
  }
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${classes[severity] ?? classes.mild}`}>
      {SEVERITY_LABEL[severity] ?? severity}
    </span>
  )
}

function EquityTable({ groups }: { groups: GroupMetrics[] }) {
  return (
    <div className="overflow-x-auto mt-2">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-zinc-800 text-zinc-500">
            <th className="text-left pb-1.5 font-medium">{TERM.equity_protected_label}</th>
            <th className="text-right pb-1.5 font-medium">Patients (n)</th>
            <th className="text-right pb-1.5 font-medium">{TERM.equity_selection_label}</th>
            <th className="text-right pb-1.5 font-medium">{TERM.equity_tpr_label}</th>
            <th className="text-right pb-1.5 font-medium">{TERM.equity_fpr_label}</th>
          </tr>
        </thead>
        <tbody>
          {groups.map((g) => (
            <tr key={g.group} className="border-b border-zinc-800/40 last:border-0">
              <td className="py-1.5 font-mono text-zinc-200">{g.group}</td>
              <td className="py-1.5 text-right text-zinc-400">{g.n_samples.toLocaleString()}</td>
              <td className="py-1.5 text-right font-mono">
                {(g.selection_rate * 100).toFixed(1)}%
              </td>
              <td className="py-1.5 text-right font-mono">
                {g.true_positive_rate != null
                  ? `${(g.true_positive_rate * 100).toFixed(1)}%`
                  : "-"}
              </td>
              <td className="py-1.5 text-right font-mono">
                {g.false_positive_rate != null
                  ? `${(g.false_positive_rate * 100).toFixed(1)}%`
                  : "-"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function AttributeEquityBlock({ result }: { result: AttributeFairnessResult }) {
  const tprValues = result.by_group
    .map((g) => g.true_positive_rate)
    .filter((v): v is number => v != null)
  const tprGap =
    tprValues.length >= 2
      ? Math.max(...tprValues) - Math.min(...tprValues)
      : null

  const gapWarning = tprGap != null && tprGap > 0.05

  return (
    <div className="rounded-lg border border-zinc-800 p-4 space-y-3">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <span className="font-mono text-sm text-zinc-200">{result.attribute}</span>
          <div className="text-xs text-zinc-500 mt-0.5 space-x-3">
            <span>
              Demographic parity diff:{" "}
              <span className="font-mono">{result.demographic_parity_diff.toFixed(3)}</span>
            </span>
            {result.equalized_odds_diff != null && (
              <span>
                Equalized odds diff:{" "}
                <span className="font-mono">{result.equalized_odds_diff.toFixed(3)}</span>
              </span>
            )}
          </div>
        </div>
        <SeverityBadge severity={result.severity} />
      </div>

      {/* Clinical gap warning */}
      {gapWarning && tprGap != null && (
        <div className="rounded border border-amber-700/40 bg-amber-950/25 px-3 py-2 text-xs text-amber-300">
          ⚠ Detection rate gap of{" "}
          <span className="font-mono font-semibold">{(tprGap * 100).toFixed(1)}%</span>{" "}
          across {result.attribute} groups. At this gap the model may systematically miss
          more patients at risk in lower-performing groups. Clinical review recommended before deployment.
        </div>
      )}

      {result.note && (
        <p className="text-xs text-zinc-500 italic">{result.note}</p>
      )}

      <EquityTable groups={result.by_group} />
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────

interface Props {
  report: FairnessReport
}

/**
 * Clinical equity report - wraps raw fairness metrics in patient-safety framing.
 *
 * Key differences from the generic FairnessReportPanel:
 * - Uses clinical language: "detection rate" instead of "TPR"
 * - Shows explicit TPR gap with clinical interpretation
 * - Severity labels are phrased as clinical review requirements
 * - Blocks-deliverables warning uses clinical safety language
 */
export function EquityReport({ report }: Props) {
  const allGroups = [...report.attributes, ...report.intersectional]

  return (
    <div className="space-y-4">
      {/* Overall status banner */}
      <div className="flex items-center justify-between rounded-lg border border-zinc-800 bg-zinc-900/60 px-4 py-3">
        <div>
          <p className="text-sm font-medium text-zinc-200">Population equity summary</p>
          <p className="text-xs text-zinc-500 mt-0.5">
            Fairness across {report.attributes.length} protected demographic attribute
            {report.attributes.length !== 1 ? "s" : ""}
          </p>
        </div>
        <SeverityBadge severity={report.overall_severity} />
      </div>

      {/* Deliverables block warning */}
      {report.blocks_deliverables && !report.acknowledged && (
        <div
          role="alert"
          className="rounded-lg border border-red-700/60 bg-red-950/25 px-4 py-3 text-sm"
        >
          <p className="font-semibold text-red-300 mb-1">⛔ Deliverable generation paused</p>
          <p className="text-red-200/80 text-xs">
            A severe equity disparity was detected. Clinical safety requires acknowledgment
            before generating patient-facing outputs. Use the chat panel to review the
            disparity and type{" "}
            <span className="font-mono bg-zinc-800 px-1 rounded">acknowledge equity gap</span>{" "}
            to proceed.
          </p>
        </div>
      )}

      {report.requires_acknowledgment && report.acknowledged && (
        <div className="rounded-lg border border-amber-700/40 bg-amber-950/20 px-4 py-3 text-xs text-amber-300">
          ⚠ Equity gap acknowledged and recorded in the clinical audit log. Deliverable
          generation will include the equity flag in the risk register.
        </div>
      )}

      {/* Per-attribute blocks */}
      {report.attributes.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
            Per-attribute equity
          </h4>
          {report.attributes.map((r) => (
            <AttributeEquityBlock key={r.attribute} result={r} />
          ))}
        </div>
      )}

      {/* Intersectional */}
      {report.intersectional.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
            Intersectional equity (subgroup combinations)
          </h4>
          {report.intersectional.map((r) => (
            <AttributeEquityBlock key={r.attribute} result={r} />
          ))}
        </div>
      )}

      {/* Clean bill */}
      {report.overall_severity === "none" && allGroups.length > 0 && (
        <p className="text-sm text-emerald-400 px-1">
          ✓ No significant equity disparities detected across protected demographic attributes.
        </p>
      )}

      {allGroups.length === 0 && (
        <p className="text-sm text-zinc-500 px-1">
          No protected demographic attributes configured. Tell the co-pilot which columns
          represent demographic groups to enable equity analysis
          (e.g. "add age_group and insurance_type to equity analysis").
        </p>
      )}

      {/* Mandatory clinical disclaimer */}
      <p className="text-[11px] text-zinc-600 border-t border-zinc-800 pt-3">
        {TERM.clinical_disclaimer}
      </p>
    </div>
  )
}
