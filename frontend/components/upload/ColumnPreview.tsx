"use client"

/** Dataset schema preview shown after upload - displays column names, dtypes,
 * and basic statistics so the user can confirm target column selection. */

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface ColumnInfo {
  name: string
  dtype: string
  missing_pct?: number
  unique?: number
  sample_values?: (string | number)[]
}

interface ColumnPreviewProps {
  columns: ColumnInfo[]
  totalRows: number
  selectedTarget?: string
  onSelectTarget?: (col: string) => void
}

const DTYPE_BADGE: Record<string, "outline" | "info"> = {
  int64: "info",
  float64: "info",
  object: "outline",
  bool: "outline",
  datetime64: "outline",
}

function dtypeBadgeVariant(dtype: string): "outline" | "info" {
  for (const [key, variant] of Object.entries(DTYPE_BADGE)) {
    if (dtype.includes(key)) return variant
  }
  return "outline"
}

export function ColumnPreview({
  columns,
  totalRows,
  selectedTarget,
  onSelectTarget,
}: ColumnPreviewProps) {
  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">
            Schema Preview - {columns.length} columns, {totalRows.toLocaleString()} rows
          </CardTitle>
          {selectedTarget && (
            <Badge variant="success" className="text-xs">
              target: {selectedTarget}
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto max-h-[320px] overflow-y-auto">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-zinc-900 z-10">
              <tr className="border-b border-zinc-700">
                <th className="text-left pb-2 font-medium text-zinc-500 pr-4">Column</th>
                <th className="text-left pb-2 font-medium text-zinc-500 pr-4">Type</th>
                <th className="text-right pb-2 font-medium text-zinc-500 pr-4">Missing</th>
                <th className="text-right pb-2 font-medium text-zinc-500 pr-4">Unique</th>
                <th className="text-left pb-2 font-medium text-zinc-500">Sample values</th>
              </tr>
            </thead>
            <tbody>
              {columns.map((col) => (
                <tr
                  key={col.name}
                  onClick={() => onSelectTarget?.(col.name)}
                  className={[
                    "border-b border-zinc-800 transition-colors",
                    onSelectTarget ? "cursor-pointer hover:bg-zinc-800/40" : "",
                    selectedTarget === col.name ? "bg-emerald-900/20 border-emerald-800/40" : "",
                  ].join(" ")}
                >
                  <td className="py-1.5 pr-4 font-mono text-zinc-200">{col.name}</td>
                  <td className="py-1.5 pr-4">
                    <Badge variant={dtypeBadgeVariant(col.dtype)} className="text-[10px]">
                      {col.dtype}
                    </Badge>
                  </td>
                  <td className="py-1.5 pr-4 text-right font-mono text-zinc-500">
                    {col.missing_pct != null
                      ? col.missing_pct > 0
                        ? <span className={col.missing_pct > 20 ? "text-amber-400" : ""}>{col.missing_pct.toFixed(1)}%</span>
                        : "0%"
                      : "-"}
                  </td>
                  <td className="py-1.5 pr-4 text-right font-mono text-zinc-500">
                    {col.unique != null ? col.unique.toLocaleString() : "-"}
                  </td>
                  <td className="py-1.5 font-mono text-zinc-600 truncate max-w-[200px]">
                    {col.sample_values?.slice(0, 4).join(", ") ?? ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {onSelectTarget && (
          <p className="mt-2 text-xs text-zinc-600">Click a column to set it as the target.</p>
        )}
      </CardContent>
    </Card>
  )
}
