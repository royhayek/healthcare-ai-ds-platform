"use client"

/** Collapsible section used to hide lower-priority content (e.g., priority-2 plots). */

import { useState } from "react"
import { ChevronDown, ChevronRight } from "lucide-react"
import { cn } from "@/lib/cn"

interface ExpandableSectionProps {
  label: string
  defaultOpen?: boolean
  children: React.ReactNode
  className?: string
  badgeCount?: number
}

export function ExpandableSection({
  label,
  defaultOpen = false,
  children,
  className,
  badgeCount,
}: ExpandableSectionProps) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className={cn("border border-zinc-800 rounded-lg overflow-hidden", className)}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-2.5 bg-zinc-900 hover:bg-zinc-800/60 transition-colors text-left"
      >
        <span className="flex items-center gap-2 text-sm font-medium text-zinc-300">
          {open ? <ChevronDown className="h-4 w-4 text-zinc-500" /> : <ChevronRight className="h-4 w-4 text-zinc-500" />}
          {label}
          {badgeCount != null && (
            <span className="ml-1 rounded-full bg-zinc-700 px-1.5 py-0.5 text-xs text-zinc-400">
              {badgeCount}
            </span>
          )}
        </span>
      </button>
      {open && <div className="bg-zinc-950 p-4">{children}</div>}
    </div>
  )
}
