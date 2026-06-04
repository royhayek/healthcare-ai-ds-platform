"use client"

/**
 * Project-level layout - activates the ChatPanel co-pilot with the most
 * recent run for this project.
 *
 * The dashboard layout mounts ChatPanel once (Hard Rule 2). This layout is
 * responsible for wiring chatStore.runId to whichever run is most relevant
 * while the user is anywhere inside /project/[id]/...
 *
 * Priority:
 *   1. The most recently updated run (any status) so the co-pilot has context.
 *   2. null if the project has no runs yet.
 *
 * The analysis page overrides this by calling setRunId(specificRunId) for its
 * own run - that's fine because this layout will restore the latest run when
 * the analysis page unmounts.
 */

import { useEffect } from "react"
import { useParams } from "next/navigation"
import useSWR from "swr"
import { fetcher } from "@/lib/api"
import { useChatStore } from "@/store/chatStore"
import type { Run } from "@/lib/types"

export default function ProjectLayout({ children }: { children: React.ReactNode }) {
  const { id: projectId } = useParams<{ id: string }>()
  const setRunId = useChatStore((s) => s.setRunId)

  const { data: runs } = useSWR<Run[]>(
    projectId ? `/api/proxy/projects/${projectId}/runs` : null,
    fetcher,
    { refreshInterval: 10_000 },
  )

  useEffect(() => {
    if (!runs || runs.length === 0) return
    // Pick the most recently active run: prefer running/awaiting over completed
    const priority = ["running", "awaiting_checkpoint", "completed", "failed", "queued"]
    const sorted = [...runs].sort(
      (a, b) => priority.indexOf(a.status) - priority.indexOf(b.status),
    )
    setRunId(sorted[0].id)
  }, [runs, setRunId])

  // Clear chat when the user leaves all project pages
  useEffect(() => {
    return () => setRunId(null)
  }, [projectId, setRunId])

  return <>{children}</>
}
