"use client"

/**
 * Shared Zustand store for active job / run state (§6).
 *
 * The analysis page is responsible for populating this store via SWR.
 * Any component that needs the current run status (e.g. ChatPanel header,
 * results widgets) can read from here without prop-drilling.
 */

import { create } from "zustand"
import type { Run } from "@/lib/types"

interface JobState {
  runId: string | null
  run: Run | null

  setRunId: (id: string | null) => void
  setRun: (run: Run | null) => void
  clearJob: () => void
}

export const useJobStore = create<JobState>()((set) => ({
  runId: null,
  run: null,

  setRunId: (id) => set({ runId: id }),
  setRun: (run) => set({ run }),
  clearJob: () => set({ runId: null, run: null }),
}))
