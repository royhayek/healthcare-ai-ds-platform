"use client"

import { create } from "zustand"
import { type StrategyDiff } from "@/lib/types"

interface StrategyState {
  strategy: Record<string, unknown> | null
  pendingDiffs: StrategyDiff[]

  setStrategy: (s: Record<string, unknown>) => void
  applyDiffs: (diffs: StrategyDiff[]) => void
  clearPendingDiffs: () => void
}

export const useStrategyStore = create<StrategyState>()((set) => ({
  strategy: null,
  pendingDiffs: [],

  setStrategy: (s) => set({ strategy: s }),

  applyDiffs: (diffs) =>
    set((state) => {
      const updated = { ...(state.strategy ?? {}) }
      for (const diff of diffs) {
        // Shallow application of field_path - deep path support added in Step 5
        const parts = diff.field_path.split(".")
        let node: Record<string, unknown> = updated
        for (let i = 0; i < parts.length - 1; i++) {
          if (typeof node[parts[i]] !== "object" || node[parts[i]] === null) {
            node[parts[i]] = {}
          }
          node = node[parts[i]] as Record<string, unknown>
        }
        node[parts[parts.length - 1]] = diff.after
      }
      return {
        strategy: updated,
        pendingDiffs: [...state.pendingDiffs, ...diffs],
      }
    }),

  clearPendingDiffs: () => set({ pendingDiffs: [] }),
}))
