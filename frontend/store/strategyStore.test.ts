import { beforeEach, describe, expect, it } from "vitest"
import { useStrategyStore } from "@/store/strategyStore"
import type { StrategyDiff } from "@/lib/types"

const mkDiff = (overrides: Partial<StrategyDiff>): StrategyDiff => ({
  field_path: "a",
  before: null,
  after: null,
  summary: "",
  run_id: "run-1",
  ...overrides,
})

beforeEach(() => {
  useStrategyStore.setState({ strategy: null, pendingDiffs: [] })
})

describe("strategyStore", () => {
  it("setStrategy stores the object", () => {
    useStrategyStore.getState().setStrategy({ task_type: "binary" })
    expect(useStrategyStore.getState().strategy).toEqual({ task_type: "binary" })
  })

  it("applyDiffs sets a top-level field", () => {
    useStrategyStore.getState().setStrategy({ threshold: 0.5 })
    useStrategyStore.getState().applyDiffs([mkDiff({ field_path: "threshold", after: 0.42 })])
    expect(useStrategyStore.getState().strategy).toMatchObject({ threshold: 0.42 })
  })

  it("applyDiffs creates nested objects for deep paths", () => {
    useStrategyStore.getState().applyDiffs([
      mkDiff({ field_path: "preprocessing.imbalance.method", after: "class_weight" }),
    ])
    expect(useStrategyStore.getState().strategy).toEqual({
      preprocessing: { imbalance: { method: "class_weight" } },
    })
  })

  it("applyDiffs starting from null strategy still works", () => {
    useStrategyStore.getState().applyDiffs([mkDiff({ field_path: "model", after: "xgboost" })])
    expect(useStrategyStore.getState().strategy).toEqual({ model: "xgboost" })
  })

  it("applyDiffs preserves sibling keys", () => {
    useStrategyStore.getState().setStrategy({ preprocessing: { scale: "standard", impute: "median" } })
    useStrategyStore.getState().applyDiffs([mkDiff({ field_path: "preprocessing.scale", after: "robust" })])
    expect(useStrategyStore.getState().strategy).toMatchObject({
      preprocessing: { scale: "robust", impute: "median" },
    })
  })

  it("applyDiffs accumulates pendingDiffs across calls", () => {
    const d1 = mkDiff({ field_path: "a", after: 1 })
    const d2 = mkDiff({ field_path: "b", after: 2 })
    useStrategyStore.getState().applyDiffs([d1])
    useStrategyStore.getState().applyDiffs([d2])
    expect(useStrategyStore.getState().pendingDiffs).toEqual([d1, d2])
  })

  it("clearPendingDiffs empties the queue without touching strategy", () => {
    useStrategyStore.getState().setStrategy({ k: "v" })
    useStrategyStore.getState().applyDiffs([mkDiff({ field_path: "x", after: 1 })])
    useStrategyStore.getState().clearPendingDiffs()
    expect(useStrategyStore.getState().pendingDiffs).toEqual([])
    expect(useStrategyStore.getState().strategy).toMatchObject({ k: "v", x: 1 })
  })
})
