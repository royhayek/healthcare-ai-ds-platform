import { describe, expect, it } from "vitest"
import { render, screen } from "@testing-library/react"
import StrategyDiffCard from "@/components/chat/StrategyDiff"
import type { StrategyDiff } from "@/lib/types"

const diff = (overrides: Partial<StrategyDiff> = {}): StrategyDiff => ({
  field_path: "preprocessing.imbalance",
  before: "smote",
  after: "class_weight",
  summary: "Switched imbalance handling",
  run_id: "run-1",
  ...overrides,
})

describe("StrategyDiffCard", () => {
  it("renders the field path and summary", () => {
    render(<StrategyDiffCard diff={diff()} />)
    expect(screen.getByText("preprocessing.imbalance")).toBeInTheDocument()
    expect(screen.getByText("Switched imbalance handling")).toBeInTheDocument()
  })

  it("renders before and after values", () => {
    render(<StrategyDiffCard diff={diff()} />)
    expect(screen.getByText("smote")).toBeInTheDocument()
    expect(screen.getByText("class_weight")).toBeInTheDocument()
  })

  it("renders a dash when before/after are null", () => {
    render(<StrategyDiffCard diff={diff({ before: null, after: null })} />)
    expect(screen.getAllByText("-")).toHaveLength(2)
  })

  it("stringifies non-string values", () => {
    render(<StrategyDiffCard diff={diff({ before: 0.5, after: 0.42 })} />)
    expect(screen.getByText("0.5")).toBeInTheDocument()
    expect(screen.getByText("0.42")).toBeInTheDocument()
  })
})
