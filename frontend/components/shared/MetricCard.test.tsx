import { describe, expect, it } from "vitest"
import { render, screen } from "@testing-library/react"
import { MetricCard } from "@/components/shared/MetricCard"

describe("MetricCard", () => {
  it("renders the label", () => {
    render(<MetricCard label="AUC" value={0.87} />)
    expect(screen.getByText("AUC")).toBeInTheDocument()
  })

  it("formats numeric values to 4 decimals", () => {
    render(<MetricCard label="AUC" value={0.8} />)
    expect(screen.getByText("0.8000")).toBeInTheDocument()
  })

  it("renders strings verbatim", () => {
    render(<MetricCard label="Model" value="XGBClassifier" />)
    expect(screen.getByText("XGBClassifier")).toBeInTheDocument()
  })

  it("shows a dash for null and undefined", () => {
    const { rerender } = render(<MetricCard label="x" value={null} />)
    expect(screen.getByText("-")).toBeInTheDocument()
    rerender(<MetricCard label="x" value={undefined} />)
    expect(screen.getByText("-")).toBeInTheDocument()
  })

  it("renders the optional sub text", () => {
    render(<MetricCard label="AUC" value={0.9} sub="mean ± std" />)
    expect(screen.getByText("mean ± std")).toBeInTheDocument()
  })

  it("applies the highlight color class", () => {
    render(<MetricCard label="AUC" value={0.9} highlight="success" />)
    expect(screen.getByText("0.9000")).toHaveClass("text-emerald-400")
  })
})
