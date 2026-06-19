import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { PatientRiskCard, type PatientRiskResult } from "@/components/results/PatientRiskCard"
import { TERM } from "@/lib/terminology"

const baseResult = (overrides: Partial<PatientRiskResult> = {}): PatientRiskResult => ({
  probability: 0.62,
  prediction: 1,
  threshold_used: 0.42,
  shap_drivers: ["age = 71", "prior_admits = 3"],
  shap_dampeners: ["bmi = 22"],
  similarity_score: 0.8,
  confidence_band: "high",
  risk_flag: false,
  row_id: "row-7",
  ...overrides,
})

describe("PatientRiskCard", () => {
  it("renders the probability as a percentage", () => {
    render(<PatientRiskCard result={baseResult({ probability: 0.625 })} />)
    expect(screen.getByText("62.5%")).toBeInTheDocument()
  })

  it("shows the high-risk tier badge for a high probability", () => {
    render(<PatientRiskCard result={baseResult({ probability: 0.62 })} />)
    expect(screen.getByText(TERM.risk_high)).toBeInTheDocument()
  })

  it("shows the low-risk tier badge for a low probability", () => {
    render(<PatientRiskCard result={baseResult({ probability: 0.05 })} />)
    expect(screen.getByText(TERM.risk_low)).toBeInTheDocument()
  })

  it("renders the outcome name in the subtitle", () => {
    render(<PatientRiskCard result={baseResult()} outcomeName="30-day readmission" />)
    expect(screen.getByText(/30-day readmission/)).toBeInTheDocument()
  })

  it("shows the dissimilarity warning only when risk_flag is set", () => {
    const { rerender } = render(<PatientRiskCard result={baseResult({ risk_flag: false })} />)
    expect(screen.queryByText(/dissimilar to the training cohort/)).not.toBeInTheDocument()
    rerender(<PatientRiskCard result={baseResult({ risk_flag: true, similarity_score: 0.12 })} />)
    expect(screen.getByText(/dissimilar to the training cohort/)).toBeInTheDocument()
    expect(screen.getByText(/12%/)).toBeInTheDocument()
  })

  it("lists at most three shap drivers", () => {
    render(
      <PatientRiskCard
        result={baseResult({ shap_drivers: ["a", "b", "c", "d", "e"], shap_dampeners: [] })}
      />,
    )
    expect(screen.getByText("a")).toBeInTheDocument()
    expect(screen.getByText("c")).toBeInTheDocument()
    expect(screen.queryByText("d")).not.toBeInTheDocument()
  })

  it("renders the threshold context with the binary prediction", () => {
    render(<PatientRiskCard result={baseResult({ prediction: 1, threshold_used: 0.42 })} />)
    expect(screen.getByText("positive")).toBeInTheDocument()
    expect(screen.getByText(/0\.420/)).toBeInTheDocument()
  })

  it("invokes onOverride with the flipped prediction", async () => {
    const onOverride = vi.fn()
    render(<PatientRiskCard result={baseResult({ prediction: 1, row_id: "row-7" })} onOverride={onOverride} />)
    await userEvent.click(screen.getByRole("button", { name: /record override/i }))
    expect(onOverride).toHaveBeenCalledWith("row-7", 0)
  })

  it("hides the override button when no handler is given", () => {
    render(<PatientRiskCard result={baseResult()} />)
    expect(screen.queryByRole("button", { name: /record override/i })).not.toBeInTheDocument()
  })
})
