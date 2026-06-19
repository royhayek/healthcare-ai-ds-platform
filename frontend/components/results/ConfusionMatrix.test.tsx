import { describe, expect, it } from "vitest"
import { render, screen } from "@testing-library/react"
import { ConfusionMatrix } from "@/components/results/ConfusionMatrix"

describe("ConfusionMatrix", () => {
  it("renders nothing when metrics are empty", () => {
    const { container } = render(<ConfusionMatrix metrics={{}} />)
    expect(container).toBeEmptyDOMElement()
  })

  it("renders classification metric tiles", () => {
    render(<ConfusionMatrix metrics={{ accuracy: 0.9, f1: 0.76, roc_auc: 0.87 }} />)
    expect(screen.getByText("ACCURACY")).toBeInTheDocument()
    expect(screen.getByText("F1")).toBeInTheDocument()
    expect(screen.getByText("ROC AUC")).toBeInTheDocument()
  })

  it("renders the 2x2 matrix when tp/tn/fp/fn are present", () => {
    render(<ConfusionMatrix metrics={{ accuracy: 0.9, tp: 50, tn: 40, fp: 5, fn: 10 }} />)
    expect(screen.getByText("Confusion matrix")).toBeInTheDocument()
    expect(screen.getByText("TP 50")).toBeInTheDocument()
    expect(screen.getByText("FN 10")).toBeInTheDocument()
    expect(screen.getByText("FP 5")).toBeInTheDocument()
    expect(screen.getByText("TN 40")).toBeInTheDocument()
  })

  it("does not render the matrix when confusion counts are missing", () => {
    render(<ConfusionMatrix metrics={{ accuracy: 0.9 }} />)
    expect(screen.queryByText("Confusion matrix")).not.toBeInTheDocument()
  })

  it("renders regression metrics when taskType is regression", () => {
    render(<ConfusionMatrix metrics={{ r2: 0.82, mae: 1200.5, rmse: 1800.25 }} taskType="regression" />)
    expect(screen.getByText("R2")).toBeInTheDocument()
    expect(screen.getByText("MAE")).toBeInTheDocument()
    expect(screen.getByText("RMSE")).toBeInTheDocument()
    // classification-only labels should be absent
    expect(screen.queryByText("Confusion matrix")).not.toBeInTheDocument()
  })

  it("returns null for regression with no recognized regression keys", () => {
    const { container } = render(<ConfusionMatrix metrics={{ foo: 1 }} taskType="regression" />)
    expect(container).toBeEmptyDOMElement()
  })
})
