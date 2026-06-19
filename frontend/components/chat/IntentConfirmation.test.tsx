import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import IntentConfirmation from "@/components/chat/IntentConfirmation"
import type { ChatIntent } from "@/lib/types"

const intent: ChatIntent = {
  intent: "modify",
  confidence: 0.92,
  category: "threshold",
  structured_payload: {},
  needs_confirmation: true,
  reasoning: "Lower the threshold to favor sensitivity.",
}

describe("IntentConfirmation", () => {
  it("renders the reasoning", () => {
    render(<IntentConfirmation intent={intent} onConfirm={() => {}} onDismiss={() => {}} />)
    expect(screen.getByText("Lower the threshold to favor sensitivity.")).toBeInTheDocument()
  })

  it("calls onConfirm when Apply is clicked", async () => {
    const onConfirm = vi.fn()
    const onDismiss = vi.fn()
    render(<IntentConfirmation intent={intent} onConfirm={onConfirm} onDismiss={onDismiss} />)
    await userEvent.click(screen.getByRole("button", { name: "Apply" }))
    expect(onConfirm).toHaveBeenCalledOnce()
    expect(onDismiss).not.toHaveBeenCalled()
  })

  it("calls onDismiss when Dismiss is clicked", async () => {
    const onConfirm = vi.fn()
    const onDismiss = vi.fn()
    render(<IntentConfirmation intent={intent} onConfirm={onConfirm} onDismiss={onDismiss} />)
    await userEvent.click(screen.getByRole("button", { name: "Dismiss" }))
    expect(onDismiss).toHaveBeenCalledOnce()
    expect(onConfirm).not.toHaveBeenCalled()
  })
})
