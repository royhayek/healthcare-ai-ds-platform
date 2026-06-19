import { describe, expect, it } from "vitest"
import { render, screen } from "@testing-library/react"
import { Badge } from "@/components/ui/badge"

describe("Badge", () => {
  it("renders its children", () => {
    render(<Badge>completed</Badge>)
    expect(screen.getByText("completed")).toBeInTheDocument()
  })

  it("applies the default variant classes", () => {
    render(<Badge>x</Badge>)
    expect(screen.getByText("x")).toHaveClass("bg-neutral-700")
  })

  it("applies a named variant", () => {
    render(<Badge variant="error">failed</Badge>)
    expect(screen.getByText("failed")).toHaveClass("bg-red-900")
  })

  it("merges a custom className", () => {
    render(<Badge className="ml-2">y</Badge>)
    expect(screen.getByText("y")).toHaveClass("ml-2")
  })

  it("forwards arbitrary html props", () => {
    render(<Badge data-testid="status-badge">z</Badge>)
    expect(screen.getByTestId("status-badge")).toBeInTheDocument()
  })
})
