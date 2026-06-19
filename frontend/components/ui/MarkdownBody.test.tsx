import { describe, expect, it } from "vitest"
import { render, screen } from "@testing-library/react"
import { MarkdownBody } from "@/components/ui/MarkdownBody"

describe("MarkdownBody", () => {
  it("renders basic markdown", () => {
    render(<MarkdownBody>{"Hello **world**"}</MarkdownBody>)
    expect(screen.getByText("world").tagName).toBe("STRONG")
  })

  it("normalizes inline numbered lists into list items", () => {
    render(<MarkdownBody>{"Steps: 1. first 2. second 3. third"}</MarkdownBody>)
    const items = screen.getAllByRole("listitem")
    expect(items.length).toBeGreaterThanOrEqual(3)
  })

  it("normalizes inline bullet markers without throwing on the dash class", () => {
    // Exercises the [•·–-] character class (regression for the out-of-order
    // range that previously threw a SyntaxError at module load).
    render(<MarkdownBody>{"Items: • alpha · beta – gamma - delta"}</MarkdownBody>)
    expect(screen.getAllByRole("listitem").length).toBeGreaterThanOrEqual(1)
  })

  it("passes text through unchanged when normalize is disabled", () => {
    render(<MarkdownBody normalize={false}>{"just text"}</MarkdownBody>)
    expect(screen.getByText("just text")).toBeInTheDocument()
  })
})
