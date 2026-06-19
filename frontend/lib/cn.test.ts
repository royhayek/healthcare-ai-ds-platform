import { describe, expect, it } from "vitest"
import { cn } from "@/lib/cn"

describe("cn", () => {
  it("joins truthy class names", () => {
    expect(cn("a", "b")).toBe("a b")
  })

  it("drops falsy values", () => {
    expect(cn("a", false, null, undefined, "", "b")).toBe("a b")
  })

  it("supports conditional object syntax", () => {
    expect(cn("base", { active: true, hidden: false })).toBe("base active")
  })

  it("merges conflicting tailwind utilities, last wins", () => {
    expect(cn("px-2 px-4")).toBe("px-4")
    expect(cn("text-red-400", "text-emerald-400")).toBe("text-emerald-400")
  })

  it("returns empty string with no args", () => {
    expect(cn()).toBe("")
  })
})
