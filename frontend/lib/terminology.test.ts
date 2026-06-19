import { describe, expect, it } from "vitest"
import {
  RISK_TIER,
  RISK_TIER_CLASSES,
  SEVERITY_LABEL,
  TERM,
  riskTier,
} from "@/lib/terminology"

describe("riskTier", () => {
  it("classifies below low_max as low", () => {
    expect(riskTier(0)).toBe("low")
    expect(riskTier(RISK_TIER.low_max - 0.001)).toBe("low")
  })

  it("treats low_max boundary as medium (exclusive lower bound)", () => {
    expect(riskTier(RISK_TIER.low_max)).toBe("medium")
  })

  it("classifies between low_max and medium_max as medium", () => {
    expect(riskTier((RISK_TIER.low_max + RISK_TIER.medium_max) / 2)).toBe("medium")
    expect(riskTier(RISK_TIER.medium_max - 0.001)).toBe("medium")
  })

  it("treats medium_max boundary as high", () => {
    expect(riskTier(RISK_TIER.medium_max)).toBe("high")
  })

  it("classifies above medium_max as high", () => {
    expect(riskTier(0.9)).toBe("high")
    expect(riskTier(1)).toBe("high")
  })
})

describe("terminology maps", () => {
  it("every risk tier has a matching style entry with a label", () => {
    for (const tier of ["low", "medium", "high"] as const) {
      expect(RISK_TIER_CLASSES[tier]).toBeDefined()
      expect(RISK_TIER_CLASSES[tier].badge).toBeTruthy()
      expect(RISK_TIER_CLASSES[tier].label).toBeTruthy()
    }
  })

  it("tier style labels match the TERM risk labels", () => {
    expect(RISK_TIER_CLASSES.low.label).toBe(TERM.risk_low)
    expect(RISK_TIER_CLASSES.medium.label).toBe(TERM.risk_medium)
    expect(RISK_TIER_CLASSES.high.label).toBe(TERM.risk_high)
  })

  it("severity labels cover all backend severities", () => {
    for (const sev of ["none", "mild", "moderate", "severe"]) {
      expect(SEVERITY_LABEL[sev]).toBeTruthy()
    }
  })

  it("clinical disclaimer is present (appended to AI outputs)", () => {
    expect(TERM.clinical_disclaimer).toMatch(/licensed clinician/i)
  })
})
