import { beforeEach, describe, expect, it } from "vitest"
import { useJobStore } from "@/store/jobStore"
import type { Run } from "@/lib/types"

const makeRun = (overrides: Partial<Run> = {}): Run =>
  ({
    id: "run-1",
    project_id: "proj-1",
    status: "running",
    progress: 0.5,
  } as Run & typeof overrides)

beforeEach(() => {
  useJobStore.setState({ runId: null, run: null })
})

describe("jobStore", () => {
  it("starts empty", () => {
    const s = useJobStore.getState()
    expect(s.runId).toBeNull()
    expect(s.run).toBeNull()
  })

  it("setRunId stores the id", () => {
    useJobStore.getState().setRunId("run-42")
    expect(useJobStore.getState().runId).toBe("run-42")
  })

  it("setRun stores the run object", () => {
    const run = makeRun()
    useJobStore.getState().setRun(run)
    expect(useJobStore.getState().run).toEqual(run)
  })

  it("clearJob resets both fields", () => {
    useJobStore.getState().setRunId("run-1")
    useJobStore.getState().setRun(makeRun())
    useJobStore.getState().clearJob()
    const s = useJobStore.getState()
    expect(s.runId).toBeNull()
    expect(s.run).toBeNull()
  })
})
