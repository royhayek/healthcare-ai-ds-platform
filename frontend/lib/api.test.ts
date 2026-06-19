import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import {
  createJoin,
  deleteProject,
  downloadDeliverable,
  fetcher,
  getProject,
  getProjects,
  predictSingle,
  uploadDataset,
} from "@/lib/api"

/** Build a minimal Response-like stub. */
function mockResponse(opts: {
  ok?: boolean
  status?: number
  json?: unknown
  text?: string
  blob?: Blob
}) {
  return {
    ok: opts.ok ?? true,
    status: opts.status ?? 200,
    json: vi.fn().mockResolvedValue(opts.json ?? {}),
    text: vi.fn().mockResolvedValue(opts.text ?? ""),
    blob: vi.fn().mockResolvedValue(opts.blob ?? new Blob()),
  }
}

const fetchMock = vi.fn()

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock)
  fetchMock.mockReset()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe("apiGet wrappers", () => {
  it("getProjects hits the proxy projects endpoint and returns parsed json", async () => {
    const data = [{ id: "p1" }]
    fetchMock.mockResolvedValue(mockResponse({ json: data }))
    const result = await getProjects()
    expect(fetchMock).toHaveBeenCalledWith("/api/proxy/projects")
    expect(result).toEqual(data)
  })

  it("getProject interpolates the id", async () => {
    fetchMock.mockResolvedValue(mockResponse({ json: { id: "p9" } }))
    await getProject("p9")
    expect(fetchMock).toHaveBeenCalledWith("/api/proxy/projects/p9")
  })

  it("throws with status when GET is not ok", async () => {
    fetchMock.mockResolvedValue(mockResponse({ ok: false, status: 500 }))
    await expect(getProjects()).rejects.toThrow(/projects failed: 500/)
  })
})

describe("apiPost wrappers", () => {
  it("createJoin posts JSON to the joins endpoint", async () => {
    fetchMock.mockResolvedValue(mockResponse({ json: { id: "ds-joined" } }))
    await createJoin("proj-1", {
      left_dataset_id: "l",
      right_dataset_id: "r",
      join_type: "inner",
      join_keys: ["id"],
    })
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe("/api/proxy/projects/proj-1/joins")
    expect(init.method).toBe("POST")
    expect(init.headers["Content-Type"]).toBe("application/json")
    expect(JSON.parse(init.body)).toMatchObject({ join_type: "inner", join_keys: ["id"] })
  })

  it("predictSingle wraps input under input_data", async () => {
    fetchMock.mockResolvedValue(mockResponse({ json: { prediction: 1 } }))
    await predictSingle("run-1", { age: 50 })
    const init = fetchMock.mock.calls[0][1]
    expect(JSON.parse(init.body)).toEqual({ input_data: { age: 50 } })
  })

  it("surfaces the server detail text on POST failure", async () => {
    fetchMock.mockResolvedValue(mockResponse({ ok: false, status: 422, text: "target_column required" }))
    await expect(predictSingle("run-1", {})).rejects.toThrow("target_column required")
  })
})

describe("apiDelete", () => {
  it("treats 204 as success", async () => {
    fetchMock.mockResolvedValue(mockResponse({ ok: false, status: 204 }))
    await expect(deleteProject("p1")).resolves.toBeUndefined()
    expect(fetchMock).toHaveBeenCalledWith("/api/proxy/projects/p1", { method: "DELETE" })
  })

  it("throws on non-204 failure", async () => {
    fetchMock.mockResolvedValue(mockResponse({ ok: false, status: 500, text: "boom" }))
    await expect(deleteProject("p1")).rejects.toThrow("boom")
  })
})

describe("multipart uploads", () => {
  it("uploadDataset builds a FormData with file, role and target_column", async () => {
    fetchMock.mockResolvedValue(mockResponse({ json: { id: "ds-1" } }))
    const file = new File(["a,b\n1,2"], "data.csv", { type: "text/csv" })
    await uploadDataset("proj-1", file, "training", "target")
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe("/api/proxy/projects/proj-1/datasets")
    expect(init.method).toBe("POST")
    const body = init.body as FormData
    expect(body.get("role")).toBe("training")
    expect(body.get("target_column")).toBe("target")
    expect((body.get("file") as File).name).toBe("data.csv")
  })

  it("uploadDataset omits target_column when not provided", async () => {
    fetchMock.mockResolvedValue(mockResponse({ json: { id: "ds-1" } }))
    const file = new File(["x"], "x.csv")
    await uploadDataset("proj-1", file, "holdout")
    const body = fetchMock.mock.calls[0][1].body as FormData
    expect(body.get("target_column")).toBeNull()
  })
})

describe("downloadDeliverable", () => {
  it("returns a Blob on success", async () => {
    const blob = new Blob(["pdf"])
    fetchMock.mockResolvedValue(mockResponse({ blob }))
    const result = await downloadDeliverable("run-1", "executive_summary")
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/proxy/runs/run-1/deliverables/executive_summary/download",
    )
    expect(result).toBe(blob)
  })

  it("throws on failure", async () => {
    fetchMock.mockResolvedValue(mockResponse({ ok: false, status: 404 }))
    await expect(downloadDeliverable("run-1", "x")).rejects.toThrow(/404/)
  })
})

describe("fetcher", () => {
  it("parses json on ok", async () => {
    fetchMock.mockResolvedValue(mockResponse({ json: { a: 1 } }))
    await expect(fetcher("/api/proxy/x")).resolves.toEqual({ a: 1 })
  })

  it("throws the status string on error", async () => {
    fetchMock.mockResolvedValue(mockResponse({ ok: false, status: 503 }))
    await expect(fetcher("/api/proxy/x")).rejects.toThrow("503")
  })
})
