import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { openChatStream, type ChatStreamHandlers } from "@/lib/chat-stream"
import type { SSEEvent } from "@/lib/types"

const fetchMock = vi.fn()

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock)
  fetchMock.mockReset()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

/** A reader that yields the given raw string chunks as encoded bytes. */
function readerFrom(chunks: string[]) {
  const encoder = new TextEncoder()
  let i = 0
  return {
    read: vi.fn().mockImplementation(() => {
      if (i < chunks.length) {
        return Promise.resolve({ done: false, value: encoder.encode(chunks[i++]) })
      }
      return Promise.resolve({ done: true, value: undefined })
    }),
  }
}

function streamResponse(chunks: string[], ok = true, status = 200, text = "") {
  return {
    ok,
    status,
    text: vi.fn().mockResolvedValue(text),
    body: { getReader: () => readerFrom(chunks) },
  }
}

/** Collect handler invocations; resolve when onDone/onError fires. */
function makeHandlers() {
  const calls = {
    text: [] as string[],
    diffs: [] as unknown[],
    intents: [] as unknown[],
    artifacts: [] as unknown[],
    errors: [] as string[],
    done: 0,
  }
  let resolve!: () => void
  const finished = new Promise<void>((r) => (resolve = r))
  const handlers: ChatStreamHandlers = {
    onTextChunk: (c) => calls.text.push(c),
    onStrategyDiff: (e) => calls.diffs.push(e),
    onIntent: (e) => calls.intents.push(e),
    onArtifactTask: (e) => calls.artifacts.push(e),
    onError: (e) => {
      calls.errors.push(e)
      resolve()
    },
    onDone: () => {
      calls.done += 1
      resolve()
    },
  }
  return { calls, handlers, finished }
}

const sse = (event: SSEEvent) => `data: ${JSON.stringify(event)}\n`

describe("openChatStream", () => {
  it("dispatches text chunks in order then done", async () => {
    fetchMock.mockResolvedValue(
      streamResponse([
        sse({ type: "text_chunk", content: "Hello " }),
        sse({ type: "text_chunk", content: "world" }),
        sse({ type: "done" }),
      ]),
    )
    const { calls, handlers, finished } = makeHandlers()
    openChatStream("run-1", "hi", handlers)
    await finished
    expect(calls.text).toEqual(["Hello ", "world"])
    expect(calls.done).toBe(1)
    expect(calls.errors).toEqual([])
  })

  it("posts content to the run chat endpoint", async () => {
    fetchMock.mockResolvedValue(streamResponse([sse({ type: "done" })]))
    const { handlers, finished } = makeHandlers()
    openChatStream("run-77", "use class_weight", handlers)
    await finished
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe("/api/proxy/runs/run-77/chat")
    expect(init.method).toBe("POST")
    expect(JSON.parse(init.body)).toEqual({ content: "use class_weight" })
  })

  it("routes strategy_diff, intent and artifact_task events", async () => {
    const diffEvent = {
      type: "strategy_diff" as const,
      diffs: [{ field_path: "threshold", before: 0.5, after: 0.4, summary: "", run_id: "run-1" }],
    }
    const intentEvent = {
      type: "intent" as const,
      intent: {
        intent: "modify" as const,
        confidence: 0.9,
        category: "threshold" as const,
        structured_payload: {},
        needs_confirmation: true,
        reasoning: "",
      },
    }
    const artifactEvent = { type: "artifact_task" as const, task_id: "t1", artifact_type: "notebook" }
    fetchMock.mockResolvedValue(
      streamResponse([sse(diffEvent), sse(intentEvent), sse(artifactEvent), sse({ type: "done" })]),
    )
    const { calls, handlers, finished } = makeHandlers()
    openChatStream("run-1", "x", handlers)
    await finished
    expect(calls.diffs).toEqual([diffEvent])
    expect(calls.intents).toEqual([intentEvent])
    expect(calls.artifacts).toEqual([artifactEvent])
  })

  it("handles SSE frames split across chunk boundaries", async () => {
    const frame = sse({ type: "text_chunk", content: "joined" })
    const mid = Math.floor(frame.length / 2)
    fetchMock.mockResolvedValue(
      streamResponse([frame.slice(0, mid), frame.slice(mid), sse({ type: "done" })]),
    )
    const { calls, handlers, finished } = makeHandlers()
    openChatStream("run-1", "x", handlers)
    await finished
    expect(calls.text).toEqual(["joined"])
  })

  it("skips malformed JSON lines without crashing", async () => {
    fetchMock.mockResolvedValue(
      streamResponse([
        "data: {not valid json}\n",
        sse({ type: "text_chunk", content: "ok" }),
        sse({ type: "done" }),
      ]),
    )
    const { calls, handlers, finished } = makeHandlers()
    openChatStream("run-1", "x", handlers)
    await finished
    expect(calls.text).toEqual(["ok"])
    expect(calls.errors).toEqual([])
  })

  it("dispatches an error event payload", async () => {
    fetchMock.mockResolvedValue(streamResponse([sse({ type: "error", error: "model overloaded" })]))
    const { calls, handlers, finished } = makeHandlers()
    openChatStream("run-1", "x", handlers)
    await finished
    expect(calls.errors).toEqual(["model overloaded"])
  })

  it("reports HTTP errors via onError using response text", async () => {
    fetchMock.mockResolvedValue(streamResponse([], false, 500, "internal error"))
    const { calls, handlers, finished } = makeHandlers()
    openChatStream("run-1", "x", handlers)
    await finished
    expect(calls.errors).toEqual(["internal error"])
  })

  it("returns an AbortController whose signal is passed to fetch", async () => {
    fetchMock.mockResolvedValue(streamResponse([sse({ type: "done" })]))
    const { handlers, finished } = makeHandlers()
    const controller = openChatStream("run-1", "x", handlers)
    await finished
    expect(controller).toBeInstanceOf(AbortController)
    expect(fetchMock.mock.calls[0][1].signal).toBe(controller.signal)
  })
})
