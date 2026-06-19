import { beforeEach, describe, expect, it } from "vitest"
import { useChatStore } from "@/store/chatStore"
import type { ChatIntent, StrategyDiff } from "@/lib/types"

const intent = (overrides: Partial<ChatIntent> = {}): ChatIntent => ({
  intent: "modify",
  confidence: 0.9,
  category: "preprocessing",
  structured_payload: {},
  needs_confirmation: false,
  reasoning: "because",
  ...overrides,
})

const diff: StrategyDiff = {
  field_path: "preprocessing.imbalance",
  before: "smote",
  after: "class_weight",
  summary: "Use class_weight",
  run_id: "run-1",
}

beforeEach(() => {
  useChatStore.setState({
    runId: null,
    messages: [],
    pendingIntent: null,
    isStreaming: false,
    streamingMessageId: null,
  })
})

describe("chatStore", () => {
  it("setRunId stores the run id", () => {
    useChatStore.getState().setRunId("run-9")
    expect(useChatStore.getState().runId).toBe("run-9")
  })

  it("addMessage appends and returns a unique id", () => {
    const id1 = useChatStore.getState().addMessage({ role: "user", content: "hi" })
    const id2 = useChatStore.getState().addMessage({ role: "assistant", content: "yo" })
    expect(id1).not.toBe(id2)
    const msgs = useChatStore.getState().messages
    expect(msgs).toHaveLength(2)
    expect(msgs[0]).toMatchObject({ id: id1, role: "user", content: "hi" })
    expect(msgs[1]).toMatchObject({ id: id2, role: "assistant", content: "yo" })
  })

  it("appendChunk concatenates to the matching message only", () => {
    const id = useChatStore.getState().addMessage({ role: "assistant", content: "" })
    const other = useChatStore.getState().addMessage({ role: "user", content: "stable" })
    useChatStore.getState().appendChunk(id, "Hel")
    useChatStore.getState().appendChunk(id, "lo")
    const msgs = useChatStore.getState().messages
    expect(msgs.find((m) => m.id === id)?.content).toBe("Hello")
    expect(msgs.find((m) => m.id === other)?.content).toBe("stable")
  })

  it("finalizeStream clears streaming flags and attaches intent + diffs", () => {
    const id = useChatStore.getState().addMessage({ role: "assistant", content: "x", isStreaming: true })
    useChatStore.setState({ isStreaming: true, streamingMessageId: id })
    const i = intent({ needs_confirmation: false })
    useChatStore.getState().finalizeStream(id, i, [diff])

    const s = useChatStore.getState()
    expect(s.isStreaming).toBe(false)
    expect(s.streamingMessageId).toBeNull()
    const msg = s.messages.find((m) => m.id === id)
    expect(msg?.isStreaming).toBe(false)
    expect(msg?.intent).toEqual(i)
    expect(msg?.diffs).toEqual([diff])
  })

  it("finalizeStream surfaces pendingIntent only when confirmation is needed", () => {
    const id = useChatStore.getState().addMessage({ role: "assistant", content: "" })
    const needs = intent({ needs_confirmation: true })
    useChatStore.getState().finalizeStream(id, needs, [])
    expect(useChatStore.getState().pendingIntent).toEqual(needs)

    const id2 = useChatStore.getState().addMessage({ role: "assistant", content: "" })
    useChatStore.getState().finalizeStream(id2, intent({ needs_confirmation: false }), [])
    expect(useChatStore.getState().pendingIntent).toBeNull()
  })

  it("finalizeStream with null intent clears pendingIntent", () => {
    const id = useChatStore.getState().addMessage({ role: "assistant", content: "" })
    useChatStore.getState().finalizeStream(id, null, [])
    expect(useChatStore.getState().pendingIntent).toBeNull()
  })

  it("setPendingIntent sets and clears", () => {
    const i = intent()
    useChatStore.getState().setPendingIntent(i)
    expect(useChatStore.getState().pendingIntent).toEqual(i)
    useChatStore.getState().setPendingIntent(null)
    expect(useChatStore.getState().pendingIntent).toBeNull()
  })

  it("loadMessages replaces history and assigns ids", () => {
    useChatStore.getState().addMessage({ role: "user", content: "old" })
    useChatStore.getState().loadMessages([
      { role: "user", content: "a" },
      { role: "assistant", content: "b" },
    ])
    const msgs = useChatStore.getState().messages
    expect(msgs).toHaveLength(2)
    expect(msgs.map((m) => m.content)).toEqual(["a", "b"])
    expect(msgs[0].id).toBeTruthy()
    expect(msgs[0].id).not.toBe(msgs[1].id)
  })

  it("clearHistory empties messages and pendingIntent", () => {
    useChatStore.getState().addMessage({ role: "user", content: "x" })
    useChatStore.getState().setPendingIntent(intent())
    useChatStore.getState().clearHistory()
    const s = useChatStore.getState()
    expect(s.messages).toEqual([])
    expect(s.pendingIntent).toBeNull()
  })
})
