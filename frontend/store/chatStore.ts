"use client"

import { create } from "zustand"
import { type ChatIntent, type Message, type StrategyDiff } from "@/lib/types"

interface ChatState {
  runId: string | null
  messages: Message[]
  pendingIntent: ChatIntent | null
  isStreaming: boolean
  streamingMessageId: string | null

  setRunId: (id: string | null) => void
  addMessage: (msg: Omit<Message, "id">) => string
  appendChunk: (messageId: string, chunk: string) => void
  finalizeStream: (messageId: string, intent: ChatIntent | null, diffs: StrategyDiff[]) => void
  setPendingIntent: (intent: ChatIntent | null) => void
  loadMessages: (msgs: Omit<Message, "id">[]) => void
  clearHistory: () => void
}

let _idCounter = 0
const uid = () => `msg-${++_idCounter}-${Date.now()}`

export const useChatStore = create<ChatState>()((set) => ({
  runId: null,
  messages: [],
  pendingIntent: null,
  isStreaming: false,
  streamingMessageId: null,

  setRunId: (id) => set({ runId: id }),

  addMessage: (msg) => {
    const id = uid()
    set((s) => ({ messages: [...s.messages, { ...msg, id }] }))
    return id
  },

  appendChunk: (messageId, chunk) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === messageId ? { ...m, content: m.content + chunk } : m,
      ),
    })),

  finalizeStream: (messageId, intent, diffs) =>
    set((s) => ({
      isStreaming: false,
      streamingMessageId: null,
      pendingIntent: intent?.needs_confirmation ? intent : null,
      messages: s.messages.map((m) =>
        m.id === messageId ? { ...m, isStreaming: false, intent, diffs } : m,
      ),
    })),

  setPendingIntent: (intent) => set({ pendingIntent: intent }),

  loadMessages: (msgs) =>
    set({ messages: msgs.map((m) => ({ ...m, id: uid() })) }),

  clearHistory: () => set({ messages: [], pendingIntent: null }),
}))
