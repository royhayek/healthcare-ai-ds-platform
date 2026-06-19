"use client"

import { useRef, useState, useEffect, useLayoutEffect, useCallback } from "react"
import { useRouter } from "next/navigation"
import { useChatStore } from "@/store/chatStore"
import { useStrategyStore } from "@/store/strategyStore"
import { useJobStore } from "@/store/jobStore"
import { type SSEEvent } from "@/lib/types"
import ChatMessage from "./ChatMessage"
import ChatComposer from "./ChatComposer"
import StrategyDiffCard from "./StrategyDiff"
import IntentConfirmation from "./IntentConfirmation"
import ContextBadge from "./ContextBadge"
import ThemeToggle from "@/components/ui/ThemeToggle"

const MIN_WIDTH = 280
const MAX_WIDTH = 960
const DEFAULT_WIDTH = 500
const STORAGE_KEY = "copilot-panel-width"

export default function ChatPanel() {
  const {
    runId,
    messages,
    isStreaming,
    pendingIntent,
    addMessage,
    appendChunk,
    finalizeStream,
    setPendingIntent,
    loadMessages,
    clearHistory,
  } = useChatStore()

  const { applyDiffs, pendingDiffs } = useStrategyStore()

  const router = useRouter()

  const [error, setError] = useState<string | null>(null)
  const [artifactNotice, setArtifactNotice] = useState<string | null>(null)
  const [historyLoading, setHistoryLoading] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  const [panelWidth, setPanelWidth] = useState(DEFAULT_WIDTH)
  const isDragging = useRef(false)

  // Restore saved width before first paint to avoid flicker
  useLayoutEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY)
    const parsed = saved ? parseInt(saved, 10) : NaN
    if (!isNaN(parsed)) {
      setPanelWidth(Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, parsed)))
    }
  }, [])

  const onDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    isDragging.current = true
    document.body.style.cursor = "col-resize"
    document.body.style.userSelect = "none"
  }, [])

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging.current) return
      const newWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, window.innerWidth - e.clientX))
      setPanelWidth(newWidth)
    }
    const onMouseUp = () => {
      if (!isDragging.current) return
      isDragging.current = false
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
      setPanelWidth((w) => {
        localStorage.setItem(STORAGE_KEY, String(w))
        return w
      })
    }
    window.addEventListener("mousemove", onMouseMove)
    window.addEventListener("mouseup", onMouseUp)
    return () => {
      window.removeEventListener("mousemove", onMouseMove)
      window.removeEventListener("mouseup", onMouseUp)
    }
  }, [])

  // Fetch persisted history from the server and map it to store messages.
  const fetchHistory = useCallback(async (): Promise<Parameters<typeof loadMessages>[0] | null> => {
    if (!runId) return null
    const r = await fetch(`/api/proxy/runs/${runId}/chat/history`)
    if (!r.ok) return null
    const data: Array<{ role: string; content: string; intent?: unknown }> = await r.json()
    return data.map((m) => ({
      role: m.role as "user" | "assistant" | "system",
      content: m.content,
      intent: (m.intent as Parameters<typeof loadMessages>[0][0]["intent"]) ?? null,
      diffs: null,
      isStreaming: false,
    }))
  }, [runId])

  const scrollToBottom = useCallback(() => {
    setTimeout(() => {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
    }, 50)
  }, [])

  // Load persisted history whenever the active run changes (full replace).
  useEffect(() => {
    if (!runId) {
      clearHistory()
      return
    }
    let cancelled = false
    setHistoryLoading(true)
    fetchHistory()
      .then((msgs) => {
        if (cancelled || !msgs) return
        loadMessages(msgs)
        scrollToBottom()
      })
      .catch(() => { /* history fetch failure is non-fatal */ })
      .finally(() => { if (!cancelled) setHistoryLoading(false) })
    return () => { cancelled = true }
  }, [runId, fetchHistory, scrollToBottom, loadMessages, clearHistory])

  // Poll for server-seeded messages while the run is active. The pipeline writes
  // proactive co-pilot messages directly to the DB (e.g. the EDA checkpoint
  // summary); without this they only appear on a remount/reload. We reconcile by
  // replacing the local list only when the server has MORE messages than we hold
  // and we are not mid-stream - the server persists every user+assistant turn, so
  // when idle it is an authoritative superset and a longer list means a new
  // server-seeded message arrived. Never replace while streaming, to avoid
  // dropping the in-flight assistant turn.
  useEffect(() => {
    if (!runId) return
    const tick = async () => {
      const status = useJobStore.getState().run?.status
      if (status === "completed" || status === "failed") return
      if (useChatStore.getState().isStreaming) return
      const msgs = await fetchHistory().catch(() => null)
      if (!msgs) return
      const cur = useChatStore.getState()
      if (cur.isStreaming || msgs.length <= cur.messages.length) return
      loadMessages(msgs)
      scrollToBottom()
    }
    const interval = setInterval(tick, 2500)
    return () => clearInterval(interval)
  }, [runId, fetchHistory, scrollToBottom, loadMessages])

  const sendMessage = async (content: string) => {
    if (!runId || isStreaming) return
    setError(null)

    // Add user message immediately
    addMessage({ role: "user", content, isStreaming: false })

    // Add empty assistant message that will be filled by streaming
    const assistantId = addMessage({ role: "assistant", content: "", isStreaming: true })

    useChatStore.setState({ isStreaming: true, streamingMessageId: assistantId })

    try {
      const response = await fetch(`/api/proxy/runs/${runId}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      let finalIntent = null
      let finalDiffs: ReturnType<typeof useChatStore.getState>["messages"][0]["diffs"] = []

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() ?? ""

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue
          const event: SSEEvent = JSON.parse(line.slice(6))

          switch (event.type) {
            case "text_chunk":
              appendChunk(assistantId, event.content)
              scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })
              break
            case "strategy_diff":
              applyDiffs(event.diffs)
              finalDiffs = event.diffs
              break
            case "intent":
              finalIntent = event.intent
              break
            case "artifact_task":
              if (event.artifact_type === "notebook") {
                setArtifactNotice("Notebook queued - check the deliverables page shortly.")
                setTimeout(() => setArtifactNotice(null), 8000)
              } else if (event.artifact_type?.startsWith("plots:")) {
                const stage = event.artifact_type.slice(6)
                setArtifactNotice(`Rendering ${stage} plots - they will appear in the results page shortly.`)
                setTimeout(() => setArtifactNotice(null), 8000)
              }
              break
            case "rerun_triggered":
              setArtifactNotice(
                `Re-running ${event.step} to apply your change - watch the progress feed; the checkpoint will refresh when it completes.`,
              )
              // If the user is reviewing this run's checkpoint, the results they
              // are looking at are now stale. Take them back to the progress feed
              // to watch the re-run rather than leaving them on a dead page.
              if (typeof window !== "undefined" && window.location.pathname.includes("/checkpoint/")) {
                router.push(window.location.pathname.split("/checkpoint/")[0])
              }
              break
            case "error":
              setError(event.error)
              break
            case "done":
              break
          }
        }
      }

      finalizeStream(assistantId, finalIntent, finalDiffs ?? [])
    } catch (err) {
      setError(err instanceof Error ? err.message : "Stream failed")
      finalizeStream(assistantId, null, [])
    }
  }

  const isPanelActive = !!runId

  return (
    <aside
      className="flex flex-col border-l border-neutral-800 bg-neutral-900 relative"
      style={{ width: panelWidth, minWidth: MIN_WIDTH, maxWidth: MAX_WIDTH }}
    >
      {/* Drag handle */}
      <div
        onMouseDown={onDragStart}
        className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize group z-10 hover:bg-indigo-500/40 transition-colors"
        title="Drag to resize"
      >
        <div className="absolute left-0 top-1/2 -translate-y-1/2 h-8 w-1 rounded-full bg-neutral-700 group-hover:bg-indigo-400 transition-colors" />
      </div>

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-neutral-800">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isPanelActive ? "bg-emerald-400" : "bg-neutral-600"}`} />
          <span className="text-sm font-medium text-neutral-200">Co-pilot</span>
        </div>
        <div className="flex items-center gap-2">
          {runId && <ContextBadge runId={runId} />}
          <ThemeToggle />
        </div>
      </div>

      {/* Pending diffs banner */}
      {pendingDiffs.length > 0 && (
        <div className="px-3 py-2 border-b border-neutral-800 space-y-1">
          {pendingDiffs.map((diff, i) => (
            <StrategyDiffCard key={i} diff={diff} />
          ))}
        </div>
      )}

      {/* Pending intent confirmation */}
      {pendingIntent && (
        <IntentConfirmation
          intent={pendingIntent}
          onConfirm={() => {
            // Confirmed modify intents without confirmation are handled server-side.
            // This handles the needs_confirmation=true case.
            setPendingIntent(null)
          }}
          onDismiss={() => setPendingIntent(null)}
        />
      )}

      {/* Artifact task notice (notebook export queued, etc.) */}
      {artifactNotice && (
        <div className="mx-3 mt-2 px-3 py-2 rounded-lg border border-indigo-900/50 bg-indigo-950/20 text-xs text-indigo-300">
          {artifactNotice}
        </div>
      )}

      {/* Message list */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 py-3 space-y-3">
        {!isPanelActive && (
          <p className="text-xs text-neutral-600 text-center pt-8">
            Start or open a run to activate the co-pilot.
          </p>
        )}
        {isPanelActive && historyLoading && messages.length === 0 && (
          <p className="text-xs text-neutral-600 text-center pt-8 animate-pulse">
            Loading conversation…
          </p>
        )}
        {isPanelActive && !historyLoading && messages.length === 0 && (
          <p className="text-xs text-neutral-600 text-center pt-8">
            No messages yet. Ask anything about this run.
          </p>
        )}
        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}
        {error && (
          <div className="rounded px-3 py-2 bg-red-950 border border-red-800 text-xs text-red-300">
            {error}
          </div>
        )}
      </div>

      {/* Composer */}
      <ChatComposer
        disabled={!isPanelActive || isStreaming}
        onSend={sendMessage}
      />
    </aside>
  )
}
