/** SSE streaming helper for the persistent chat co-pilot (§2, §21).
 *
 * Extracted from ChatPanel.tsx so any component (or test) can consume the
 * chat SSE stream without the full panel. Handles text chunks, strategy diffs,
 * intent events, artifact tasks, and done/error signals.
 */

import type { SSEEvent } from "@/lib/types"

export interface ChatStreamHandlers {
  onTextChunk: (chunk: string) => void
  onStrategyDiff: (diffs: SSEEvent & { type: "strategy_diff" }) => void
  onIntent: (event: SSEEvent & { type: "intent" }) => void
  onArtifactTask: (event: SSEEvent & { type: "artifact_task" }) => void
  onError: (error: string) => void
  onDone: () => void
}

/**
 * Open a chat SSE stream for a given run and message. Calls the appropriate
 * handler for each event type. Returns an AbortController so the caller can
 * cancel the stream (e.g., on component unmount).
 */
export function openChatStream(
  runId: string,
  content: string,
  handlers: ChatStreamHandlers,
): AbortController {
  const controller = new AbortController()

  fetch(`/api/proxy/runs/${runId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        const text = await res.text().catch(() => String(res.status))
        handlers.onError(text)
        return
      }
      if (!res.body) {
        handlers.onError("No response body from chat endpoint")
        return
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() ?? ""

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue
          const raw = line.slice(6).trim()
          if (!raw) continue
          try {
            const event = JSON.parse(raw) as SSEEvent
            dispatchEvent(event, handlers)
          } catch {
            // Malformed SSE line - skip
          }
        }
      }
    })
    .catch((err: unknown) => {
      if (err instanceof Error && err.name === "AbortError") return
      handlers.onError(err instanceof Error ? err.message : String(err))
    })

  return controller
}

function dispatchEvent(event: SSEEvent, handlers: ChatStreamHandlers): void {
  switch (event.type) {
    case "text_chunk":
      handlers.onTextChunk(event.content)
      break
    case "strategy_diff":
      handlers.onStrategyDiff(event)
      break
    case "intent":
      handlers.onIntent(event)
      break
    case "artifact_task":
      handlers.onArtifactTask(event)
      break
    case "error":
      handlers.onError(event.error)
      break
    case "done":
      handlers.onDone()
      break
  }
}
