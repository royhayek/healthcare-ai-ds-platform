"use client"

/** Reusable progress event feed - renders a vertical list of StepCards from SSE events.
 *
 * Consumed by the analysis run page and any component that wants to show
 * live pipeline progress. Separate from the inline implementation in the
 * analysis page so it can be reused in checkpoint cards and tests.
 */

import { useEffect, useRef } from "react"
import { StepCard, type StepStatus } from "./StepCard"

export interface FeedEvent {
  id: string
  eventType: string
  message: string
  pct?: number
  timestamp: number
}

interface ProgressFeedProps {
  events: FeedEvent[]
  currentStep: string | null
  status: "queued" | "running" | "awaiting_checkpoint" | "completed" | "failed" | null
  autoScroll?: boolean
  className?: string
}

function stepStatus(
  event: FeedEvent,
  isLatest: boolean,
  runStatus: ProgressFeedProps["status"],
): StepStatus {
  if (event.eventType === "error") return "error"
  if (event.eventType === "checkpoint") return "checkpoint"
  if (isLatest && runStatus === "running") return "running"
  if (isLatest && runStatus === "failed") return "error"
  return "done"
}

export function ProgressFeed({
  events,
  currentStep,
  status,
  autoScroll = true,
  className,
}: ProgressFeedProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }, [events.length, autoScroll])

  const startTimes = useRef<Record<string, number>>({})

  const eventsWithElapsed = events.map((ev, i) => {
    const isFirst = i === 0 || events[i - 1].eventType !== ev.eventType
    if (isFirst) {
      startTimes.current[ev.id] = ev.timestamp
    }
    const start = startTimes.current[ev.id] ?? ev.timestamp
    return { ...ev, elapsed: (ev.timestamp - start) / 1000 }
  })

  return (
    <div className={className}>
      {eventsWithElapsed.map((ev, i) => {
        const isLatest = i === eventsWithElapsed.length - 1
        return (
          <StepCard
            key={ev.id}
            eventType={ev.eventType}
            message={ev.message}
            pct={ev.pct}
            status={stepStatus(ev, isLatest, status)}
            elapsed={isLatest ? ev.elapsed : undefined}
            isLatest={isLatest}
          />
        )
      })}
      <div ref={bottomRef} />
    </div>
  )
}
