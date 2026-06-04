"use client"

import { type ChatIntent } from "@/lib/types"

interface Props {
  intent: ChatIntent
  onConfirm: () => void
  onDismiss: () => void
}

export default function IntentConfirmation({ intent, onConfirm, onDismiss }: Props) {
  return (
    <div className="mx-3 my-2 rounded-md border border-indigo-700/50 bg-indigo-950/40 px-3 py-2.5 text-xs">
      <p className="text-indigo-300 font-medium mb-1">Apply this change?</p>
      <p className="text-neutral-300 mb-2">{intent.reasoning}</p>
      <div className="flex gap-2">
        <button
          onClick={onConfirm}
          className="rounded px-2.5 py-1 bg-indigo-600 text-white text-xs font-medium hover:bg-indigo-500"
        >
          Apply
        </button>
        <button
          onClick={onDismiss}
          className="rounded px-2.5 py-1 bg-neutral-700 text-neutral-300 text-xs font-medium hover:bg-neutral-600"
        >
          Dismiss
        </button>
      </div>
    </div>
  )
}
