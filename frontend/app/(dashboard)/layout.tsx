/**
 * Dashboard layout - two-pane: canvas left, ChatPanel right.
 *
 * ChatPanel is mounted ONCE here and persists across all child routes.
 * It must never be moved into a page component or sub-layout.
 * The chat history and streaming state live in Zustand (chatStore),
 * not in component state, so they survive client-side navigation.
 *
 * See project hard rule 2.
 */

import ChatPanel from "@/components/chat/ChatPanel"
import AuthGuard from "@/components/auth/AuthGuard"

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <AuthGuard>
      <div className="flex h-screen overflow-hidden bg-neutral-950">
        {/* Canvas - all project pages render here */}
        <main className="flex-1 min-w-0 overflow-auto">{children}</main>

        {/* ChatPanel - mounted once, persists across all dashboard routes */}
        <ChatPanel />
      </div>
    </AuthGuard>
  )
}
