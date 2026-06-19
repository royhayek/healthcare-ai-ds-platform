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
import AccountMenu, { AccountMenuBrand } from "@/components/auth/AccountMenu"

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <AuthGuard>
      <div className="flex h-screen overflow-hidden bg-neutral-950">
        {/* Canvas column - global top bar + scrollable page content */}
        <div className="flex flex-1 min-w-0 flex-col">
          {/* Global top bar - brand left, account menu right (always available) */}
          <header className="flex h-11 shrink-0 items-center justify-between border-b border-neutral-800 bg-neutral-950 px-4">
            <AccountMenuBrand />
            <AccountMenu />
          </header>

          {/* Canvas - all project pages render here */}
          <main className="flex-1 min-w-0 overflow-auto">{children}</main>
        </div>

        {/* ChatPanel - mounted once, persists across all dashboard routes */}
        <ChatPanel />
      </div>
    </AuthGuard>
  )
}
