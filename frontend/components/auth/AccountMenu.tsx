"use client"

/**
 * Account menu - the always-available identity + session control in the
 * dashboard top bar.
 *
 * Shows the signed-in user's email behind an avatar button. Opening the menu
 * exposes "Account settings" (-> /account) and "Sign out". Sign-out clears the
 * Supabase session and sends the user to /login; AuthGuard's onAuthStateChange
 * listener also catches SIGNED_OUT, so this works even from a stale tab.
 *
 * Dev mode (NEXT_PUBLIC_DEV_MODE=true): AuthGuard does not require a session,
 * so there may be no Supabase user. We still try to read the real user first -
 * if you actually signed in, your real email is shown - and only fall back to a
 * placeholder identity (mirroring the backend's `dev-user-1`) when no session
 * exists. Sign-out always routes to /login for parity with production.
 */

import { useEffect, useRef, useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { LogOut, Settings } from "lucide-react"
import { getUser, signOut } from "@/lib/supabase"

const DEV_MODE = process.env.NEXT_PUBLIC_DEV_MODE === "true"
const DEV_EMAIL = "dev@local"

export default function AccountMenu() {
  const router = useRouter()
  const [email, setEmail] = useState<string | null>(null)
  const [open, setOpen] = useState(false)
  const [signingOut, setSigningOut] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Always prefer the real signed-in user; fall back to the dev placeholder
    // only when there is genuinely no session.
    getUser().then((user) => {
      setEmail(user?.email ?? (DEV_MODE ? DEV_EMAIL : null))
    })
  }, [])

  // Close on outside click or Escape
  useEffect(() => {
    if (!open) return
    const onClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false)
    }
    document.addEventListener("mousedown", onClick)
    document.addEventListener("keydown", onKey)
    return () => {
      document.removeEventListener("mousedown", onClick)
      document.removeEventListener("keydown", onKey)
    }
  }, [open])

  const handleSignOut = async () => {
    setSigningOut(true)
    try {
      await signOut()
    } finally {
      router.replace("/login")
    }
  }

  const initial = (email ?? "?").charAt(0).toUpperCase()

  return (
    <div className="relative" ref={menuRef}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-label="Account menu"
        aria-haspopup="menu"
        aria-expanded={open}
        data-testid="account-menu-button"
        className="flex h-7 w-7 items-center justify-center rounded-full bg-indigo-700 text-xs font-semibold text-white transition-colors hover:bg-indigo-600 focus:outline-none focus-visible:ring-1 focus-visible:ring-indigo-400"
      >
        {initial}
      </button>

      {open && (
        <div
          role="menu"
          className="absolute right-0 top-9 z-50 w-56 overflow-hidden rounded-lg border border-neutral-800 bg-neutral-900 shadow-xl"
        >
          <div className="border-b border-neutral-800 px-3 py-2.5">
            <p className="text-[11px] uppercase tracking-wide text-neutral-600">Signed in as</p>
            <p className="mt-0.5 truncate text-sm text-neutral-200" title={email ?? undefined}>
              {email ?? "Unknown user"}
            </p>
          </div>

          <Link
            href="/account"
            role="menuitem"
            onClick={() => setOpen(false)}
            className="flex items-center gap-2 px-3 py-2 text-sm text-neutral-300 transition-colors hover:bg-neutral-800 hover:text-neutral-100"
          >
            <Settings className="h-4 w-4 text-neutral-500" />
            Account settings
          </Link>

          <button
            type="button"
            role="menuitem"
            onClick={handleSignOut}
            disabled={signingOut}
            data-testid="sign-out-button"
            className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-neutral-300 transition-colors hover:bg-neutral-800 hover:text-red-300 disabled:opacity-50"
          >
            <LogOut className="h-4 w-4 text-neutral-500" />
            {signingOut ? "Signing out…" : "Sign out"}
          </button>
        </div>
      )}
    </div>
  )
}

/** Small static brand used beside the account menu in the dashboard top bar. */
export function AccountMenuBrand() {
  return (
    <Link
      href="/"
      className="flex items-center gap-2 text-neutral-300 transition-colors hover:text-neutral-100"
    >
      <span className="flex h-5 w-5 items-center justify-center rounded bg-indigo-600">
        <span className="h-2 w-2 rounded-full bg-white" />
      </span>
      <span className="text-sm font-semibold tracking-tight">AI Data Science Co-Pilot</span>
    </Link>
  )
}
