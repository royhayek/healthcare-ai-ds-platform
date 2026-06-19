"use client"

/**
 * Client-side auth guard for dashboard routes.
 *
 * Checks the Supabase session on mount. If no session exists, redirects to
 * /login. Renders a blank screen while checking to avoid layout flash.
 *
 * Dev mode (NEXT_PUBLIC_DEV_MODE=true): the session check is skipped entirely.
 * This mirrors the backend `DEV_MODE` auth stub and the API proxy, which both
 * trust a hard-coded `dev-user-1` identity. Without this, the dashboard is
 * unreachable locally because no Supabase project is configured to sign in
 * against. Production builds set NEXT_PUBLIC_DEV_MODE=false (or leave it unset)
 * to restore the real session gate.
 */

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { supabase } from "@/lib/supabase"

const DEV_MODE = process.env.NEXT_PUBLIC_DEV_MODE === "true"

export default function AuthGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const [ready, setReady] = useState(DEV_MODE)

  useEffect(() => {
    // In dev mode the backend trusts a stub user and there is no Supabase
    // session to check - render immediately.
    if (DEV_MODE) {
      setReady(true)
      return
    }

    // Check initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) {
        router.replace("/login")
      } else {
        setReady(true)
      }
    })

    // Also listen for sign-out events that happen while the user is on a page
    const { data: { subscription } } = supabase.auth.onAuthStateChange((event) => {
      if (event === "SIGNED_OUT") {
        router.replace("/login")
      }
    })

    return () => subscription.unsubscribe()
  }, [router])

  if (!ready) return null
  return <>{children}</>
}
