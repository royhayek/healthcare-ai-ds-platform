"use client"

/**
 * Client-side auth guard for dashboard routes.
 *
 * Checks the Supabase session on mount. If no session exists, redirects to
 * /login. Renders a blank screen while checking to avoid layout flash.
 */

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { supabase } from "@/lib/supabase"

export default function AuthGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const [ready, setReady] = useState(false)

  useEffect(() => {
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
