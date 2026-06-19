"use client"

/**
 * Account settings page.
 *
 * Lets the signed-in user review their identity and change their password via
 * Supabase Auth. Sign-out is available here as well as in the top-bar account
 * menu. This is a self-service settings surface, not an admin panel - it only
 * ever operates on the current session's own user.
 *
 * Dev mode (NEXT_PUBLIC_DEV_MODE=true): a session is not required by AuthGuard.
 * We still load the real user when one exists; password changes are only
 * disabled when there is genuinely no session to mutate.
 */

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { ArrowLeft, LogOut } from "lucide-react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { getUser, signOut, updatePassword } from "@/lib/supabase"

const DEV_MODE = process.env.NEXT_PUBLIC_DEV_MODE === "true"
const MIN_PASSWORD_LENGTH = 8

export default function AccountPage() {
  const router = useRouter()
  const [email, setEmail] = useState<string | null>(null)
  const [createdAt, setCreatedAt] = useState<string | null>(null)
  const [hasSession, setHasSession] = useState(false)
  const [loading, setLoading] = useState(true)

  const [password, setPassword] = useState("")
  const [confirm, setConfirm] = useState("")
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)
  const [signingOut, setSigningOut] = useState(false)

  useEffect(() => {
    // Load the real user when a session exists, regardless of dev mode.
    getUser()
      .then((user) => {
        setHasSession(Boolean(user))
        setEmail(user?.email ?? (DEV_MODE ? "dev@local" : null))
        setCreatedAt(user?.created_at ?? null)
      })
      .finally(() => setLoading(false))
  }, [])

  const handleChangePassword = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setSuccess(false)

    if (password.length < MIN_PASSWORD_LENGTH) {
      setError(`Password must be at least ${MIN_PASSWORD_LENGTH} characters.`)
      return
    }
    if (password !== confirm) {
      setError("Passwords do not match.")
      return
    }

    setSaving(true)
    try {
      const { error: updateError } = await updatePassword(password)
      if (updateError) {
        setError(updateError.message)
        return
      }
      setSuccess(true)
      setPassword("")
      setConfirm("")
    } catch {
      setError("Something went wrong. Please try again.")
    } finally {
      setSaving(false)
    }
  }

  const handleSignOut = async () => {
    setSigningOut(true)
    try {
      await signOut()
    } finally {
      router.replace("/login")
    }
  }

  const inputClass =
    "w-full rounded-lg border border-neutral-700 bg-neutral-800 px-3 py-2 text-sm text-neutral-100 placeholder-neutral-600 focus:border-indigo-600 focus:outline-none focus:ring-1 focus:ring-indigo-600 disabled:opacity-50"

  return (
    <div className="mx-auto max-w-2xl p-8">
      <Link
        href="/"
        className="inline-flex items-center gap-1.5 text-xs text-neutral-500 transition-colors hover:text-neutral-300"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        Back to projects
      </Link>

      <h1 className="mt-4 text-lg font-semibold text-neutral-100">Account</h1>
      <p className="mt-0.5 text-sm text-neutral-500">
        Manage your identity, password, and session.
      </p>

      {/* ── Profile ── */}
      <section className="mt-6 rounded-xl border border-neutral-800 bg-neutral-900 p-5">
        <h2 className="text-sm font-semibold text-neutral-200">Profile</h2>
        <dl className="mt-3 space-y-2.5 text-sm">
          <div className="flex items-center justify-between gap-4">
            <dt className="text-neutral-500">Email</dt>
            <dd className="truncate text-neutral-200" title={email ?? undefined}>
              {loading ? "Loading…" : email ?? "Unknown"}
            </dd>
          </div>
          {createdAt && (
            <div className="flex items-center justify-between gap-4">
              <dt className="text-neutral-500">Member since</dt>
              <dd className="text-neutral-300">
                {new Date(createdAt).toLocaleDateString(undefined, {
                  year: "numeric",
                  month: "long",
                  day: "numeric",
                })}
              </dd>
            </div>
          )}
        </dl>
      </section>

      {/* ── Change password ── */}
      <section className="mt-4 rounded-xl border border-neutral-800 bg-neutral-900 p-5">
        <h2 className="text-sm font-semibold text-neutral-200">Change password</h2>
        {!loading && !hasSession ? (
          <p className="mt-2 text-xs text-neutral-500">
            Password changes require an active session. Sign in to change your password.
          </p>
        ) : (
          <form onSubmit={handleChangePassword} className="mt-3 space-y-4">
            <div className="space-y-1.5">
              <label className="block text-xs text-neutral-400" htmlFor="new-password">
                New password
              </label>
              <input
                id="new-password"
                type="password"
                autoComplete="new-password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className={inputClass}
                placeholder="••••••••"
              />
            </div>

            <div className="space-y-1.5">
              <label className="block text-xs text-neutral-400" htmlFor="confirm-password">
                Confirm new password
              </label>
              <input
                id="confirm-password"
                type="password"
                autoComplete="new-password"
                required
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                className={inputClass}
                placeholder="••••••••"
              />
            </div>

            {error && (
              <p className="rounded-lg border border-red-900/50 bg-red-950/40 px-3 py-2 text-xs text-red-400">
                {error}
              </p>
            )}
            {success && (
              <p className="rounded-lg border border-emerald-900/50 bg-emerald-950/40 px-3 py-2 text-xs text-emerald-400">
                Password updated.
              </p>
            )}

            <Button type="submit" size="sm" disabled={saving}>
              {saving ? "Updating…" : "Update password"}
            </Button>
          </form>
        )}
      </section>

      {/* ── Session ── */}
      <section className="mt-4 rounded-xl border border-neutral-800 bg-neutral-900 p-5">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h2 className="text-sm font-semibold text-neutral-200">Session</h2>
            <p className="mt-0.5 text-xs text-neutral-500">
              Sign out of this device. You&apos;ll need to sign in again to continue.
            </p>
          </div>
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={handleSignOut}
            disabled={signingOut}
            data-testid="account-sign-out"
          >
            <LogOut className="mr-1.5 h-3.5 w-3.5" />
            {signingOut ? "Signing out…" : "Sign out"}
          </Button>
        </div>
      </section>
    </div>
  )
}
