import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"

// Mocks shared across dynamic imports.
const replace = vi.fn()
const getSession = vi.fn()
const onAuthStateChange = vi.fn((_cb?: unknown) => ({
  data: { subscription: { unsubscribe: vi.fn() } },
}))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace }),
}))

vi.mock("@/lib/supabase", () => ({
  supabase: {
    auth: {
      getSession: () => getSession(),
      onAuthStateChange: (cb: unknown) => onAuthStateChange(cb as never),
    },
  },
}))

const ORIGINAL = process.env.NEXT_PUBLIC_DEV_MODE

/** Re-import AuthGuard so the module-level DEV_MODE constant is recomputed. */
async function loadGuard(devMode: string | undefined) {
  if (devMode === undefined) delete process.env.NEXT_PUBLIC_DEV_MODE
  else process.env.NEXT_PUBLIC_DEV_MODE = devMode
  vi.resetModules()
  return (await import("@/components/auth/AuthGuard")).default
}

beforeEach(() => {
  replace.mockReset()
  getSession.mockReset()
  onAuthStateChange.mockClear()
})

afterEach(() => {
  if (ORIGINAL === undefined) delete process.env.NEXT_PUBLIC_DEV_MODE
  else process.env.NEXT_PUBLIC_DEV_MODE = ORIGINAL
})

describe("AuthGuard", () => {
  it("renders children immediately in dev mode without checking the session", async () => {
    const AuthGuard = await loadGuard("true")
    render(
      <AuthGuard>
        <div>protected</div>
      </AuthGuard>,
    )
    expect(screen.getByText("protected")).toBeInTheDocument()
    expect(getSession).not.toHaveBeenCalled()
    expect(replace).not.toHaveBeenCalled()
  })

  it("redirects to /login when not in dev mode and no session exists", async () => {
    getSession.mockResolvedValue({ data: { session: null } })
    const AuthGuard = await loadGuard("false")
    render(
      <AuthGuard>
        <div>protected</div>
      </AuthGuard>,
    )
    await waitFor(() => expect(replace).toHaveBeenCalledWith("/login"))
    expect(screen.queryByText("protected")).not.toBeInTheDocument()
  })

  it("renders children when a real session exists outside dev mode", async () => {
    getSession.mockResolvedValue({ data: { session: { user: { id: "u1" } } } })
    const AuthGuard = await loadGuard("false")
    render(
      <AuthGuard>
        <div>protected</div>
      </AuthGuard>,
    )
    await waitFor(() => expect(screen.getByText("protected")).toBeInTheDocument())
    expect(replace).not.toHaveBeenCalled()
  })
})
