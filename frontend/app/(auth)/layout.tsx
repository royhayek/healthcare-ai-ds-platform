/** Centered layout for login and signup pages. */
export default function AuthLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-neutral-950 flex items-center justify-center px-4">
      <div className="w-full max-w-sm">
        <div className="mb-8 text-center">
          <div className="inline-flex items-center gap-2 mb-2">
            <div className="w-5 h-5 rounded bg-indigo-600 flex items-center justify-center">
              <div className="w-2 h-2 rounded-full bg-white" />
            </div>
            <span className="text-sm font-semibold text-neutral-200 tracking-tight">
              AI Data Science Co-Pilot
            </span>
          </div>
          <p className="text-xs text-neutral-500">
            Senior-grade ML pipeline automation
          </p>
        </div>
        {children}
      </div>
    </div>
  )
}
