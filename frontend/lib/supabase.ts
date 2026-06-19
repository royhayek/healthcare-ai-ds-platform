/**
 * Supabase JS client (§5).
 *
 * Browser-side: uses the anon key - respects Row-Level Security policies.
 * Server-side (Route Handlers / Server Components): import the service-role
 * variant from this file's named export `supabaseAdmin` when you need to
 * bypass RLS (e.g., signed-URL generation for deliverable downloads).
 *
 * SUPABASE_URL and SUPABASE_ANON_KEY are public env vars (prefixed NEXT_PUBLIC_).
 * SUPABASE_SERVICE_ROLE_KEY must NOT be exposed to the browser - it is only
 * available in server-side Route Handlers via process.env.
 */

import { createClient, SupabaseClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL ?? "";
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "";

if (!supabaseUrl || !supabaseAnonKey) {
  if (typeof window !== "undefined") {
    console.warn(
      "[supabase] NEXT_PUBLIC_SUPABASE_URL or NEXT_PUBLIC_SUPABASE_ANON_KEY is not set. " +
        "Auth and storage will not work. Set these in .env.local."
    );
  }
}

/** Browser-safe Supabase client (uses anon key, respects RLS). */
export const supabase: SupabaseClient = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    persistSession: true,
    autoRefreshToken: true,
    detectSessionInUrl: true,
  },
});

/**
 * Name of the cookie that mirrors the current Supabase access token.
 *
 * The session itself lives in localStorage (the supabase-js default), which is
 * only readable in the browser. The Next.js API proxy runs server-side and
 * cannot see localStorage, so it cannot attach the user's JWT on its own. To
 * close that gap we mirror the access token into a first-party cookie that the
 * browser sends automatically on every same-origin `/api/proxy/*` request; the
 * proxy reads it and forwards `Authorization: Bearer <token>` to FastAPI.
 *
 * This cookie is JS-readable (not httpOnly) by necessity, but that is no weaker
 * than the token already sitting in localStorage. SameSite=Lax keeps it from
 * riding cross-site requests.
 */
export const ACCESS_TOKEN_COOKIE = "sb-access-token";

function writeAccessTokenCookie(token: string | null) {
  if (typeof document === "undefined") return;
  const secure = window.location.protocol === "https:" ? "; Secure" : "";
  if (token) {
    document.cookie = `${ACCESS_TOKEN_COOKIE}=${token}; Path=/; SameSite=Lax${secure}`;
  } else {
    document.cookie = `${ACCESS_TOKEN_COOKIE}=; Path=/; Max-Age=0; SameSite=Lax${secure}`;
  }
}

// Keep the mirrored cookie in sync with the live session. This runs once per
// browser tab: it seeds the cookie from any persisted session and updates it on
// sign-in, token refresh, and sign-out so the proxy always forwards a fresh JWT.
if (typeof window !== "undefined") {
  supabase.auth.getSession().then(({ data: { session } }) => {
    writeAccessTokenCookie(session?.access_token ?? null);
  });
  supabase.auth.onAuthStateChange((_event, session) => {
    writeAccessTokenCookie(session?.access_token ?? null);
  });
}

/**
 * Return the current session's JWT access token, or null when the user is not
 * authenticated. Pass this as `Authorization: Bearer <token>` when calling the
 * FastAPI backend so the backend's Supabase JWT verification can decode it.
 */
export async function getAccessToken(): Promise<string | null> {
  const {
    data: { session },
  } = await supabase.auth.getSession();
  return session?.access_token ?? null;
}

/**
 * Sign in with email + password.  Returns the session on success.
 */
export async function signIn(email: string, password: string) {
  return supabase.auth.signInWithPassword({ email, password });
}

/**
 * Sign up with email + password.
 */
export async function signUp(email: string, password: string) {
  return supabase.auth.signUp({ email, password });
}

/**
 * Sign out and clear the local session.
 */
export async function signOut() {
  return supabase.auth.signOut();
}

/**
 * Return the currently authenticated user, or null when not signed in.
 * Used by the account menu and settings page to display identity.
 */
export async function getUser() {
  const {
    data: { user },
  } = await supabase.auth.getUser();
  return user;
}

/**
 * Update the signed-in user's password. The user must have an active session;
 * Supabase rejects the call otherwise. Returns the standard `{ data, error }`
 * shape so callers can surface `error.message` directly.
 */
export async function updatePassword(newPassword: string) {
  return supabase.auth.updateUser({ password: newPassword });
}
