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
