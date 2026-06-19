// Force dynamic rendering so Next.js doesn't buffer SSE responses.
export const dynamic = "force-dynamic"

/**
 * Auth-gated proxy: all frontend API calls route through here.
 *
 * Real auth (preferred): the browser mirrors the Supabase access token into the
 * `sb-access-token` cookie (see lib/supabase.ts). This handler reads that cookie
 * and forwards `Authorization: Bearer <token>` to FastAPI, which verifies the
 * JWT when DEV_MODE=false. No bypass: if the token is missing the backend will
 * reject the request.
 *
 * Dev fallback: when no token cookie is present AND NEXT_PUBLIC_DEV_MODE=true,
 * inject X-User-Id: dev-user-1 to match the backend's dev auth stub. With the
 * backend in production mode (DEV_MODE=false) this header is ignored and the
 * request is rejected, which is the intended behavior.
 */

import { type NextRequest, NextResponse } from "next/server";

const FASTAPI_URL = process.env.FASTAPI_URL ?? "http://127.0.0.1:8001";
console.log("[proxy] FASTAPI_URL =", FASTAPI_URL);
const DEV_MODE = process.env.NEXT_PUBLIC_DEV_MODE !== "false";
// Dev-mode fallback identity, used only when no real session token is present.
const DEV_USER_ID = "dev-user-1";
const ACCESS_TOKEN_COOKIE = "sb-access-token";

async function proxyRequest(
  request: NextRequest,
  path: string,
): Promise<NextResponse> {
  const { searchParams } = new URL(request.url);
  const query = searchParams.toString();
  const targetUrl = `${FASTAPI_URL}/${path}${query ? `?${query}` : ""}`;

  const headers = new Headers();
  const accessToken = request.cookies.get(ACCESS_TOKEN_COOKIE)?.value;
  if (accessToken) {
    // Real authenticated request: forward the Supabase JWT for verification.
    headers.set("Authorization", `Bearer ${accessToken}`);
  } else if (DEV_MODE) {
    // No session token, but dev mode is on: fall back to the stub identity.
    headers.set("X-User-Id", DEV_USER_ID);
  }

  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("Content-Type", contentType);

  let body: ArrayBuffer | undefined;
  if (request.method !== "GET" && request.method !== "HEAD") {
    body = await request.arrayBuffer();
  }

  const upstream = await fetch(targetUrl, {
    method: request.method,
    headers,
    body: body ? Buffer.from(body) : undefined,
  });

  const responseHeaders = new Headers();
  upstream.headers.forEach((value, key) => {
    if (!["connection", "transfer-encoding"].includes(key.toLowerCase())) {
      responseHeaders.set(key, value);
    }
  });

  return new NextResponse(upstream.body, {
    status: upstream.status,
    headers: responseHeaders,
  });
}

type RouteContext = { params: { path: string[] } };

export async function GET(req: NextRequest, ctx: RouteContext): Promise<NextResponse> {
  return proxyRequest(req, ctx.params.path.join("/"));
}
export async function POST(req: NextRequest, ctx: RouteContext): Promise<NextResponse> {
  return proxyRequest(req, ctx.params.path.join("/"));
}
export async function PUT(req: NextRequest, ctx: RouteContext): Promise<NextResponse> {
  return proxyRequest(req, ctx.params.path.join("/"));
}
export async function PATCH(req: NextRequest, ctx: RouteContext): Promise<NextResponse> {
  return proxyRequest(req, ctx.params.path.join("/"));
}
export async function DELETE(req: NextRequest, ctx: RouteContext): Promise<NextResponse> {
  return proxyRequest(req, ctx.params.path.join("/"));
}
