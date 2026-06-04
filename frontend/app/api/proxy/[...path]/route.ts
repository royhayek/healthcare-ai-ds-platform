// Force dynamic rendering so Next.js doesn't buffer SSE responses.
export const dynamic = "force-dynamic"

/**
 * Auth-gated proxy: all frontend API calls route through here.
 *
 * Dev mode: injects X-User-Id: dev-user-1 header (matches backend auth stub).
 * Production: replace DEV_USER_ID block with Supabase session extraction:
 *   const supabase = createRouteHandlerClient({ cookies })
 *   const { data: { session } } = await supabase.auth.getSession()
 *   if (!session) return new Response('Unauthorized', { status: 401 })
 *   headers.set('Authorization', `Bearer ${session.access_token}`)
 */

import { type NextRequest, NextResponse } from "next/server";

const FASTAPI_URL = process.env.FASTAPI_URL ?? "http://127.0.0.1:8001";
console.log("[proxy] FASTAPI_URL =", FASTAPI_URL);
// Dev-mode user. Remove when wiring real auth.
const DEV_USER_ID = "dev-user-1";

async function proxyRequest(
  request: NextRequest,
  path: string,
): Promise<NextResponse> {
  const { searchParams } = new URL(request.url);
  const query = searchParams.toString();
  const targetUrl = `${FASTAPI_URL}/${path}${query ? `?${query}` : ""}`;

  const headers = new Headers({ "X-User-Id": DEV_USER_ID });

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
