// GET /api/catalog
// Same-origin proxy for the AISimulators catalog. Fetches /systems and
// /models (both with specs) server-side via AISIMULATORS_GATEWAY_URL and
// returns their raw shapes combined as { systems, models }.
//
// The browser must call this route rather than AISimulators directly, so the
// per-host gateway (host.containers.internal on each deployment) is resolved
// server-side. That keeps the .dev/.xyz two-host split correct client-side and
// removes the need for a build-time NEXT_PUBLIC_AISIMULATORS_API_URL.

import { NextResponse } from 'next/server'
import { gatewayTimeoutSeconds } from '@/lib/api/timeout'

// This is the catalog fetch's own timeout (30s). The value surfaced to the
// client below is gatewayTimeoutSeconds() — the longer recommend/estimate timeout.
const DEFAULT_TIMEOUT_SECONDS = 30

export async function GET() {
  // Require the gateway to be configured; fail loud (like /recommend) rather
  // than silently falling back to the public domain, which masks a misconfig
  // and bypasses the intended per-host internal gateway.
  const baseUrl = process.env.AISIMULATORS_GATEWAY_URL
  if (!baseUrl) {
    return NextResponse.json(
      { status: 'failed', error: { code: 'AISIM_NOT_CONFIGURED', message: 'AISimulators API URL is not configured' } },
      { status: 503, headers: { 'Cache-Control': 'no-store', 'Access-Control-Allow-Origin': '*' } },
    )
  }
  // Shares the shared resolver's positive-integer validation, with the catalog
  // fetch's own 30s baseline (a negative env value would break AbortSignal).
  const timeoutSeconds = gatewayTimeoutSeconds(DEFAULT_TIMEOUT_SECONDS)

  try {
    const [systemsRes, modelsRes, backendsResult] = await Promise.all([
      fetch(`${baseUrl}/systems?include=specs`, {
        headers: { Accept: 'application/json' },
        cache: 'no-store',
        signal: AbortSignal.timeout(timeoutSeconds * 1000),
      }),
      fetch(`${baseUrl}/models?include=specs`, {
        headers: { Accept: 'application/json' },
        cache: 'no-store',
        signal: AbortSignal.timeout(timeoutSeconds * 1000),
      }),
      fetch(`${baseUrl}/backends`, {
        headers: { Accept: 'application/json' },
        cache: 'no-store',
        signal: AbortSignal.timeout(timeoutSeconds * 1000),
      }).then(response => ({ response })).catch(() => ({ response: null })),
    ])

    if (!systemsRes.ok || !modelsRes.ok) {
      return NextResponse.json(
        {
          status: 'failed',
          error: {
            code: 'AISIM_ERROR',
            message: `AISimulators catalog fetch failed (systems ${systemsRes.status}, models ${modelsRes.status})`,
          },
        },
        { status: 502, headers: { 'Cache-Control': 'no-store', 'Access-Control-Allow-Origin': '*' } },
      )
    }

    let systemsData: { systems?: unknown[] }
    let modelsData: { models?: unknown[] }
    let backendsData: { backends?: unknown[] } = {}
    try {
      systemsData = await systemsRes.json()
      modelsData = await modelsRes.json()
    } catch {
      return NextResponse.json(
        { status: 'failed', error: { code: 'AISIM_INVALID_RESPONSE', message: 'AISimulators returned non-JSON response' } },
        { status: 502, headers: { 'Cache-Control': 'no-store', 'Access-Control-Allow-Origin': '*' } },
      )
    }
    if (backendsResult.response?.ok) {
      try {
        backendsData = await backendsResult.response.json()
      } catch {
        backendsData = {}
      }
    }

    return NextResponse.json(
      {
        systems: systemsData.systems ?? [],
        models: modelsData.models ?? [],
        backends: backendsData.backends ?? [],
        // Effective AISimulators request timeout (recommend/estimate), for the loader hint.
        timeoutSeconds: gatewayTimeoutSeconds(),
      },
      {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          // Mirrors the useCatalog client-side cache TTL (10 min).
          'Cache-Control': 'public, s-maxage=600, stale-while-revalidate=3600',
        },
      },
    )
  } catch (err: unknown) {
    if (err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError')) {
      return NextResponse.json(
        { status: 'failed', error: { code: 'AISIM_TIMEOUT', message: 'AISimulators API timed out' } },
        { status: 504 },
      )
    }
    return NextResponse.json(
      { status: 'failed', error: { code: 'AISIM_UNAVAILABLE', message: 'AISimulators API is unreachable' } },
      { status: 502 },
    )
  }
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  })
}
