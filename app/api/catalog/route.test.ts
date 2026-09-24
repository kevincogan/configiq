import { afterEach, describe, expect, it, vi } from 'vitest'
import { GET } from './route'

afterEach(() => {
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
})

describe('GET /api/catalog', () => {
  it('preserves the shared backend catalogue alongside systems and models', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://aisimulators.dev')
    const responses: Record<string, unknown> = {
      '/systems?include=specs': { systems: [{ id: 'h200_sxm' }] },
      '/models?include=specs': { models: [{ id: 'Qwen/Qwen3-8B' }] },
      '/backends': { backends: [{ id: 'vllm' }, { id: 'sglang' }, { id: 'tensorrt-llm' }] },
    }
    const fetchMock = vi.fn((url: string) => {
      const path = new URL(url).pathname + new URL(url).search
      return Promise.resolve(Response.json(responses[path]))
    })
    vi.stubGlobal('fetch', fetchMock)

    const response = await GET()

    expect(response.status).toBe(200)
    expect(await response.json()).toMatchObject({
      systems: [{ id: 'h200_sxm' }],
      models: [{ id: 'Qwen/Qwen3-8B' }],
      backends: [{ id: 'vllm' }, { id: 'sglang' }, { id: 'tensorrt-llm' }],
    })
    expect(fetchMock).toHaveBeenCalledTimes(3)
  })

  it('preserves backends when using a combined ConfigIQ catalogue gateway', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://configiq.dev/api')
    const fetchMock = vi.fn(() => Promise.resolve(Response.json({
      systems: [{ id: 'h200_sxm' }],
      models: ['Qwen/Qwen3-8B'],
      backends: [{ id: 'vllm' }],
    })))
    vi.stubGlobal('fetch', fetchMock)

    const response = await GET()

    expect(response.status).toBe(200)
    expect(await response.json()).toMatchObject({
      systems: [{ id: 'h200_sxm' }],
      models: ['Qwen/Qwen3-8B'],
      backends: [{ id: 'vllm' }],
    })
    expect(fetchMock).toHaveBeenCalledWith('https://configiq.dev/api/catalog', expect.any(Object))
  })
})
