// @vitest-environment happy-dom
import * as React from 'react'
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/app-config', () => ({
  loadAppConfig: async () => ({
    defaultModel: 'Qwen/Qwen3-32B',
    defaultSystem: 'h200_sxm',
    defaultOpenModel: 'Qwen/Qwen3-32B',
    defaultFrontierModel: 'claude-fable-5',
    defaultBackend: 'vllm',
    testedModels: [],
    huggingFaceModels: [],
    suggestedModelNames: [],
    modelRequestUrl: '',
    workloadPresets: [],
  }),
}))

import { SettingsProvider, useSettings } from './SettingsContext'

const VERSION_KEY = 'settings_backend_version_v2'
const LEGACY_VERSION_KEY = 'settings_backend_version'

let root: ReturnType<typeof createRoot>
let host: HTMLDivElement

function Consumer() {
  const { hydrated, backendVersion, setBackendVersion, setInferenceBackend } = useSettings()
  return <>
    <output>{JSON.stringify({ hydrated, backendVersion })}</output>
    <button onClick={() => setBackendVersion('0.24.0')}>Set version</button>
    <button onClick={() => setInferenceBackend('sglang')}>Set backend</button>
  </>
}

const state = () => JSON.parse(host.querySelector('output')!.textContent!)

async function render() {
  await act(async () => root.render(<SettingsProvider><Consumer /></SettingsProvider>))
}

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  localStorage.clear()
  host = document.createElement('div')
  document.body.append(host)
  root = createRoot(host)
})

afterEach(async () => {
  await act(async () => root.unmount())
  host.remove()
  vi.restoreAllMocks()
})

describe('Backend version persistence', () => {
  it('restores a version the user explicitly saved', async () => {
    localStorage.setItem(VERSION_KEY, '0.24.0')

    await render()

    expect(state()).toEqual({ hydrated: true, backendVersion: '0.24.0' })
  })

  it('writes nothing when the user never chose a version', async () => {
    await render()

    expect(state().backendVersion).toBe('')
    expect(localStorage.getItem(VERSION_KEY)).toBeNull()
  })

  it('ignores the legacy fleet-wide pin so a stale literal is not resent', async () => {
    // 0.31.0 was the fleet-wide newest version, unqueryable on a100_sxm and b60.
    localStorage.setItem(LEGACY_VERSION_KEY, '0.31.0')

    await render()

    expect(state().backendVersion).toBe('')
  })

  it('clears the persisted version when the backend changes', async () => {
    localStorage.setItem(VERSION_KEY, '0.24.0')
    await render()

    await act(async () => (host.querySelectorAll('button')[1] as HTMLButtonElement).click())

    expect(state().backendVersion).toBe('')
    expect(localStorage.getItem(VERSION_KEY)).toBe('')
  })

  it('persists an explicit user choice', async () => {
    await render()

    await act(async () => (host.querySelector('button') as HTMLButtonElement).click())

    expect(state().backendVersion).toBe('0.24.0')
    expect(localStorage.getItem(VERSION_KEY)).toBe('0.24.0')
  })
})
