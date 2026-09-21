import { describe, expect, it } from 'vitest'
import {
  hasServerPurchaseConfiguration,
  resolveCompatibleServerPurchaseConfigurations,
  resolveServerPurchaseConfiguration,
} from './purchase-catalog'

describe('purchased server catalogue', () => {
  it('returns a complete one-GPU L40S server rather than the GPU-card price', () => {
    const result = resolveServerPurchaseConfiguration('l40s', 1)
    expect(result?.purchasePrice).toBe(35_000)
    expect(result?.installationCost).toBe(2_500)
  })

  it('does not invent unsupported server topologies', () => {
    expect(resolveServerPurchaseConfiguration('b200_sxm', 1)).toBeNull()
    expect(resolveServerPurchaseConfiguration('h100_sxm', 4)).toBeNull()
  })

  it('retains larger complete servers as explicit packing candidates', () => {
    const configurations = resolveCompatibleServerPurchaseConfigurations('l40s', 1)
    expect(configurations.map(configuration => configuration.gpuCount)).toEqual([1, 2, 4, 8])
    expect(configurations.find(configuration => configuration.gpuCount === 8)?.purchasePrice)
      .toBe(150_000)
  })

  it('distinguishes complete-server coverage from bare-GPU pricing', () => {
    expect(hasServerPurchaseConfiguration('l40s')).toBe(true)
    expect(hasServerPurchaseConfiguration('gb300')).toBe(false)
    expect(hasServerPurchaseConfiguration('future_gpu', {
      new_usd: 40_000,
      new_usd_low: null,
      new_usd_high: null,
      indicative: true,
      source_label: 'Bare GPU only',
      source_url: null,
      source_date: null,
    })).toBe(false)
  })

  it('accepts an exact complete-server record from the shared catalogue', () => {
    const result = resolveServerPurchaseConfiguration('future_gpu', 2, {
      new_usd: 20_000,
      new_usd_low: 18_000,
      new_usd_high: 22_000,
      indicative: true,
      source_label: 'Per-GPU feed',
      source_url: null,
      source_date: null,
      server_configurations: {
        '2': {
          new_usd: 65_000,
          new_usd_low: 60_000,
          new_usd_high: 70_000,
          installation_usd: 4_000,
          source_label: 'Complete system quote',
          source_url: 'https://example.com/system',
          source_date: '2026-09-01',
        },
      },
    })

    expect(result?.purchasePrice).toBe(65_000)
    expect(result?.installationCost).toBe(4_000)
    expect(result?.sourceLabel).toBe('Complete system quote')
  })
})
