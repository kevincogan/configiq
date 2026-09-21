import { describe, expect, it } from 'vitest'
import {
  hasRentedCloudOffer,
  resolveRentedCloudOffers,
} from './cloud-offer-catalog'

const liveSystems = [
  'a100_sxm',
  'b200_sxm',
  'b300_sxm',
  'b60',
  'gb200',
  'gb300',
  'h100_sxm',
  'h200_sxm',
  'l40s',
  'rtx_pro_6000_server',
]

describe('complete rented cloud offers', () => {
  it('has defensible stable whole-instance offers for seven of the ten live systems', () => {
    const ready = liveSystems.filter(systemId => hasRentedCloudOffer(systemId))
    expect(ready).toEqual([
      'a100_sxm',
      'b200_sxm',
      'gb200',
      'h100_sxm',
      'h200_sxm',
      'l40s',
      'rtx_pro_6000_server',
    ])
  })

  it('retains different whole-instance sizes until after sizing', () => {
    const offers = resolveRentedCloudOffers('l40s')
    expect(new Set(offers.map(offer => offer.gpuCount))).toEqual(new Set([1, 4, 8]))
    expect(offers.every(offer => offer.hourlyCost > 0)).toBe(true)
  })

  it('honors the selected provider without collapsing its instance shapes', () => {
    const offers = resolveRentedCloudOffers('a100_sxm', undefined, 'gcp.us-central1')
    expect(offers.every(offer => offer.provider === 'gcp')).toBe(true)
    expect(new Set(offers.map(offer => offer.gpuCount))).toEqual(new Set([1, 2, 4, 8]))
  })

  it('does not let a spot rate silently beat stable on-demand offers', () => {
    const offers = resolveRentedCloudOffers('h200_sxm', {
      'vast.marketplace': {
        on_demand: null,
        reserved_1yr: null,
        reserved_3yr: null,
        spot_median: 1,
        rate_basis: 'gpu_hour',
        gpus_per_instance: 1,
      },
    })
    expect(offers.some(offer => offer.rateKind === 'spot')).toBe(false)
  })

  it('does not treat a spot-only system as cost-ready by default', () => {
    expect(resolveRentedCloudOffers('b300_sxm')).toEqual([])
    expect(hasRentedCloudOffer('b300_sxm')).toBe(false)
  })

  it('does not rank an aggregate rate without a complete instance identity', () => {
    const offers = resolveRentedCloudOffers('gb300', {
      'future.region': {
        on_demand: 12,
        reserved_1yr: null,
        reserved_3yr: null,
        spot_median: null,
        rate_basis: 'gpu_hour',
        gpus_per_instance: 4,
      },
    })
    expect(offers).toHaveLength(0)
  })
})
