import { describe, expect, it } from 'vitest'
import {
  bestOwnedAtVolume,
  bestRentedAtVolume,
  calculateHybridComparison,
  candidateCapacityTokens,
  hostedCostAtVolume,
  workloadFacts,
  type CostAssumptions,
  type HostedPrice,
  type HybridWorkload,
  type InfrastructureCandidate,
} from './calc'

const workload: HybridWorkload = {
  monthlyInputTokens: 20_000_000,
  monthlyOutputTokens: 5_000_000,
  averageInputTokens: 4_000,
  averageOutputTokens: 1_000,
  activeHoursPerMonth: 730,
  peakToAverage: 1,
}

const hostedPrice: HostedPrice = {
  modelId: 'test/model',
  label: 'Hosted API',
  inputPerMillion: 1,
  outputPerMillion: 3,
}

const assumptions: CostAssumptions = {
  costLens: 'fully-loaded',
  cloudBillingMode: 'always-on',
  cloudRuntimeBufferPct: 15,
  planningCapacityUsePct: 100,
  hoursPerMonth: 730,
  analysisMonths: 36,
  loadedMonthlyCostPerFte: 18_000,
  hostedOperationsFte: 0,
  hostedImplementation: 0,
  rentedDirectInfrastructureMonthly: 0,
  rentedOperationsFte: 0,
  rentedImplementation: 0,
  hardwareLifeYears: 5,
  hardwareResidualPct: 0,
  annualCostOfCapitalPct: 0,
  annualMaintenancePct: 8,
  electricityPerKwh: 0.12,
  pue: 1.4,
  ownedBaseSystemPowerWattsPerServer: 0,
  ownedInstallationPerServer: 0,
  ownedFacilityMonthlyPerServer: 0,
  ownedDirectInfrastructureMonthly: 0,
  ownedOperationsFte: 0,
  ownedImplementation: 0,
  hostedFixedMonthly: 0,
  rentedFixedMonthly: 0,
  ownedFixedMonthly: 0,
}

const candidate: InfrastructureCandidate = {
  systemId: 'test_gpu',
  label: 'Test GPU',
  gpusPerReplica: 1,
  replicasNeeded: 1,
  clusterOutputTokensPerSecond: 100,
  cloudRatePerGpuHour: 2,
  cloudGpusPerInstance: 1,
  cloudProvider: 'test.region',
  cloudRateKind: 'on_demand',
  purchasePricePerReplica: 12_000,
  purchaseInstallationPerReplica: 0,
  purchasePriceIndicative: true,
  purchasePriceSource: 'Test catalogue',
  purchasePriceSourceUrl: null,
  purchasePriceSourceDate: '2026-09-16',
  tdpWattsPerGpu: 500,
  ttftMs: 300,
  tpotMs: 40,
  source: 'AISimulators',
}

describe('hybrid savings calculations', () => {
  it('derives request demand from billed tokens and the request shape', () => {
    const facts = workloadFacts(workload)
    expect(facts.monthlyTokens).toBe(25_000_000)
    expect(facts.monthlyRequests).toBe(5_000)
    expect(facts.peakRequestsPerSecond).toBeCloseTo(5_000 / (730 * 3_600))
  })

  it('prices hosted input and output tokens independently', () => {
    const result = hostedCostAtVolume(workload, hostedPrice, assumptions, 25_000_000)
    expect(result.monthlyCost).toBe(35)
    expect(result.costPerMillionTokens).toBe(1.4)
  })

  it('treats invalid hosted feed prices as zero in totals and breakdown rows', () => {
    const result = hostedCostAtVolume(
      workload,
      { ...hostedPrice, inputPerMillion: Number.NaN, outputPerMillion: Number.POSITIVE_INFINITY },
      assumptions,
      25_000_000,
    )

    expect(result.monthlyCost).toBe(0)
    expect(result.breakdown.every(item => Number.isFinite(item.monthlyCost))).toBe(true)
  })

  it('uses an offer-specific rented infrastructure allowance before the global fallback', () => {
    const priced = bestRentedAtVolume(
      workload,
      [{ ...candidate, cloudDirectInfrastructureMonthly: 500 }],
      { ...assumptions, rentedDirectInfrastructureMonthly: 1_800 },
      25_000_000,
    )

    expect(
      priced?.breakdown.find(item => item.label === 'Infrastructure and observability')?.monthlyCost,
    ).toBe(500)
  })

  it('converts AISimulators output throughput into billed-token capacity', () => {
    const capacity = candidateCapacityTokens(workload, candidate)
    expect(capacity).toBe(100 * 5 * 730 * 3_600)
  })

  it('selects the least-cost eligible rented and owned candidates independently', () => {
    const expensive = {
      ...candidate,
      systemId: 'expensive',
      label: 'Expensive GPU',
      cloudRatePerGpuHour: 4,
      purchasePricePerReplica: 24_000,
    }
    expect(bestRentedAtVolume(workload, [expensive, candidate], assumptions, 25_000_000)?.candidate?.systemId)
      .toBe('test_gpu')
    expect(bestOwnedAtVolume(workload, [expensive, candidate], assumptions, 25_000_000)?.candidate?.systemId)
      .toBe('test_gpu')
  })

  it('uses workload-linked GPU hours for scale-to-zero billing', () => {
    const result = bestRentedAtVolume(
      workload,
      [candidate],
      { ...assumptions, cloudBillingMode: 'scale-to-zero', cloudRuntimeBufferPct: 0 },
      25_000_000,
    )
    const expectedGpuHours = 5_000_000 / 100 / 3_600
    expect(result?.monthlyCost).toBeCloseTo(expectedGpuHours * 2)
  })

  it('bills a whole multi-GPU instance for sparse scale-to-zero demand', () => {
    const result = bestRentedAtVolume(
      workload,
      [{
        ...candidate,
        cloudGpusPerInstance: 8,
        cloudHourlyCostPerInstance: 16,
      }],
      { ...assumptions, cloudBillingMode: 'scale-to-zero', cloudRuntimeBufferPct: 0 },
      25_000_000,
    )
    const expectedInstanceHours = 5_000_000 / 100 / 3_600
    expect(result?.monthlyCost).toBeCloseTo(expectedInstanceHours * 16)
    expect(result?.billedGpuCount).toBe(8)
  })

  it('does not mis-rank an eight-GPU B200 instance as a one-GPU bargain', () => {
    const l40s = {
      ...candidate,
      systemId: 'l40s',
      label: 'NVIDIA L40S',
      clusterOutputTokensPerSecond: 698.279,
      cloudHourlyCostPerInstance: 1.861,
      cloudRatePerGpuHour: 1.861,
      cloudGpusPerInstance: 1,
    }
    const b200 = {
      ...candidate,
      systemId: 'b200_sxm',
      label: 'NVIDIA B200',
      clusterOutputTokensPerSecond: 9_164.991,
      cloudHourlyCostPerInstance: 68.8,
      cloudRatePerGpuHour: 8.6,
      cloudGpusPerInstance: 8,
    }
    const result = bestRentedAtVolume(
      workload,
      [b200, l40s],
      {
        ...assumptions,
        cloudBillingMode: 'scale-to-zero',
        cloudRuntimeBufferPct: 10,
        planningCapacityUsePct: 90,
      },
      2_500_000_000,
    )
    expect(result?.candidate?.systemId).toBe('l40s')
    expect(result?.candidate?.cloudGpusPerInstance).toBe(1)
  })

  it('shares a whole instance only across replicas the workload actually needs', () => {
    const capacity = candidateCapacityTokens(workload, candidate)
    const result = bestRentedAtVolume(
      workload,
      [{
        ...candidate,
        cloudGpusPerInstance: 8,
        cloudHourlyCostPerInstance: 16,
      }],
      { ...assumptions, cloudBillingMode: 'scale-to-zero', cloudRuntimeBufferPct: 0 },
      capacity * 8,
    )
    const outputShare = workload.monthlyOutputTokens /
      (workload.monthlyInputTokens + workload.monthlyOutputTokens)
    const replicaHours = capacity * 8 * outputShare / 100 / 3_600
    expect(result?.replicas).toBe(8)
    expect(result?.monthlyCost).toBeCloseTo(replicaHours / 8 * 16)
  })

  it('never lowers scale-to-zero rented cost when demand crosses replica boundaries', () => {
    const packedCandidate = {
      ...candidate,
      cloudGpusPerInstance: 8,
      cloudHourlyCostPerInstance: 16,
    }
    const scaleToZero = {
      ...assumptions,
      cloudBillingMode: 'scale-to-zero' as const,
      cloudRuntimeBufferPct: 0,
    }
    const capacity = candidateCapacityTokens(workload, packedCandidate)
    const volumes = [
      capacity * 0.5,
      capacity,
      capacity + 1,
      capacity * 2,
      capacity * 4,
      capacity * 8,
      capacity * 8 + 1,
      capacity * 12,
      capacity * 16,
    ]
    const costs = volumes.map(volume => (
      bestRentedAtVolume(workload, [packedCandidate], scaleToZero, volume)?.monthlyCost ?? -1
    ))

    for (let index = 1; index < costs.length; index += 1) {
      expect(costs[index]).toBeGreaterThanOrEqual(costs[index - 1] - 0.000001)
    }
  })

  it('adds whole replicas when demand exceeds one replica capacity', () => {
    const capacity = candidateCapacityTokens(workload, candidate)
    const result = bestOwnedAtVolume(workload, [candidate], assumptions, capacity + 1)
    expect(result?.replicas).toBe(2)
    expect(result?.gpuCount).toBe(2)
  })

  it('packs compatible purchased replicas into the same complete server', () => {
    const packedCandidate = {
      ...candidate,
      purchaseGpusPerServer: 8,
      purchasePricePerReplica: 80_000,
    }
    const capacity = candidateCapacityTokens(workload, packedCandidate)
    const oneReplica = bestOwnedAtVolume(workload, [packedCandidate], assumptions, capacity)
    const eightReplicas = bestOwnedAtVolume(workload, [packedCandidate], assumptions, capacity * 8)

    expect(oneReplica?.billedGpuCount).toBe(8)
    expect(eightReplicas?.replicas).toBe(8)
    expect(eightReplicas?.billedGpuCount).toBe(8)
    expect(eightReplicas?.breakdown.find(item => item.label.includes('depreciation'))?.monthlyCost)
      .toBeCloseTo(oneReplica?.breakdown.find(item => item.label.includes('depreciation'))?.monthlyCost ?? 0)
  })

  it('rounds active-window rented capacity to a whole cloud instance', () => {
    const result = bestRentedAtVolume(
      workload,
      [{ ...candidate, cloudGpusPerInstance: 8 }],
      { ...assumptions, cloudBillingMode: 'active-window' },
      25_000_000,
    )

    expect(result?.gpuCount).toBe(1)
    expect(result?.billedGpuCount).toBe(8)
    expect(result?.monthlyCost).toBe(8 * 730 * 2)
  })

  it('uses exact replica packing rather than aggregate GPU rounding', () => {
    const fiveGpuReplica = {
      ...candidate,
      gpusPerReplica: 5,
      cloudGpusPerInstance: 8,
      cloudHourlyCostPerInstance: 16,
      clusterOutputTokensPerSecond: 300,
    }
    const capacity = candidateCapacityTokens(workload, fiveGpuReplica)
    const result = bestRentedAtVolume(
      workload,
      [fiveGpuReplica],
      { ...assumptions, cloudBillingMode: 'always-on' },
      capacity * 3,
    )
    expect(result?.replicas).toBe(3)
    expect(result?.billedGpuCount).toBe(24)
    expect(result?.monthlyCost).toBe(3 * 730 * 16)
  })

  it('excludes a cloud shape that cannot span enough instances for one replica', () => {
    const result = bestRentedAtVolume(
      workload,
      [{
        ...candidate,
        gpusPerReplica: 16,
        cloudGpusPerInstance: 8,
        cloudHourlyCostPerInstance: 16,
        cloudMaxInstancesPerReplica: 1,
      }],
      assumptions,
      25_000_000,
    )
    expect(result).toBeNull()
  })

  it('forces capacity-block offers to remain on for the full month', () => {
    const result = bestRentedAtVolume(
      workload,
      [{
        ...candidate,
        cloudHourlyCostPerInstance: 10,
        cloudRateKind: 'capacity_block',
      }],
      { ...assumptions, cloudBillingMode: 'scale-to-zero' },
      25_000_000,
    )
    expect(result?.monthlyCost).toBe(730 * 10)
  })

  it('applies explicit planning headroom once to modeled capacity', () => {
    expect(candidateCapacityTokens(workload, candidate, 90)).toBeCloseTo(
      candidateCapacityTokens(workload, candidate) * 0.9,
    )
  })

  it('builds a zero-based chart and finds infrastructure crossovers', () => {
    const highApiPrice = { ...hostedPrice, inputPerMillion: 100, outputPerMillion: 100 }
    const result = calculateHybridComparison(workload, highApiPrice, [candidate], assumptions)
    expect(result.chartPoints[0].tokens).toBe(0)
    expect(result.chartPoints[1].tokens).toBe(1)
    expect(result.rentedBreakEvenTokens).not.toBeNull()
    expect(result.ownedBreakEvenTokens).not.toBeNull()
    const lowestCostTransitions = [result.rentedLowestCostTokens, result.ownedLowestCostTokens]
      .filter((value): value is number => value !== null)
    expect(lowestCostTransitions.length).toBeGreaterThan(0)
    expect(result.chartPoints.some(point => point.tokens === result.rentedBreakEvenTokens)).toBe(true)
    expect(lowestCostTransitions.every(transition =>
      result.chartPoints.some(point => point.tokens === transition),
    )).toBe(true)
    expect(result.chartMaximumTokens).toBeGreaterThanOrEqual(result.monthlyTokens)
    const furthestRelevantPoint = Math.max(
      result.monthlyTokens,
      result.rentedBreakEvenTokens ?? 0,
      result.ownedBreakEvenTokens ?? 0,
      result.rentedLowestCostTokens ?? 0,
      result.ownedLowestCostTokens ?? 0,
    )
    expect(result.chartMaximumTokens).toBeGreaterThan(furthestRelevantPoint)
    expect(result.chartMaximumTokens).toBeLessThanOrEqual(furthestRelevantPoint * 2)
  })

  it('uses the same lowest-cost formulas for every plotted chart point', () => {
    const alternative = {
      ...candidate,
      systemId: 'alternative_gpu',
      label: 'Alternative GPU',
      clusterOutputTokensPerSecond: 250,
      cloudRatePerGpuHour: 3,
      purchasePricePerReplica: 20_000,
    }
    const candidates = [candidate, alternative]
    const result = calculateHybridComparison(workload, hostedPrice, candidates, assumptions)
    const currentPoint = result.chartPoints.find(point => point.tokens === result.monthlyTokens)

    expect(currentPoint).toBeDefined()
    for (const point of result.chartPoints) {
      expect(point.hosted).toBeCloseTo(
        hostedCostAtVolume(workload, hostedPrice, assumptions, point.tokens).monthlyCost,
      )
      expect(point.rented).toBeCloseTo(
        bestRentedAtVolume(workload, candidates, assumptions, point.tokens)?.monthlyCost ?? 0,
      )
      expect(point.owned).toBeCloseTo(
        bestOwnedAtVolume(workload, candidates, assumptions, point.tokens)?.monthlyCost ?? 0,
      )
    }
  })

  it('keeps every plotted cost path monotonic as workload increases', () => {
    const packed = {
      ...candidate,
      cloudGpusPerInstance: 8,
      cloudHourlyCostPerInstance: 16,
      purchaseGpusPerServer: 8,
      purchasePricePerReplica: 80_000,
    }
    const alternative = {
      ...candidate,
      systemId: 'alternative_gpu',
      label: 'Alternative GPU',
      clusterOutputTokensPerSecond: 160,
      cloudGpusPerInstance: 4,
      cloudHourlyCostPerInstance: 13,
      purchaseGpusPerServer: 4,
      purchasePricePerReplica: 58_000,
    }
    const result = calculateHybridComparison(
      workload,
      hostedPrice,
      [packed, alternative],
      {
        ...assumptions,
        cloudBillingMode: 'scale-to-zero',
        cloudRuntimeBufferPct: 10,
      },
    )

    for (let index = 1; index < result.chartPoints.length; index += 1) {
      const previous = result.chartPoints[index - 1]
      const current = result.chartPoints[index]
      expect(current.hosted ?? Infinity).toBeGreaterThanOrEqual((previous.hosted ?? 0) - 0.000001)
      expect(current.rented ?? Infinity).toBeGreaterThanOrEqual((previous.rented ?? 0) - 0.000001)
      expect(current.owned ?? Infinity).toBeGreaterThanOrEqual((previous.owned ?? 0) - 0.000001)
    }
  })

  it('keeps full TCO and marginal costs distinct without changing GPU capacity', () => {
    const fullTco = calculateHybridComparison(
      workload,
      hostedPrice,
      [candidate],
      {
        ...assumptions,
        hostedOperationsFte: 0.05,
        hostedImplementation: 15_000,
        rentedDirectInfrastructureMonthly: 1_800,
        rentedOperationsFte: 0.2,
        rentedImplementation: 40_000,
        ownedOperationsFte: 0.25,
        ownedImplementation: 50_000,
        ownedDirectInfrastructureMonthly: 1_500,
      },
    )
    const marginal = calculateHybridComparison(
      workload,
      hostedPrice,
      [candidate],
      {
        ...assumptions,
        costLens: 'marginal',
        hostedOperationsFte: 0.05,
        hostedImplementation: 15_000,
        rentedDirectInfrastructureMonthly: 1_800,
        rentedOperationsFte: 0.2,
        rentedImplementation: 40_000,
        ownedOperationsFte: 0.25,
        ownedImplementation: 50_000,
        ownedDirectInfrastructureMonthly: 1_500,
      },
    )

    expect(fullTco.options.find(option => option.key === 'hosted')?.monthlyCost).toBeGreaterThan(
      marginal.options.find(option => option.key === 'hosted')?.monthlyCost ?? 0,
    )
    expect(fullTco.options.find(option => option.key === 'owned')?.monthlyCost).toBeGreaterThan(
      marginal.options.find(option => option.key === 'owned')?.monthlyCost ?? 0,
    )
    expect(fullTco.options.find(option => option.key === 'owned')?.monthlyCapacityTokens).toBe(
      marginal.options.find(option => option.key === 'owned')?.monthlyCapacityTokens,
    )
  })

  it('uses a complete-server Qwen3 8B estimate for purchased TCO', () => {
    const qwenPrice: HostedPrice = {
      modelId: 'Qwen/Qwen3-8B',
      label: 'Qwen hosted API',
      inputPerMillion: 0.117,
      outputPerMillion: 0.455,
    }
    const l40s: InfrastructureCandidate = {
      ...candidate,
      systemId: 'l40s',
      label: 'NVIDIA L40S',
      clusterOutputTokensPerSecond: 470.039,
      cloudRatePerGpuHour: null,
      purchasePricePerReplica: 35_000,
      purchaseInstallationPerReplica: 2_500,
      tdpWattsPerGpu: 350,
      ttftMs: 590.247,
      tpotMs: 58.979,
    }
    const result = calculateHybridComparison(
      workload,
      qwenPrice,
      [l40s],
      {
        ...assumptions,
        hardwareLifeYears: 4,
        hardwareResidualPct: 20,
        annualCostOfCapitalPct: 8,
        annualMaintenancePct: 5,
        ownedInstallationPerServer: 0,
        ownedFacilityMonthlyPerServer: 200,
        ownedDirectInfrastructureMonthly: 1_500,
        ownedOperationsFte: 0.25,
        ownedImplementation: 50_000,
      },
    )
    const hosted = result.options.find(option => option.key === 'hosted')
    const owned = result.options.find(option => option.key === 'owned')

    expect(hosted?.monthlyCost).toBeCloseTo(4.615, 3)
    expect(owned?.candidate?.systemId).toBe('l40s')
    expect(owned?.monthlyCost).toBeGreaterThan(7_500)
    expect(owned?.monthlyCost).toBe(owned?.fullyLoadedMonthlyCost)
    expect(owned?.marginalMonthlyCost).toBeLessThan(owned?.fullyLoadedMonthlyCost ?? 0)
    expect(owned?.breakdown.find(item => item.label === 'Hardware depreciation net of residual')?.monthlyCost)
      .toBeCloseTo((35_000 - 7_000) / 48)
  })

  it('prices each complete server once rather than multiplying a bare GPU price', () => {
    const fourGpuServer: InfrastructureCandidate = {
      ...candidate,
      gpusPerReplica: 4,
      purchasePricePerReplica: 94_000,
      purchaseInstallationPerReplica: 6_600,
      tdpWattsPerGpu: 0,
    }
    const result = bestOwnedAtVolume(
      workload,
      [fourGpuServer],
      {
        ...assumptions,
        hardwareLifeYears: 4,
        annualMaintenancePct: 0,
      },
      25_000_000,
    )

    expect(result?.breakdown.find(item => item.label === 'Hardware depreciation net of residual')?.monthlyCost)
      .toBeCloseTo(94_000 / 48)
    expect(result?.breakdown.find(item => item.label === 'Installation and commissioning')?.monthlyCost)
      .toBeCloseTo(6_600 / 48)
  })

  it('uses the four-year UI default when hardware life is invalid', () => {
    const result = bestOwnedAtVolume(
      workload,
      [{ ...candidate, purchasePricePerReplica: 48_000, tdpWattsPerGpu: 0 }],
      {
        ...assumptions,
        hardwareLifeYears: 0,
        hardwareResidualPct: 0,
        annualMaintenancePct: 0,
      },
      25_000_000,
    )

    expect(result?.breakdown.find(item => item.label === 'Hardware depreciation net of residual')?.monthlyCost)
      .toBeCloseTo(1_000)
  })

  it('accounts for replacement purchases when the analysis exceeds useful life', () => {
    const result = bestOwnedAtVolume(
      workload,
      [{ ...candidate, purchasePricePerReplica: 12_000, purchaseInstallationPerReplica: 1_200, tdpWattsPerGpu: 0 }],
      {
        ...assumptions,
        analysisMonths: 24,
        hardwareLifeYears: 1,
        annualMaintenancePct: 0,
      },
      25_000_000,
    )

    expect(result?.breakdown.find(item => item.label === 'Hardware depreciation net of residual')?.monthlyCost)
      .toBeCloseTo(1_000)
    expect(result?.breakdown.find(item => item.label === 'Installation and commissioning')?.monthlyCost)
      .toBeCloseTo(100)
  })
})
