import type { HardwareCost } from '@/lib/hooks/useCostings'

/**
 * A complete, purchasable server configuration.
 *
 * The shared costing feed currently exposes indicative per-GPU prices. Those
 * are useful catalogue data, but they are not a defensible substitute for the
 * price of a complete server (chassis, CPUs, memory, networking and support).
 * Hybrid Savings therefore ranks purchased hardware only when the exact GPU
 * topology has a complete-server estimate.
 */
export interface ServerPurchaseConfiguration {
  gpuCount: number
  purchasePrice: number
  purchasePriceLow: number
  purchasePriceHigh: number
  installationCost: number
  sourceLabel: string
  sourceUrl: string | null
  sourceDate: string
  indicative: boolean
}

type PurchaseCatalogue = Record<string, Record<number, ServerPurchaseConfiguration>>

const FALLBACK_SOURCE_DATE = '2026-08-01'

function configuration(
  gpuCount: number,
  purchasePrice: number,
  purchasePriceLow: number,
  purchasePriceHigh: number,
  installationCost: number,
  sourceLabel: string,
  sourceUrl: string | null,
  sourceDate = FALLBACK_SOURCE_DATE,
): ServerPurchaseConfiguration {
  return {
    gpuCount,
    purchasePrice,
    purchasePriceLow,
    purchasePriceHigh,
    installationCost,
    sourceLabel,
    sourceUrl,
    sourceDate,
    indicative: true,
  }
}

const l40s = {
  1: configuration(1, 35_000, 30_000, 42_000, 2_500, 'Thinkmate NVIDIA L40S server catalogue', 'https://www.thinkmate.com/systems/servers/gpx/l40s'),
  2: configuration(2, 56_000, 48_000, 65_000, 3_900, 'Thinkmate NVIDIA L40S server catalogue', 'https://www.thinkmate.com/systems/servers/gpx/l40s'),
  4: configuration(4, 94_000, 78_000, 110_000, 6_600, 'Thinkmate NVIDIA L40S server catalogue', 'https://www.thinkmate.com/systems/servers/gpx/l40s'),
  8: configuration(8, 150_000, 125_000, 175_000, 10_500, 'Thinkmate NVIDIA L40S server catalogue', 'https://www.thinkmate.com/systems/servers/gpx/l40s'),
}

const a100 = {
  1: configuration(1, 34_000, 27_000, 42_000, 2_400, 'Exxact public-sector server price schedule', 'https://www.gsaadvantage.gov/ref_text/GS35F0278Y/0ZQME9.3VGZ8X_GS-35F0278Y_EXXACTPRICELIST2024.PDF'),
  2: configuration(2, 59_000, 48_000, 70_000, 4_100, 'Exxact A100 server catalogue and price schedule', 'https://configurator.exxactcorp.com/configure/TWS-194019223'),
  4: configuration(4, 95_000, 75_000, 125_000, 6_700, 'Public four-GPU A100 80 GB server and component-price range', 'https://www.gsaadvantage.gov/ref_text/GS35F0278Y/0ZQME9.3VGZ8X_GS-35F-0278Y_EXXACTPRICELIST2024.PDF'),
  8: configuration(8, 160_000, 90_000, 220_000, 11_200, 'Public eight-GPU A100 80 GB server market range', 'https://www.qubrid.com/bare-metal-gpu-servers'),
}

const h100 = {
  1: configuration(1, 48_000, 40_000, 55_000, 3_900, 'Exxact HGX H100 server catalogue', 'https://www.exxactcorp.com/Exxact-TS4-101818584-E101818584'),
  2: configuration(2, 98_000, 80_000, 115_000, 6_900, 'Exxact HGX H100 server catalogue', 'https://www.exxactcorp.com/Exxact-TS4-101818584-E101818584'),
  8: configuration(8, 285_000, 250_000, 320_000, 22_300, 'Exxact eight-GPU HGX H100 configured system price', 'https://configurator.exxactcorp.com/configure/TS4-193475697'),
}

const h200 = {
  1: configuration(1, 65_000, 55_000, 75_000, 6_000, 'H200 SXM GPU market price plus single-GPU server chassis', 'https://cpq.exxactcorp.com/quote/view_quote_revs.php?qo_id=173436', '2026-04-28'),
  2: configuration(2, 110_000, 95_000, 125_000, 7_700, 'Derived from Exxact one- and four-GPU H200 system prices', 'https://www.exxactcorp.com/Exxact-TS4-180208180-E180208180'),
  8: configuration(8, 370_000, 320_000, 420_000, 18_900, 'Exxact and GSA eight-GPU H200 system prices', 'https://www.exxactcorp.com/Exxact-TS4-118380266-E118380266'),
}

export const PURCHASE_CATALOGUE: PurchaseCatalogue = {
  l40s,
  a100_sxm: a100,
  a100_pcie: a100,
  h100_sxm: h100,
  h100_pcie: h100,
  h200_sxm: h200,
  b200_sxm: {
    8: configuration(8, 404_000, 390_200, 431_000, 28_300, 'Exeton eight-GPU HGX B200 optimized configuration range', 'https://exeton.com/configure/ts4-169219634', '2026-09-09'),
  },
  l4: {
    1: configuration(1, 18_000, 14_000, 22_000, 1_300, 'OEM and GPU-server reseller market range', 'https://www.thinkmate.com/systems/servers/gpx/a10'),
    4: configuration(4, 24_000, 18_000, 30_000, 1_700, 'Public four-GPU L4 server and OEM chassis market range', 'https://jarvislabs.ai/blog/l4-gpu-price'),
  },
  rtx_pro_6000_server: {
    1: configuration(1, 34_766, 30_000, 40_000, 2_434, 'B3IQ complete one-GPU RTX PRO 6000 Blackwell server configuration', 'https://www.b3iq.org/machines/rtx-pro-6000', '2026-09-09'),
    8: configuration(8, 165_000, 149_000, 175_000, 11_550, 'Blackwell Cloud complete eight-GPU RTX PRO 6000 server configuration', 'https://www.blackwellcloud.com/nvidia-rtx-pro-6000-gpu-servers', '2026-09-09'),
  },
}

/** Whether at least one complete-server topology can be purchased. */
export function hasServerPurchaseConfiguration(
  systemId: string,
  hardwareCost?: HardwareCost,
): boolean {
  const shared = hardwareCost?.server_configurations
  if (shared && Object.values(shared).some(configuration => (
    configuration.new_usd > 0 && configuration.installation_usd >= 0
  ))) {
    return true
  }
  return Object.keys(PURCHASE_CATALOGUE[systemId] ?? {}).length > 0
}

function validTopology(value: number): number | null {
  if (!Number.isFinite(value) || value < 0.5) return null
  const rounded = Math.round(value)
  return Math.abs(rounded - value) <= 0.1 ? rounded : null
}

/** Prefer a complete-server record from the shared API when it supplies one. */
export function resolveServerPurchaseConfiguration(
  systemId: string,
  gpusPerReplica: number,
  hardwareCost?: HardwareCost,
): ServerPurchaseConfiguration | null {
  const gpuCount = validTopology(gpusPerReplica)
  if (gpuCount == null) return null

  const shared = hardwareCost?.server_configurations?.[String(gpuCount)]
  if (shared && shared.new_usd > 0 && shared.installation_usd >= 0) {
    return {
      gpuCount,
      purchasePrice: shared.new_usd,
      purchasePriceLow: shared.new_usd_low ?? shared.new_usd,
      purchasePriceHigh: shared.new_usd_high ?? shared.new_usd,
      installationCost: shared.installation_usd,
      sourceLabel: shared.source_label ?? hardwareCost?.source_label ?? 'Shared costing catalogue',
      sourceUrl: shared.source_url ?? hardwareCost?.source_url ?? null,
      sourceDate: shared.source_date ?? hardwareCost?.source_date ?? 'Unknown',
      indicative: shared.indicative ?? hardwareCost?.indicative ?? true,
    }
  }

  return PURCHASE_CATALOGUE[systemId]?.[gpuCount] ?? null
}

/**
 * Return every complete server that can contain one model replica. Larger
 * servers remain separate candidates because they may pack several replicas
 * and become cheaper at higher demand; their full server price is preserved.
 */
export function resolveCompatibleServerPurchaseConfigurations(
  systemId: string,
  gpusPerReplica: number,
  hardwareCost?: HardwareCost,
): ServerPurchaseConfiguration[] {
  const requiredGpus = validTopology(gpusPerReplica)
  if (requiredGpus == null) return []

  const configurations = new Map<number, ServerPurchaseConfiguration>()
  for (const [gpuCountText, shared] of Object.entries(
    hardwareCost?.server_configurations ?? {},
  )) {
    const gpuCount = validTopology(Number(gpuCountText))
    if (gpuCount == null || gpuCount < requiredGpus || shared.new_usd <= 0) continue
    configurations.set(gpuCount, {
      gpuCount,
      purchasePrice: shared.new_usd,
      purchasePriceLow: shared.new_usd_low ?? shared.new_usd,
      purchasePriceHigh: shared.new_usd_high ?? shared.new_usd,
      installationCost: shared.installation_usd,
      sourceLabel: shared.source_label ?? hardwareCost?.source_label ?? 'Shared costing catalogue',
      sourceUrl: shared.source_url ?? hardwareCost?.source_url ?? null,
      sourceDate: shared.source_date ?? hardwareCost?.source_date ?? 'Unknown',
      indicative: shared.indicative ?? hardwareCost?.indicative ?? true,
    })
  }

  for (const configuration of Object.values(PURCHASE_CATALOGUE[systemId] ?? {})) {
    if (configuration.gpuCount >= requiredGpus && !configurations.has(configuration.gpuCount)) {
      configurations.set(configuration.gpuCount, configuration)
    }
  }

  return [...configurations.values()].sort((left, right) => (
    left.gpuCount - right.gpuCount || left.purchasePrice - right.purchasePrice
  ))
}
