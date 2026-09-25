// Three-band scenario search: conservative / typical / aggressive.
// Ties together model × precision × gpu × concurrency → full result per band.
import type { ModelSpec } from '@/data/models';
import { PRECISIONS } from '@/data/models';
import type { GpuSpec } from '@/data/gpus';
import { estimateVram, usableVramGB, kvBytesPerToken as kvBytesPerTokenOf } from './vram';
import { findMinimalPlan, plansForDevices, type ParallelPlan } from './parallelism';
import { estimateThroughput, type ThroughputResult } from './throughput';
import { OPTS_OFF, tierOffloadGB, type Optimizations } from './optimizations';

export type Band = 'conservative' | 'typical' | 'aggressive';

export interface BandConfig {
  eta: number; // bandwidth efficiency
  computeUtil: number; // prefill utilization
  overheadFactor: number; // fraction of weights
  overheadBaseGB: number;
}

export const BANDS: Record<Band, BandConfig> = {
  conservative: { eta: 0.7, computeUtil: 0.3, overheadFactor: 0.10, overheadBaseGB: 4 },
  typical: { eta: 0.8, computeUtil: 0.4, overheadFactor: 0.075, overheadBaseGB: 3 },
  aggressive: { eta: 0.9, computeUtil: 0.5, overheadFactor: 0.05, overheadBaseGB: 2 },
};

export interface ScenarioInput {
  model: ModelSpec;
  gpu: GpuSpec;
  precisionId: string;
  concurrency: number;
  avgCtxTokens: number;
  inputTokens: number;
}

export interface ScenarioResult {
  band: Band;
  plan: ParallelPlan;
  vram: ReturnType<typeof estimateVram>;
  throughput: ThroughputResult;
  config: BandConfig;
}

export function runScenario(input: ScenarioInput, band: Band, opts: Optimizations = OPTS_OFF): ScenarioResult {
  const config = BANDS[band];
  const prec = PRECISIONS[input.precisionId] ?? PRECISIONS.bf16;
  const plan = findMinimalPlan(buildSearchInput(input, config, prec, opts));
  return scenarioForPlan(input, band, opts, plan);
}

function buildSearchInput(
  input: ScenarioInput,
  config: BandConfig,
  prec: { bytesPerParam: number },
  opts: Optimizations,
) {
  const { model, gpu, precisionId, concurrency, avgCtxTokens } = input;
  return {
    model,
    gpu,
    precisionId: input.precisionId,
    bytesPerParam: prec.bytesPerParam,
    kvBytesPerToken: kvBytesPerTokenOf(model, precisionId) * (opts.kvFp8 ? 0.5 : 1),
    kvOffloadGB: tierOffloadGB(opts),
    concurrency,
    avgCtxTokens,
    overheadFactor: config.overheadFactor,
    overheadBaseGB: config.overheadBaseGB,
  };
}

/** 按给定切分方案计算场景结果（不搜索最少卡数）。 */
export function scenarioForPlan(
  input: ScenarioInput,
  band: Band,
  opts: Optimizations,
  plan: ParallelPlan,
): ScenarioResult {
  const { model, gpu, precisionId, concurrency, avgCtxTokens, inputTokens } = input;
  const config = BANDS[band];
  const prec = PRECISIONS[precisionId] ?? PRECISIONS.bf16;

  const vram = estimateVram({
    model,
    precisionId,
    concurrency,
    avgCtxTokens,
    overheadFactor: config.overheadFactor,
    overheadBaseGB: config.overheadBaseGB,
  });

  const activeWeightBytes = model.activeParamsB * 1e9 * prec.bytesPerParam;
  const effKvPerTok = kvBytesPerTokenOf(model, precisionId) * (opts.kvFp8 ? 0.5 : 1);
  const kvTotalGB = (input.concurrency * input.avgCtxTokens * effKvPerTok) / 1e9;
  const coldRetrieveMs =
    opts.coldCache.enabled && kvTotalGB > opts.hotCache.sizeGB
      ? (Math.min(opts.coldCache.sizeGB, kvTotalGB - opts.hotCache.sizeGB) / Math.max(opts.coldCache.ssdGBs, 1)) * 1000
      : 0;
  const replicaGpus: GpuSpec[] = Array.from({ length: plan.devicesPerReplica }, () => gpu);

  const throughput = estimateThroughput({
    model,
    gpus: replicaGpus,
    concurrency: plan.replicaConcurrency,
    avgCtxTokens,
    inputTokens,
    eta: config.eta,
    computeUtil: config.computeUtil,
    activeWeightBytes,
    kvBytesPerToken: effKvPerTok,
    opts: {
      chunked: opts.chunkedPrefill,
      chunkSize: opts.chunkSize,
      hotHitRate: opts.hotCache.enabled ? opts.hotCache.hitRate : 0,
      coldRetrieveMs,
    },
  });

  return { band, plan, vram, throughput, config };
}

export function runAllBands(input: ScenarioInput, opts: Optimizations = OPTS_OFF): Record<Band, ScenarioResult> {
  return {
    conservative: runScenario(input, 'conservative', opts),
    typical: runScenario(input, 'typical', opts),
    aggressive: runScenario(input, 'aggressive', opts),
  };
}

/**
 * 反向模式：给定实际拥有的设备数 N，在该卡数约束下搜索最优 (tp, pp, dp) 切分，
 * 返回单用户吞吐最高的方案；装不下时 best 为 null 并给出最少所需卡数。
 */
export function runForDevices(
  input: ScenarioInput,
  opts: Optimizations,
  devices: number,
  band: Band = 'typical',
): { best: ScenarioResult | null; minNeeded: number; candidates: number } {
  const config = BANDS[band];
  const prec = PRECISIONS[input.precisionId] ?? PRECISIONS.bf16;
  const searchInput = buildSearchInput(input, config, prec, opts);
  const plans = plansForDevices(searchInput, devices);

  let best: ScenarioResult | null = null;
  for (const plan of plans) {
    const r = scenarioForPlan(input, band, opts, plan);
    if (!best || r.throughput.perUserTokPerSec > best.throughput.perUserTokPerSec) best = r;
  }
  return { best, minNeeded: findMinimalPlan(searchInput).totalDevices, candidates: plans.length };
}

/** 该切分方案在余量耗尽前最多可承载的并发用户总数（含 DP 副本扩容）。 */
export function maxConcurrentUsers(
  input: ScenarioInput,
  opts: Optimizations,
  plan: ParallelPlan,
  band: Band = 'typical',
): number {
  const { model, gpu, precisionId, avgCtxTokens } = input;
  const config = BANDS[band];
  const prec = PRECISIONS[precisionId] ?? PRECISIONS.bf16;
  const usable = usableVramGB(gpu);
  const weights = (model.paramsB * 1e9 * prec.bytesPerParam) / plan.devicesPerReplica / 1e9;
  const overhead = config.overheadBaseGB + weights * config.overheadFactor;
  const kvBudgetGB = Math.max(0, usable - weights - overhead);
  const perUserKvGB = (avgCtxTokens * kvBytesPerTokenOf(model, precisionId) * (opts.kvFp8 ? 0.5 : 1)) / 1e9;
  if (perUserKvGB <= 0) return Infinity;
  return Math.floor(kvBudgetGB / perUserKvGB) * plan.dp;
}

export function fmtGB(v: number): string {
  if (v >= 100) return v.toFixed(0);
  if (v >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

export function fmtTok(v: number): string {
  if (v >= 100) return v.toFixed(0);
  if (v >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

export function fmtMs(v: number): string {
  if (v >= 1000) return `${(v / 1000).toFixed(2)} s`;
  if (v >= 100) return `${v.toFixed(0)} ms`;
  if (v >= 10) return `${v.toFixed(1)} ms`;
  return `${v.toFixed(2)} ms`;
}
