// Three-band scenario search: conservative / typical / aggressive.
// Ties together model × precision × gpu × concurrency → full result per band.
import type { ModelSpec } from '@/data/models';
import { PRECISIONS } from '@/data/models';
import type { GpuSpec } from '@/data/gpus';
import { estimateVram, kvBytesPerToken as kvBytesPerTokenOf } from './vram';
import { findMinimalPlan, type ParallelPlan } from './parallelism';
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

  const plan = findMinimalPlan({
    model,
    gpu,
    precisionId,
    bytesPerParam: prec.bytesPerParam,
    kvBytesPerToken: kvBytesPerTokenOf(model, precisionId),
    kvOffloadGB: tierOffloadGB(opts),
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
