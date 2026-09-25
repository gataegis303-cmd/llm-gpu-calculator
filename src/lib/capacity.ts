// 反向推演：用户给定实际持有的设备数 N，求最佳部署 (tp, pp, dp) 与可获得的吞吐。
// 与 search.ts 的“最少卡数”方向互补：那边固定负载搜最小 N，这边固定 N 搜最快方案。
// 口径与 runScenario 保持一致：显存判定用未折减 KV 字节，吞吐用 KV FP8 折减后的字节；
// 唯一差别是冷缓存检索延迟按“每副本 KV”计（本模块一切都是单副本口径）。
import type { ModelSpec } from '@/data/models';
import { PRECISIONS } from '@/data/models';
import type { GpuSpec } from '@/data/gpus';
import { kvBytesPerToken, usableVramGB } from './vram';
import {
  fitPlan,
  maxReplicaConcurrency,
  perDeviceFixedGB,
  type ParallelPlan,
  type SearchInput,
} from './parallelism';
import { estimateThroughput, type ThroughputResult } from './throughput';
import { BANDS, type Band, type BandConfig } from './search';
import { OPTS_OFF, tierOffloadGB, type Optimizations } from './optimizations';

export interface CapacityInput {
  model: ModelSpec;
  gpu: GpuSpec;
  precisionId: string;
  concurrency: number;
  avgCtxTokens: number;
  inputTokens: number;
  /** 用户实际持有的设备数 N（≥1） */
  devices: number;
}

/** 单卡显存构成（选定方案、当前承载并发下） */
export interface DeviceVram {
  weightsGB: number;
  kvGB: number;
  overheadGB: number;
  totalGB: number;
}

export interface CapacityPlan {
  tp: number;
  pp: number;
  dp: number;
  devicesPerReplica: number; // tp × pp
  totalDevices: number; // 实际用上的卡数 ≤ N
  idleDevices: number; // N − totalDevices（无法整除且更大分解不可行时闲置的卡）
  replicaConcurrency: number; // 每副本承载的并发路数
  vramPerDeviceGB: number;
  feasible: boolean;
  reason?: string;
}

export interface CapacityResult {
  band: Band;
  config: BandConfig;
  devices: number;
  /** 服务当前并发 C 的方案（C 超容量时回落为满载方案） */
  plan: CapacityPlan;
  /** C 是否 ≤ 容量；false 时下方数据按满载并发计 */
  feasibleAtC: boolean;
  vramPerDevice: DeviceVram;
  /** 每副本吞吐（estimateThroughput 原样） */
  throughput: ThroughputResult;
  /** 全集群总吞吐 = 单用户 tok/s × C（C 超容量时 × Cmax） */
  clusterAggregateTokPerSec: number;
  /** N 张卡的满载能力 */
  capacity: {
    plan: CapacityPlan;
    /** dp × 每副本 KV 容量上限；generic（无 KV）为 Infinity */
    maxConcurrency: number;
    throughput: ThroughputResult;
    clusterAggregateTokPerSec: number;
  };
  /** 权重本身在该硬件上所需的最少卡数（TP×PP 枚举内都放不下为 null） */
  minDevicesToFit: number | null;
  /** 连 1 路并发都承载不了 / C 超容量时的原因说明 */
  reason?: string;
}

const ZERO_THROUGHPUT: ThroughputResult = { tpotMs: 0, perUserTokPerSec: 0, aggregateTokPerSec: 0, ttftMs: 0, kvTotalGB: 0 };

interface Candidate {
  tp: number;
  pp: number;
  dp: number;
  dpr: number; // devicesPerReplica = tp × pp
  maxReplicaConc: number; // 每副本 KV 容量上限（显存约束；权重放不下为 0）
  planAtC: ParallelPlan; // fitPlan(用户并发 C)
}

function searchInputOf(input: CapacityInput, config: BandConfig, opts: Optimizations): SearchInput {
  const prec = PRECISIONS[input.precisionId] ?? PRECISIONS.bf16;
  return {
    model: input.model,
    gpu: input.gpu,
    precisionId: input.precisionId,
    bytesPerParam: prec.bytesPerParam,
    kvBytesPerToken: kvBytesPerToken(input.model, input.precisionId),
    kvOffloadGB: tierOffloadGB(opts),
    concurrency: input.concurrency,
    avgCtxTokens: input.avgCtxTokens,
    overheadFactor: config.overheadFactor,
    overheadBaseGB: config.overheadBaseGB,
  };
}

function enumerateCandidates(n: number, si: SearchInput): Candidate[] {
  const maxTp = si.gpu.vendor === 'apple' ? 1 : si.gpu.gpusPerNode;
  const usable = usableVramGB(si.gpu);
  const out: Candidate[] = [];
  for (let tp = 1; tp <= maxTp; tp++) {
    for (let pp = 1; pp <= 8; pp++) {
      const dpr = tp * pp;
      if (dpr > n) break; // pp 再大只会更占卡
      const dp = Math.floor(n / dpr); // 剩余卡闲置也不单独成副本（成副本见更大 dpr）
      const fitsWeights = perDeviceFixedGB(tp, pp, si) <= usable;
      out.push({
        tp, pp, dp, dpr,
        maxReplicaConc: fitsWeights ? maxReplicaConcurrency(tp, pp, si) : 0,
        planAtC: fitPlan(tp, pp, dp, si),
      });
    }
  }
  return out;
}

/** 单副本吞吐。与 runScenario 的吞吐段同口径（那边 coldRetrieve 用全集群 KV，这里用每副本 KV）。 */
function throughputAt(input: CapacityInput, config: BandConfig, opts: Optimizations, dpr: number, replicaConc: number): ThroughputResult {
  if (!Number.isFinite(replicaConc) || replicaConc <= 0) return ZERO_THROUGHPUT;
  const prec = PRECISIONS[input.precisionId] ?? PRECISIONS.bf16;
  const activeWeightBytes = input.model.activeParamsB * 1e9 * prec.bytesPerParam;
  const effKvPerTok = kvBytesPerToken(input.model, input.precisionId) * (opts.kvFp8 ? 0.5 : 1);
  const kvTotalGB = (replicaConc * input.avgCtxTokens * effKvPerTok) / 1e9;
  const coldRetrieveMs =
    opts.coldCache.enabled && kvTotalGB > opts.hotCache.sizeGB
      ? (Math.min(opts.coldCache.sizeGB, kvTotalGB - opts.hotCache.sizeGB) / Math.max(opts.coldCache.ssdGBs, 1)) * 1000
      : 0;
  return estimateThroughput({
    model: input.model,
    gpus: Array.from({ length: dpr }, () => input.gpu),
    concurrency: replicaConc,
    avgCtxTokens: input.avgCtxTokens,
    inputTokens: input.inputTokens,
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
}

function asCapacityPlan(p: ParallelPlan, n: number): CapacityPlan {
  return {
    tp: p.tp,
    pp: p.pp,
    dp: p.dp,
    devicesPerReplica: p.devicesPerReplica,
    totalDevices: p.totalDevices,
    idleDevices: n - p.totalDevices,
    replicaConcurrency: p.replicaConcurrency,
    vramPerDeviceGB: p.vramPerDeviceGB,
    feasible: p.feasible && p.replicaConcurrency >= 1,
    reason: p.reason,
  };
}

function deviceVram(p: ParallelPlan, si: SearchInput): DeviceVram {
  const dpr = p.devicesPerReplica;
  const weightsGB = (si.model.paramsB * 1e9 * si.bytesPerParam) / dpr / 1e9;
  const replicaConc = Number.isFinite(p.replicaConcurrency) ? p.replicaConcurrency : 0;
  const kvGB = Math.max(0, replicaConc * si.avgCtxTokens * si.kvBytesPerToken - (si.kvOffloadGB ?? 0) * 1e9) / dpr / 1e9;
  const overheadGB = si.overheadBaseGB + weightsGB * si.overheadFactor;
  return { weightsGB, kvGB, overheadGB, totalGB: weightsGB + kvGB + overheadGB };
}

/** 权重本身（含开销、不含 KV）在该硬件上所需的最少卡数；TP×PP 枚举内放不下为 null。 */
function minDevicesToFitOf(si: SearchInput): number | null {
  const maxTp = si.gpu.vendor === 'apple' ? 1 : si.gpu.gpusPerNode;
  const usable = usableVramGB(si.gpu);
  for (let dpr = 1; dpr <= maxTp * 8; dpr++) {
    const decomposable = Array.from({ length: dpr }, (_, i) => i + 1).some(
      (tp) => tp <= maxTp && dpr % tp === 0 && dpr / tp <= 8,
    );
    if (decomposable && perDeviceFixedGB(1, dpr, si) <= usable) return dpr;
  }
  return null;
}

export function runCapacity(input: CapacityInput, band: Band, opts: Optimizations = OPTS_OFF): CapacityResult {
  const n = Math.max(1, Math.floor(input.devices));
  const config = BANDS[band];
  const si = searchInputOf(input, config, opts);
  const candidates = enumerateCandidates(n, si);
  const minDevicesToFit = minDevicesToFitOf(si);

  // 满载能力：每副本 KV 容量上限 × dp，取最大者；平手取满载时更快（tpot 小）的分解
  const capCandidates = candidates.filter((c) => c.maxReplicaConc >= 1);
  let capacityPlan: CapacityPlan | null = null;
  let capacityThpt = ZERO_THROUGHPUT;
  let maxConcurrency = 0;
  if (capCandidates.length > 0) {
    const ranked = [...capCandidates].sort((a, b) => {
      const ca = a.maxReplicaConc * a.dp;
      const cb = b.maxReplicaConc * b.dp;
      if (cb !== ca) return cb - ca;
      const ta = throughputAt(input, config, opts, a.dpr, a.maxReplicaConc).tpotMs;
      const tb = throughputAt(input, config, opts, b.dpr, b.maxReplicaConc).tpotMs;
      if (ta !== tb) return ta - tb;
      return a.tp - b.tp || a.pp - b.pp || a.dp - b.dp;
    });
    const best = ranked[0];
    maxConcurrency = best.maxReplicaConc * best.dp;
    capacityThpt = throughputAt(input, config, opts, best.dpr, best.maxReplicaConc);
    if (Number.isFinite(best.maxReplicaConc)) {
      const atCap = fitPlan(best.tp, best.pp, best.dp, { ...si, concurrency: maxConcurrency });
      capacityPlan = asCapacityPlan(atCap, n);
    } else {
      // generic（无 KV）：容量不受限，每卡占用 = 权重分片 + 开销
      const fixed = perDeviceFixedGB(best.tp, best.pp, si);
      capacityPlan = {
        tp: best.tp, pp: best.pp, dp: best.dp,
        devicesPerReplica: best.dpr,
        totalDevices: best.dpr * best.dp,
        idleDevices: n - best.dpr * best.dp,
        replicaConcurrency: Number.POSITIVE_INFINITY,
        vramPerDeviceGB: fixed,
        feasible: true,
      };
    }
  }
  const capacityClusterAggregate = Number.isFinite(maxConcurrency)
    ? capacityThpt.perUserTokPerSec * maxConcurrency
    : 0;

  // 服务当前并发 C：在装得下 C 的方案里选 tpot 最小者（= 单用户最快 = 集群总吞吐最高）
  const servingCandidates = candidates.filter((c) => c.planAtC.feasible && c.planAtC.replicaConcurrency >= 1);
  let plan: CapacityPlan;
  let vram: DeviceVram;
  let thpt: ThroughputResult;
  let clusterAggregate: number;
  let feasibleAtC: boolean;
  let reason: string | undefined;

  if (servingCandidates.length > 0) {
    const ranked = [...servingCandidates].sort((a, b) => {
      const ta = throughputAt(input, config, opts, a.dpr, a.planAtC.replicaConcurrency).tpotMs;
      const tb = throughputAt(input, config, opts, b.dpr, b.planAtC.replicaConcurrency).tpotMs;
      if (ta !== tb) return ta - tb;
      const idleA = n - a.planAtC.totalDevices;
      const idleB = n - b.planAtC.totalDevices;
      if (idleA !== idleB) return idleA - idleB;
      return a.tp - b.tp || a.pp - b.pp || a.dp - b.dp;
    });
    const best = ranked[0];
    plan = asCapacityPlan(best.planAtC, n);
    vram = deviceVram(best.planAtC, si);
    thpt = throughputAt(input, config, opts, best.dpr, best.planAtC.replicaConcurrency);
    clusterAggregate = thpt.perUserTokPerSec * input.concurrency;
    feasibleAtC = true;
  } else if (capacityPlan && maxConcurrency >= 1) {
    // C 超容量但卡本身放得下：按满载口径给出数据
    plan = capacityPlan;
    vram = deviceVram(
      { ...capacityPlan, devicesPerReplica: capacityPlan.devicesPerReplica, replicaConcurrency: capacityPlan.replicaConcurrency } as ParallelPlan,
      si,
    );
    thpt = capacityThpt;
    clusterAggregate = capacityClusterAggregate;
    feasibleAtC = false;
    reason = `当前并发 ${input.concurrency} 路超出 ${n} 张 ${si.gpu.name} 的容量：最多承载 ${maxConcurrency} 路，以下数据按满载 ${maxConcurrency} 路计`;
  } else {
    // 连 1 路并发都放不下
    const fallback = fitPlan(1, 1, n, si);
    plan = asCapacityPlan(fallback, n);
    vram = deviceVram(fallback, si);
    thpt = ZERO_THROUGHPUT;
    clusterAggregate = 0;
    feasibleAtC = false;
    const hint = minDevicesToFit
      ? `该硬件部署此模型最少需要 ${minDevicesToFit} 张（不含 KV 常驻）`
      : '已超出该硬件 TP×PP 枚举上限，无法部署';
    reason = `${fallback.reason ?? '显存不足'}；${hint}`;
  }

  return {
    band,
    config,
    devices: n,
    plan,
    feasibleAtC,
    vramPerDevice: vram,
    throughput: thpt,
    clusterAggregateTokPerSec: clusterAggregate,
    capacity: {
      plan: capacityPlan ?? plan,
      maxConcurrency,
      throughput: capacityThpt,
      clusterAggregateTokPerSec: capacityClusterAggregate,
    },
    minDevicesToFit,
    reason,
  };
}

export function runCapacityAllBands(input: CapacityInput, opts: Optimizations = OPTS_OFF): Record<Band, CapacityResult> {
  return {
    conservative: runCapacity(input, 'conservative', opts),
    typical: runCapacity(input, 'typical', opts),
    aggressive: runCapacity(input, 'aggressive', opts),
  };
}
