// Parallelism search: find the minimal device count that fits VRAM and report
// the (tp, pp, dp) decomposition. TP within a node, PP across nodes.
import type { ModelSpec } from '@/data/models';
import type { GpuSpec } from '@/data/gpus';
import { usableVramGB } from './vram';

export interface ParallelPlan {
  tp: number; // tensor parallel size (within one node)
  pp: number; // pipeline stages = nodes per replica
  dp: number; // data replicas serving different user subsets
  devicesPerReplica: number; // tp × pp
  totalDevices: number; // devicesPerReplica × dp
  replicaConcurrency: number; // users per replica
  vramPerDeviceGB: number; // worst-case device usage
  feasible: boolean;
  reason?: string;
}

export interface SearchInput {
  model: ModelSpec;
  gpu: GpuSpec;
  precisionId: string;
  bytesPerParam: number;
  kvBytesPerToken: number;
  concurrency: number;
  avgCtxTokens: number;
  overheadFactor: number;
  overheadBaseGB: number;
  /** 热缓存+冷缓存卸载的 KV 字节（GB）：这部分不占 GPU 显存 */
  kvOffloadGB?: number;
}

/** 单方案显存校验：给定 (tp, pp, dp) 与用户并发，算每卡占用并判定是否装得下。 */
export function fitPlan(
  tp: number,
  pp: number,
  dp: number,
  input: SearchInput,
): ParallelPlan {
  const { model, gpu, bytesPerParam, kvBytesPerToken, concurrency, avgCtxTokens, overheadFactor, overheadBaseGB, kvOffloadGB } = input;

  const totalWeightBytes = model.paramsB * 1e9 * bytesPerParam;
  const devicesPerReplica = tp * pp;
  const replicaConcurrency = Math.ceil(concurrency / Math.max(dp, 1));
  // 分层 KV：热/冷缓存卸载的部分不占 GPU 显存
  const kvTotalBytes = Math.max(0, replicaConcurrency * avgCtxTokens * kvBytesPerToken - (kvOffloadGB ?? 0) * 1e9);

  // per-device: weight shard + KV slice + overhead share
  const weightPerDeviceGB = totalWeightBytes / devicesPerReplica / 1e9;
  const kvPerDeviceGB = kvTotalBytes / devicesPerReplica / 1e9;
  const overheadPerDeviceGB = overheadBaseGB + weightPerDeviceGB * overheadFactor;
  const vramPerDeviceGB = weightPerDeviceGB + kvPerDeviceGB + overheadPerDeviceGB;

  const usable = usableVramGB(gpu);
  const feasible = vramPerDeviceGB <= usable;

  return {
    tp, pp, dp,
    devicesPerReplica,
    totalDevices: devicesPerReplica * dp,
    replicaConcurrency,
    vramPerDeviceGB,
    feasible,
    reason: feasible ? undefined : `单卡需 ${vramPerDeviceGB.toFixed(1)}GB > 可用 ${usable.toFixed(1)}GB`,
  };
}

/** Enumerate (tp, pp, dp) and return the minimal-device feasible plan (ties → smaller tp). */
export function findMinimalPlan(input: SearchInput): ParallelPlan {
  const { gpu } = input;
  const maxTp = gpu.vendor === 'apple' ? 1 : gpu.gpusPerNode;
  const candidates: ParallelPlan[] = [];

  for (let tp = 1; tp <= maxTp; tp++) {
    for (let pp = 1; pp <= 8; pp++) {
      for (let dp = 1; dp <= 16; dp++) {
        if (dp * input.concurrency < input.concurrency) continue; // noop guard
        const plan = fitPlan(tp, pp, dp, input);
        if (!plan.feasible) continue;
        // every replica must host at least 1 user
        if (plan.replicaConcurrency < 1) continue;
        candidates.push(plan);
      }
    }
  }
  if (candidates.length === 0) {
    // report the tightest infeasible config for the UI
    const best = fitPlan(Math.min(maxTp, 1), 8, 1, input);
    return best;
  }
  candidates.sort((a, b) =>
    a.totalDevices - b.totalDevices || a.tp - b.tp || a.pp - b.pp || a.dp - b.dp,
  );
  return candidates[0];
}

export function allFeasiblePlans(input: SearchInput): ParallelPlan[] {
  const { gpu } = input;
  const maxTp = gpu.vendor === 'apple' ? 1 : gpu.gpusPerNode;
  const plans: ParallelPlan[] = [];
  for (let tp = 1; tp <= maxTp; tp++) {
    for (let pp = 1; pp <= 8; pp++) {
      for (let dp = 1; dp <= 16; dp++) {
        const plan = fitPlan(tp, pp, dp, input);
        if (plan.feasible && plan.replicaConcurrency >= 1) plans.push(plan);
      }
    }
  }
  return plans;
}

/** 单卡固定占用（权重分片 + 运行时开销），不含 KV。返回 GB。 */
export function perDeviceFixedGB(tp: number, pp: number, input: SearchInput): number {
  const weightPerDeviceGB = (input.model.paramsB * 1e9 * input.bytesPerParam) / (tp * pp) / 1e9;
  return weightPerDeviceGB + input.overheadBaseGB + weightPerDeviceGB * input.overheadFactor;
}

/**
 * 显存容量上限（反向推演用）：给定 (tp, pp) 分解时，单个副本最多可承载的并发路数。
 * KV 预算 = (可用显存 − 固定占用) × 单副本卡数 + 缓存卸载量，除以每路 KV 字节后向下取整。
 * 与 fitPlan 的判定式互为逆运算（fitPlan 里 kvTotalBytes 的 max(0, …) 与此处的预算口径一致）。
 * generic 模型无 KV → Infinity。
 */
export function maxReplicaConcurrency(tp: number, pp: number, input: SearchInput): number {
  const { gpu, avgCtxTokens, kvBytesPerToken, kvOffloadGB } = input;
  if (kvBytesPerToken <= 0 || avgCtxTokens <= 0) return Number.POSITIVE_INFINITY;
  const fixedGB = perDeviceFixedGB(tp, pp, input);
  const kvBudgetBytes = Math.max(0, usableVramGB(gpu) - fixedGB) * tp * pp * 1e9 + (kvOffloadGB ?? 0) * 1e9;
  return Math.floor(kvBudgetBytes / (avgCtxTokens * kvBytesPerToken));
}

/**
 * 反向模式：给定实际设备数 N，枚举所有 (tp, pp, dp) 且 tp×pp×dp === N 的可行切分。
 * 约束：tp ≤ 节点内卡数、pp ≤ 8、dp = N/(tp×pp) 为整数。
 */
export function plansForDevices(input: SearchInput, devices: number): ParallelPlan[] {
  const { gpu } = input;
  const maxTp = gpu.vendor === 'apple' ? 1 : gpu.gpusPerNode;
  const plans: ParallelPlan[] = [];
  if (!Number.isFinite(devices) || devices < 1 || devices > 128) return plans;
  for (let tp = 1; tp <= Math.min(maxTp, devices); tp++) {
    if (devices % tp !== 0) continue;
    const rest = devices / tp;
    for (let pp = 1; pp <= Math.min(rest, 8); pp++) {
      if (rest % pp !== 0) continue;
      const dp = rest / pp;
      const plan = fitPlan(tp, pp, dp, input);
      if (plan.feasible && plan.replicaConcurrency >= 1) plans.push(plan);
    }
  }
  // 单用户吞吐代理排序：每步读取字节越少、聚合带宽越高越好
  plans.sort((a, b) => {
    const scoreA = a.replicaConcurrency / (a.tp * gpu.memBandwidthGBs);
    const scoreB = b.replicaConcurrency / (b.tp * gpu.memBandwidthGBs);
    return scoreA - scoreB;
  });
  return plans;
}
