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

function fitPlan(
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
