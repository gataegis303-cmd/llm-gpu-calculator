// Roofline VRAM estimation engine — pure functions.
// Formulas:
//   weightsGB   = paramsB × bytesPerParam (quantized) ; MoE stores ALL experts
//   kvPerTokKB  = GQA/MHA: 2 × layers × kvHeads × headDim × kvDtypeBytes
//                 MLA   : (kvLoraRank + ropeDim) × kvDtypeBytes   [per layer, × layers]
//   overheadGB  = base (2–4) + weights × 5–10 %
//   totalGB     = weights + concurrency × avgCtx × kvPerTok + overhead
import type { ModelSpec } from '@/data/models';
import { PRECISIONS } from '@/data/models';
import type { GpuSpec } from '@/data/gpus';

export interface VramInput {
  model: ModelSpec;
  precisionId: string;
  concurrency: number;
  avgCtxTokens: number;
  /** overhead aggressiveness: conservative = big margin, aggressive = tight */
  overheadFactor: number; // fraction of weights, e.g. 0.05–0.10
  overheadBaseGB: number; // e.g. 2–4
}

export interface VramBreakdown {
  weightsGB: number;
  kvPerTokKB: number;
  kvTotalGB: number;
  overheadGB: number;
  totalGB: number;
}

export function weightsBytesGB(paramsB: number, precisionId: string): number {
  const p = PRECISIONS[precisionId] ?? PRECISIONS.bf16;
  return (paramsB * 1e9 * p.bytesPerParam) / 1e9;
}

/** KV cache bytes per token for the WHOLE model (all layers). */
export function kvBytesPerToken(model: ModelSpec, precisionId: string): number {
  const p = PRECISIONS[precisionId] ?? PRECISIONS.bf16;
  const dtype = p.kvDtypeBytes; // KV usually stays fp16/bf16 even for quantized weights
  // 非自回归模型（generic）：无 KV 缓存
  if (!model.isMLA && (model.numKvHeads === 0 || model.headDim === 0)) return 0;
  if (model.isMLA) {
    const rank = model.mlaKvLoraRank ?? 512;
    const rope = model.mlaRopeHeadDim ?? 0;
    // DeepSeek MLA: c_KV (rank) + k_rope shared across heads, per layer
    return (rank + rope) * dtype * model.numHiddenLayers;
  }
  return 2 * model.numHiddenLayers * model.numKvHeads * model.headDim * dtype;
}

export function estimateVram(input: VramInput): VramBreakdown {
  const { model, precisionId, concurrency, avgCtxTokens, overheadFactor, overheadBaseGB } = input;
  const weightsGB = weightsBytesGB(model.paramsB, precisionId);
  const kvPerTokKB = kvBytesPerToken(model, precisionId) / 1024;
  const kvTotalGB = (concurrency * avgCtxTokens * kvBytesPerToken(model, precisionId)) / 1e9;
  const overheadGB = overheadBaseGB + weightsGB * overheadFactor;
  return {
    weightsGB,
    kvPerTokKB,
    kvTotalGB,
    overheadGB,
    totalGB: weightsGB + kvTotalGB + overheadGB,
  };
}

/** Usable VRAM per device. */
export function usableVramGB(gpu: GpuSpec): number {
  if (gpu.usableFraction !== undefined) {
    // 统一内存架构（Apple M 系列 / GB10）：按可用占比扣减
    return gpu.vramGB * gpu.usableFraction - (gpu.vendor === 'apple' ? 2 : 1.2);
  }
  if (gpu.vendor === 'apple') {
    return gpu.vramGB * 0.75 - 2; // OS reserve inside the usable slice
  }
  return gpu.vramGB - 1.2; // CUDA/ROCm/CANN context + fragmentation reserve
}
