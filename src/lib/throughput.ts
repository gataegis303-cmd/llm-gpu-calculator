// Throughput roofline: decode (bandwidth-bound TPOT) + prefill (compute-bound TTFT).
// All byte counts passed in are PRECISION-ADJUSTED by the caller (search.ts).
import type { ModelSpec } from '@/data/models';
import type { GpuSpec } from '@/data/gpus';

export interface ThroughputInput {
  model: ModelSpec;
  gpus: GpuSpec[]; // devices in one serving replica (tp × pp)
  concurrency: number;
  avgCtxTokens: number;
  inputTokens: number; // prompt length for TTFT
  eta: number; // aggregate bandwidth efficiency (0.7–0.9)
  computeUtil: number; // prefill compute utilization (0.3–0.5)
  /** precision-adjusted bytes of ACTIVE weights read per decode step */
  activeWeightBytes: number;
  /** KV bytes per token (whole model, precision-adjusted KV dtype, 已含 KV FP8 折减) */
  kvBytesPerToken: number;
  opts?: {
    chunked: boolean;
    chunkSize: number;
    hotHitRate: number; // 0 = off；命中部分免重复预填充
    coldRetrieveMs: number; // 0 = none；冷缓存检索延迟计入 TTFT
  };
}

export interface ThroughputResult {
  tpotMs: number; // per output token, per user
  perUserTokPerSec: number;
  aggregateTokPerSec: number;
  ttftMs: number;
  kvTotalGB: number;
}

export function estimateThroughput(input: ThroughputInput): ThroughputResult {
  const { model, gpus, concurrency, avgCtxTokens, inputTokens, eta, computeUtil, activeWeightBytes, kvBytesPerToken, opts } = input;

  const bwAggregateGBs = gpus.reduce((s, g) => s + g.memBandwidthGBs, 0) * eta;

  // decode step: read active weights once + every concurrent sequence's KV
  const kvTotalBytes = concurrency * avgCtxTokens * kvBytesPerToken;
  const stepMs = ((activeWeightBytes + kvTotalBytes) / (bwAggregateGBs * 1e9)) * 1000;

  // prefill: 2 × active_params × input_tokens FLOPs, floored by cold weights load
  const flops = 2 * model.activeParamsB * 1e9 * inputTokens;
  const aggregateFlops = gpus.reduce((s, g) => s + (g.fp8Tflops ?? g.bf16Tflops), 0) * 1e12;
  // 热缓存命中：命中比例的前缀免重复预填充
  const hotHit = opts?.hotHitRate ?? 0;
  const prefillComputeMs = ((flops / (aggregateFlops * computeUtil)) * 1000) * (1 - hotHit);
  const weightLoadMs = (activeWeightBytes / (bwAggregateGBs * 1e9)) * 1000;
  // 分块预填充：解码步与预填充块交错，TPOT 轻微上浮
  const tpotFinal = opts?.chunked ? stepMs * 1.05 : stepMs;
  const ttftBase = opts?.chunked
    ? Math.ceil(inputTokens / Math.max(opts.chunkSize, 1)) *
      Math.max((prefillComputeMs * opts.chunkSize) / Math.max(inputTokens, 1), tpotFinal)
    : Math.max(prefillComputeMs, weightLoadMs);
  // 冷缓存检索：超出热层的 KV 从 SSD 回读
  const ttft = ttftBase + (opts?.coldRetrieveMs ?? 0);

  return {
    tpotMs: tpotFinal,
    perUserTokPerSec: 1000 / Math.max(tpotFinal, 1e-6),
    aggregateTokPerSec: (1000 / Math.max(tpotFinal, 1e-6)) * concurrency,
    ttftMs: ttft,
    kvTotalGB: kvTotalBytes / 1e9,
  };
}
