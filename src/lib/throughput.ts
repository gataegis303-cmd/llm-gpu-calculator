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
  /** KV bytes per token (whole model, precision-adjusted KV dtype) */
  kvBytesPerToken: number;
}

export interface ThroughputResult {
  tpotMs: number; // per output token, per user
  perUserTokPerSec: number;
  aggregateTokPerSec: number;
  ttftMs: number;
  kvTotalGB: number;
}

export function estimateThroughput(input: ThroughputInput): ThroughputResult {
  const { model, gpus, concurrency, avgCtxTokens, inputTokens, eta, computeUtil, activeWeightBytes, kvBytesPerToken } = input;

  const bwAggregateGBs = gpus.reduce((s, g) => s + g.memBandwidthGBs, 0) * eta;

  // decode step: read active weights once + every concurrent sequence's KV
  const kvTotalBytes = concurrency * avgCtxTokens * kvBytesPerToken;
  const stepMs = ((activeWeightBytes + kvTotalBytes) / (bwAggregateGBs * 1e9)) * 1000;
  const perUser = 1000 / Math.max(stepMs, 1e-6);

  // prefill: 2 × active_params × input_tokens FLOPs, floored by cold weights load
  const flops = 2 * model.activeParamsB * 1e9 * inputTokens;
  const aggregateFlops = gpus.reduce((s, g) => s + (g.fp8Tflops ?? g.bf16Tflops), 0) * 1e12;
  const prefillComputeMs = (flops / (aggregateFlops * computeUtil)) * 1000;
  const weightLoadMs = (activeWeightBytes / (bwAggregateGBs * 1e9)) * 1000;
  const ttft = Math.max(prefillComputeMs, weightLoadMs);

  return {
    tpotMs: stepMs,
    perUserTokPerSec: perUser,
    aggregateTokPerSec: perUser * concurrency,
    ttftMs: ttft,
    kvTotalGB: kvTotalBytes / 1e9,
  };
}
