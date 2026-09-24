// 反查：给定一台设备，列出能部署哪些模型。
// 口径：显存 = 权重(指定精度) + 并发×上下文×KV/token + 开销(激进档系数)，≤ 设备可用显存。
import type { ModelSpec } from '@/data/models';
import { MODELS } from '@/data/models';
import type { GpuSpec } from '@/data/gpus';
import { usableVramGB, estimateVram } from './vram';

export interface FitOptions {
  concurrency: number;
  avgCtxTokens: number;
  /** 开销系数档：默认激进（给个人设备更接近真实的上限） */
  overheadFactor: number;
  overheadBaseGB: number;
}

export interface FitResult {
  model: ModelSpec;
  precisionId: string;
  totalGB: number;
  /** 剩余显存还够几路并发（≥1 表示至少放得下） */
  headroomConcurrency: number;
}

const DEFAULTS: FitOptions = {
  concurrency: 1,
  avgCtxTokens: 8192,
  overheadFactor: 0.05,
  overheadBaseGB: 2,
};

/**
 * 遍历内置模型库，返回在某精度档下可部署于该设备的模型（按参数量降序）。
 * 每个模型取其「可用的最大精度档」：优先 bf16，其次 fp8/int8，最后 int4/gguf_q4km。
 */
export function deployableModels(gpu: GpuSpec, options: Partial<FitOptions> = {}): {
  full: FitResult[]; // 高精度档（bf16 / 原生 fp8）
  quantized: FitResult[]; // 量化档（int4 / gguf_q4km）
  usableGB: number;
} {
  const opts = { ...DEFAULTS, ...options };
  const usable = usableVramGB(gpu);
  const full: FitResult[] = [];
  const quantized: FitResult[] = [];

  for (const model of MODELS_SORTED) {
    const preferred = model.precisions.map((p) => p.id);
    // 高精度档：优先原生顺序（bf16 或 fp8 原生权重）
    const fullPrec = preferred.includes('bf16') ? 'bf16' : preferred.includes('fp8') ? 'fp8' : null;
    const quantPrec = ['int4', 'gguf_q4km'].find((p) => preferred.includes(p)) ?? null;

    if (fullPrec) {
      const v = estimateVram({
        model, precisionId: fullPrec, concurrency: opts.concurrency, avgCtxTokens: opts.avgCtxTokens,
        overheadFactor: opts.overheadFactor, overheadBaseGB: opts.overheadBaseGB,
      });
      if (v.totalGB <= usable) {
        full.push({
          model, precisionId: fullPrec, totalGB: v.totalGB,
          headroomConcurrency: Math.max(1, Math.floor((usable - v.weightsGB - v.overheadGB) / Math.max(v.kvTotalGB, 1e-9))),
        });
      }
    }
    if (quantPrec) {
      const v = estimateVram({
        model, precisionId: quantPrec, concurrency: opts.concurrency, avgCtxTokens: opts.avgCtxTokens,
        overheadFactor: opts.overheadFactor, overheadBaseGB: opts.overheadBaseGB,
      });
      if (v.totalGB <= usable) {
        quantized.push({
          model, precisionId: quantPrec, totalGB: v.totalGB,
          headroomConcurrency: Math.max(1, Math.floor((usable - v.weightsGB - v.overheadGB) / Math.max(v.kvTotalGB, 1e-9))),
        });
      }
    }
  }

  return { full, quantized, usableGB: usable };
}

const MODELS_SORTED = [...MODELS].sort((a, b) => b.paramsB - a.paramsB);
