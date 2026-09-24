import type { ModelSpec } from '@/data/models';
import type { GpuSpec } from '@/data/gpus';
import { modelById } from '@/data/models';
import { gpuById } from '@/data/gpus';

export interface PresetScenario {
  id: string;
  title: string;
  description: string;
  modelId: string;
  gpuId: string;
  precisionId: string;
  concurrency: number;
  avgCtxTokens: number;
  inputTokens: number;
  /** model ids whose config was fetched but derived numbers are estimates */
  caveat?: string;
}

export const PRESET_SCENARIOS: PresetScenario[] = [
  {
    id: 'v4flash-h100',
    title: 'DeepSeek-V4-Flash-0731 × H100',
    description: '1M 上下文 MLA 稀疏注意力模型，FP8 原生权重，32 并发 × 32K 上下文',
    modelId: 'deepseek-ai/DeepSeek-V4-Flash-0731',
    gpuId: 'h100-sxm',
    precisionId: 'fp8',
    concurrency: 32,
    avgCtxTokens: 32768,
    inputTokens: 4096,
    caveat: '参数量与激活参数由 checkpoint 体量推算，官方口径以模型卡为准',
  },
  {
    id: 'glm53flash-h100',
    title: 'GLM-5.3-Flash × H100',
    description: '288 专家 MoE（激活 8），原生 FP8，1M 上下文，16 并发 × 64K 上下文',
    modelId: 'zai-org/GLM-5.3-Flash',
    gpuId: 'h100-sxm',
    precisionId: 'fp8',
    concurrency: 16,
    avgCtxTokens: 65536,
    inputTokens: 8192,
    caveat: '参数量与激活参数由 checkpoint 体量推算，官方口径以模型卡为准',
  },
];

export function resolveScenario(s: PresetScenario): { model: ModelSpec; gpu: GpuSpec } | null {
  const model = modelById(s.modelId);
  const gpu = gpuById(s.gpuId);
  if (!model || !gpu) return null;
  return { model, gpu };
}
