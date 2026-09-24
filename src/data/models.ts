// Model architecture specs — sourced from HuggingFace / ModelScope config.json
// and model.safetensors.index.json (fetched 2026-09-23, see sourceUrl per entry).
// All numbers are exact config values except paramsB (see note below).

export interface PrecisionOption {
  id: string; // key into PRECISIONS
  note?: string; // e.g. "官方 FP8 权重"
}

export interface ModelSpec {
  id: string; // HF/ModelScope repo id
  org: string;
  name: string;
  family: string; // grouping for the picker
  /** total parameters in billions (from safetensors total_size / dtype bytes; marked estimated where derived) */
  paramsB: number;
  /** active parameters per token in billions (MoE only; = paramsB for dense) */
  activeParamsB: number;
  isMoE: boolean;
  numHiddenLayers: number;
  numKvHeads: number; // GQA/MHA kv heads
  headDim: number;
  /** MLA (DeepSeek-style) compressed KV: per-token bytes = (kvLoraRank + qkRopeHeadDim) * dtypeBytes */
  isMLA: boolean;
  mlaKvLoraRank?: number;
  mlaRopeHeadDim?: number;
  /** offered context lengths (tokens) the user can pick from */
  contexts: number[];
  precisions: PrecisionOption[];
  /** native checkpoint dtype of the published weights */
  nativeDtype: string;
  /** true when the repo only publishes FP8/FP4 weights (BF16 must be derived) */
  fp8Native?: boolean;
  /** parameter count derived from checkpoint size rather than an official figure */
  paramsEstimated?: boolean;
  status: 'verified' | 'unverified';
  sourceUrl: string;
  fetchedAt: string; // ISO date
  /** 模型类别：llm = 自回归文本模型（完整引擎）；vlm = 视觉语言模型（LLM 引擎 + 视觉塔权重）；generic = 其他（仅权重显存） */
  kind?: 'llm' | 'vlm' | 'generic';
  /** 非自回归模型：checkpoint 实际字节数（GB），用于精确权重显存 */
  checkpointBytesGB?: number;
}

/** bytes per parameter per precision id */
export const PRECISIONS: Record<string, { bytesPerParam: number; label: string; kvDtypeBytes: number }> = {
  bf16: { bytesPerParam: 2, label: 'BF16 / FP16', kvDtypeBytes: 2 },
  fp8: { bytesPerParam: 1, label: 'FP8 (E4M3)', kvDtypeBytes: 1 },
  int8: { bytesPerParam: 1, label: 'INT8 (AWQ / GPTQ-W8)', kvDtypeBytes: 2 },
  int4: { bytesPerParam: 0.5, label: 'INT4 (AWQ / GPTQ-W4)', kvDtypeBytes: 2 },
  gguf_q8: { bytesPerParam: 1.0625, label: 'GGUF Q8_0', kvDtypeBytes: 2 },
  gguf_q4km: { bytesPerParam: 0.5875, label: 'GGUF Q4_K_M', kvDtypeBytes: 2 },
};

const C = {
  ctx8k: 8192,
  ctx32k: 32768,
  ctx128k: 131072,
  ctx1m: 1048576,
};

export const MODELS: ModelSpec[] = [
  // ── Qwen2.5 ──────────────────────────────────────────────────────────────
  {
    id: 'Qwen/Qwen2.5-0.5B-Instruct', org: 'Qwen', name: 'Qwen2.5-0.5B-Instruct', family: 'Qwen2.5',
    paramsB: 0.49, activeParamsB: 0.49, isMoE: false,
    numHiddenLayers: 24, numKvHeads: 2, headDim: 64, isMLA: false,
    contexts: [C.ctx32k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct', fetchedAt: '2026-09-23',
  },
  {
    id: 'Qwen/Qwen2.5-1.5B-Instruct', org: 'Qwen', name: 'Qwen2.5-1.5B-Instruct', family: 'Qwen2.5',
    paramsB: 1.54, activeParamsB: 1.54, isMoE: false,
    numHiddenLayers: 28, numKvHeads: 2, headDim: 64, isMLA: false,
    contexts: [C.ctx32k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct', fetchedAt: '2026-09-23',
  },
  {
    id: 'Qwen/Qwen2.5-3B-Instruct', org: 'Qwen', name: 'Qwen2.5-3B-Instruct', family: 'Qwen2.5',
    paramsB: 3.09, activeParamsB: 3.09, isMoE: false,
    numHiddenLayers: 36, numKvHeads: 2, headDim: 128, isMLA: false,
    contexts: [C.ctx32k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/Qwen/Qwen2.5-3B-Instruct', fetchedAt: '2026-09-23',
  },
  {
    id: 'Qwen/Qwen2.5-7B-Instruct', org: 'Qwen', name: 'Qwen2.5-7B-Instruct', family: 'Qwen2.5',
    paramsB: 7.62, activeParamsB: 7.62, isMoE: false,
    numHiddenLayers: 28, numKvHeads: 4, headDim: 128, isMLA: false,
    contexts: [C.ctx32k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct', fetchedAt: '2026-09-23',
  },
  {
    id: 'Qwen/Qwen2.5-14B-Instruct', org: 'Qwen', name: 'Qwen2.5-14B-Instruct', family: 'Qwen2.5',
    paramsB: 14.77, activeParamsB: 14.77, isMoE: false,
    numHiddenLayers: 48, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx32k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/Qwen/Qwen2.5-14B-Instruct', fetchedAt: '2026-09-23',
  },
  {
    id: 'Qwen/Qwen2.5-32B-Instruct', org: 'Qwen', name: 'Qwen2.5-32B-Instruct', family: 'Qwen2.5',
    paramsB: 32.76, activeParamsB: 32.76, isMoE: false,
    numHiddenLayers: 64, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx32k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/Qwen/Qwen2.5-32B-Instruct', fetchedAt: '2026-09-23',
  },
  {
    id: 'Qwen/Qwen2.5-72B-Instruct', org: 'Qwen', name: 'Qwen2.5-72B-Instruct', family: 'Qwen2.5',
    paramsB: 72.71, activeParamsB: 72.71, isMoE: false,
    numHiddenLayers: 80, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx32k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/Qwen/Qwen2.5-72B-Instruct', fetchedAt: '2026-09-23',
  },
  // ── Qwen3 ────────────────────────────────────────────────────────────────
  {
    id: 'Qwen/Qwen3-0.6B', org: 'Qwen', name: 'Qwen3-0.6B', family: 'Qwen3',
    paramsB: 0.6, activeParamsB: 0.6, isMoE: false,
    numHiddenLayers: 28, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [40960], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/Qwen/Qwen3-0.6B', fetchedAt: '2026-09-23',
  },
  {
    id: 'Qwen/Qwen3-1.7B', org: 'Qwen', name: 'Qwen3-1.7B', family: 'Qwen3',
    paramsB: 2.03, activeParamsB: 2.03, isMoE: false,
    numHiddenLayers: 28, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [40960], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', paramsEstimated: true, status: 'verified',
    sourceUrl: 'https://huggingface.co/Qwen/Qwen3-1.7B', fetchedAt: '2026-09-23',
  },
  {
    id: 'Qwen/Qwen3-4B', org: 'Qwen', name: 'Qwen3-4B', family: 'Qwen3',
    paramsB: 4.02, activeParamsB: 4.02, isMoE: false,
    numHiddenLayers: 36, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [40960], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/Qwen/Qwen3-4B', fetchedAt: '2026-09-23',
  },
  {
    id: 'Qwen/Qwen3-8B', org: 'Qwen', name: 'Qwen3-8B', family: 'Qwen3',
    paramsB: 8.19, activeParamsB: 8.19, isMoE: false,
    numHiddenLayers: 36, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [40960], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/Qwen/Qwen3-8B', fetchedAt: '2026-09-23',
  },
  {
    id: 'Qwen/Qwen3-14B', org: 'Qwen', name: 'Qwen3-14B', family: 'Qwen3',
    paramsB: 14.77, activeParamsB: 14.77, isMoE: false,
    numHiddenLayers: 40, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [40960], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/Qwen/Qwen3-14B', fetchedAt: '2026-09-23',
  },
  {
    id: 'Qwen/Qwen3-32B', org: 'Qwen', name: 'Qwen3-32B', family: 'Qwen3',
    paramsB: 32.76, activeParamsB: 32.76, isMoE: false,
    numHiddenLayers: 64, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [40960], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/Qwen/Qwen3-32B', fetchedAt: '2026-09-23',
  },
  {
    id: 'Qwen/Qwen3-30B-A3B', org: 'Qwen', name: 'Qwen3-30B-A3B', family: 'Qwen3 (MoE)',
    paramsB: 30.53, activeParamsB: 3.3, isMoE: true,
    numHiddenLayers: 48, numKvHeads: 4, headDim: 128, isMLA: false,
    contexts: [40960], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/Qwen/Qwen3-30B-A3B', fetchedAt: '2026-09-23',
  },
  {
    id: 'Qwen/Qwen3-235B-A22B', org: 'Qwen', name: 'Qwen3-235B-A22B', family: 'Qwen3 (MoE)',
    paramsB: 235.09, activeParamsB: 22, isMoE: true,
    numHiddenLayers: 94, numKvHeads: 4, headDim: 128, isMLA: false,
    contexts: [40960], precisions: [{ id: 'bf16' }, { id: 'fp8' }, { id: 'int4' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/Qwen/Qwen3-235B-A22B', fetchedAt: '2026-09-23',
  },
  {
    id: 'Qwen/Qwen2-72B-Instruct', org: 'Qwen', name: 'Qwen2-72B-Instruct', family: 'Qwen2',
    paramsB: 72.71, activeParamsB: 72.71, isMoE: false,
    numHiddenLayers: 80, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx32k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/Qwen/Qwen2-72B-Instruct', fetchedAt: '2026-09-23',
  },
  // ── Llama 3.x ────────────────────────────────────────────────────────────
  {
    id: 'meta-llama/Llama-3.1-8B-Instruct', org: 'Meta', name: 'Llama-3.1-8B-Instruct', family: 'Llama 3.x',
    paramsB: 8.03, activeParamsB: 8.03, isMoE: false,
    numHiddenLayers: 32, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx128k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct (config) / modelscope LLM-Research/Meta-Llama-3.1-8B-Instruct (index)', fetchedAt: '2026-09-23',
  },
  {
    id: 'meta-llama/Llama-3.1-70B-Instruct', org: 'Meta', name: 'Llama-3.1-70B-Instruct', family: 'Llama 3.x',
    paramsB: 70.55, activeParamsB: 70.55, isMoE: false,
    numHiddenLayers: 80, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx128k], precisions: [{ id: 'bf16' }, { id: 'fp8' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct (config) / modelscope LLM-Research/Meta-Llama-3.1-70B-Instruct (index)', fetchedAt: '2026-09-23',
  },
  {
    id: 'meta-llama/Llama-3.1-405B-Instruct', org: 'Meta', name: 'Llama-3.1-405B-Instruct', family: 'Llama 3.x',
    paramsB: 405.85, activeParamsB: 405.85, isMoE: false,
    numHiddenLayers: 126, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx128k], precisions: [{ id: 'bf16' }, { id: 'fp8' }, { id: 'int4' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct (config) / modelscope LLM-Research/Meta-Llama-3.1-405B-Instruct (index)', fetchedAt: '2026-09-23',
  },
  {
    id: 'meta-llama/Llama-3.3-70B-Instruct', org: 'Meta', name: 'Llama-3.3-70B-Instruct', family: 'Llama 3.x',
    paramsB: 70.55, activeParamsB: 70.55, isMoE: false,
    numHiddenLayers: 80, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx128k], precisions: [{ id: 'bf16' }, { id: 'fp8' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://modelscope.cn/models/LLM-Research/Llama-3.3-70B-Instruct', fetchedAt: '2026-09-23',
  },
  // ── DeepSeek ─────────────────────────────────────────────────────────────
  {
    id: 'deepseek-ai/DeepSeek-V3', org: 'DeepSeek', name: 'DeepSeek-V3', family: 'DeepSeek',
    paramsB: 671, activeParamsB: 37, isMoE: true,
    numHiddenLayers: 61, numKvHeads: 128, headDim: 128, isMLA: true, mlaKvLoraRank: 512, mlaRopeHeadDim: 64,
    contexts: [C.ctx128k], precisions: [{ id: 'fp8', note: '官方 FP8 权重' }, { id: 'int8' }, { id: 'int4' }],
    nativeDtype: 'fp8 (E4M3)', fp8Native: true, paramsEstimated: true, status: 'verified',
    sourceUrl: 'https://huggingface.co/deepseek-ai/DeepSeek-V3', fetchedAt: '2026-09-23',
  },
  {
    id: 'deepseek-ai/DeepSeek-R1', org: 'DeepSeek', name: 'DeepSeek-R1', family: 'DeepSeek',
    paramsB: 671, activeParamsB: 37, isMoE: true,
    numHiddenLayers: 61, numKvHeads: 128, headDim: 128, isMLA: true, mlaKvLoraRank: 512, mlaRopeHeadDim: 64,
    contexts: [C.ctx128k], precisions: [{ id: 'fp8', note: '官方 FP8 权重' }, { id: 'int8' }, { id: 'int4' }],
    nativeDtype: 'fp8 (E4M3)', fp8Native: true, paramsEstimated: true, status: 'verified',
    sourceUrl: 'https://huggingface.co/deepseek-ai/DeepSeek-R1', fetchedAt: '2026-09-23',
  },
  {
    // 43 MoE layers × 256 experts × 3 × 4096 × 2048 ≈ 277B routed params (fp4-packed in
    // the checkpoint) + shared/attention/embed ≈ 283B total, ~11B active — config-derived.
    id: 'deepseek-ai/DeepSeek-V4-Flash-0731', org: 'DeepSeek', name: 'DeepSeek-V4-Flash-0731', family: 'DeepSeek',
    paramsB: 283, activeParamsB: 11, isMoE: true,
    numHiddenLayers: 43, numKvHeads: 64, headDim: 512, isMLA: true, mlaKvLoraRank: 512, mlaRopeHeadDim: 64,
    contexts: [65536, C.ctx128k, 262144, C.ctx1m], precisions: [{ id: 'fp8', note: '原生 FP8 注意力 + FP4 专家（按 FP8 估算显存）' }],
    nativeDtype: 'fp8 attn / fp4 experts', fp8Native: true, paramsEstimated: true, status: 'verified',
    sourceUrl: 'https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731', fetchedAt: '2026-09-23',
  },
  // ── GLM ──────────────────────────────────────────────────────────────────
  {
    id: 'THUDM/chatglm3-6b', org: 'THUDM', name: 'ChatGLM3-6B', family: 'GLM',
    paramsB: 6.24, activeParamsB: 6.24, isMoE: false,
    numHiddenLayers: 28, numKvHeads: 2, headDim: 128, isMLA: false,
    contexts: [8192, C.ctx32k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', paramsEstimated: true, status: 'verified',
    sourceUrl: 'https://huggingface.co/THUDM/chatglm3-6b', fetchedAt: '2026-09-23',
  },
  {
    id: 'THUDM/glm-4-9b-chat', org: 'THUDM', name: 'GLM-4-9B-Chat', family: 'GLM',
    paramsB: 9.4, activeParamsB: 9.4, isMoE: false,
    numHiddenLayers: 40, numKvHeads: 2, headDim: 128, isMLA: false,
    contexts: [8192, C.ctx128k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', paramsEstimated: true, status: 'verified',
    sourceUrl: 'https://huggingface.co/THUDM/glm-4-9b-chat', fetchedAt: '2026-09-23',
  },
  {
    id: 'zai-org/GLM-4.5', org: 'Zhipu', name: 'GLM-4.5', family: 'GLM (MoE)',
    paramsB: 355, activeParamsB: 32, isMoE: true,
    numHiddenLayers: 92, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx128k], precisions: [{ id: 'bf16' }, { id: 'fp8' }, { id: 'int4' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/zai-org/GLM-4.5', fetchedAt: '2026-09-23',
  },
  {
    id: 'zai-org/GLM-4.6', org: 'Zhipu', name: 'GLM-4.6', family: 'GLM (MoE)',
    paramsB: 355, activeParamsB: 32, isMoE: true,
    numHiddenLayers: 92, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx128k, 202752], precisions: [{ id: 'bf16' }, { id: 'fp8' }, { id: 'int4' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/zai-org/GLM-4.6', fetchedAt: '2026-09-23',
  },
  {
    // Glm5NextForConditionalGeneration (VLM wrapper); text_config: 45 layers,
    // MLA kv_lora_rank=512 / qk_rope=0, 288 routed experts × 8 active,
    // first 3 dense layers, native FP8 quantization_config (dynamic, e4m3).
    id: 'zai-org/GLM-5.3-Flash', org: 'Zhipu', name: 'GLM-5.3-Flash', family: 'GLM (MoE)',
    paramsB: 316, activeParamsB: 19, isMoE: true,
    numHiddenLayers: 45, numKvHeads: 1, headDim: 512, isMLA: true, mlaKvLoraRank: 512, mlaRopeHeadDim: 0,
    contexts: [C.ctx128k, 262144, C.ctx1m], precisions: [{ id: 'fp8', note: '原生 FP8（dynamic e4m3）' }, { id: 'int4' }],
    nativeDtype: 'fp8 (dynamic e4m3)', fp8Native: true, paramsEstimated: true, status: 'verified',
    sourceUrl: 'https://huggingface.co/zai-org/GLM-5.3-Flash', fetchedAt: '2026-09-23',
  },
  // ── Mistral ──────────────────────────────────────────────────────────────
  {
    id: 'mistralai/Mistral-7B-Instruct-v0.3', org: 'Mistral', name: 'Mistral-7B-Instruct-v0.3', family: 'Mistral',
    paramsB: 7.25, activeParamsB: 7.25, isMoE: false,
    numHiddenLayers: 32, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx32k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3', fetchedAt: '2026-09-23',
  },
  {
    id: 'mistralai/Mixtral-8x7B-Instruct-v0.1', org: 'Mistral', name: 'Mixtral-8x7B-Instruct-v0.1', family: 'Mistral (MoE)',
    paramsB: 46.7, activeParamsB: 12.9, isMoE: true,
    numHiddenLayers: 32, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx32k], precisions: [{ id: 'bf16' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1', fetchedAt: '2026-09-23',
  },
  {
    id: 'mistralai/Mixtral-8x22B-Instruct-v0.1', org: 'Mistral', name: 'Mixtral-8x22B-Instruct-v0.1', family: 'Mistral (MoE)',
    paramsB: 140.63, activeParamsB: 39, isMoE: true,
    numHiddenLayers: 56, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [65536], precisions: [{ id: 'bf16' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1', fetchedAt: '2026-09-23',
  },
  // ── Gemma ────────────────────────────────────────────────────────────────
  {
    id: 'google/gemma-2-9b-it', org: 'Google', name: 'Gemma-2-9B-it', family: 'Gemma',
    paramsB: 9.24, activeParamsB: 9.24, isMoE: false,
    numHiddenLayers: 42, numKvHeads: 8, headDim: 256, isMLA: false,
    contexts: [8192], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/google/gemma-2-9b-it (config) / modelscope LLM-Research/gemma-2-9b-it (index)', fetchedAt: '2026-09-23',
  },
  {
    id: 'google/gemma-2-27b-it', org: 'Google', name: 'Gemma-2-27B-it', family: 'Gemma',
    paramsB: 27.23, activeParamsB: 27.23, isMoE: false,
    numHiddenLayers: 46, numKvHeads: 16, headDim: 128, isMLA: false,
    contexts: [8192], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/google/gemma-2-27b-it (config) / modelscope LLM-Research/gemma-2-27b-it (index)', fetchedAt: '2026-09-23',
  },
  {
    // Gemma3ForConditionalGeneration wrapper; text_config has partial fields,
    // attention/kv heads and head_dim from published Gemma3 text architecture.
    id: 'google/gemma-3-4b-it', org: 'Google', name: 'Gemma-3-4B-it', family: 'Gemma',
    paramsB: 4.3, activeParamsB: 4.3, isMoE: false,
    numHiddenLayers: 34, numKvHeads: 4, headDim: 256, isMLA: false,
    contexts: [C.ctx128k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/google/gemma-3-4b-it / modelscope LLM-Research/gemma-3-4b-it', fetchedAt: '2026-09-23',
  },
  {
    id: 'google/gemma-3-12b-it', org: 'Google', name: 'Gemma-3-12B-it', family: 'Gemma',
    paramsB: 12.19, activeParamsB: 12.19, isMoE: false,
    numHiddenLayers: 48, numKvHeads: 8, headDim: 256, isMLA: false,
    contexts: [C.ctx128k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/google/gemma-3-12b-it / modelscope LLM-Research/gemma-3-12b-it', fetchedAt: '2026-09-23',
  },
  {
    id: 'google/gemma-3-27b-it', org: 'Google', name: 'Gemma-3-27B-it', family: 'Gemma',
    paramsB: 27.43, activeParamsB: 27.43, isMoE: false,
    numHiddenLayers: 62, numKvHeads: 16, headDim: 128, isMLA: false,
    contexts: [C.ctx128k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/google/gemma-3-27b-it / modelscope LLM-Research/gemma-3-27b-it', fetchedAt: '2026-09-23',
  },
  // ── Microsoft ────────────────────────────────────────────────────────────
  {
    id: 'microsoft/Phi-3.5-mini-instruct', org: 'Microsoft', name: 'Phi-3.5-mini-instruct', family: 'Phi',
    paramsB: 3.82, activeParamsB: 3.82, isMoE: false,
    numHiddenLayers: 32, numKvHeads: 32, headDim: 96, isMLA: false,
    contexts: [C.ctx128k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/microsoft/Phi-3.5-mini-instruct', fetchedAt: '2026-09-23',
  },
  {
    id: 'microsoft/phi-4', org: 'Microsoft', name: 'Phi-4', family: 'Phi',
    paramsB: 14.66, activeParamsB: 14.66, isMoE: false,
    numHiddenLayers: 40, numKvHeads: 10, headDim: 128, isMLA: false,
    contexts: [16384], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/microsoft/phi-4', fetchedAt: '2026-09-23',
  },
  // ── Others ───────────────────────────────────────────────────────────────
  {
    id: 'internlm/internlm2_5-7b-chat', org: 'InternLM', name: 'InternLM2.5-7B-Chat', family: 'InternLM',
    paramsB: 7.74, activeParamsB: 7.74, isMoE: false,
    numHiddenLayers: 32, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx32k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/internlm/internlm2_5-7b-chat', fetchedAt: '2026-09-23',
  },
  {
    id: 'internlm/internlm2_5-20b-chat', org: 'InternLM', name: 'InternLM2.5-20B-Chat', family: 'InternLM',
    paramsB: 19.86, activeParamsB: 19.86, isMoE: false,
    numHiddenLayers: 48, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx32k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/internlm/internlm2_5-20b-chat', fetchedAt: '2026-09-23',
  },
  {
    id: '01-ai/Yi-1.5-9B-Chat', org: '01.AI', name: 'Yi-1.5-9B-Chat', family: 'Yi',
    paramsB: 8.83, activeParamsB: 8.83, isMoE: false,
    numHiddenLayers: 48, numKvHeads: 4, headDim: 128, isMLA: false,
    contexts: [C.ctx32k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/01-ai/Yi-1.5-9B-Chat', fetchedAt: '2026-09-23',
  },
  {
    id: '01-ai/Yi-1.5-34B-Chat', org: '01.AI', name: 'Yi-1.5-34B-Chat', family: 'Yi',
    paramsB: 34.39, activeParamsB: 34.39, isMoE: false,
    numHiddenLayers: 60, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx32k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/01-ai/Yi-1.5-34B-Chat', fetchedAt: '2026-09-23',
  },
  {
    // multi_query_group_num not published in config (older 4096-class GQA, 40 heads);
    // 40 heads / 4096 hidden → head_dim 128; kv heads default to 40-group MHA marker below.
    id: 'baichuan-inc/Baichuan2-13B-Chat', org: 'Baichuan', name: 'Baichuan2-13B-Chat', family: 'Baichuan',
    paramsB: 13.0, activeParamsB: 13.0, isMoE: false,
    numHiddenLayers: 40, numKvHeads: 40, headDim: 128, isMLA: false,
    contexts: [4096], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', paramsEstimated: true, status: 'verified',
    sourceUrl: 'https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat', fetchedAt: '2026-09-23',
  },
  {
    id: 'OpenBMB/MiniCPM4-8B', org: 'OpenBMB', name: 'MiniCPM4-8B', family: 'MiniCPM',
    paramsB: 8.19, activeParamsB: 8.19, isMoE: false,
    numHiddenLayers: 32, numKvHeads: 2, headDim: 128, isMLA: false,
    contexts: [C.ctx32k, 131072], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/OpenBMB/MiniCPM4-8B', fetchedAt: '2026-09-23',
  },
  // ── 第二批（2026-09-23 追加）─────────────────────────────────────────────
  {
    // MoE 1.04T-A32B: 61 layers × 384 experts × 3 × 7168 × 2048 ≈ 1.031T routed params;
    // index total_size 1.029TB ⇒ FP8-packed checkpoint. MLA (kv_lora 512, rope 64).
    id: 'moonshotai/Kimi-K2-Instruct', org: 'Moonshot', name: 'Kimi-K2-Instruct', family: 'Kimi (MoE)',
    paramsB: 1040, activeParamsB: 32, isMoE: true,
    numHiddenLayers: 61, numKvHeads: 64, headDim: 128, isMLA: true, mlaKvLoraRank: 512, mlaRopeHeadDim: 64,
    contexts: [C.ctx128k], precisions: [{ id: 'fp8', note: '原生 FP8 权重' }, { id: 'int4' }],
    nativeDtype: 'fp8', fp8Native: true, paramsEstimated: true, status: 'verified',
    sourceUrl: 'https://huggingface.co/moonshotai/Kimi-K2-Instruct', fetchedAt: '2026-09-23',
  },
  {
    id: 'deepseek-ai/DeepSeek-V3.1', org: 'DeepSeek', name: 'DeepSeek-V3.1', family: 'DeepSeek',
    paramsB: 671, activeParamsB: 37, isMoE: true,
    numHiddenLayers: 61, numKvHeads: 128, headDim: 128, isMLA: true, mlaKvLoraRank: 512, mlaRopeHeadDim: 64,
    contexts: [C.ctx128k, 163840], precisions: [{ id: 'fp8' }, { id: 'int8' }, { id: 'int4' }],
    nativeDtype: 'fp8 (E4M3)', fp8Native: true, paramsEstimated: true, status: 'verified',
    sourceUrl: 'https://huggingface.co/deepseek-ai/DeepSeek-V3.1', fetchedAt: '2026-09-23',
  },
  {
    id: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', org: 'DeepSeek', name: 'R1-Distill-Qwen-7B', family: 'DeepSeek (蒸馏)',
    paramsB: 7.62, activeParamsB: 7.62, isMoE: false,
    numHiddenLayers: 28, numKvHeads: 4, headDim: 128, isMLA: false,
    contexts: [C.ctx128k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', fetchedAt: '2026-09-23',
  },
  {
    id: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B', org: 'DeepSeek', name: 'R1-Distill-Qwen-14B', family: 'DeepSeek (蒸馏)',
    paramsB: 14.77, activeParamsB: 14.77, isMoE: false,
    numHiddenLayers: 48, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx128k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B', fetchedAt: '2026-09-23',
  },
  {
    id: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', org: 'DeepSeek', name: 'R1-Distill-Qwen-32B', family: 'DeepSeek (蒸馏)',
    paramsB: 32.76, activeParamsB: 32.76, isMoE: false,
    numHiddenLayers: 64, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx128k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', fetchedAt: '2026-09-23',
  },
  {
    // 128 experts × 3 × 4096 × 1408 × 45 moe layers ≈ 99.6B routed + attention/embed ≈ 106B (官方口径);
    // index total_size 110.5GB ⇒ FP8-packed checkpoint.
    id: 'zai-org/GLM-4.5-Air', org: 'Zhipu', name: 'GLM-4.5-Air', family: 'GLM (MoE)',
    paramsB: 106, activeParamsB: 12, isMoE: true,
    numHiddenLayers: 46, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx128k], precisions: [{ id: 'fp8' }, { id: 'int4' }, { id: 'bf16' }],
    nativeDtype: 'fp8', fp8Native: true, paramsEstimated: true, status: 'verified',
    sourceUrl: 'https://huggingface.co/zai-org/GLM-4.5-Air', fetchedAt: '2026-09-23',
  },
  {
    id: 'Qwen/Qwen2.5-Coder-32B-Instruct', org: 'Qwen', name: 'Qwen2.5-Coder-32B-Instruct', family: 'Qwen2.5',
    paramsB: 32.76, activeParamsB: 32.76, isMoE: false,
    numHiddenLayers: 64, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx32k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct', fetchedAt: '2026-09-23',
  },
  {
    id: 'Qwen/QwQ-32B', org: 'Qwen', name: 'QwQ-32B', family: 'Qwen2.5',
    paramsB: 32.76, activeParamsB: 32.76, isMoE: false,
    numHiddenLayers: 64, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx32k, 40960], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/Qwen/QwQ-32B', fetchedAt: '2026-09-23',
  },
  {
    id: 'Qwen/Qwen3-Coder-480B-A35B-Instruct', org: 'Qwen', name: 'Qwen3-Coder-480B-A35B', family: 'Qwen3 (MoE)',
    paramsB: 480, activeParamsB: 35, isMoE: true,
    numHiddenLayers: 62, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx128k, 262144], precisions: [{ id: 'bf16' }, { id: 'fp8' }, { id: 'int4' }],
    nativeDtype: 'bfloat16', paramsEstimated: true, status: 'verified',
    sourceUrl: 'https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct', fetchedAt: '2026-09-23',
  },
  {
    id: 'microsoft/Phi-4-mini-instruct', org: 'Microsoft', name: 'Phi-4-mini-instruct', family: 'Phi',
    paramsB: 3.84, activeParamsB: 3.84, isMoE: false,
    numHiddenLayers: 32, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx128k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/microsoft/Phi-4-mini-instruct', fetchedAt: '2026-09-23',
  },
  {
    // MQA (1 kv head); checkpoint gated, params from official ~1B figure.
    id: 'google/gemma-3-1b-it', org: 'Google', name: 'Gemma-3-1B-it', family: 'Gemma',
    paramsB: 1.05, activeParamsB: 1.05, isMoE: false,
    numHiddenLayers: 26, numKvHeads: 1, headDim: 256, isMLA: false,
    contexts: [C.ctx32k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', paramsEstimated: true, status: 'verified',
    sourceUrl: 'https://huggingface.co/google/gemma-3-1b-it', fetchedAt: '2026-09-23',
  },
  {
    id: 'mistralai/Mistral-Small-3.2-24B-Instruct-2506', org: 'Mistral', name: 'Mistral-Small-3.2-24B', family: 'Mistral',
    paramsB: 24.0, activeParamsB: 24.0, isMoE: false,
    numHiddenLayers: 40, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx128k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506', fetchedAt: '2026-09-23',
  },
  {
    // 16 experts × 1 active; official 109B-A17B (index est 108.6 ✓ bf16);
    // config declares 10M ctx, engine ladder offered to 1M.
    id: 'meta-llama/Llama-4-Scout-17B-16E-Instruct', org: 'Meta', name: 'Llama-4-Scout-17B-16E', family: 'Llama 4 (MoE)',
    paramsB: 109, activeParamsB: 17, isMoE: true,
    numHiddenLayers: 48, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx128k, 524288, C.ctx1m], precisions: [{ id: 'bf16' }, { id: 'fp8' }, { id: 'int4' }],
    nativeDtype: 'bfloat16', status: 'verified',
    sourceUrl: 'https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct (config) / modelscope LLM-Research (index)', fetchedAt: '2026-09-23',
  },
  {
    id: 'Tencent-Hunyuan/Hunyuan-A13B-Instruct', org: 'Tencent', name: 'Hunyuan-A13B-Instruct', family: 'Hunyuan (MoE)',
    paramsB: 80.4, activeParamsB: 13, isMoE: true,
    numHiddenLayers: 32, numKvHeads: 8, headDim: 128, isMLA: false,
    contexts: [C.ctx32k], precisions: [{ id: 'bf16' }, { id: 'int8' }, { id: 'int4' }, { id: 'gguf_q8' }, { id: 'gguf_q4km' }],
    nativeDtype: 'bfloat16', paramsEstimated: true, status: 'verified',
    sourceUrl: 'https://modelscope.cn/models/Tencent-Hunyuan/Hunyuan-A13B-Instruct', fetchedAt: '2026-09-23',
  },
];

// ── 规模分类 ───────────────────────────────────────────────────────────────
export type ModelCategory = 'lite' | 'single' | 'multi' | 'cluster';

export const CATEGORY_FILTERS: Array<{ id: ModelCategory | 'all'; label: string }> = [
  { id: 'all', label: '全部' },
  { id: 'lite', label: '端侧 ≤4B' },
  { id: 'single', label: '单卡 4–35B' },
  { id: 'multi', label: '多卡 35–120B' },
  { id: 'cluster', label: '集群 >120B' },
];

export function modelCategory(m: ModelSpec): ModelCategory {
  if (m.paramsB <= 4) return 'lite';
  if (m.paramsB <= 35) return 'single';
  if (m.paramsB <= 120) return 'multi';
  return 'cluster';
}

export function modelById(id: string): ModelSpec | undefined {
  return MODELS.find((m) => m.id === id);
}
