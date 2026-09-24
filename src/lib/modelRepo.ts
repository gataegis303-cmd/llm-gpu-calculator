// Live model repository: search HuggingFace, fetch config.json on demand,
// derive a ModelSpec from the config. Embedded snapshot stays as offline fallback.
// All upstream calls go through the vite dev proxy (/hf-api, /hf-file, /ms-file).
//
// 支持三类模型（kind）：
//   llm      — 自回归文本模型：KV/TPOT/TTFT 全引擎可用
//   vlm      — 视觉语言模型（vision_config + 文本塔）：LLM 引擎 + 视觉塔权重并入
//   generic  — 其余（diffusers/DiT/VAE…）：按 checkpoint 实际字节数估权重显存，
//              无 KV、无 TPOT/TTFT，仅给最小持有设备数

import type { ModelSpec } from '@/data/models';

export interface OnlineModelHit {
  repoId: string;
  downloads: number;
  likes: number;
  pipelineTag?: string;
}

const derivedCache = new Map<string, ModelSpec>();

export async function searchOnlineModels(query: string, signal?: AbortSignal): Promise<OnlineModelHit[]> {
  // 不按 pipeline_tag 过滤：非 LLM（如视频/图像模型）也返回并标注类型，
  // 避免「HF 上有但搜不到」的困惑；是否可计算在选中时校验。
  const url = `/hf-api/api/models?search=${encodeURIComponent(query)}&limit=30&sort=downloads&direction=-1`;
  let lastErr: unknown;
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const res = await fetch(url, { signal });
      if (!res.ok) throw new Error(`HF search failed: ${res.status}`);
      const text = await res.text();
      const list = JSON.parse(text) as Array<{ id?: string; downloads?: number; likes?: number; pipeline_tag?: string }>;
      return (list || [])
        .filter((m) => typeof m.id === 'string')
        .map((m) => ({ repoId: m.id as string, downloads: m.downloads ?? 0, likes: m.likes ?? 0, pipelineTag: m.pipeline_tag }));
    } catch (e) {
      if (signal?.aborted) throw e;
      lastErr = e;
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error('HF search failed');
}

async function fetchJson(url: string, signal?: AbortSignal): Promise<unknown> {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return res.json();
}

/** Fetch config.json: HF raw first, ModelScope mirror as fallback. */
export async function fetchModelConfig(repoId: string, signal?: AbortSignal): Promise<Record<string, any>> {
  try {
    return (await fetchJson(`/hf-file/${repoId}/raw/main/config.json`, signal)) as Record<string, any>;
  } catch (e) {
    if (signal?.aborted) throw e;
    return (await fetchJson(`/ms-file/${repoId}/resolve/master/config.json`, signal)) as Record<string, any>;
  }
}

export interface RepoFile {
  path: string;
  sizeBytes: number;
}

/** 仓库文件树（含文件体积），用于非 transformers 仓库的权重字节求和。 */
export async function fetchRepoTree(repoId: string, signal?: AbortSignal): Promise<RepoFile[]> {
  const res = await fetch(`/hf-api/api/models/${repoId}?blobs=true&files_metadata=true`, { signal });
  if (!res.ok) throw new Error(`HF repo tree failed: ${res.status}`);
  const j = (await res.json()) as { siblings?: Array<{ rfilename?: string; size?: number; lfs?: { size?: number } }> };
  return (j.siblings || [])
    .filter((f) => typeof f.rfilename === 'string')
    .map((f) => ({ path: f.rfilename as string, sizeBytes: f.size ?? f.lfs?.size ?? 0 }));
}

export function sumCheckpointBytes(files: RepoFile[]): number {
  return files.filter((f) => f.path.endsWith('.safetensors')).reduce((s, f) => s + f.sizeBytes, 0);
}

const CTX_LADDER = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 163840, 262144, 524288, 1048576];

function visionTowerParams(v: Record<string, any> | undefined): number {
  if (!v) return 0;
  const layers: number = v.depth ?? v.num_hidden_layers ?? 0;
  const hidden: number = v.hidden_size ?? 0;
  if (!layers || !hidden) return 0;
  const perLayer = 12 * hidden * hidden; // ViT: MHSA 4d² + MLP 8d²
  const ps = v.patch_size ?? 16;
  const inCh = v.in_chans ?? v.num_channels ?? 3;
  const patchEmbed = ps * ps * inCh * hidden;
  return layers * perLayer + patchEmbed;
}

/**
 * Derive a ModelSpec from a live config.json (LLM/VLM), or from the checkpoint
 * byte total (generic). Parameters are estimated from architecture math and
 * marked paramsEstimated unless checkpoint bytes are exact.
 */
export function deriveModelSpec(
  repoId: string,
  cfg: Record<string, any> | null,
  today: string,
  checkpointBytes?: number,
): ModelSpec {
  const c: Record<string, any> = cfg?.text_config ? { ...cfg, ...cfg.text_config } : cfg ?? {};

  const hidden: number = c.hidden_size;
  const layers: number = c.num_hidden_layers ?? c.num_layers;
  const vocab: number = c.vocab_size;
  const pipelineTag: string | undefined = cfg?.pipeline_tag;

  // ── LLM / VLM：标准 transformers 文本塔 ─────────────────────────────────
  if (hidden && layers && vocab) {
    const visionParams = cfg?.vision_config ? visionTowerParams(cfg.vision_config) : 0;
    const kind: 'llm' | 'vlm' = visionParams > 0 ? 'vlm' : 'llm';

    const heads: number = c.num_attention_heads ?? 32;
    const kvHeads: number = c.num_key_value_heads ?? c.multi_query_group_num ?? (c.multi_query_attention ? 1 : heads);
    const headDim: number = c.head_dim ?? Math.round(hidden / heads);
    const maxPos: number = c.max_position_embeddings ?? c.seq_length ?? 4096;
    const tie = !!c.tie_word_embeddings;
    const inter: number = c.intermediate_size ?? 4 * hidden;

    const isMLA = c.kv_lora_rank !== undefined;
    const mlaKvLoraRank: number | undefined = c.kv_lora_rank;
    const mlaRopeHeadDim: number | undefined = c.qk_rope_head_dim;

    const numExperts: number | null = c.num_local_experts ?? c.n_routed_experts ?? c.num_experts ?? null;
    const expertsPerTok: number | null = c.num_experts_per_tok ?? c.n_experts_per_tok ?? null;
    const moeInter: number | null = c.moe_intermediate_size ?? null;
    const sharedCount: number = c.n_shared_experts ?? (c.shared_expert_intermediate_size ? 1 : 0);
    const firstDense: number = c.first_k_dense_replace ?? (numExperts ? 1 : layers);
    const isMoE = !!numExperts && !!moeInter;
    const moeLayers = isMoE ? Math.max(layers - firstDense, 0) : 0;
    const denseLayers = layers - moeLayers;

    let params = vocab * hidden * (tie ? 1 : 2);
    const attnPerLayer = isMLA
      ? hidden * (c.q_lora_rank ?? hidden) +
        (c.q_lora_rank ?? hidden) * heads * headDim +
        hidden * ((mlaKvLoraRank ?? 512) + (mlaRopeHeadDim ?? 0)) +
        (mlaKvLoraRank ?? 512) * heads * headDim +
        heads * headDim * hidden
      : 2 * hidden * hidden + 2 * hidden * kvHeads * headDim;
    params += attnPerLayer * layers;
    params += denseLayers * 3 * hidden * inter;
    if (isMoE) params += moeLayers * (numExperts + sharedCount) * 3 * hidden * (moeInter ?? inter);
    params += visionParams;
    const paramsB = Math.round((params / 1e9) * 100) / 100;

    const activeB = isMoE
      ? Math.round(((attnPerLayer * layers + denseLayers * 3 * hidden * inter + moeLayers * ((expertsPerTok ?? 1) + sharedCount) * 3 * hidden * (moeInter ?? inter)) / 1e9) * 10) / 10
      : Math.round(((params - visionParams) / 1e9) * 100) / 100;

    const quantFmt = c.quantization_config?.fmt ?? cfg?.quantization_config?.fmt;
    const fp8Native = quantFmt === 'e4m3' || quantFmt === 'fp8';
    const precisions = fp8Native
      ? [{ id: 'fp8' as const, note: '原生 FP8 权重' }, { id: 'int8' as const }, { id: 'int4' as const }]
      : [{ id: 'bf16' as const }, { id: 'int8' as const }, { id: 'int4' as const }, { id: 'gguf_q8' as const }, { id: 'gguf_q4km' as const }];

    const contexts = CTX_LADDER.filter((n) => n <= (maxPos || 4096));
    if (contexts.length === 0) contexts.push(Math.min(maxPos || 2048, 2048));

    const [org, ...rest] = repoId.split('/');
    return {
      id: repoId,
      org: org || repoId,
      name: rest.join('/') || repoId,
      family: `${org} (在线)`,
      paramsB,
      activeParamsB: activeB,
      isMoE,
      numHiddenLayers: layers,
      numKvHeads: kvHeads,
      headDim,
      isMLA,
      mlaKvLoraRank,
      mlaRopeHeadDim,
      contexts,
      precisions,
      nativeDtype: String(c.torch_dtype ?? c.dtype ?? 'bfloat16'),
      fp8Native,
      paramsEstimated: true,
      status: 'verified',
      sourceUrl: `https://huggingface.co/${repoId}`,
      fetchedAt: today,
      kind,
    };
  }

  // ── generic：无标准文本塔，但有 checkpoint 字节（diffusers/DiT/VAE…）────
  if (checkpointBytes && checkpointBytes > 0) {
    // paramsB 存「FP16 等效参数量」= checkpoint 字节 ÷ 2，使 BF16 档 = 原生体积
    const paramsB = Math.round((checkpointBytes / 2 / 1e9) * 100) / 100;
    const [org, ...rest] = repoId.split('/');
    return {
      id: repoId,
      org: org || repoId,
      name: rest.join('/') || repoId,
      family: `${org} (在线)`,
      paramsB,
      activeParamsB: paramsB,
      isMoE: false,
      numHiddenLayers: 0,
      numKvHeads: 0,
      headDim: 0,
      isMLA: false,
      contexts: [],
      precisions: [
        { id: 'bf16', note: '按 checkpoint 原生体积' },
        { id: 'fp8', note: '需推理框架支持量化' },
        { id: 'int4', note: '需推理框架支持量化' },
      ],
      nativeDtype: 'checkpoint 原生',
      paramsEstimated: false,
      status: 'verified',
      sourceUrl: `https://huggingface.co/${repoId}`,
      fetchedAt: today,
      kind: 'generic',
      checkpointBytesGB: Math.round((checkpointBytes / 1e9) * 10) / 10,
    };
  }

  throw new Error(
    `NOT_LLM:${pipelineTag ?? '未知类型'}:该仓库既无标准 transformers 文本塔（缺 hidden_size / num_hidden_layers / vocab_size）` +
      `，也未找到 .safetensors 权重文件，无法估算显存` +
      (pipelineTag ? `（类别：${pipelineTag}）` : ''),
  );
}

/**
 * Cached config → spec pipeline used by the picker.
 * LLM/VLM 走 config 推导；失败或非文本塔时回落到仓库权重字节（generic）。
 */
export async function resolveOnlineModel(repoId: string, today: string, signal?: AbortSignal): Promise<ModelSpec> {
  const hit = derivedCache.get(repoId);
  if (hit) return hit;

  const cfg = await fetchModelConfig(repoId, signal).catch(() => null);
  if (cfg) {
    try {
      const spec = deriveModelSpec(repoId, cfg, today);
      derivedCache.set(repoId, spec);
      return spec;
    } catch (e) {
      if (signal?.aborted) throw e;
      // 非文本塔（无 vocab 等）：继续尝试按权重字节求和
    }
  }

  // config.json 缺失或非 LLM/VLM 形态（多为 diffusers 仓库）：按权重文件字节求和
  const files = await fetchRepoTree(repoId, signal);
  const bytes = sumCheckpointBytes(files);
  const spec = deriveModelSpec(repoId, null, today, bytes || undefined);
  derivedCache.set(repoId, spec);
  return spec;
}
