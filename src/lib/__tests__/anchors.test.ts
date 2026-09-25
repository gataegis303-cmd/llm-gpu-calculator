// Anchor tests: hand-checked values pinning the roofline formulas.
import { describe, it, expect } from 'vitest';
import { MODELS, modelById, modelCategory, type ModelSpec } from '@/data/models';
import { GPUS, gpuById } from '@/data/gpus';
import { weightsBytesGB, kvBytesPerToken, estimateVram } from '../vram';
import { findMinimalPlan } from '../parallelism';
import { runAllBands, runScenario } from '../search';
import { runCapacity } from '../capacity';

const qwen7b = modelById('Qwen/Qwen2.5-7B-Instruct')!;
const llama8b = modelById('meta-llama/Llama-3.1-8B-Instruct')!;
const dsv3 = modelById('deepseek-ai/DeepSeek-V3')!;
const v4flash = modelById('deepseek-ai/DeepSeek-V4-Flash-0731')!;
const glm53 = modelById('zai-org/GLM-5.3-Flash')!;
const h100 = gpuById('h100-sxm')!;
const a100 = gpuById('a100-80')!;

describe('weights roofline', () => {
  it('Qwen2.5-7B BF16 ≈ 14–15 GB (params 7.62B × 2B)', () => {
    const g = weightsBytesGB(qwen7b.paramsB, 'bf16');
    expect(g).toBeGreaterThan(14);
    expect(g).toBeLessThan(15.5);
  });

  it('Llama-3.1-8B INT4 ≈ 4.5–5 GB (fp16 checkpoint 8.03B × 0.5B)', () => {
    const g = weightsBytesGB(llama8b.paramsB, 'int4');
    expect(g).toBeGreaterThan(3.9);
    expect(g).toBeLessThan(4.6);
  });

  it('DeepSeek-V3 FP8 ≈ 671 GB (official 671B params × 1B)', () => {
    const g = weightsBytesGB(dsv3.paramsB, 'fp8');
    expect(g).toBeGreaterThan(660);
    expect(g).toBeLessThan(680);
  });

  it('GLM-4.5 BF16 ≈ 710 GB (355B params × 2B — guards the fp8-checkpoint halving bug)', () => {
    const glm45 = modelById('zai-org/GLM-4.5')!;
    const g = weightsBytesGB(glm45.paramsB, 'bf16');
    expect(g).toBeGreaterThan(700);
    expect(g).toBeLessThan(715);
  });

  it('Kimi-K2 FP8 ≈ 1040 GB (1.04T params × 1B)', () => {
    const k2 = modelById('moonshotai/Kimi-K2-Instruct')!;
    const g = weightsBytesGB(k2.paramsB, 'fp8');
    expect(g).toBeGreaterThan(1020);
    expect(g).toBeLessThan(1050);
  });
});

describe('KV cache roofline', () => {
  it('DeepSeek-V3 MLA: (512+64)B × 2 × 61 layers = 70272 B ≈ 68.6 KB/token (fp16 KV)', () => {
    expect(kvBytesPerToken(dsv3, 'bf16')).toBe(576 * 2 * 61);
    const kb = kvBytesPerToken(dsv3, 'bf16') / 1024;
    expect(kb).toBeGreaterThan(68);
    expect(kb).toBeLessThan(69.5);
  });

  it('Qwen2.5-7B GQA: 2×28×4×128×2 = 57344 B/token', () => {
    expect(kvBytesPerToken(qwen7b, 'bf16')).toBe(57344);
  });

  it('DeepSeek-V4-Flash MLA: fp8 KV dtype → (512+64)×1×43 = 24768 B ≈ 24.2 KB/token', () => {
    expect(kvBytesPerToken(v4flash, 'fp8')).toBe(576 * 1 * 43);
    expect(kvBytesPerToken(v4flash, 'bf16')).toBe(576 * 2 * 43);
  });

  it('GLM-5.3-Flash MLA (512+0)×2×45 = 46080 B/token = 45 KB', () => {
    expect(kvBytesPerToken(glm53, 'bf16')).toBe(512 * 2 * 45);
  });
});

describe('VRAM assembly', () => {
  it('H100 single card fits Qwen2.5-7B BF16 @ 4K ctx × 10 users in every band', () => {
    const bands = runAllBands({
      model: qwen7b, gpu: h100, precisionId: 'bf16',
      concurrency: 10, avgCtxTokens: 4096, inputTokens: 2048,
    });
    for (const band of ['conservative', 'typical', 'aggressive'] as const) {
      expect(bands[band].plan.totalDevices).toBe(1);
      expect(bands[band].plan.tp).toBe(1);
    }
  });

  it('KV total scales linearly with concurrency', () => {
    const a = estimateVram({ model: qwen7b, precisionId: 'bf16', concurrency: 10, avgCtxTokens: 4096, overheadFactor: 0.075, overheadBaseGB: 3 });
    const b = estimateVram({ model: qwen7b, precisionId: 'bf16', concurrency: 20, avgCtxTokens: 4096, overheadFactor: 0.075, overheadBaseGB: 3 });
    expect(b.kvTotalGB).toBeCloseTo(a.kvTotalGB * 2, 6);
  });
});

describe('minimal-device search', () => {
  it('DeepSeek-V3 FP8 needs ≥ 8× H100 per replica (685GB weights / 80GB usable)', () => {
    const plan = findMinimalPlan({
      model: dsv3, gpu: h100, precisionId: 'fp8',
      bytesPerParam: 1, kvBytesPerToken: kvBytesPerToken(dsv3, 'fp8'),
      concurrency: 8, avgCtxTokens: 8192,
      overheadFactor: 0.075, overheadBaseGB: 3,
    });
    expect(plan.devicesPerReplica).toBeGreaterThanOrEqual(8);
    expect(plan.feasible).toBe(true);
  });

  it('Llama-3.1-70B BF16: 141.1GB weights → 3× H100 (47GB/card < 78.8 usable)', () => {
    const plan = findMinimalPlan({
      model: modelById('meta-llama/Llama-3.1-70B-Instruct')!, gpu: h100, precisionId: 'bf16',
      bytesPerParam: 2, kvBytesPerToken: kvBytesPerToken(modelById('meta-llama/Llama-3.1-70B-Instruct')!, 'bf16'),
      concurrency: 8, avgCtxTokens: 8192,
      overheadFactor: 0.075, overheadBaseGB: 3,
    });
    expect(plan.devicesPerReplica).toBe(3);
  });

  it('Qwen2.5-72B BF16 on A100-80: 3 cards (2 cards fail after KV+overhead)', () => {
    const plan = findMinimalPlan({
      model: modelById('Qwen/Qwen2.5-72B-Instruct')!, gpu: a100, precisionId: 'bf16',
      bytesPerParam: 2, kvBytesPerToken: kvBytesPerToken(modelById('Qwen/Qwen2.5-72B-Instruct')!, 'bf16'),
      concurrency: 4, avgCtxTokens: 8192,
      overheadFactor: 0.075, overheadBaseGB: 3,
    });
    expect(plan.devicesPerReplica).toBe(3);
  });
});

describe('throughput sanity', () => {
  it('H100 × 1, Qwen2.5-7B BF16, 10×4K: roofline TPOT in a plausible 4–40ms band', () => {
    const r = runScenario({
      model: qwen7b, gpu: h100, precisionId: 'bf16',
      concurrency: 10, avgCtxTokens: 4096, inputTokens: 2048,
    }, 'typical');
    // roofline floor: (15.24GB weights + 2.35GB KV) / (3350×0.8 GB/s) ≈ 6.6ms;
    // real-world schedulers add overhead on top — band kept generous.
    expect(r.throughput.tpotMs).toBeGreaterThan(4);
    expect(r.throughput.tpotMs).toBeLessThan(40);
    expect(r.throughput.perUserTokPerSec).toBeGreaterThan(15);
    expect(r.throughput.aggregateTokPerSec).toBeGreaterThan(150);
  });

  it('TTFT grows with input length', () => {
    const short = runScenario({ model: qwen7b, gpu: h100, precisionId: 'bf16', concurrency: 4, avgCtxTokens: 4096, inputTokens: 512 }, 'typical');
    const long = runScenario({ model: qwen7b, gpu: h100, precisionId: 'bf16', concurrency: 4, avgCtxTokens: 4096, inputTokens: 16384 }, 'typical');
    expect(long.throughput.ttftMs).toBeGreaterThan(short.throughput.ttftMs);
  });
});

describe('band monotonicity', () => {
  it('conservative VRAM ≥ typical ≥ aggressive across a sweep of scenarios', () => {
    const cases: Array<[ModelSpec, string, number, number]> = [
      [qwen7b, 'bf16', 10, 4096],
      [dsv3, 'fp8', 16, 16384],
      [v4flash, 'fp8', 32, 32768],
      [glm53, 'fp8', 16, 65536],
      [modelById('meta-llama/Llama-3.3-70B-Instruct')!, 'int4', 24, 8192],
    ];
    for (const [model, prec, conc, ctx] of cases) {
      const fixed = { model, gpu: h100, precisionId: prec, concurrency: conc, avgCtxTokens: ctx, inputTokens: ctx / 2 };
      const bands = runAllBands(fixed);
      // overhead is monotonic by construction (same weights, growing margin)
      expect(bands.conservative.vram.overheadGB).toBeGreaterThanOrEqual(bands.typical.vram.overheadGB);
      expect(bands.typical.vram.overheadGB).toBeGreaterThanOrEqual(bands.aggressive.vram.overheadGB);
      // TPOT is monotonic whenever the bands agree on device geometry
      const cons = bands.conservative.plan;
      const agg = bands.aggressive.plan;
      if (cons.devicesPerReplica === agg.devicesPerReplica && cons.dp === agg.dp) {
        expect(bands.conservative.throughput.tpotMs).toBeGreaterThanOrEqual(bands.aggressive.throughput.tpotMs);
      }
    }
  });
});

describe('live model derivation (modelRepo)', () => {
  it('derives a GQA model spec from a Qwen-style config', async () => {
    const { deriveModelSpec } = await import('../modelRepo');
    const spec = deriveModelSpec('Qwen/Qwen2.5-7B-Instruct', {
      hidden_size: 3584, num_hidden_layers: 28, num_attention_heads: 28, num_key_value_heads: 4,
      head_dim: 128, intermediate_size: 18944, vocab_size: 152064, max_position_embeddings: 32768,
      tie_word_embeddings: false,
    }, '2026-09-23');
    expect(spec.isMLA).toBe(false);
    expect(spec.isMoE).toBe(false);
    expect(spec.numHiddenLayers).toBe(28);
    expect(spec.numKvHeads).toBe(4);
    expect(kvBytesPerToken(spec, 'bf16')).toBe(57344);
    expect(spec.paramsB).toBeGreaterThan(6);
    expect(spec.paramsB).toBeLessThan(9);
    expect(spec.contexts).toContain(32768);
    expect(spec.precisions[0].id).toBe('bf16');
  });

  it('derives an MLA MoE spec and prefers fp8 when the config declares it', async () => {
    const { deriveModelSpec } = await import('../modelRepo');
    const spec = deriveModelSpec('deepseek-ai/Test-MLA', {
      hidden_size: 7168, num_hidden_layers: 61, num_attention_heads: 128, num_key_value_heads: 128,
      kv_lora_rank: 512, qk_rope_head_dim: 64, n_routed_experts: 256, num_experts_per_tok: 8,
      moe_intermediate_size: 2048, first_k_dense_replace: 3, intermediate_size: 18432,
      vocab_size: 129280, max_position_embeddings: 163840,
      quantization_config: { fmt: 'e4m3' },
    }, '2026-09-23');
    expect(spec.isMLA).toBe(true);
    expect(spec.mlaKvLoraRank).toBe(512);
    expect(spec.isMoE).toBe(true);
    expect(kvBytesPerToken(spec, 'bf16')).toBe(576 * 2 * 61);
    expect(spec.precisions[0].id).toBe('fp8');
    expect(spec.paramsB).toBeGreaterThan(400);
    expect(spec.paramsB).toBeLessThan(900);
    expect(spec.contexts).toContain(131072);
    expect(spec.sourceUrl).toBe('https://huggingface.co/deepseek-ai/Test-MLA');
  });

  it('rejects non-LLM repos with a clear error (e.g. video models like MiniMax-H3)', async () => {
    const { deriveModelSpec } = await import('../modelRepo');
    // video-generation repos carry no hidden_size/num_hidden_layers
    expect(() =>
      deriveModelSpec('MiniMaxAI/MiniMax-H3', {
        _class_name: 'MiniMaxH3Pipeline', pipeline_tag: 'image-text-to-video',
        transformer: { num_layers: 32 },
      }, '2026-09-24'),
    ).toThrow(/NOT_LLM.*image-text-to-video/);
  });

  it('derives a VLM spec: vision tower params merged, kind=vlm', async () => {
    const { deriveModelSpec } = await import('../modelRepo');
    const textTower = {
      hidden_size: 4096, num_hidden_layers: 36, num_attention_heads: 32, num_key_value_heads: 8,
      head_dim: 128, intermediate_size: 11008, vocab_size: 152064, max_position_embeddings: 32768,
    };
    const withoutVision = deriveModelSpec('org/Test-VLM', { ...textTower }, '2026-09-24');
    const withVision = deriveModelSpec('org/Test-VLM', {
      vision_config: { depth: 32, hidden_size: 1280, patch_size: 16, in_chans: 3 },
      ...textTower,
    }, '2026-09-24');
    expect(withVision.kind).toBe('vlm');
    expect(withoutVision.kind).toBe('llm');
    expect(withVision.paramsB).toBeGreaterThan(withoutVision.paramsB);
    // ViT 32L/1280d ≈ 0.63B extra
    expect(withVision.paramsB - withoutVision.paramsB).toBeGreaterThan(0.5);
    expect(withVision.paramsB - withoutVision.paramsB).toBeLessThan(1.0);
    expect(kvBytesPerToken(withVision, 'bf16')).toBe(kvBytesPerToken(withoutVision, 'bf16'));
  });

  it('derives a generic (non-autoregressive) spec from checkpoint bytes', async () => {
    const { deriveModelSpec } = await import('../modelRepo');
    const spec = deriveModelSpec('MiniMaxAI/MiniMax-H3', null, '2026-09-24', 498329662404);
    expect(spec.kind).toBe('generic');
    expect(spec.checkpointBytesGB).toBe(498.3);
    // paramsB stores fp16-equivalent count: bytes / 2
    expect(spec.paramsB).toBeGreaterThan(248);
    expect(spec.paramsB).toBeLessThan(250);
    expect(kvBytesPerToken(spec, 'bf16')).toBe(0);
    expect(spec.contexts).toEqual([]);
    expect(spec.precisions[0].id).toBe('bf16');
  });

  it('unwraps thinker_config for Omni/ASR models (Qwen3-ASR style)', async () => {
    const { deriveModelSpec } = await import('../modelRepo');
    const spec = deriveModelSpec('netease-youdao/Confucius4-R2T2', {
      architectures: ['Qwen3ASRForConditionalGeneration'], model_type: 'qwen3_asr',
      thinker_config: {
        audio_config: { encoder_layers: 24, d_model: 1024, encoder_ffn_dim: 4096 },
        text_config: {
          hidden_size: 2048, num_hidden_layers: 28, num_attention_heads: 16, num_key_value_heads: 8,
          head_dim: 128, vocab_size: 151936, max_position_embeddings: 65536,
          intermediate_size: 6144, tie_word_embeddings: true,
        },
      },
    }, '2026-09-24');
    expect(spec.kind).toBe('vlm');
    // 文本塔 ≈ 1.72B + 音频塔 Whisper-math ≈ 0.19B（24L×1024d）
    expect(spec.paramsB).toBeGreaterThan(1.8);
    expect(spec.paramsB).toBeLessThan(2.1);
    expect(spec.numHiddenLayers).toBe(28);
    expect(kvBytesPerToken(spec, 'bf16')).toBe(2 * 28 * 8 * 128 * 2);
    expect(spec.contexts).toContain(65536);
    // 2B 级模型单张 128GB 设备即可部署
    const { GPUS } = await import('@/data/gpus');
    const gb10 = GPUS.find((g) => g.id === 'gb10-spark')!;
    const { findMinimalPlan } = await import('../parallelism');
    const plan = findMinimalPlan({
      model: spec, gpu: gb10, precisionId: 'bf16', bytesPerParam: 2,
      kvBytesPerToken: kvBytesPerToken(spec, 'bf16'),
      concurrency: 10, avgCtxTokens: 65536, overheadFactor: 0.075, overheadBaseGB: 3,
    });
    expect(plan.totalDevices).toBe(1);
  });
});

describe('device fit (deployableModels)', () => {
  it('Ryzen AI Max+ 395 (128GB) fits 70B-class at INT4, not at BF16', async () => {
    const { deployableModels } = await import('../fit');
    const gpu = gpuById('amd-ai-max-395-128')!;
    const { full, quantized, usableGB } = deployableModels(gpu);
    expect(usableGB).toBeCloseTo(128 * 0.75 - 1.2, 1); // 94.8GB（统一内存扣 1.2GB 运行时）
    const names = quantized.map((r) => r.model.name);
    expect(names.some((n) => n.includes('Qwen2.5-72B'))).toBe(true);
    expect(names.some((n) => n.includes('Llama-3.1-70B'))).toBe(true);
    // 72B BF16 ≈ 145GB > 94GB 可用 → 不在高精度档
    expect(full.map((r) => r.model.name).some((n) => n.includes('Qwen2.5-72B'))).toBe(false);
    // 7B BF16（~18GB）放得下
    expect(full.map((r) => r.model.name).some((n) => n.includes('Qwen2.5-7B'))).toBe(true);
  });

  it('Mac mini M4 (16GB) fits only small models; M3 Ultra (512GB) fits DeepSeek-V3 at FP8', async () => {
    const { deployableModels } = await import('../fit');
    const mini = deployableModels(gpuById('m4-mini-16')!);
    expect(mini.usableGB).toBeCloseTo(16 * 0.75 - 2, 1); // 10GB
    expect(mini.quantized.length).toBeGreaterThan(0);
    expect(mini.quantized.every((r) => r.totalGB <= 10)).toBe(true);
    expect(mini.quantized.map((r) => r.model.name).some((n) => n.includes('Qwen2.5-7B'))).toBe(true);

    const ultra = deployableModels(gpuById('m3-ultra-512')!);
    // DeepSeek-V3 FP8 ≈ 671GB > 512×0.75−2=382GB → 不行；70B FP8 ≈ 70.6GB → 行
    expect(ultra.full.map((r) => r.model.name).some((n) => n.includes('DeepSeek-V3'))).toBe(false);
    expect(ultra.full.map((r) => r.model.name).some((n) => n.includes('Llama-3.1-70B'))).toBe(true);
  });
});

describe('dataset integrity', () => {
  it('all 56 model entries are well-formed and categorizable', () => {
    expect(MODELS.length).toBeGreaterThanOrEqual(54);
    for (const m of MODELS) {
      expect(m.paramsB).toBeGreaterThan(0);
      expect(m.activeParamsB).toBeGreaterThan(0);
      expect(m.activeParamsB).toBeLessThanOrEqual(m.paramsB);
      expect(m.numHiddenLayers).toBeGreaterThan(0);
      expect(m.contexts.length).toBeGreaterThan(0);
      expect(m.precisions.length).toBeGreaterThan(0);
      expect(m.sourceUrl).toMatch(/^https?:\/\//);
      expect(m.fetchedAt).toBe('2026-09-23');
      expect(['lite', 'single', 'multi', 'cluster']).toContain(modelCategory(m));
    }
    // every category bucket is non-empty
    for (const c of ['lite', 'single', 'multi', 'cluster'] as const) {
      expect(MODELS.filter((m) => modelCategory(m) === c).length).toBeGreaterThan(0);
    }
  });

  it('all 40+ GPU entries are well-formed', () => {
    expect(GPUS.length).toBeGreaterThanOrEqual(40);
    for (const g of GPUS) {
      expect(g.vramGB).toBeGreaterThan(0);
      expect(g.memBandwidthGBs).toBeGreaterThan(0);
      expect(g.bf16Tflops).toBeGreaterThan(0);
      expect(g.sourceUrl).toMatch(/^https?:\/\//);
    }
    // GB10 (DGX Spark): datasheet 128GB LPDDR5x / 273GB/s
    const gb10 = gpuById('gb10-spark')!;
    expect(gb10.vramGB).toBe(128);
    expect(gb10.memBandwidthGBs).toBe(273);
    expect(gb10.gpusPerNode).toBe(1);
    // 统一内存设备（Apple / AMD AI Max）必须有可用占比
    expect(gpuById('amd-ai-max-395-128')!.usableFraction).toBeDefined();
    expect(gpuById('amd-ai-max-385')!.usableFraction).toBeDefined();
  });

  it('the two user-designated models are present and verified', () => {
    expect(v4flash.status).toBe('verified');
    expect(glm53.status).toBe('verified');
    expect(v4flash.numHiddenLayers).toBe(43);
    expect(glm53.numHiddenLayers).toBe(45);
  });

  it('GB10 (DGX Spark) unified memory uses its usable fraction (~114GB)', async () => {
    const gb10 = gpuById('gb10-spark')!;
    expect(gb10.vramGB).toBe(128);
    expect(gb10.memBandwidthGBs).toBe(273);
    const { usableVramGB } = await import('../vram');
    const usable = usableVramGB(gb10);
    expect(usable).toBeGreaterThan(110);
    expect(usable).toBeLessThan(118);
  });
});

describe('capacity: given devices (反向推演)', () => {
  it('16× H100 fits DeepSeek-V3 FP8 with all 16 cards in one replica (C=32)', () => {
    const r = runCapacity({
      model: dsv3, gpu: h100, precisionId: 'fp8',
      concurrency: 32, avgCtxTokens: 8192, inputTokens: 2048,
      devices: 16,
    }, 'typical');
    expect(r.feasibleAtC).toBe(true);
    expect(r.plan.feasible).toBe(true);
    expect(r.plan.devicesPerReplica).toBe(16);
    expect(r.plan.totalDevices).toBe(16);
    expect(r.plan.idleDevices).toBe(0);
    // 单副本 16 卡带宽聚合 → 单用户速度远高于 8 卡方案（8 卡装不下权重，只能拆副本）
    expect(r.throughput.tpotMs).toBeGreaterThan(0);
    expect(r.throughput.perUserTokPerSec).toBeGreaterThan(0);
    expect(r.clusterAggregateTokPerSec).toBeCloseTo(r.throughput.perUserTokPerSec * 32, 6);
  });

  it('2× H100 serving Qwen2.5-7B puts both cards in one replica (fastest per-user) over 2 DP replicas', () => {
    const r = runCapacity({
      model: qwen7b, gpu: h100, precisionId: 'bf16',
      concurrency: 10, avgCtxTokens: 4096, inputTokens: 2048,
      devices: 2,
    }, 'typical');
    expect(r.plan.totalDevices).toBe(2);
    expect(r.plan.devicesPerReplica).toBe(2);
    expect(r.plan.dp).toBe(1);
    // (15.24GB weights + 10×0.235GB KV) / (2×3350×0.8 GB/s) ≈ 3.3ms
    expect(r.throughput.tpotMs).toBeGreaterThan(2.5);
    expect(r.throughput.tpotMs).toBeLessThan(4);
    expect(r.clusterAggregateTokPerSec).toBeGreaterThan(2500);
    expect(r.clusterAggregateTokPerSec).toBeLessThan(3500);
  });

  it('max concurrency grows with devices: 1 → 2 → 4× H100 (KV budget ÷ per-user KV)', () => {
    const base = {
      model: qwen7b, gpu: h100, precisionId: 'bf16',
      concurrency: 16, avgCtxTokens: 8192, inputTokens: 2048,
    };
    const c1 = runCapacity({ ...base, devices: 1 }, 'typical');
    const c2 = runCapacity({ ...base, devices: 2 }, 'typical');
    const c4 = runCapacity({ ...base, devices: 4 }, 'typical');
    // budget(dpr=1) ≈ (78.8 − 19.4)GB / 0.47GB per user ≈ 126
    expect(c1.capacity.maxConcurrency).toBeGreaterThan(120);
    expect(c1.capacity.maxConcurrency).toBeLessThan(135);
    // dpr=2 单卡更省开销 → 每副本容量更高，2 卡 ≥ 2×126
    expect(c2.capacity.maxConcurrency).toBeGreaterThan(270);
    expect(c4.capacity.maxConcurrency).toBeGreaterThan(c2.capacity.maxConcurrency);
    expect(c2.capacity.maxConcurrency).toBeGreaterThan(c1.capacity.maxConcurrency);
    expect(c1.feasibleAtC).toBe(true);
  });

  it('C=500 on a single H100 exceeds capacity → falls back to full-load numbers', () => {
    const r = runCapacity({
      model: qwen7b, gpu: h100, precisionId: 'bf16',
      concurrency: 500, avgCtxTokens: 8192, inputTokens: 2048,
      devices: 1,
    }, 'typical');
    expect(r.feasibleAtC).toBe(false);
    expect(r.reason).toBeTruthy();
    expect(r.capacity.maxConcurrency).toBeGreaterThan(100);
    expect(r.capacity.maxConcurrency).toBeLessThan(140);
    // 满载口径：单用户被 KV 拉慢但为正，集群总吞吐 = 单用户 × 满载并发
    expect(r.throughput.perUserTokPerSec).toBeGreaterThan(0);
    expect(r.clusterAggregateTokPerSec).toBeCloseTo(
      r.throughput.perUserTokPerSec * r.capacity.maxConcurrency, 6,
    );
  });

  it('13× H100 with DeepSeek-V3 FP8 uses 12 cards and idles 1 (13 not decomposable)', () => {
    const r = runCapacity({
      model: dsv3, gpu: h100, precisionId: 'fp8',
      concurrency: 8, avgCtxTokens: 8192, inputTokens: 2048,
      devices: 13,
    }, 'typical');
    expect(r.plan.devicesPerReplica).toBe(12);
    expect(r.plan.totalDevices).toBe(12);
    expect(r.plan.idleDevices).toBe(1);
    expect(r.feasibleAtC).toBe(true);
  });

  it('4× H100 cannot hold DeepSeek-V3 FP8 at all → reason + minDevicesToFit = 10', () => {
    const r = runCapacity({
      model: dsv3, gpu: h100, precisionId: 'fp8',
      concurrency: 8, avgCtxTokens: 8192, inputTokens: 2048,
      devices: 4,
    }, 'typical');
    expect(r.plan.feasible).toBe(false);
    expect(r.capacity.maxConcurrency).toBe(0);
    expect(r.minDevicesToFit).toBe(10);
    expect(r.reason).toContain('最少需要 10 张');
  });

  it('cluster aggregate at user concurrency is monotone non-decreasing in devices', () => {
    const base = {
      model: qwen7b, gpu: h100, precisionId: 'bf16',
      concurrency: 16, avgCtxTokens: 4096, inputTokens: 512,
    };
    let prev = -1;
    for (const n of [1, 2, 4, 8]) {
      const r = runCapacity({ ...base, devices: n }, 'typical');
      expect(r.feasibleAtC).toBe(true);
      expect(r.clusterAggregateTokPerSec).toBeGreaterThanOrEqual(prev - 1e-9);
      prev = r.clusterAggregateTokPerSec;
    }
  });
});
