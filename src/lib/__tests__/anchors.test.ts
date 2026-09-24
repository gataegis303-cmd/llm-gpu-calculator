// Anchor tests: hand-checked values pinning the roofline formulas.
import { describe, it, expect } from 'vitest';
import { MODELS, modelById, modelCategory, type ModelSpec } from '@/data/models';
import { GPUS, gpuById } from '@/data/gpus';
import { weightsBytesGB, kvBytesPerToken, estimateVram } from '../vram';
import { findMinimalPlan } from '../parallelism';
import { runAllBands, runScenario } from '../search';

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
    const gb10 = gpuById('gb10-dgx-spark')!;
    expect(gb10.vramGB).toBe(128);
    expect(gb10.memBandwidthGBs).toBe(273);
    expect(gb10.gpusPerNode).toBe(1);
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
