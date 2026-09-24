import { useState } from 'react';
import ModelPicker from '@/sections/ModelPicker';
import ConfigPicker from '@/sections/ConfigPicker';
import GpuPicker from '@/sections/GpuPicker';
import ResultsPanel from '@/sections/ResultsPanel';
import type { ModelSpec } from '@/data/models';
import type { GpuSpec } from '@/data/gpus';
import { modelById } from '@/data/models';
import { gpuById } from '@/data/gpus';
import { PRESET_SCENARIOS, type PresetScenario } from '@/data/scenarios';
import { OPTS_OFF, type Optimizations } from '@/lib/optimizations';
import OptPanel from '@/sections/OptPanel';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';

export default function Home() {
  const [model, setModel] = useState<ModelSpec | null>(modelById('Qwen/Qwen2.5-7B-Instruct') ?? null);
  const [gpu, setGpu] = useState<GpuSpec | null>(gpuById('h100-sxm') ?? null);
  const [precisionId, setPrecisionId] = useState('bf16');
  const [avgCtxTokens, setAvgCtxTokens] = useState(8192);
  const [concurrency, setConcurrency] = useState(10);
  const [inputTokens, setInputTokens] = useState(2048);
  const [caveat, setCaveat] = useState<string | undefined>(undefined);
  const [opts, setOpts] = useState<Optimizations>(OPTS_OFF);

  function pickModel(m: ModelSpec) {
    setModel(m);
    setCaveat(undefined);
    // clamp selections into the model's own options
    if (!m.precisions.some((p) => p.id === precisionId)) {
      setPrecisionId(m.precisions[0].id);
    }
    if (avgCtxTokens > Math.max(...m.contexts)) {
      setAvgCtxTokens(Math.max(...m.contexts));
    }
  }

  function loadPreset(s: PresetScenario) {
    const m = modelById(s.modelId);
    const g = gpuById(s.gpuId);
    if (!m || !g) return;
    setModel(m);
    setGpu(g);
    setPrecisionId(s.precisionId);
    setAvgCtxTokens(s.avgCtxTokens);
    setConcurrency(s.concurrency);
    setInputTokens(s.inputTokens);
    setCaveat(s.caveat);
  }

  const ready = model && gpu;

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">LLM 推理显存与 GPU 需求计算器</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          选模型 → 配精度与负载 → 选硬件，估算最少设备数与 TTFT / TPOT / 吞吐。数据取自 HuggingFace 与 ModelScope 的 config.json 及厂商官方规格。
        </p>
      </header>

      <Card className="mb-6">
        <CardContent className="flex flex-wrap items-center gap-2 py-3">
          <span className="text-xs font-medium text-muted-foreground">样例场景：</span>
          {PRESET_SCENARIOS.map((s) => (
            <Button key={s.id} variant="outline" size="sm" onClick={() => loadPreset(s)} title={s.description}>
              {s.title}
            </Button>
          ))}
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <ModelPicker selected={model} onSelect={pickModel} />
        <GpuPicker selected={gpu} onSelect={setGpu} />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <div className="space-y-4">
        <ConfigPicker
          model={model}
          precisionId={precisionId}
          onPrecision={setPrecisionId}
          avgCtxTokens={avgCtxTokens}
          onAvgCtx={setAvgCtxTokens}
          concurrency={concurrency}
          onConcurrency={setConcurrency}
          inputTokens={inputTokens}
          onInputTokens={setInputTokens}
        />
          <OptPanel gpu={gpu} opts={opts} onOptsChange={setOpts} />
        </div>
        {ready ? (
          <ResultsPanel
            model={model}
            gpu={gpu}
            input={{ precisionId, concurrency, avgCtxTokens, inputTokens }}
            caveat={caveat}
            opts={opts}
          />
        ) : (
          <div className="flex items-center justify-center rounded-lg border border-dashed p-8 text-sm text-muted-foreground">
            先选择模型与硬件
          </div>
        )}
      </div>

      <footer className="mt-8 border-t pt-4 text-xs text-muted-foreground">
        <p>估算口径：显存 = 权重 + 并发 × 平均上下文 × KV/token + 运行时开销；TPOT 按带宽屋顶线（权重 + KV ÷ η·带宽），TTFT 按 prefill 算力屋顶线与权重载入时间的较大值。三档区间对应 η=0.7/0.8/0.9 与不同开销系数。结果为估算值，实际部署请以目标框架实测为准。</p>
        <p className="mt-1">模型数据抓取于 2026-09-23（HuggingFace / ModelScope）；硬件规格来自厂商官方页面。DeepSeek-V4-Flash-0731 与 GLM-5.3-Flash 的参数量由 checkpoint 体量推算。</p>
      </footer>
    </div>
  );
}
