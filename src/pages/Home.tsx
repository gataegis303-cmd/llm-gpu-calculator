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
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { fleetEntrySummary, type FleetEntry } from '@/lib/fleet';
import FleetConfigPicker from '@/sections/FleetConfigPicker';
import FleetResultsPanel from '@/sections/FleetResultsPanel';
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
  const [deployMode, setDeployMode] = useState<'min' | 'given'>('min');
  const [devices, setDevices] = useState(8);
  const [appMode, setAppMode] = useState<'single' | 'multi'>('single');
  const [fleetEntries, setFleetEntries] = useState<FleetEntry[]>(() => {
    const qwen = modelById('Qwen/Qwen2.5-7B-Instruct');
    const dsv3 = modelById('deepseek-ai/DeepSeek-V3');
    const entries: FleetEntry[] = [];
    if (qwen) {
      entries.push({ id: 'fleet-qwen7b', model: qwen, precisionId: 'bf16', concurrency: 10, avgCtxTokens: 8192, inputTokens: 2048 });
    }
    if (dsv3) {
      const prec = dsv3.precisions.some((x) => x.id === 'fp8') ? 'fp8' : dsv3.precisions[0].id;
      entries.push({ id: 'fleet-dsv3', model: dsv3, precisionId: prec, concurrency: 32, avgCtxTokens: Math.min(32768, Math.max(...dsv3.contexts)), inputTokens: 2048 });
    }
    return entries;
  });

  function addFleetModel(m: ModelSpec) {
    setFleetEntries((prev) => {
      if (prev.length >= 6 || prev.some((e) => e.model.id === m.id)) return prev;
      return [
        ...prev,
        {
          id: `fleet-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
          model: m,
          precisionId: m.precisions[0].id,
          concurrency: 10,
          avgCtxTokens: Math.min(8192, Math.max(...m.contexts)),
          inputTokens: 2048,
        },
      ];
    });
  }

  /** chips 传模型 id，配置行传条目 id —— 两者都接受（模型 id 在 fleet 内唯一） */
  function removeFleetEntry(idOrModelId: string) {
    setFleetEntries((prev) => prev.filter((e) => e.id !== idOrModelId && e.model.id !== idOrModelId));
  }

  function patchFleetEntry(id: string, patch: Partial<Omit<FleetEntry, 'id' | 'model'>>) {
    setFleetEntries((prev) => prev.map((e) => (e.id === id ? { ...e, ...patch } : e)));
  }

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
          选模型 → 配精度与负载 → 选硬件，估算最少设备数与 TTFT / TPOT / 吞吐；也可以反过来输入手上的实际卡数，推算能获得的并发容量与 tok/s；支持多模型共享集群，把一批卡按需求分给多个模型分别推演。数据取自 HuggingFace 与 ModelScope 的 config.json 及厂商官方规格。
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
        <ModelPicker
          selected={model}
          onSelect={pickModel}
          mode={appMode}
          selectedList={appMode === 'multi' ? fleetEntries.map((e) => e.model) : undefined}
          summaryOf={appMode === 'multi' && gpu ? (m) => {
            const entry = fleetEntries.find((e) => e.model.id === m.id);
            if (!entry) return undefined;
            const sum = fleetEntrySummary(entry, gpu, opts);
            return { kindLabel: sum.kindLabel, kind: sum.kind, vramGB: sum.totalGB, minDevices: sum.need ?? sum.floor };
          } : undefined}
          onAdd={appMode === 'multi' ? addFleetModel : undefined}
          onRemove={appMode === 'multi' ? (modelId) => { removeFleetEntry(modelId); } : undefined}
          headerExtra={
            <Tabs value={appMode} onValueChange={(v) => setAppMode(v === 'multi' ? 'multi' : 'single')}>
              <TabsList className="h-7">
                <TabsTrigger value="single" className="h-6 px-2.5 text-xs">单模型</TabsTrigger>
                <TabsTrigger value="multi" className="h-6 px-2.5 text-xs">多模型</TabsTrigger>
              </TabsList>
            </Tabs>
          }
        />
        <GpuPicker selected={gpu} onSelect={setGpu} />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <div className="space-y-4">
        {appMode === 'multi' ? (
          gpu ? (
          <FleetConfigPicker
            gpu={gpu}
            entries={fleetEntries}
            opts={opts}
            onPatch={patchFleetEntry}
            onRemove={removeFleetEntry}
          />
          ) : null
        ) : (
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
        )}
          <OptPanel gpu={gpu} opts={opts} onOptsChange={setOpts} />
        </div>
        {ready && appMode === 'multi' ? (
          <FleetResultsPanel
            gpu={gpu}
            entries={fleetEntries}
            devices={devices}
            deployMode={deployMode}
            onDeployMode={setDeployMode}
            onDevices={setDevices}
            opts={opts}
          />
        ) : ready ? (
          <ResultsPanel
            model={model}
            gpu={gpu}
            input={{ precisionId, concurrency, avgCtxTokens, inputTokens }}
            caveat={caveat}
            opts={opts}
            deployMode={deployMode}
            devices={devices}
            onDeployMode={setDeployMode}
            onDevices={setDevices}
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
