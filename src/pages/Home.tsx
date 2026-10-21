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
import type { FleetEntry } from '@/lib/fleet';
import FleetConfigPicker from '@/sections/FleetConfigPicker';
import FleetResultsPanel from '@/sections/FleetResultsPanel';
import OptPanel from '@/sections/OptPanel';
import { Button } from '@/components/ui/button';
import { Cpu, Sparkles, Boxes, Settings2, BarChart3 } from 'lucide-react';
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from '@/components/ui/empty';
import { cn } from '@/lib/utils';

const STEP_ITEMS = [
  { id: 'model', label: '选择模型', short: '模型', icon: Boxes },
  { id: 'gpu', label: '选择硬件', short: '硬件', icon: Cpu },
  { id: 'config', label: '配置负载', short: '负载', icon: Settings2 },
  { id: 'result', label: '推演结果', short: '结果', icon: BarChart3 },
];

function Stepper({
  steps,
  current,
}: {
  steps: { id: string; label: string; short: string; icon: React.ComponentType<{ className?: string }> }[];
  current: number;
}) {
  return (
    <div className="flex items-center gap-2 rounded-2xl border border-border/60 bg-card p-2 shadow-sm sm:p-3">
      {steps.map((step, idx) => {
        const Icon = step.icon;
        const active = idx <= current;
        const isCurrent = idx === current;
        return (
          <div key={step.id} className="flex flex-1 items-center gap-1 sm:gap-2">
            <div
              className={cn(
                'flex flex-1 items-center gap-2 rounded-xl px-2 py-2 transition-colors sm:px-3',
                isCurrent && 'bg-primary/5'
              )}
              aria-label={`步骤 ${idx + 1}：${step.label}${active ? '（已完成）' : ''}`}
            >
              <div
                className={cn(
                  'flex h-6 w-6 shrink-0 items-center justify-center rounded-lg text-[10px] font-semibold transition-colors sm:h-7 sm:w-7 sm:text-xs',
                  active
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-secondary text-muted-foreground'
                )}
              >
                {active && !isCurrent ? (
                  <Icon className="h-3 w-3 sm:h-3.5 sm:w-3.5" />
                ) : (
                  idx + 1
                )}
              </div>
              <div className="min-w-0">
                <p
                  className={cn(
                    'hidden text-xs font-medium sm:block',
                    active ? 'text-foreground' : 'text-muted-foreground'
                  )}
                >
                  {step.label}
                </p>
                <p
                  className={cn(
                    'block text-[10px] font-medium sm:hidden',
                    active ? 'text-foreground' : 'text-muted-foreground'
                  )}
                >
                  {step.short}
                </p>
              </div>
            </div>
            {idx < steps.length - 1 && (
              <div
                className={cn(
                  'step-connector hidden sm:block',
                  idx < current && 'step-connector-active'
                )}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

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

  function removeFleetEntry(idOrModelId: string) {
    setFleetEntries((prev) => prev.filter((e) => e.id !== idOrModelId && e.model.id !== idOrModelId));
  }

  function patchFleetEntry(id: string, patch: Partial<Omit<FleetEntry, 'id' | 'model'>>) {
    setFleetEntries((prev) => prev.map((e) => (e.id === id ? { ...e, ...patch } : e)));
  }

  function pickModel(m: ModelSpec) {
    setModel(m);
    setCaveat(undefined);
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

  const hasModel = appMode === 'multi' ? fleetEntries.length > 0 : !!model;
  const hasGpu = !!gpu;
  const ready = hasModel && hasGpu;

  const currentStep = !hasModel ? 0 : !hasGpu ? 1 : ready ? 3 : 2;

  return (
    <div className="page-container">
      {/* Header */}
      <header className="section-gap animate-fade-in-up">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
            <Cpu className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">LLM GPU 推理计算器</h1>
            <p className="text-sm text-muted-foreground">显存、卡数、TTFT / TPOT / 吞吐一键推演</p>
          </div>
        </div>
      </header>

      {/* Stepper */}
      <section className="section-gap animate-fade-in-up" style={{ animationDelay: '40ms' }}>
        <Stepper steps={STEP_ITEMS} current={currentStep} />
      </section>

      {/* Preset scenarios toolbar */}
      <section className="section-gap animate-fade-in-up" style={{ animationDelay: '80ms' }}>
        <div className="flex flex-wrap items-center gap-2">
          <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            <Sparkles className="h-3.5 w-3.5" />
            样例场景
          </span>
          {PRESET_SCENARIOS.map((s) => (
            <Button
              key={s.id}
              variant="outline"
              size="sm"
              onClick={() => loadPreset(s)}
              title={s.description}
              className="h-7 rounded-full border-border/80 bg-card px-3 text-xs font-normal shadow-none transition-all duration-fast hover:border-primary/30 hover:bg-primary/5 hover:text-primary hover:shadow-card"
            >
              {s.title}
            </Button>
          ))}
        </div>
      </section>

      {/* 选择模型 */}
      <section className="section-gap animate-fade-in-up">
        <ModelPicker
          selected={model}
          onSelect={pickModel}
          mode={appMode}
          onModeChange={setAppMode}
          selectedList={appMode === 'multi' ? fleetEntries.map((e) => e.model) : undefined}
          onAdd={appMode === 'multi' ? addFleetModel : undefined}
          onRemove={appMode === 'multi' ? removeFleetEntry : undefined}
        />
      </section>

      {/* 选择硬件 */}
      <section className="section-gap animate-fade-in-up" style={{ animationDelay: '40ms' }}>
        <GpuPicker selected={gpu} onSelect={setGpu} />
      </section>

      {/* 配置负载 */}
      <section className="section-gap animate-fade-in-up" style={{ animationDelay: '80ms' }}>
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
        ) : model ? (
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
        ) : null}
      </section>

      {/* 性能优化选项 */}
      <section className="section-gap animate-fade-in-up" style={{ animationDelay: '120ms' }}>
        <OptPanel gpu={gpu} opts={opts} onOptsChange={setOpts} />
      </section>

      {/* 推演结果 */}
      <section className="section-gap animate-fade-in-up" style={{ animationDelay: '160ms' }}>
        {ready && appMode === 'multi' && gpu ? (
          <FleetResultsPanel
            gpu={gpu}
            entries={fleetEntries}
            devices={devices}
            deployMode={deployMode}
            onDeployMode={setDeployMode}
            onDevices={setDevices}
            opts={opts}
          />
        ) : ready && model && gpu ? (
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
          <Empty className="min-h-[360px] rounded-2xl border-dashed border-border/60 bg-secondary/30">
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <Cpu className="h-6 w-6" />
              </EmptyMedia>
              <EmptyTitle>先完成前置步骤</EmptyTitle>
              <EmptyDescription>
                {!hasModel && !hasGpu && '选择模型与 GPU 后，即可查看显存构成、最少卡数与吞吐推演。'}
                {!hasModel && hasGpu && '请选择一个模型（多模型模式下可添加多个）。'}
                {hasModel && !hasGpu && '请选择要部署的 GPU 硬件。'}
              </EmptyDescription>
            </EmptyHeader>
          </Empty>
        )}
      </section>

      {/* Footer */}
      <footer className="section-gap border-t border-border/60 pt-6 text-xs leading-relaxed text-muted-foreground/70 animate-fade-in">
        <p>估算口径：显存 = 权重 + 并发 × 平均上下文 × KV/token + 运行时开销；TPOT 按带宽屋顶线（权重 + KV ÷ η·带宽），TTFT 按 prefill 算力屋顶线与权重载入时间的较大值。三档区间对应 η=0.7/0.8/0.9 与不同开销系数。结果为估算值，实际部署请以目标框架实测为准。</p>
        <p className="mt-1">模型数据抓取于 2026-09-23（HuggingFace / ModelScope）；硬件规格来自厂商官方页面。DeepSeek-V4-Flash-0731 与 GLM-5.3-Flash 的参数量由 checkpoint 体量推算。</p>
      </footer>
    </div>
  );
}
