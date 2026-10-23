import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { runAllBands, fmtGB, fmtTok, fmtMs, type ScenarioInput } from '@/lib/search';
import { runCapacityAllBands } from '@/lib/capacity';
import { OPTS_OFF, anyOptEnabled, type Optimizations } from '@/lib/optimizations';
import AnimatedNumber from '@/components/result/AnimatedNumber';
import BandCard, { type BandMeta } from '@/components/result/BandCard';
import Callout from '@/components/result/Callout';
import VramBar from '@/components/result/VramBar';
import ResultHeader from '@/components/result/ResultHeader';
import type { ModelSpec } from '@/data/models';
import type { GpuSpec } from '@/data/gpus';
import InfoTip from '@/components/info-tip';
import { Rocket, AlertCircle, AlertTriangle, Info } from 'lucide-react';

interface Props {
  model: ModelSpec;
  gpu: GpuSpec;
  input: Omit<ScenarioInput, 'model' | 'gpu'>;
  caveat?: string;
  opts: Optimizations;
  deployMode: 'min' | 'given';
  devices: number;
  onDeployMode: (m: 'min' | 'given') => void;
  onDevices: (n: number) => void;
}

const BAND_LABEL: Record<string, BandMeta> = {
  conservative: { name: '保守', hint: 'η=0.7 · 开销大', cls: 'text-amber-600 dark:text-amber-400', tip: '带宽效率按 0.7、运行时开销按 4GB + 权重 10% 计：为调度、碎片与峰值并发留足余量，适合采购预算决策。', dot: 'bg-amber-500' },
  typical: { name: '典型', hint: 'η=0.8', cls: 'text-primary', tip: '带宽效率 0.8、开销 3GB + 权重 7.5%：对应 vLLM / SGLang 常见实测水平，推荐以本档作为规划基准。', dot: 'bg-primary' },
  aggressive: { name: '激进', hint: 'η=0.9 · 开销小', cls: 'text-emerald-600 dark:text-emerald-400', tip: '带宽效率 0.9、开销 2GB + 权重 5%：理论接近上限，仅在深度调优（量化 KV、chunked prefill 等）后可能达到。', dot: 'bg-emerald-500' },
};

const MIN_TIP = '屋顶线估算结果：在满足显存与延迟的前提下搜索的最少设备组合（TP 节点内并行 / PP 跨节点流水 / DP 副本）。三档对应不同效率与开销假设，均为估算值而非实测。';
const GIVEN_TIP = '反向推演：在实际持有的 N 张卡内枚举 TP/PP/DP 分解，选单用户最快（即集群总吞吐最高）的方案；最大并发按显存 KV 预算反推（dp × 单副本容量上限）。输入并发超出容量时按满载口径展示。均为估算值。';

function clampDevices(n: number): number {
  return Math.min(4096, Math.max(1, Number.isFinite(n) ? Math.round(n) : 1));
}

const BAND_KEYS = ['conservative', 'typical', 'aggressive'] as const;

export default function ResultsPanel({ model, gpu, input, caveat, opts, deployMode, devices, onDeployMode, onDevices }: Props) {
  const given = deployMode === 'given';
  const isGeneric = model.kind === 'generic';
  const isVlm = model.kind === 'vlm';
  const n = Math.max(1, Math.floor(devices));

  const bands = runAllBands({ model, gpu, ...input }, opts);
  const baseBands = runAllBands({ model, gpu, ...input }, OPTS_OFF);
  const capBands = runCapacityAllBands({ model, gpu, ...input, devices: n }, opts);
  const typical = bands.typical;
  const cap = capBands.typical;

  const minDevices = Math.min(...BAND_KEYS.map((b) => bands[b].plan.totalDevices));
  const maxDevices = Math.max(...BAND_KEYS.map((b) => bands[b].plan.totalDevices));

  // 单卡显存构成条：最少卡数模式用整模型口径，实际卡数模式用选定方案的每卡口径
  const barWeights = given ? cap.vramPerDevice.weightsGB : typical.vram.weightsGB;
  const barKv = given ? cap.vramPerDevice.kvGB : typical.vram.kvTotalGB;
  const barOverhead = given ? cap.vramPerDevice.overheadGB : typical.vram.overheadGB;
  const barTotal = barWeights + barKv + barOverhead || 1;

  const capFits = cap.plan.feasible;

  return (
    <div className="flex h-full flex-col gap-5">
      <Card className="section-card">
        <ResultHeader
          icon={<Rocket className="h-4 w-4 text-primary" />}
          title={`推演结果 — ${model.name} @ ${gpu.name}`}
          infoTip={given ? GIVEN_TIP : MIN_TIP}
          subtitle={
            given
              ? `负载：并发 ${input.concurrency} 路 · 平均上下文 ${input.avgCtxTokens.toLocaleString()} tokens —— 在 ${n} 张卡内挑总吞吐最高的部署`
              : `典型方案：${typical.plan.tp} 路张量并行 × ${typical.plan.pp} 级流水 × ${typical.plan.dp} 数据副本（单副本 ${typical.plan.devicesPerReplica} 卡，每副本承载 ${typical.plan.replicaConcurrency} 路并发）`
          }
          deployMode={deployMode}
          onDeployMode={onDeployMode}
          devices={devices}
          onDevices={(v) => onDevices(clampDevices(v))}
          gpuName={gpu.name}
          actualLabel="实际持有"
        />
        <CardContent className="space-y-5">
          {isGeneric && (
            <Callout variant="info" icon={<Info className="h-4 w-4" />}>
              非自回归模型（diffusers / 生成管线等）：仅估算权重显存与设备占用；
              无 KV 缓存，TTFT / TPOT / 单用户 tok/s 与最大并发不适用，实际吞吐请按目标框架（diffusers、ComfyUI 等）实测。
            </Callout>
          )}
          {isVlm && (
            <Callout variant="info" icon={<Info className="h-4 w-4" />}>
              多模态模型：显存 = 视觉/音频塔 + 文本塔权重（架构估算），KV/吞吐按文本塔计算；图像/音频输入会额外占用上下文 token。
            </Callout>
          )}

          {given ? (
            <>
              {!capFits && (
                <Callout variant="error" icon={<AlertCircle className="h-4 w-4" />}>
                  {cap.reason ?? `${n} 张 ${gpu.name} 无法承载当前配置，请减少卡数外的其他变量或更换更大显存的设备。`}
                </Callout>
              )}
              {capFits && !cap.feasibleAtC && (
                <Callout variant="warning" icon={<AlertTriangle className="h-4 w-4" />}>
                  {cap.reason}
                </Callout>
              )}

              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <div className="rounded-2xl bg-secondary/30 p-5">
                  <div className="metric-value">
                    {capFits && !isGeneric ? (
                      <>
                        <AnimatedNumber value={Math.round(cap.clusterAggregateTokPerSec)} formatter={(v) => v.toLocaleString()} />
                        <span className="metric-label ml-2">
                          tok/s · {n} 张 {gpu.name} 合计{!cap.feasibleAtC ? '（满载口径）' : ''}
                        </span>
                      </>
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </div>
                  {capFits && !isGeneric && (
                    <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                      方案 {cap.plan.tp}×TP · {cap.plan.pp}×PP · {cap.plan.dp}×DP{cap.plan.idleDevices > 0 ? `（闲置 ${cap.plan.idleDevices} 卡）` : ''} · 单用户 {fmtTok(cap.throughput.perUserTokPerSec)} tok/s · TPOT {fmtMs(cap.throughput.tpotMs)} · TTFT {fmtMs(cap.throughput.ttftMs)}
                    </p>
                  )}
                </div>
                <VramBar
                  className="rounded-2xl bg-secondary/30 p-5"
                  weights={barWeights}
                  kv={barKv}
                  overhead={barOverhead}
                  total={barTotal}
                  gpu={gpu}
                />
              </div>

              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                {BAND_KEYS.map((band) => {
                  const r = capBands[band];
                  const meta = BAND_LABEL[band];
                  const fits = r.plan.feasible;
                  const atCap = fits && !r.feasibleAtC;
                  const isTypical = band === 'typical';
                  return (
                    <BandCard key={band} meta={meta} highlighted={isTypical} className={`h-full ${band === 'aggressive' ? 'md:col-span-2 xl:col-span-1' : ''}`}>
                      {fits ? (
                        <dl className="mt-2 space-y-1.5 text-xs">
                          <div className="flex justify-between"><dt className="text-muted-foreground">方案</dt><dd className="font-medium tabular-nums">{r.plan.tp}×TP · {r.plan.pp}×PP · {r.plan.dp}×DP</dd></div>
                          <div className="flex justify-between"><dt className="text-muted-foreground">用卡</dt><dd className="font-medium tabular-nums">{r.plan.totalDevices}{r.plan.idleDevices > 0 ? `（闲置 ${r.plan.idleDevices}）` : ''}</dd></div>
                          <div className="flex justify-between"><dt className="text-muted-foreground">每副本并发</dt><dd className="font-medium tabular-nums">{Number.isFinite(r.plan.replicaConcurrency) ? `${r.plan.replicaConcurrency} 路` : '—'}</dd></div>
                          <div className="flex justify-between">
                            <dt className="flex items-center gap-1 text-muted-foreground">最大并发 <InfoTip text="满载能力：按显存 KV 预算反推（dp × 单副本容量上限），含缓存卸载的影响。" /></dt>
                            <dd className="font-medium tabular-nums">{Number.isFinite(r.capacity.maxConcurrency) ? `${r.capacity.maxConcurrency.toLocaleString()} 路` : '—'}</dd>
                          </div>
                          <div className="flex justify-between"><dt className="text-muted-foreground">TPOT</dt><dd className="font-medium tabular-nums">{isGeneric ? '—' : fmtMs(r.throughput.tpotMs)}</dd></div>
                          <div className="flex justify-between"><dt className="text-muted-foreground">TTFT</dt><dd className="font-medium tabular-nums">{isGeneric ? '—' : fmtMs(r.throughput.ttftMs)}</dd></div>
                          <div className="flex justify-between"><dt className="text-muted-foreground">单用户</dt><dd className="font-medium tabular-nums">{isGeneric ? '—' : `${fmtTok(r.throughput.perUserTokPerSec)} tok/s`}</dd></div>
                          <div className="flex justify-between"><dt className="text-muted-foreground">{atCap ? '满载总吞吐' : '总吞吐'}</dt><dd className="font-medium tabular-nums">{isGeneric ? '—' : `${fmtTok(r.clusterAggregateTokPerSec)} tok/s`}</dd></div>
                          <div className="flex justify-between"><dt className="text-muted-foreground">单卡占用</dt><dd className="font-medium tabular-nums">{fmtGB(r.vramPerDevice.totalGB)} GB</dd></div>
                        </dl>
                      ) : (
                        <p className="mt-2 text-xs text-destructive">{r.reason ?? '该档位下无法承载'}</p>
                      )}
                      {atCap && (
                        <p className="mt-2 text-[11px] font-medium text-amber-600 dark:text-amber-400">
                          并发 {input.concurrency} 路超出容量，以上按满载 {r.capacity.maxConcurrency.toLocaleString()} 路计
                        </p>
                      )}
                    </BandCard>
                  );
                })}
              </div>

              {capFits && !isGeneric && Number.isFinite(cap.capacity.maxConcurrency) && (
                <Callout variant="info" icon={<Info className="h-4 w-4" />}>
                  满载能力（典型档）：{n} 张卡最多承载 <span className="font-medium tabular-nums text-foreground">{cap.capacity.maxConcurrency.toLocaleString()} 路并发</span>
                  （方案 {cap.capacity.plan.tp}×TP · {cap.capacity.plan.pp}×PP · {cap.capacity.plan.dp}×DP），满载总吞吐 <span className="font-medium tabular-nums text-foreground">{fmtTok(cap.capacity.clusterAggregateTokPerSec)} tok/s</span>
                </Callout>
              )}
            </>
          ) : (
            <>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <div className="rounded-2xl bg-secondary/30 p-5">
                  <div className="metric-value">
                    <AnimatedNumber value={minDevices === maxDevices ? minDevices : `${minDevices}–${maxDevices}`} />
                    <span className="metric-label ml-2">张 {gpu.name}</span>
                  </div>
                </div>
                <VramBar
                  className="rounded-2xl bg-secondary/30 p-5"
                  weights={barWeights}
                  kv={barKv}
                  overhead={barOverhead}
                  total={barTotal}
                  gpu={gpu}
                />
              </div>

              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                {BAND_KEYS.map((band) => {
                  const r = bands[band];
                  const meta = BAND_LABEL[band];
                  const isTypical = band === 'typical';
                  return (
                    <BandCard key={band} meta={meta} highlighted={isTypical} className={`h-full ${band === 'aggressive' ? 'md:col-span-2 xl:col-span-1' : ''}`}>
                      <dl className="mt-2 space-y-1.5 text-xs">
                        <div className="flex justify-between"><dt className="text-muted-foreground">设备数</dt><dd className="font-medium tabular-nums">{r.plan.totalDevices}</dd></div>
                        <div className="flex justify-between"><dt className="text-muted-foreground">TPOT</dt><dd className="font-medium tabular-nums">{isGeneric ? '—' : fmtMs(r.throughput.tpotMs)}</dd></div>
                        <div className="flex justify-between"><dt className="text-muted-foreground">TTFT</dt><dd className="font-medium tabular-nums">{isGeneric ? '—' : fmtMs(r.throughput.ttftMs)}</dd></div>
                        <div className="flex justify-between"><dt className="text-muted-foreground">单用户</dt><dd className="font-medium tabular-nums">{isGeneric ? '—' : `${fmtTok(r.throughput.perUserTokPerSec)} tok/s`}</dd></div>
                        <div className="flex justify-between"><dt className="text-muted-foreground">总吞吐</dt><dd className="font-medium tabular-nums">{isGeneric ? '—' : `${fmtTok(r.throughput.aggregateTokPerSec)} tok/s`}</dd></div>
                        <div className="flex justify-between"><dt className="text-muted-foreground">单卡占用</dt><dd className="font-medium tabular-nums">{fmtGB(r.plan.vramPerDeviceGB)} GB</dd></div>
                      </dl>
                    </BandCard>
                  );
                })}
              </div>

              {anyOptEnabled(opts) && (() => {
                const o = bands.typical.throughput, b = baseBands.typical.throughput;
                const pct = (a: number, c: number) => c === 0 ? '—' : ((a - c) / c * 100).toFixed(0);
                return (
                  <Callout variant="info" icon={<Info className="h-4 w-4" />}>
                    优化已开启（典型档 vs 关闭）：单用户吞吐 <span className="font-medium tabular-nums text-foreground">{fmtTok(o.perUserTokPerSec)} tok/s</span>（基线 {fmtTok(b.perUserTokPerSec)}，{pct(o.perUserTokPerSec, b.perUserTokPerSec)}%）· TTFT <span className="font-medium tabular-nums text-foreground">{fmtMs(o.ttftMs)}</span>（基线 {fmtMs(b.ttftMs)}，{pct(o.ttftMs, b.ttftMs)}%）
                    {bands.typical.throughput.kvTotalGB > opts.hotCache.sizeGB && opts.coldCache.enabled ? ' · 冷缓存 SSD 检索延迟已计入 TTFT' : ''}
                  </Callout>
                );
              })()}

              {!typical.plan.feasible && (
                <Callout variant="error" icon={<AlertCircle className="h-4 w-4" />}>
                  {typical.plan.reason ?? '该硬件无法承载当前配置，请换更大显存的设备或降低并发/上下文。'}
                </Callout>
              )}
            </>
          )}

          {caveat && (
            <Callout variant="warning" icon={<AlertTriangle className="h-4 w-4" />}>
              {caveat}
            </Callout>
          )}

          <div className="flex flex-wrap gap-1.5">
            <Badge variant="outline" className="rounded-md text-[10px] font-normal">屋顶线估算：吞吐随 η 取值浮动，非实测</Badge>
            {isGeneric ? (
              <Badge variant="outline" className="rounded-md text-[10px] font-normal">非自回归模型：无 KV 常驻项</Badge>
            ) : (
              <Badge variant="outline" className="rounded-md text-[10px] font-normal">KV 按 {model.isMLA ? 'MLA 压缩' : 'GQA'} 全层常驻计算</Badge>
            )}
            <Badge variant="outline" className="rounded-md text-[10px] font-normal">PP 跨节点开销未计入带宽</Badge>
            {given && <Badge variant="outline" className="rounded-md text-[10px] font-normal">实际卡数模式：TP/PP/DP 不整除时允许闲置少量卡</Badge>}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
