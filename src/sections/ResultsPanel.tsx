import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { runAllBands, fmtGB, fmtTok, fmtMs, type ScenarioInput } from '@/lib/search';
import { usableVramGB } from '@/lib/vram';
import type { ModelSpec } from '@/data/models';
import type { GpuSpec } from '@/data/gpus';

interface Props {
  model: ModelSpec;
  gpu: GpuSpec;
  input: Omit<ScenarioInput, 'model' | 'gpu'>;
  caveat?: string;
}

const BAND_LABEL: Record<string, { name: string; hint: string; cls: string }> = {
  conservative: { name: '保守', hint: 'η=0.7 · 开销大', cls: 'text-amber-600 dark:text-amber-400' },
  typical: { name: '典型', hint: 'η=0.8', cls: 'text-primary' },
  aggressive: { name: '激进', hint: 'η=0.9 · 开销小', cls: 'text-emerald-600 dark:text-emerald-400' },
};

export default function ResultsPanel({ model, gpu, input, caveat }: Props) {
  const bands = runAllBands({ model, gpu, ...input });
  const typical = bands.typical;
  const minDevices = Math.min(...Object.values(bands).map((b) => b.plan.totalDevices));
  const maxDevices = Math.max(...Object.values(bands).map((b) => b.plan.totalDevices));
  const isGeneric = model.kind === 'generic';
  const isVlm = model.kind === 'vlm';

  const stack = typical.vram;
  const total = stack.weightsGB + stack.kvTotalGB + stack.overheadGB;
  const wp = (stack.weightsGB / total) * 100;
  const kp = (stack.kvTotalGB / total) * 100;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">④ 部署需求 — {model.name} @ {gpu.name}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-3xl font-semibold tracking-tight">
              {minDevices === maxDevices ? minDevices : `${minDevices}–${maxDevices}`}{' '}
              <span className="text-lg font-normal text-muted-foreground">张 {gpu.name}</span>
            </div>
            <div className="mt-1 text-xs text-muted-foreground">
              典型方案：{typical.plan.tp} 路张量并行 × {typical.plan.pp} 级流水 × {typical.plan.dp} 数据副本
              （单副本 {typical.plan.devicesPerReplica} 卡，每副本承载 {typical.plan.replicaConcurrency} 路并发）
            </div>
          </div>

          {isGeneric && (
            <div className="rounded-md border border-primary/40 bg-primary/5 p-3 text-xs text-muted-foreground">
              非自回归模型（diffusers / 生成管线等）：仅估算权重显存与最小持有设备数；
              无 KV 缓存，TTFT / TPOT / 单用户 tok/s 不适用，实际吞吐请按目标框架（diffusers、ComfyUI 等）实测。
            </div>
          )}
          {isVlm && (
            <div className="rounded-md border border-primary/40 bg-primary/5 p-3 text-xs text-muted-foreground">
              视觉语言模型：显存 = 视觉塔 + 文本塔权重（架构估算），KV/吞吐按文本塔计算；图像输入会额外占用上下文 token。
            </div>
          )}

          <div>
            <div className="mb-1.5 flex items-center justify-between text-xs text-muted-foreground">
              <span>单卡显存构成（典型档）</span>
              <span>{fmtGB(stack.totalGB)} GB / 可用 {usableVramGB(gpu).toFixed(1)} GB</span>
            </div>
            <div className="flex h-4 w-full overflow-hidden rounded-full bg-muted" role="img" aria-label={`显存构成：权重 ${wp.toFixed(0)}%，KV ${kp.toFixed(0)}%，开销 ${(100 - wp - kp).toFixed(0)}%`}>
              <div className="bg-primary" style={{ width: `${wp}%` }} title={`权重 ${fmtGB(stack.weightsGB)}GB`} />
              <div className="bg-primary/50" style={{ width: `${kp}%` }} title={`KV 缓存 ${fmtGB(stack.kvTotalGB)}GB`} />
              <div className="bg-muted-foreground/40" style={{ width: `${100 - wp - kp}%` }} title={`开销 ${fmtGB(stack.overheadGB)}GB`} />
            </div>
            <div className="mt-1.5 flex flex-wrap gap-3 text-[11px] text-muted-foreground">
              <span>■ 权重 {fmtGB(stack.weightsGB)} GB{isGeneric && model.checkpointBytesGB ? `（checkpoint 实测 ${model.checkpointBytesGB} GB）` : ''}</span>
              {!isGeneric && <span>■ KV 缓存 {fmtGB(stack.kvTotalGB)} GB（{fmtGB(stack.kvPerTokKB / 1024)} MB/token）</span>}
              <span>■ 运行时开销 {fmtGB(stack.overheadGB)} GB</span>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2">
            {(['conservative', 'typical', 'aggressive'] as const).map((band) => {
              const r = bands[band];
              const meta = BAND_LABEL[band];
              return (
                <div key={band} className="rounded-lg border p-3">
                  <div className="flex items-center justify-between">
                    <span className={`text-xs font-medium ${meta.cls}`}>{meta.name}档</span>
                    <span className="text-[10px] text-muted-foreground">{meta.hint}</span>
                  </div>
                  <dl className="mt-2 space-y-1.5 text-xs">
                    <div className="flex justify-between"><dt className="text-muted-foreground">设备数</dt><dd className="font-medium">{r.plan.totalDevices}</dd></div>
                    <div className="flex justify-between"><dt className="text-muted-foreground">TPOT</dt><dd className="font-medium">{isGeneric ? '—' : fmtMs(r.throughput.tpotMs)}</dd></div>
                    <div className="flex justify-between"><dt className="text-muted-foreground">TTFT</dt><dd className="font-medium">{isGeneric ? '—' : fmtMs(r.throughput.ttftMs)}</dd></div>
                    <div className="flex justify-between"><dt className="text-muted-foreground">单用户</dt><dd className="font-medium">{isGeneric ? '—' : `${fmtTok(r.throughput.perUserTokPerSec)} tok/s`}</dd></div>
                    <div className="flex justify-between"><dt className="text-muted-foreground">总吞吐</dt><dd className="font-medium">{isGeneric ? '—' : `${fmtTok(r.throughput.aggregateTokPerSec)} tok/s`}</dd></div>
                    <div className="flex justify-between"><dt className="text-muted-foreground">单卡占用</dt><dd className="font-medium">{fmtGB(r.plan.vramPerDeviceGB)} GB</dd></div>
                  </dl>
                </div>
              );
            })}
          </div>

          {!typical.plan.feasible && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive">
              {typical.plan.reason ?? '该硬件无法承载当前配置，请换更大显存的设备或降低并发/上下文。'}
            </div>
          )}

          {caveat && (
            <div className="text-xs text-muted-foreground">⚠ {caveat}</div>
          )}

          <div className="flex flex-wrap gap-1.5">
            <Badge variant="outline" className="text-[10px]">屋顶线估算：吞吐随 η 取值浮动，非实测</Badge>
            {isGeneric ? (
              <Badge variant="outline" className="text-[10px]">非自回归模型：无 KV 常驻项</Badge>
            ) : (
              <Badge variant="outline" className="text-[10px]">KV 按 {model.isMLA ? 'MLA 压缩' : 'GQA'} 全层常驻计算</Badge>
            )}
            <Badge variant="outline" className="text-[10px]">PP 跨节点开销未计入带宽</Badge>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
