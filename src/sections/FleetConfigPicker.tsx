import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Empty, EmptyHeader, EmptyMedia, EmptyTitle, EmptyDescription } from '@/components/ui/empty';
import { ModelLoadFields } from '@/sections/ConfigPicker';
import { fleetEntrySummary, fleetKindOf, fleetKindLabel, FLEET_KIND_ORDER, type FleetEntry } from '@/lib/fleet';
import type { GpuSpec } from '@/data/gpus';
import type { Optimizations } from '@/lib/optimizations';
import InfoTip from '@/components/info-tip';
import { LayoutGrid, X } from 'lucide-react';

const FLEET_CAP = 6;

function fmtParams(b: number): string {
  return b >= 1000 ? `${(b / 1000).toFixed(2)}T` : `${b.toFixed(1)}B`;
}

function fmtGB(v: number): string {
  return v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
}

/** 多模型模式：逐模型负载配置行，按模型种类分组；行头带类型 / 显存 / 最少卡数。 */
export default function FleetConfigPicker({ gpu, entries, opts, onPatch, onRemove }: Props) {
  const summaries = new Map(entries.map((e) => [e.id, fleetEntrySummary(e, gpu, opts)]));
  const groups = FLEET_KIND_ORDER
    .map((kind) => ({ kind, items: entries.filter((e) => fleetKindOf(e.model) === kind) }))
    .filter((g) => g.items.length > 0);

  return (
    <Card className="section-card flex-1 gap-0">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base font-semibold tracking-tight">
          <LayoutGrid className="h-4 w-4 text-primary" />
          逐模型负载
          <InfoTip text="每个模型独立配置精度、上下文与并发；行头显存按当前配置估算（权重 + KV + 开销），最少卡数为服务该负载的最小设备数。引擎按各自需求分配集群卡数，再逐模型推演。" />
        </CardTitle>
        <CardDescription>
          每个模型独立配置精度、上下文与并发；行头显存按当前配置估算，最少卡数为服务该负载的最小设备数。
        </CardDescription>
        {entries.length > 0 && (
          <CardAction>
            <Button
              variant="ghost"
              size="sm"
              className="text-destructive hover:bg-destructive/10 hover:text-destructive"
              onClick={() => entries.forEach((e) => onRemove(e.id))}
            >
              清除全部
            </Button>
          </CardAction>
        )}
      </CardHeader>
      <CardContent>
        {entries.length === 0 && (
          <Empty>
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <LayoutGrid />
              </EmptyMedia>
              <EmptyTitle>还没有模型</EmptyTitle>
              <EmptyDescription>
                在模型选择中搜索并添加（可多选，最多 {FLEET_CAP} 个）
              </EmptyDescription>
            </EmptyHeader>
          </Empty>
        )}
        {groups.map((g) => (
          <div key={g.kind} className="space-y-3">
            <div className="flex items-center gap-2 text-meta">
              {fleetKindLabel(g.kind)}
              <span className="rounded-full bg-secondary px-2 py-0.5 text-[10px] opacity-90">{g.items.length} 个</span>
              <div className="h-px flex-1 bg-border/60" />
            </div>
            {g.items.map((e) => {
              const sum = summaries.get(e.id)!;
              return (
                <div key={e.id} className="rounded-2xl border border-border/60 bg-card p-4 transition-shadow duration-fast hover:shadow-card">
                  <div className="mb-3 grid grid-cols-[1fr_auto] items-start gap-3">
                    <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
                      <span className="text-value">{e.model.name}</span>
                      <span className="text-meta">{fmtParams(e.model.paramsB)} 参数</span>
                      {e.model.isMoE && <Badge variant="outline" className="rounded-md border-amber-500/30 text-[10px] font-normal text-amber-600 dark:text-amber-400">MoE</Badge>}
                      {e.model.isMLA && <Badge variant="outline" className="rounded-md text-[10px] font-normal">MLA</Badge>}
                      <span className="text-meta">·</span>
                      <span className="text-meta">显存</span>
                      <span className="text-value">{fmtGB(sum.totalGB)} GB</span>
                      <span className="text-xs text-muted-foreground">（权重 {fmtGB(sum.weightsGB)} + KV {fmtGB(sum.kvGB)}）</span>
                      <span className="text-meta">·</span>
                      {sum.floor === null ? (
                        <span className="text-xs font-medium text-destructive">本硬件无法装载</span>
                      ) : (
                        <>
                          <span className="text-meta">最少</span>
                          <span className="text-value">{sum.need ?? sum.floor} 张卡</span>
                        </>
                      )}
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 rounded-full text-muted-foreground/60 transition-colors hover:bg-destructive/10 hover:text-destructive"
                      onClick={() => onRemove(e.id)}
                      aria-label={`移除 ${e.model.name}`}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                  <ModelLoadFields
                    model={e.model}
                    precisionId={e.precisionId}
                    onPrecision={(id) => onPatch(e.id, { precisionId: id })}
                    avgCtxTokens={e.avgCtxTokens}
                    onAvgCtx={(n) => onPatch(e.id, { avgCtxTokens: n })}
                    concurrency={e.concurrency}
                    onConcurrency={(n) => onPatch(e.id, { concurrency: n })}
                    inputTokens={e.inputTokens}
                    onInputTokens={(n) => onPatch(e.id, { inputTokens: n })}
                    compact
                  />
                </div>
              );
            })}
          </div>
        ))}
        {entries.length > 0 && entries.length < FLEET_CAP && (
          <p className="text-xs text-muted-foreground">在模型选择中继续添加（最多 {FLEET_CAP} 个）</p>
        )}
      </CardContent>
    </Card>
  );
}

interface Props {
  gpu: GpuSpec;
  entries: FleetEntry[];
  opts: Optimizations;
  onPatch: (id: string, patch: Partial<Omit<FleetEntry, 'id' | 'model'>>) => void;
  onRemove: (id: string) => void;
}
