import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ModelLoadFields } from '@/sections/ConfigPicker';
import { fleetEntrySummary, fleetKindOf, fleetKindLabel, FLEET_KIND_ORDER, type FleetEntry } from '@/lib/fleet';
import type { GpuSpec } from '@/data/gpus';
import type { Optimizations } from '@/lib/optimizations';
import InfoTip from '@/components/info-tip';

const FLEET_CAP = 6;

interface Props {
  gpu: GpuSpec;
  entries: FleetEntry[];
  opts: Optimizations;
  onPatch: (id: string, patch: Partial<Omit<FleetEntry, 'id' | 'model'>>) => void;
  onRemove: (id: string) => void;
}

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
    <Card>
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-1.5 text-base">
          ② 逐模型负载（{entries.length} 个） <InfoTip text="每个模型独立配置精度、上下文与并发；行头显存按当前配置估算（权重 + KV + 开销），最少卡数为服务该负载的最小设备数。引擎按各自需求分配集群卡数，再逐模型推演。" />
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {entries.length === 0 && (
          <div className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
            还没有模型：在上方 ① 卡片搜索并添加（可多选，最多 {FLEET_CAP} 个）
          </div>
        )}
        {groups.map((g) => (
          <div key={g.kind} className="space-y-2">
            <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
              {fleetKindLabel(g.kind)}
              <span className="opacity-70">{g.items.length} 个</span>
              <div className="h-px flex-1 bg-border" />
            </div>
            {g.items.map((e) => {
              const sum = summaries.get(e.id)!;
              return (
                <div key={e.id} className="rounded-lg border p-3">
                  <div className="mb-2 flex flex-wrap items-center gap-x-2 gap-y-1">
                    <span className="text-sm font-medium">{e.model.name}</span>
                    <span className="text-xs text-muted-foreground">{fmtParams(e.model.paramsB)} 参数</span>
                    {e.model.isMoE && <Badge variant="outline" className="text-[10px] text-amber-600 dark:text-amber-400">MoE</Badge>}
                    {e.model.isMLA && <Badge variant="outline" className="text-[10px]">MLA</Badge>}
                    <span className="text-xs text-muted-foreground">·</span>
                    <span className="text-xs">
                      显存 <span className="font-medium text-foreground">{fmtGB(sum.totalGB)}</span> GB
                      <span className="ml-1 text-muted-foreground">（权重 {fmtGB(sum.weightsGB)} + KV {fmtGB(sum.kvGB)}）</span>
                    </span>
                    <span className="text-xs text-muted-foreground">·</span>
                    {sum.floor === null ? (
                      <span className="text-xs text-destructive">本硬件无法装载</span>
                    ) : (
                      <span className="text-xs">最少 <span className="font-medium text-foreground">{sum.need ?? sum.floor}</span> 张卡</span>
                    )}
                    <Button
                      variant="ghost"
                      size="sm"
                      className="ml-auto h-7 px-2 text-xs text-muted-foreground hover:text-destructive"
                      onClick={() => onRemove(e.id)}
                      aria-label={`移除 ${e.model.name}`}
                    >
                      移除
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
                  />
                </div>
              );
            })}
          </div>
        ))}
        {entries.length > 0 && entries.length < FLEET_CAP && (
          <p className="text-xs text-muted-foreground">在上方 ① 卡片继续添加模型（最多 {FLEET_CAP} 个）</p>
        )}
      </CardContent>
    </Card>
  );
}
