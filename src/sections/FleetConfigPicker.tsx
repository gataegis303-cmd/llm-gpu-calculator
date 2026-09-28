import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ModelLoadFields } from '@/sections/ConfigPicker';
import type { FleetEntry } from '@/lib/fleet';
import InfoTip from '@/components/info-tip';

const FLEET_CAP = 6;

interface Props {
  entries: FleetEntry[];
  onPatch: (id: string, patch: Partial<Omit<FleetEntry, 'id' | 'model'>>) => void;
  onRemove: (id: string) => void;
}

function fmtParams(b: number): string {
  return b >= 1000 ? `${(b / 1000).toFixed(2)}T` : `${b.toFixed(1)}B`;
}

/** 多模型模式：逐模型负载配置行（精度 / 上下文 / 并发 / TTFT 输入）。 */
export default function FleetConfigPicker({ entries, onPatch, onRemove }: Props) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-1.5 text-base">
          ② 逐模型负载（{entries.length} 个） <InfoTip text="每个模型独立配置精度、上下文与并发；引擎按各自需求分配集群卡数，再对每个模型单独推演吞吐与容量。" />
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {entries.length === 0 && (
          <div className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
            还没有模型：在上方 ① 卡片搜索并添加（可多选，最多 {FLEET_CAP} 个）
          </div>
        )}
        {entries.map((e) => (
          <div key={e.id} className="rounded-lg border p-3">
            <div className="mb-2 flex items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-1.5 text-sm font-medium">
                {e.model.name}
                <span className="text-xs text-muted-foreground">{fmtParams(e.model.paramsB)} 参数</span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 px-2 text-xs text-muted-foreground hover:text-destructive"
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
        ))}
        {entries.length > 0 && entries.length < FLEET_CAP && (
          <p className="text-xs text-muted-foreground">在上方 ① 卡片继续添加模型（最多 {FLEET_CAP} 个）</p>
        )}
      </CardContent>
    </Card>
  );
}
