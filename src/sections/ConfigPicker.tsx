import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { PRECISIONS, type ModelSpec } from '@/data/models';

interface Props {
  model: ModelSpec | null;
  precisionId: string;
  onPrecision: (id: string) => void;
  avgCtxTokens: number;
  onAvgCtx: (n: number) => void;
  concurrency: number;
  onConcurrency: (n: number) => void;
  inputTokens: number;
  onInputTokens: (n: number) => void;
}

const CTX_LADDER = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 163840, 262144, 524288, 1048576];

function fmtCtx(n: number): string {
  if (n >= 1048576) return `1M (${n.toLocaleString()})`;
  if (n >= 1024) return `${n / 1024}K`;
  return String(n);
}

export default function ConfigPicker(p: Props) {
  const [customCtx, setCustomCtx] = useState(false);
  if (!p.model) return null;
  const m = p.model;

  const declared = new Set(m.contexts);
  const ctxOptions = [...new Set([...m.contexts, ...CTX_LADDER.filter((c) => c <= Math.max(...m.contexts))])].sort((a, b) => a - b);
  const isCustom = !ctxOptions.includes(p.avgCtxTokens);
  const ctxValue = customCtx || isCustom ? 'custom' : String(p.avgCtxTokens);

  function clamp(n: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, Number.isFinite(n) ? Math.round(n) : min));
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">② 计算精度与负载</CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="space-y-2">
          <Label>计算精度</Label>
          <Select value={p.precisionId} onValueChange={p.onPrecision}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              {m.precisions.map((opt) => (
                <SelectItem key={opt.id} value={opt.id}>
                  {PRECISIONS[opt.id]?.label ?? opt.id}
                  {opt.note ? ` — ${opt.note}` : ''}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            KV 缓存按 {PRECISIONS[p.precisionId]?.kvDtypeBytes === 1 ? 'FP8' : 'FP16'} 存储；模型上下文上限 {fmtCtx(Math.max(...m.contexts))}
          </p>
        </div>

        {m.kind !== 'generic' && (
          <div className="space-y-2">
            <Label>平均上下文长度</Label>
            <Select
              value={ctxValue}
              onValueChange={(v) => {
                if (v === 'custom') {
                  setCustomCtx(true);
                } else {
                  setCustomCtx(false);
                  p.onAvgCtx(Number(v));
                }
              }}
            >
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                {ctxOptions.map((c) => (
                  <SelectItem key={c} value={String(c)}>
                    {fmtCtx(c)}{declared.has(c) ? ' · 官方档位' : ''}
                  </SelectItem>
                ))}
                <SelectItem value="custom">自定义…</SelectItem>
              </SelectContent>
            </Select>
            {(customCtx || isCustom) && (
              <Input
                type="number"
                min={128}
                max={Math.max(...m.contexts)}
                step={128}
                value={p.avgCtxTokens}
                onChange={(e) => p.onAvgCtx(clamp(Number(e.target.value), 128, Math.max(...m.contexts)))}
              />
            )}
            <p className="text-xs text-muted-foreground">
              该模型声明的档位：{m.contexts.map(fmtCtx).join(' · ')}
            </p>
          </div>
        )}

        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-2">
            <Label>{m.kind === 'generic' ? '并发实例数（副本）' : '并发用户数'}</Label>
            <Input
              type="number"
              min={1}
              max={4096}
              value={p.concurrency}
              onChange={(e) => p.onConcurrency(clamp(Number(e.target.value), 1, 4096))}
            />
          </div>
          <div className="space-y-2">
            <Label>TTFT 测算输入长度</Label>
            <Input
              type="number"
              min={128}
              max={Math.max(p.avgCtxTokens, 1024)}
              step={128}
              value={p.inputTokens}
              onChange={(e) => p.onInputTokens(clamp(Number(e.target.value), 128, Math.max(p.avgCtxTokens, 1024)))}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
