import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { PRECISIONS, type ModelSpec } from '@/data/models';
import { SlidersHorizontal } from 'lucide-react';
import InfoTip from '@/components/info-tip';
import { cn } from '@/lib/utils';

const CTX_LADDER = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 163840, 262144, 524288, 1048576];

function fmtCtx(n: number): string {
  if (n >= 1048576) return `1M (${n.toLocaleString()})`;
  if (n >= 1024) return `${n / 1024}K`;
  return String(n);
}

function clamp(n: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Number.isFinite(n) ? Math.round(n) : min));
}

/** 单个模型的负载配置字段落：精度 / 上下文 / 并发 / TTFT 输入长度。 */
export function ModelLoadFields(p: {
  model: ModelSpec;
  precisionId: string;
  onPrecision: (id: string) => void;
  avgCtxTokens: number;
  onAvgCtx: (n: number) => void;
  concurrency: number;
  onConcurrency: (n: number) => void;
  inputTokens: number;
  onInputTokens: (n: number) => void;
  compact?: boolean;
}) {
  const [customCtx, setCustomCtx] = useState(false);
  const m = p.model;

  const declared = new Set(m.contexts);
  const ctxOptions = [...new Set([...m.contexts, ...CTX_LADDER.filter((c) => c <= Math.max(...m.contexts))])].sort((a, b) => a - b);
  const isCustom = !ctxOptions.includes(p.avgCtxTokens);
  const ctxValue = customCtx || isCustom ? 'custom' : String(p.avgCtxTokens);

  // TTFT 固定档位：不超过当前平均上下文，当前值不在档位内时一并保留
  const ttftOptions = [...new Set([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536].filter((n) => n <= p.avgCtxTokens).concat(p.inputTokens))].sort((a, b) => a - b);

  return (
    <div className={cn('grid grid-cols-1', !p.compact && 'sm:grid-cols-2', p.compact ? 'gap-3' : 'gap-4')}>
      <div className="space-y-1.5">
        <Label className="text-meta">
          计算精度
          <InfoTip text="权重存储位宽：BF16/FP16 约 2 字节/参数，FP8/INT8 约 1 字节，INT4 约 0.5 字节，GGUF Q4_K_M 约 0.59 字节。KV 缓存仍按 FP16（部分模型 FP8）单独计算。" />
        </Label>
        <Select value={p.precisionId} onValueChange={p.onPrecision}>
          <SelectTrigger className="h-10 w-full rounded-xl text-value transition-colors focus:border-primary focus:ring-primary">
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="rounded-xl">
            {m.precisions.map((opt) => (
              <SelectItem key={opt.id} value={opt.id} className="rounded-lg">
                {PRECISIONS[opt.id]?.label ?? opt.id}
                {opt.note ? ` — ${opt.note}` : ''}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-xs leading-relaxed text-muted-foreground">
          KV 缓存按 {PRECISIONS[p.precisionId]?.kvDtypeBytes === 1 ? 'FP8' : 'FP16'} 存储；模型上下文上限 {fmtCtx(Math.max(...m.contexts))}
        </p>
      </div>

      {m.kind !== 'generic' && (
        <div className="space-y-1.5">
          <Label className="text-meta">
            平均上下文长度
            <InfoTip text="每路请求的输入 + 输出 token 总量，KV 缓存显存与之线性正比：并发 10 路 × 8K 上下文即 80K token 的 KV 常驻显存。" />
          </Label>
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
            <SelectTrigger className="h-10 w-full rounded-xl text-value transition-colors focus:border-primary focus:ring-primary">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="rounded-xl">
              {ctxOptions.map((c) => (
                <SelectItem key={c} value={String(c)} className="rounded-lg">
                  {fmtCtx(c)}{declared.has(c) ? ' · 官方档位' : ''}
                </SelectItem>
              ))}
              <SelectItem value="custom" className="rounded-lg">自定义…</SelectItem>
            </SelectContent>
          </Select>
          <div
            className={cn(
              'grid transition-all duration-normal ease-out-quart',
              customCtx || isCustom ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0',
            )}
          >
            <div className="overflow-hidden">
              <Input
                type="number"
                min={128}
                max={Math.max(...m.contexts)}
                step={128}
                value={p.avgCtxTokens}
                onChange={(e) => p.onAvgCtx(clamp(Number(e.target.value), 128, Math.max(...m.contexts)))}
                className="mt-2 h-10 rounded-xl tabular-nums text-value transition-colors focus:border-primary focus:ring-primary"
              />
            </div>
          </div>
          <p className="text-xs leading-relaxed text-muted-foreground">
            该模型声明的档位：{m.contexts.map(fmtCtx).join(' · ')}
          </p>
        </div>
      )}

      <div className="space-y-1.5">
        <Label className="text-meta">
          {m.kind === 'generic' ? '并发实例数（副本）' : '并发用户数'}
            <InfoTip text={m.kind === 'generic' ? '非自回归模型无多用户 KV 分摊，此处按并行副本数计算总显存。' : '同时在推理的用户路数：KV 缓存按并发数倍增，吞吐按副本分摊；显存不够时会自动增加设备或副本。'} />
          </Label>
          <Input
            type="number"
            min={1}
            max={4096}
            value={p.concurrency}
            onChange={(e) => p.onConcurrency(clamp(Number(e.target.value), 1, 4096))}
            className="h-10 rounded-xl tabular-nums text-value transition-colors focus:border-primary focus:ring-primary"
          />
        </div>
        <div className="space-y-1.5">
          <Label className="text-meta">
            TTFT 测算输入长度
            <InfoTip text="首字延迟（TTFT）测算所用的提示词长度：prefill 计算量随输入长度线性增长，典型对话约 512–4096，长文档场景可设更高。" />
          </Label>
          <Select value={String(p.inputTokens)} onValueChange={(v) => p.onInputTokens(Number(v))}>
            <SelectTrigger className="h-10 w-full rounded-xl text-value transition-colors focus:border-primary focus:ring-primary">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="rounded-xl">
              {ttftOptions.map((n) => (
                <SelectItem key={n} value={String(n)} className="rounded-lg">
                  {n >= 1024 ? `${n / 1024}K (${n.toLocaleString()})` : `${n} tokens`}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
    </div>
  );
}

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

export default function ConfigPicker(p: Props) {
  if (!p.model) return null;

  return (
    <Card className="section-card flex-1 gap-0">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base font-semibold tracking-tight">
          <SlidersHorizontal className="h-4 w-4 text-primary" />
          计算精度与负载
          <InfoTip text="精度决定权重的每参数字节数；上下文与并发决定 KV 缓存总量。显存 = 权重 + KV + 运行时开销，三者共同决定最少设备数。" />
        </CardTitle>
        <CardDescription>
          精度决定权重的每参数字节数；上下文与并发决定 KV 缓存总量。
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ModelLoadFields
          model={p.model}
          precisionId={p.precisionId}
          onPrecision={p.onPrecision}
          avgCtxTokens={p.avgCtxTokens}
          onAvgCtx={p.onAvgCtx}
          concurrency={p.concurrency}
          onConcurrency={p.onConcurrency}
          inputTokens={p.inputTokens}
          onInputTokens={p.onInputTokens}
        />
      </CardContent>
    </Card>
  );
}
