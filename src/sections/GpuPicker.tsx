import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command';
import { GPUS, VENDOR_LABELS, type GpuSpec } from '@/data/gpus';
import { deployableModels } from '@/lib/fit';
import { ChevronsUpDown, HardDrive, ExternalLink } from 'lucide-react';
import { cn } from '@/lib/utils';
import InfoTip from '@/components/info-tip';

interface Props {
  selected: GpuSpec | null;
  onSelect: (g: GpuSpec) => void;
  className?: string;
}

const GROUPS: Array<{ key: GpuSpec['vendor']; heading: string }> = [
  { key: 'nvidia', heading: 'NVIDIA' },
  { key: 'amd', heading: 'AMD' },
  { key: 'ascend', heading: '华为昇腾' },
  { key: 'apple', heading: 'Apple M 系列' },
];

export default function GpuPicker({ selected, onSelect, className }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <Card className={cn('section-card', className)}>
      <CardHeader>
        <CardTitle className="flex items-center gap-1.5 text-base font-semibold tracking-tight">
          <HardDrive className="h-4 w-4 text-primary" />
          选择硬件
          <InfoTip text="内置 55+ 款设备的官方规格（显存 / 带宽 / 稠密算力 / 互联 / 每节点卡数）；统一内存设备（Apple、AMD AI Max、DGX Spark）按可用占比折算实际可分配显存。" />
        </CardTitle>
        <p className="mt-1 text-xs text-muted-foreground">
          按厂商分组，选择 GPU 后查看可部署模型与关键性能指标。
        </p>
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <button
              type="button"
              role="combobox"
              aria-expanded={open}
              className="group mt-3 flex h-11 w-full items-center justify-between rounded-xl border border-border/80 bg-secondary/50 px-3 py-2 text-sm transition-all duration-fast hover:border-primary/30 hover:bg-primary/5 focus-visible:border-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
            >
              <span className={cn('truncate', !selected && 'text-muted-foreground')}>
                {selected ? `${selected.name} · ${selected.vramGB}GB` : '搜索或选择 GPU…'}
              </span>
              <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 text-muted-foreground/50 transition-colors group-hover:text-muted-foreground" />
            </button>
          </PopoverTrigger>
          <PopoverContent className="w-[min(28rem,calc(100vw-2rem))] origin-top-left animate-scale-in p-0 rounded-xl border-border/80 shadow-elevated" align="start">
            <Command>
              <CommandInput placeholder="搜索 GPU 型号…" className="h-10" />
              <CommandList className="max-h-80">
                <CommandEmpty>没有匹配的 GPU</CommandEmpty>
                {GROUPS.map(({ key, heading }) => (
                  <CommandGroup key={key} heading={heading} className="text-xs font-medium text-muted-foreground">
                    {GPUS.filter((g) => g.vendor === key).map((g) => (
                      <CommandItem
                        key={g.id}
                        value={`${g.name} ${VENDOR_LABELS[key]}`}
                        onSelect={() => {
                          onSelect(g);
                          setOpen(false);
                        }}
                        className="cursor-pointer rounded-lg transition-colors"
                      >
                        <div className="flex w-full items-center justify-between gap-2">
                          <span className="truncate">
                            {g.name}
                            <span className="ml-2 text-xs text-muted-foreground">
                              {g.vramGB}GB · {g.memBandwidthGBs}GB/s
                            </span>
                          </span>
                          {selected?.id === g.id && <span className="shrink-0 text-[10px] font-medium text-primary">已选</span>}
                        </div>
                      </CommandItem>
                    ))}
                  </CommandGroup>
                ))}
              </CommandList>
            </Command>
          </PopoverContent>
        </Popover>
      </CardHeader>
      <CardContent className="border-t border-border/60 bg-secondary/20">
        {selected ? (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="secondary" className="rounded-md font-normal">{VENDOR_LABELS[selected.vendor]}</Badge>
              <Badge variant="outline" className="rounded-md font-normal">{selected.memType}</Badge>
              <Badge variant="outline" className="rounded-md font-normal">{selected.vramGB} GB 显存</Badge>
              <Badge variant="outline" className="rounded-md font-normal">{selected.memBandwidthGBs} GB/s</Badge>
              <Badge variant="outline" className="rounded-md font-normal">稠密 BF16 {selected.bf16Tflops} TFLOPS</Badge>
              {selected.fp8Tflops && <Badge variant="outline" className="rounded-md font-normal">FP8 {selected.fp8Tflops} TFLOPS</Badge>}
            </div>
            <div className="mt-2.5 flex items-center gap-1 text-xs text-muted-foreground">
              <span>来源：</span>
              <a
                href={selected.sourceUrl.startsWith('http') ? selected.sourceUrl : '#'}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-0.5 text-primary transition-colors hover:underline"
              >
                {selected.sourceUrl}
                <ExternalLink className="h-3 w-3" />
              </a>
            </div>
            <DeployableList gpu={selected} />
          </>
        ) : (
          <div className="py-1 text-sm text-muted-foreground">尚未选择硬件</div>
        )}
      </CardContent>
    </Card>
  );
}

/** 按设备反查：该设备（单并发、8K 上下文）能部署哪些内置模型。 */
function DeployableList({ gpu }: { gpu: GpuSpec }) {
  const [expanded, setExpanded] = useState(true);
  const { full, quantized, usableGB } = deployableModels(gpu);
  const show = (list: typeof full, label: string) => {
    if (list.length === 0) return null;
    const top = list.slice(0, 10);
    return (
      <div className="mt-2">
        <div className="text-xs font-medium text-muted-foreground">{label}（{list.length} 款）</div>
        <div className="mt-1.5 flex flex-wrap gap-1">
          {top.map((r) => (
            <span
              key={r.model.id}
              title={`${r.precisionId.toUpperCase()} · 需 ${r.totalGB.toFixed(1)}GB · 剩余显存约可扩至 ${r.headroomConcurrency} 路并发`}
              className="rounded-md border border-border/60 bg-background/60 px-1.5 py-0.5 text-[11px] text-muted-foreground transition-colors hover:border-primary/30 hover:text-foreground"
            >
              {r.model.name} <span className="text-[10px]">{r.model.paramsB >= 1000 ? `${(r.model.paramsB / 1000).toFixed(1)}T` : `${r.model.paramsB.toFixed(0)}B`}</span>
            </span>
          ))}
          {list.length > top.length && <span className="px-1 py-0.5 text-[11px] text-muted-foreground">…等 {list.length} 款</span>}
        </div>
      </div>
    );
  };
  return (
    <div className="mt-3 overflow-hidden rounded-xl border border-border/60 bg-background/50">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center justify-between px-3 py-2.5 text-left text-xs font-medium transition-colors hover:bg-secondary/50"
      >
        <span>这台设备能部署什么？（单并发 · 8K 上下文 · 可用 {usableGB.toFixed(0)}GB）</span>
        <ChevronsUpDown className={cn('h-3.5 w-3.5 text-muted-foreground/60 transition-transform duration-fast', expanded && 'rotate-180')} />
      </button>
      <div
        className={cn(
          'grid transition-all duration-normal ease-out-quart',
          expanded ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]',
        )}
      >
        <div className="overflow-hidden">
          <div className="border-t border-border/60 px-3 py-2.5">
            {show(full, '高精度档（BF16 / 原生 FP8）')}
            {show(quantized, '量化档（INT4 / Q4）')}
            {full.length === 0 && quantized.length === 0 && (
              <div className="text-xs text-muted-foreground">当前内置库中没有能装下的模型（显存不足）</div>
            )}
            <p className="mt-2 text-[10px] leading-relaxed text-muted-foreground/80">
              口径：INT4/Q4 量化按激进开销档估算；「能部署」指权重+开销+单路 KV 放得下，实际吞吐受带宽限制。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
