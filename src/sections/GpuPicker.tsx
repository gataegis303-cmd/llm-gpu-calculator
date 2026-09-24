import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command';
import { GPUS, VENDOR_LABELS, type GpuSpec } from '@/data/gpus';
import { deployableModels } from '@/lib/fit';
import { ChevronsUpDown } from 'lucide-react';
import { cn } from '@/lib/utils';
import InfoTip from '@/components/info-tip';

interface Props {
  selected: GpuSpec | null;
  onSelect: (g: GpuSpec) => void;
}

const GROUPS: Array<{ key: GpuSpec['vendor']; heading: string }> = [
  { key: 'nvidia', heading: 'NVIDIA' },
  { key: 'amd', heading: 'AMD' },
  { key: 'ascend', heading: '华为昇腾' },
  { key: 'apple', heading: 'Apple M 系列' },
];

export default function GpuPicker({ selected, onSelect }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-1.5 text-base">③ 选择硬件 <InfoTip text="内置 55+ 款设备的官方规格（显存 / 带宽 / 稠密算力 / 互联 / 每节点卡数）；统一内存设备（Apple、AMD AI Max、DGX Spark）按可用占比折算实际可分配显存。" /></CardTitle>
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <button
              type="button"
              role="combobox"
              aria-expanded={open}
              className="mt-2 flex h-10 w-full items-center justify-between rounded-md border border-border bg-background px-3 py-2 text-sm hover:bg-muted"
            >
              <span className={cn('truncate', !selected && 'text-muted-foreground')}>
                {selected ? `${selected.name} · ${selected.vramGB}GB` : '搜索或选择 GPU…'}
              </span>
              <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
            </button>
          </PopoverTrigger>
          <PopoverContent className="w-[420px] p-0" align="start">
            <Command>
              <CommandInput placeholder="搜索 GPU 型号…" />
              <CommandList className="max-h-80">
                <CommandEmpty>没有匹配的 GPU</CommandEmpty>
                {GROUPS.map(({ key, heading }) => (
                  <CommandGroup key={key} heading={heading}>
                    {GPUS.filter((g) => g.vendor === key).map((g) => (
                      <CommandItem
                        key={g.id}
                        value={`${g.name} ${VENDOR_LABELS[key]}`}
                        onSelect={() => {
                          onSelect(g);
                          setOpen(false);
                        }}
                      >
                        <div className="flex w-full items-center justify-between gap-2">
                          <span className="truncate">
                            {g.name}
                            <span className="ml-2 text-xs text-muted-foreground">
                              {g.vramGB}GB · {g.memBandwidthGBs}GB/s
                            </span>
                          </span>
                          {selected?.id === g.id && <span className="shrink-0 text-[10px] text-primary">已选</span>}
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
      <CardContent className="border-t bg-muted/40">
        {selected ? (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="secondary">{VENDOR_LABELS[selected.vendor]}</Badge>
              <Badge variant="outline">{selected.memType}</Badge>
              <Badge variant="outline">{selected.memBandwidthGBs} GB/s</Badge>
              <Badge variant="outline">稠密 BF16 {selected.bf16Tflops} TFLOPS</Badge>
              {selected.fp8Tflops && <Badge variant="outline">FP8 {selected.fp8Tflops} TFLOPS</Badge>}
              <Badge variant="outline">{selected.interconnect}</Badge>
              <Badge variant="outline">{selected.gpusPerNode} 卡/节点</Badge>
            </div>
            <div className="mt-2 text-xs text-muted-foreground">
              来源：
              <a href={selected.sourceUrl.startsWith('http') ? selected.sourceUrl : '#'} target="_blank" rel="noreferrer" className="underline">
                {selected.sourceUrl}
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
  const { full, quantized, usableGB } = deployableModels(gpu);
  const show = (list: typeof full, label: string) => {
    if (list.length === 0) return null;
    const top = list.slice(0, 10);
    return (
      <div className="mt-2">
        <div className="text-xs font-medium text-muted-foreground">{label}（{list.length} 款）</div>
        <div className="mt-1 flex flex-wrap gap-1">
          {top.map((r) => (
            <span key={r.model.id} title={`${r.precisionId.toUpperCase()} · 需 ${r.totalGB.toFixed(1)}GB · 剩余显存约可扩至 ${r.headroomConcurrency} 路并发`} className="rounded border border-border px-1.5 py-0.5 text-[11px] text-muted-foreground">
              {r.model.name} <span className="text-[10px]">{r.model.paramsB >= 1000 ? `${(r.model.paramsB / 1000).toFixed(1)}T` : `${r.model.paramsB.toFixed(0)}B`}</span>
            </span>
          ))}
          {list.length > top.length && <span className="px-1 py-0.5 text-[11px] text-muted-foreground">…等 {list.length} 款</span>}
        </div>
      </div>
    );
  };
  return (
    <details className="mt-3 rounded-md border bg-background/50 p-2.5">
      <summary className="cursor-pointer text-xs font-medium">
        这台设备能部署什么？（单并发 · 8K 上下文 · 可用 {usableGB.toFixed(0)}GB）
      </summary>
      <div className="mt-1">
        {show(full, '高精度档（BF16 / 原生 FP8）')}
        {show(quantized, '量化档（INT4 / Q4）')}
        {full.length === 0 && quantized.length === 0 && (
          <div className="mt-1 text-xs text-muted-foreground">当前内置库中没有能装下的模型（显存不足）</div>
        )}
        <p className="mt-2 text-[10px] text-muted-foreground">
          口径：INT4/Q4 量化按激进开销档估算；「能部署」指权重+开销+单路 KV 放得下，实际吞吐受带宽限制。
        </p>
      </div>
    </details>
  );
}
