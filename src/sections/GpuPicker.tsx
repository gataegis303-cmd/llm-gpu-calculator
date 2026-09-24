import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command';
import { GPUS, VENDOR_LABELS, type GpuSpec } from '@/data/gpus';
import { ChevronsUpDown } from 'lucide-react';
import { cn } from '@/lib/utils';

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
        <CardTitle className="text-base">③ 选择硬件</CardTitle>
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
          </>
        ) : (
          <div className="py-1 text-sm text-muted-foreground">尚未选择硬件</div>
        )}
      </CardContent>
    </Card>
  );
}
