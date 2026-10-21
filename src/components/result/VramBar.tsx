import { fmtGB } from '@/lib/search';
import { usableVramGB } from '@/lib/vram';
import type { GpuSpec } from '@/data/gpus';

interface VramBarProps {
  weights: number;
  kv: number;
  overhead: number;
  total: number;
  gpu: GpuSpec;
  className?: string;
}

export default function VramBar({ weights, kv, overhead, total, gpu, className }: VramBarProps) {
  const wp = (weights / total) * 100;
  const kp = (kv / total) * 100;
  const op = 100 - wp - kp;

  return (
    <div className={className}>
      <div className="mb-2 flex items-center justify-between text-xs text-muted-foreground">
        <span className="font-medium">单卡显存构成（典型档）</span>
        <span className="tabular-nums">{fmtGB(total)} GB / 可用 {usableVramGB(gpu).toFixed(1)} GB</span>
      </div>
      <div
        className="flex h-4 w-full overflow-hidden rounded-full bg-muted"
        role="img"
        aria-label={`显存构成：权重 ${wp.toFixed(0)}%，KV ${kp.toFixed(0)}%，开销 ${op.toFixed(0)}%`}
      >
        <div className="bg-primary transition-all duration-slow" style={{ width: `${wp}%` }} title={`权重 ${fmtGB(weights)}GB`} />
        <div className="bg-primary/50 transition-all duration-slow" style={{ width: `${kp}%` }} title={`KV 缓存 ${fmtGB(kv)}GB`} />
        <div className="bg-muted-foreground/30 transition-all duration-slow" style={{ width: `${op}%` }} title={`开销 ${fmtGB(overhead)}GB`} />
      </div>
      <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted-foreground">
        <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-full bg-primary" /> 权重 {fmtGB(weights)} GB</span>
        <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-full bg-primary/50" /> KV 缓存 {fmtGB(kv)} GB</span>
        <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-full bg-muted-foreground/30" /> 运行时开销 {fmtGB(overhead)} GB</span>
      </div>
    </div>
  );
}
