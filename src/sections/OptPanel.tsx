// ⑤ 性能优化选项（按所选硬件自动判定可用性）
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import InfoTip from '@/components/info-tip';
import { hardwareOptSupport, type Optimizations } from '@/lib/optimizations';
import type { GpuSpec } from '@/data/gpus';
import { cn } from '@/lib/utils';

interface Props {
  gpu: GpuSpec | null;
  opts: Optimizations;
  onOptsChange: (o: Optimizations) => void;
}

export default function OptPanel({ gpu, opts, onOptsChange }: Props) {
  const sup = hardwareOptSupport(gpu);
  const gpuName = gpu ? gpu.name : null;

  const row = (
    label: string,
    tip: string,
    controls: React.ReactNode,
    opts_: { disabled?: boolean; reason?: string; on?: boolean; toggle?: (v: boolean) => void; id?: string },
  ) => (
    <div className={cn('flex items-center justify-between gap-2 rounded-md border p-2.5', opts_.disabled && 'opacity-50')}>
      <div className="flex items-center gap-1 text-xs font-medium">
        {label} <InfoTip text={opts_.disabled && opts_.reason ? `${tip}（${opts_.reason}）` : tip} />
      </div>
      <div className="flex items-center gap-2">
        {controls}
        {opts_.id && (
          <Switch
            checked={!!opts_.on}
            disabled={!!opts_.disabled}
            onCheckedChange={(v) => opts_.toggle?.(v)}
            aria-label={opts_.id}
          />
        )}
      </div>
    </div>
  );

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-1.5 text-base">
          性能优化选项
          <InfoTip text="推理服务常见优化，按所选硬件自动判定可用性：分块预填充改善并发下 TTFT（解码轻微上浮）；KV FP8 量化使 KV 字节减半（无 FP8 的卡可用 INT8）；热缓存（内存）命中免重复预填充；冷缓存（SSD）扩展 KV 容量，检索延迟计入 TTFT。" />
        </CardTitle>
        {gpuName && <p className="text-xs text-muted-foreground">已按「{gpuName}」适配可用性</p>}
      </CardHeader>
      <CardContent className="grid gap-2">
        {row(
          '分块预填充',
          '预填充块与解码步交错：每步处理一个块而非阻塞解码，降低并发请求的 TTFT；解码步被分块挤占，TPOT 轻微上浮约 5%。vLLM/SGLang 默认开启，llama.cpp 由 ubatch 承担。',
          <Select value={String(opts.chunkSize)} onValueChange={(v) => onOptsChange({ ...opts, chunkSize: Number(v), chunkedPrefill: true })}>
            <SelectTrigger className="h-8 w-28 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              {[512, 1024, 2048, 4096].map((n) => <SelectItem key={n} value={String(n)}>{n >= 1024 ? n / 1024 + 'K' : n} /块</SelectItem>)}
            </SelectContent>
          </Select>,
          { on: opts.chunkedPrefill, toggle: (v) => onOptsChange({ ...opts, chunkedPrefill: v }), id: '分块预填充' },
        )}
        {row(
          sup.kvFp8 === 'int8' ? 'KV INT8 量化（该卡无 FP8）' : 'KV FP8 量化',
          'KV 缓存按 8bit 存储：字节数减半，同显存可容纳双倍并发或上下文，解码读取 KV 变快；有轻微精度代价。Hopper/Blackwell/Ada 与 MI300 原生 FP8；Ampere（A100/A30/T4）可用 INT8 KV；V100 不支持；Apple 走 MLX q8/q4 KV。',
          null,
          {
            disabled: sup.kvFp8 === 'no',
            reason: sup.kvFp8 === 'no' ? `${gpuName ?? '该卡'}无 8bit KV 支持` : undefined,
            on: opts.kvFp8,
            toggle: (v) => onOptsChange({ ...opts, kvFp8: v }),
            id: 'KV 量化',
          },
        )}
        {row(
          '热缓存（内存）',
          '频繁访问的 KV/前缀块驻留 CPU 内存：命中部分免重复预填充，TTFT 下降；同时作为 KV 卸载层扩展容量。独立显存服务器适用（LMCache / vLLM cpu-offload / Mooncake）；统一内存设备（Apple / GB10 / AI Max）内存本就同一池，无独立内存层可分层。',
          <div className="flex items-center gap-2">
            <Select value={String(opts.hotCache.sizeGB)} onValueChange={(v) => onOptsChange({ ...opts, hotCache: { ...opts.hotCache, sizeGB: Number(v), enabled: true } })}>
              <SelectTrigger className="h-8 w-24 text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>{[64, 128, 256, 512, 1024].map((n) => <SelectItem key={n} value={String(n)}>{n >= 1024 ? '1TB' : n + 'GB'}</SelectItem>)}</SelectContent>
            </Select>
            <Select value={String(opts.hotCache.hitRate)} onValueChange={(v) => onOptsChange({ ...opts, hotCache: { ...opts.hotCache, hitRate: Number(v), enabled: true } })}>
              <SelectTrigger className="h-8 w-16 text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="0.2">低</SelectItem><SelectItem value="0.4">中</SelectItem><SelectItem value="0.6">高</SelectItem>
              </SelectContent>
            </Select>
          </div>,
          {
            disabled: sup.hotTier === 'no',
            reason: sup.hotTier === 'no' ? '统一内存设备无独立主机内存层' : undefined,
            on: opts.hotCache.enabled,
            toggle: (v) => onOptsChange({ ...opts, hotCache: { ...opts.hotCache, enabled: v } }),
            id: '热缓存',
          },
        )}
        {row(
          '冷缓存（SSD）',
          '被驱逐的 KV 块持久化到 SSD：大幅扩展可服务上下文总量；命中冷层时按 SSD 带宽回读，延迟计入 TTFT。dGPU 服务器有成熟方案（LMCache disk backend / Mooncake）；统一内存设备工具链支持有限，标注仅供参考。',
          <div className="flex items-center gap-2">
            <Select value={String(opts.coldCache.sizeGB)} onValueChange={(v) => onOptsChange({ ...opts, coldCache: { ...opts.coldCache, sizeGB: Number(v), enabled: true } })}>
              <SelectTrigger className="h-8 w-24 text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>{[512, 1024, 2048, 4096].map((n) => <SelectItem key={n} value={String(n)}>{n >= 1024 ? n / 1024 + 'TB' : n + 'GB'}</SelectItem>)}</SelectContent>
            </Select>
            <Select value={String(opts.coldCache.ssdGBs)} onValueChange={(v) => onOptsChange({ ...opts, coldCache: { ...opts.coldCache, ssdGBs: Number(v), enabled: true } })}>
              <SelectTrigger className="h-8 w-20 text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="3">SATA</SelectItem><SelectItem value="7">NVMe</SelectItem>
              </SelectContent>
            </Select>
          </div>,
          {
            disabled: sup.coldTier === 'no',
            reason: sup.coldTier === 'limited' ? undefined : undefined,
            on: opts.coldCache.enabled,
            toggle: (v) => onOptsChange({ ...opts, coldCache: { ...opts.coldCache, enabled: v } }),
            id: '冷缓存',
          },
        )}
        {sup.hotTier === 'no' && (
          <p className="text-[10px] text-muted-foreground">
            当前为统一内存设备：显存与系统内存同一池，无独立内存卸载层；KV FP8 量化与 SSD 冷缓存仍可用。
          </p>
        )}
      </CardContent>
    </Card>
  );
}
