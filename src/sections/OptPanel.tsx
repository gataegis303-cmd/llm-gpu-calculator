// 性能优化选项（按所选硬件自动判定可用性）
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import InfoTip from '@/components/info-tip';
import { hardwareOptSupport, type Optimizations } from '@/lib/optimizations';
import type { GpuSpec } from '@/data/gpus';
import { cn } from '@/lib/utils';
import { Zap, MemoryStick, Flame, Snowflake } from 'lucide-react';

interface Props {
  gpu: GpuSpec | null;
  opts: Optimizations;
  onOptsChange: (o: Optimizations) => void;
}

interface SettingRowProps {
  icon: React.ReactNode;
  iconBg: string;
  label: string;
  tip: string;
  disabled?: boolean;
  disabledReason?: string;
  children?: React.ReactNode;
  switchId?: string;
  switchOn?: boolean;
  onSwitch?: (v: boolean) => void;
}

function SettingRow({
  icon,
  iconBg,
  label,
  tip,
  disabled,
  disabledReason,
  children,
  switchId,
  switchOn,
  onSwitch,
}: SettingRowProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-start gap-3 rounded-xl border border-border/60 bg-card p-3 transition-colors sm:flex-row sm:flex-wrap sm:items-center sm:justify-between',
        disabled && 'bg-secondary/40 opacity-60',
      )}
    >
      <div className="flex min-w-0 items-center gap-3">
        <div className={cn('flex h-9 w-9 shrink-0 items-center justify-center rounded-lg', iconBg)}>
          {icon}
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-1 text-meta">
            <span className="truncate">{label}</span>
            <InfoTip text={disabled && disabledReason ? `${tip}（${disabledReason}）` : tip} />
          </div>
          {disabled && disabledReason && (
            <p className="mt-0.5 text-meta">{disabledReason}</p>
          )}
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {children}
        {switchId && (
          <Switch
            checked={!!switchOn}
            disabled={!!disabled}
            onCheckedChange={(v) => onSwitch?.(v)}
            aria-label={switchId}
            className="data-[state=checked]:bg-primary"
          />
        )}
      </div>
    </div>
  );
}

export default function OptPanel({ gpu, opts, onOptsChange }: Props) {
  const sup = hardwareOptSupport(gpu);
  const gpuName = gpu ? gpu.name : null;

  return (
    <Card className="section-card flex-1 gap-0">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Zap className="h-4 w-4 text-primary" />
          <CardTitle className="text-base font-semibold tracking-tight">性能优化选项</CardTitle>
          <InfoTip text="推理服务常见优化，按所选硬件自动判定可用性：分块预填充改善并发下 TTFT（解码轻微上浮）；KV FP8 量化使 KV 字节减半（无 FP8 的卡可用 INT8）；热缓存（内存）命中免重复预填充；冷缓存（SSD）扩展 KV 容量，检索延迟计入 TTFT。" />
        </div>
        {gpuName && <p className="text-meta">已按「{gpuName}」适配可用性</p>}
      </CardHeader>
      <CardContent className="grid gap-2.5">
        <SettingRow
          icon={<Zap className="h-4 w-4 text-amber-600 dark:text-amber-400" />}
          iconBg="bg-amber-500/10"
          label="分块预填充"
          tip="预填充块与解码步交错：每步处理一个块而非阻塞解码，降低并发请求的 TTFT；解码步被分块挤占，TPOT 轻微上浮约 5%。vLLM/SGLang 默认开启，llama.cpp 由 ubatch 承担。"
          switchId="分块预填充"
          switchOn={opts.chunkedPrefill}
          onSwitch={(v) => onOptsChange({ ...opts, chunkedPrefill: v })}
        >
          <Select value={String(opts.chunkSize)} onValueChange={(v) => onOptsChange({ ...opts, chunkSize: Number(v), chunkedPrefill: true })}>
            <SelectTrigger className="h-8 min-w-[5rem] rounded-lg text-value text-xs transition-colors focus:border-primary focus:ring-primary">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="rounded-xl">
              {[512, 1024, 2048, 4096].map((n) => <SelectItem key={n} value={String(n)} className="rounded-lg">{n >= 1024 ? n / 1024 + 'K' : n} /块</SelectItem>)}
            </SelectContent>
          </Select>
        </SettingRow>

        <SettingRow
          icon={<MemoryStick className="h-4 w-4 text-sky-600 dark:text-sky-400" />}
          iconBg="bg-sky-500/10"
          label={sup.kvFp8 === 'int8' ? 'KV INT8 量化（该卡无 FP8）' : 'KV FP8 量化'}
          tip="KV 缓存按 8bit 存储：字节数减半，同显存可容纳双倍并发或上下文，解码读取 KV 变快；有轻微精度代价。Hopper/Blackwell/Ada 与 MI300 原生 FP8；Ampere（A100/A30/T4）可用 INT8 KV；V100 不支持；Apple 走 MLX q8/q4 KV。"
          disabled={sup.kvFp8 === 'no'}
          disabledReason={sup.kvFp8 === 'no' ? `${gpuName ?? '该卡'}无 8bit KV 支持` : undefined}
          switchId="KV 量化"
          switchOn={opts.kvFp8}
          onSwitch={(v) => onOptsChange({ ...opts, kvFp8: v })}
        />

        <SettingRow
          icon={<Flame className="h-4 w-4 text-orange-600 dark:text-orange-400" />}
          iconBg="bg-orange-500/10"
          label="热缓存（内存）"
          tip="频繁访问的 KV/前缀块驻留 CPU 内存：命中部分免重复预填充，TTFT 下降；同时作为 KV 卸载层扩展容量。独立显存服务器适用（LMCache / vLLM cpu-offload / Mooncake）；统一内存设备（Apple / GB10 / AI Max）内存本就同一池，无独立内存层可分层。"
          disabled={sup.hotTier === 'no'}
          disabledReason={sup.hotTier === 'no' ? '统一内存设备无独立主机内存层' : undefined}
          switchId="热缓存"
          switchOn={opts.hotCache.enabled}
          onSwitch={(v) => onOptsChange({ ...opts, hotCache: { ...opts.hotCache, enabled: v } })}
        >
          <div className="flex items-center gap-2">
            <Select value={String(opts.hotCache.sizeGB)} onValueChange={(v) => onOptsChange({ ...opts, hotCache: { ...opts.hotCache, sizeGB: Number(v), enabled: true } })}>
              <SelectTrigger className="h-8 min-w-[5rem] rounded-lg text-value text-xs transition-colors focus:border-primary focus:ring-primary">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="rounded-xl">{[64, 128, 256, 512, 1024].map((n) => <SelectItem key={n} value={String(n)} className="rounded-lg">{n >= 1024 ? '1TB' : n + 'GB'}</SelectItem>)}</SelectContent>
            </Select>
            <Select value={String(opts.hotCache.hitRate)} onValueChange={(v) => onOptsChange({ ...opts, hotCache: { ...opts.hotCache, hitRate: Number(v), enabled: true } })}>
              <SelectTrigger className="h-8 min-w-[5rem] rounded-lg text-value text-xs transition-colors focus:border-primary focus:ring-primary">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="rounded-xl">
                <SelectItem value="0.2" className="rounded-lg">低</SelectItem>
                <SelectItem value="0.4" className="rounded-lg">中</SelectItem>
                <SelectItem value="0.6" className="rounded-lg">高</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </SettingRow>

        <SettingRow
          icon={<Snowflake className="h-4 w-4 text-cyan-600 dark:text-cyan-400" />}
          iconBg="bg-cyan-500/10"
          label="冷缓存（SSD）"
          tip="被驱逐的 KV 块持久化到 SSD：大幅扩展可服务上下文总量；命中冷层时按 SSD 带宽回读，延迟计入 TTFT。dGPU 服务器有成熟方案（LMCache disk backend / Mooncake）；统一内存设备工具链支持有限，标注仅供参考。"
          disabled={sup.coldTier === 'no'}
          switchId="冷缓存"
          switchOn={opts.coldCache.enabled}
          onSwitch={(v) => onOptsChange({ ...opts, coldCache: { ...opts.coldCache, enabled: v } })}
        >
          <div className="flex items-center gap-2">
            <Select value={String(opts.coldCache.sizeGB)} onValueChange={(v) => onOptsChange({ ...opts, coldCache: { ...opts.coldCache, sizeGB: Number(v), enabled: true } })}>
              <SelectTrigger className="h-8 min-w-[5rem] rounded-lg text-value text-xs transition-colors focus:border-primary focus:ring-primary">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="rounded-xl">{[512, 1024, 2048, 4096].map((n) => <SelectItem key={n} value={String(n)} className="rounded-lg">{n >= 1024 ? n / 1024 + 'TB' : n + 'GB'}</SelectItem>)}</SelectContent>
            </Select>
            <Select value={String(opts.coldCache.ssdGBs)} onValueChange={(v) => onOptsChange({ ...opts, coldCache: { ...opts.coldCache, ssdGBs: Number(v), enabled: true } })}>
              <SelectTrigger className="h-8 min-w-[5rem] rounded-lg text-value text-xs transition-colors focus:border-primary focus:ring-primary">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="rounded-xl">
                <SelectItem value="3" className="rounded-lg">SATA</SelectItem>
                <SelectItem value="7" className="rounded-lg">NVMe</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </SettingRow>

        {sup.hotTier === 'no' && (
          <p className="text-[10px] leading-relaxed text-muted-foreground/80">
            当前为统一内存设备：显存与系统内存同一池，无独立内存卸载层；KV FP8 量化与 SSD 冷缓存仍可用。
          </p>
        )}
      </CardContent>
    </Card>
  );
}
