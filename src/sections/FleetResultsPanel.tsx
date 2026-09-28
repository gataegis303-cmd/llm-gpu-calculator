import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { runFleetAllBands, type FleetEntry } from '@/lib/fleet';
import { fmtGB, fmtTok, fmtMs } from '@/lib/search';
import { usableVramGB } from '@/lib/vram';
import type { Optimizations } from '@/lib/optimizations';
import type { GpuSpec } from '@/data/gpus';
import InfoTip from '@/components/info-tip';

const FLEET_CAP = 6;

interface Props {
  gpu: GpuSpec;
  entries: FleetEntry[];
  devices: number;
  deployMode: 'min' | 'given';
  onDeployMode: (m: 'min' | 'given') => void;
  onDevices: (n: number) => void;
  opts: Optimizations;
}

const BAND_KEYS = ['conservative', 'typical', 'aggressive'] as const;
const BAND_LABEL: Record<string, { name: string; cls: string; tip: string }> = {
  conservative: { name: '保守', cls: 'text-amber-600 dark:text-amber-400', tip: '带宽效率 0.7、开销 4GB + 权重 10%。' },
  typical: { name: '典型', cls: 'text-primary', tip: '带宽效率 0.8、开销 3GB + 权重 7.5%，对应 vLLM / SGLang 常见实测水平。' },
  aggressive: { name: '激进', cls: 'text-emerald-600 dark:text-emerald-400', tip: '带宽效率 0.9、开销 2GB + 权重 5%，理论接近上限。' },
};
const SEG_CLS = ['bg-primary', 'bg-sky-500', 'bg-emerald-500', 'bg-amber-500', 'bg-fuchsia-500', 'bg-cyan-600'];

function clampDevices(n: number): number {
  return Math.min(4096, Math.max(1, Number.isFinite(n) ? Math.round(n) : 1));
}

export default function FleetResultsPanel({ gpu, entries, devices, deployMode, onDeployMode, onDevices, opts }: Props) {
  const given = deployMode === 'given';
  const n = Math.max(1, Math.floor(devices));
  const bands = runFleetAllBands({ gpu, devices: n, entries, opts });
  const typical = bands.typical;

  const minNeed = Math.min(...BAND_KEYS.map((b) => bands[b].totalNeed));
  const maxNeed = Math.max(...BAND_KEYS.map((b) => bands[b].totalNeed));
  const deficit = typical.deficit;
  const idle = typical.idleDevices;
  const sumShare = typical.outcomes.reduce((s, o) => s + o.share, 0);

  const undeployed = typical.outcomes.filter((o) => o.floor === null);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex flex-wrap items-center gap-1.5 text-base">
            ④ 集群推演 — {gpu.name} · {entries.length} 个模型
            <InfoTip text="多模型共享集群：按「权重保底 + 需求比例」把 N 张卡分给各模型，再逐模型推演。三档对应不同效率与开销假设；Σ 保底超过 N 时报缺口，不做软分配。均为估算值。" />
          </CardTitle>
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <Tabs value={deployMode} onValueChange={(v) => onDeployMode(v === 'given' ? 'given' : 'min')}>
              <TabsList className="h-7">
                <TabsTrigger value="min" className="h-6 px-2.5 text-xs">按最少卡数</TabsTrigger>
                <TabsTrigger value="given" className="h-6 px-2.5 text-xs">按实际卡数</TabsTrigger>
              </TabsList>
            </Tabs>
            {given && (
              <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
                集群持有
                <Input
                  type="number"
                  min={1}
                  max={4096}
                  className="h-7 w-24 text-xs"
                  value={devices}
                  onChange={(e) => onDevices(clampDevices(Number(e.target.value)))}
                />
                张 {gpu.name}
              </label>
            )}
          </div>
          <p className="text-xs text-muted-foreground">
            {given
              ? `把 ${n} 张卡分给 ${entries.length} 个模型：保底合计 ${typical.outcomes.reduce((s, o) => s + (o.floor ?? 0), 0)} 张，需求合计 ${typical.totalNeed} 张`
              : `分别满足 ${entries.length} 个模型的并发与上下文所需的最少设备数（三档区间）`}
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          {entries.length === 0 ? (
            <div className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
              先在左侧 ① 卡片添加模型（可多选，最多 {FLEET_CAP} 个）
            </div>
          ) : (
            <>
              {undeployed.length > 0 && (
                <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive">
                  {undeployed.map((o) => o.entry.model.name).join('、')} 在 {gpu.name} 上无法装载（权重超出 TP×PP 枚举上限），已排除出分配
                </div>
              )}
              {given && deficit > 0 && (
                <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive">
                  {n} 张卡不足以放下全部 {entries.length} 个模型：保底合计 {typical.outcomes.reduce((s, o) => s + (o.floor ?? 0), 0)} 张
                  （{typical.outcomes.filter((o) => o.floor !== null).map((o) => `${o.entry.model.name} 需 ${o.floor}`).join(' · ')}），差 {deficit} 张。请增加卡数或移除模型。
                </div>
              )}

              {given ? (
                <>
                  <div className="rounded-lg border bg-muted/40 p-4">
                    <div className="text-3xl font-semibold tracking-tight">
                      {deficit > 0 ? '—' : sumShare.toLocaleString()}
                      <span className="text-lg font-normal text-muted-foreground"> 张卡已分配 · 闲置 {idle} 张</span>
                    </div>
                    <p className="mt-1.5 text-xs text-muted-foreground">
                      分配规则：每模型保底（权重装载下界）直配，剩余按需求比例 need / Σ need 分配
                    </p>
                  </div>
                  {deficit === 0 && (
                    <div>
                      <div className="mb-1.5 text-xs text-muted-foreground">分卡比例（典型档）</div>
                      <div className="flex h-4 w-full overflow-hidden rounded-full bg-muted" role="img" aria-label={`分卡比例：共 ${n} 张，闲置 ${idle} 张`}>
                        {typical.outcomes.map((o, i) => (
                          <div
                            key={o.entry.id}
                            className={SEG_CLS[i % SEG_CLS.length]}
                            style={{ width: `${(o.share / n) * 100}%` }}
                            title={`${o.entry.model.name}：${o.share} 张`}
                          />
                        ))}
                        {idle > 0 && <div className="bg-muted-foreground/30" style={{ width: `${(idle / n) * 100}%` }} title={`闲置 ${idle} 张`} />}
                      </div>
                      <div className="mt-1.5 flex flex-wrap gap-3 text-[11px] text-muted-foreground">
                        {typical.outcomes.map((o, i) => (
                          <span key={o.entry.id}>
                            <span className={`mr-1 inline-block h-2 w-2 rounded-full align-middle ${SEG_CLS[i % SEG_CLS.length]}`} />
                            {o.entry.model.name} {o.share} 张
                          </span>
                        ))}
                        {idle > 0 && <span>□ 闲置 {idle} 张</span>}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="rounded-lg border bg-muted/40 p-4">
                  <div className="text-3xl font-semibold tracking-tight">
                    {minNeed === maxNeed ? minNeed : `${minNeed}–${maxNeed}`}{' '}
                    <span className="text-lg font-normal text-muted-foreground">张 {gpu.name}（全部模型合计）</span>
                  </div>
                  <p className="mt-1.5 text-xs text-muted-foreground">
                    逐模型最少卡数见下方明细；切换「按实际卡数」可查看 N 张卡如何分配
                  </p>
                </div>
              )}

              <div className="space-y-3">
                {typical.outcomes.map((o, i) => {
                  const seg = SEG_CLS[i % SEG_CLS.length];
                  return (
                    <div key={o.entry.id} className="rounded-lg border p-3">
                      <div className="mb-2 flex flex-wrap items-center gap-2">
                        <span className={`inline-block h-2.5 w-2.5 rounded-full ${o.floor === null ? 'bg-muted-foreground/30' : seg}`} />
                        <span className="text-sm font-medium">{o.entry.model.name}</span>
                        <Badge variant="outline" className="text-[10px]">
                          {o.entry.precisionId.toUpperCase()} · C={o.entry.concurrency} · ctx {o.entry.avgCtxTokens.toLocaleString()}
                        </Badge>
                        {o.floor !== null && given && deficit === 0 && (
                          <Badge variant="secondary" className="text-[10px]">分到 {o.share} 张</Badge>
                        )}
                        {o.floor !== null && !given && (
                          <Badge variant="secondary" className="text-[10px]">最少 {o.need ?? o.floor} 张</Badge>
                        )}
                        {o.floor === null && <Badge variant="destructive" className="text-[10px]">无法装载</Badge>}
                      </div>
                      {o.result ? (
                        <div className="grid grid-cols-3 gap-2">
                          {BAND_KEYS.map((band) => {
                            const r = bands[band].outcomes[i].result;
                            const meta = BAND_LABEL[band];
                            const atCap = !!r && !r.feasibleAtC;
                            if (!r) {
                              return (
                                <div key={band} className="rounded-lg border p-2.5">
                                  <span className="text-xs font-medium ${meta.cls}">{meta.name}档</span>
                                  <p className="mt-1.5 text-[11px] text-muted-foreground">该档位下无法承载（保底或并发超限）</p>
                                </div>
                              );
                            }
                            return (
                              <div key={band} className="rounded-lg border p-2.5">
                                <div className="flex items-center justify-between">
                                  <span className={`flex items-center gap-1 text-xs font-medium ${meta.cls}`}>{meta.name}档 <InfoTip text={meta.tip} /></span>
                                </div>
                                <dl className="mt-1.5 space-y-1 text-xs">
                                  <div className="flex justify-between"><dt className="text-muted-foreground">方案</dt><dd className="font-medium">{r.plan.tp}×TP · {r.plan.pp}×PP · {r.plan.dp}×DP</dd></div>
                                  <div className="flex justify-between"><dt className="text-muted-foreground">用卡</dt><dd className="font-medium">{r.plan.totalDevices}{r.plan.idleDevices > 0 ? `（闲置 ${r.plan.idleDevices}）` : ''}</dd></div>
                                  <div className="flex justify-between"><dt className="text-muted-foreground">TPOT</dt><dd className="font-medium">{fmtMs(r.throughput.tpotMs)}</dd></div>
                                  <div className="flex justify-between"><dt className="text-muted-foreground">TTFT</dt><dd className="font-medium">{fmtMs(r.throughput.ttftMs)}</dd></div>
                                  <div className="flex justify-between"><dt className="text-muted-foreground">单用户</dt><dd className="font-medium">{fmtTok(r.throughput.perUserTokPerSec)} tok/s</dd></div>
                                  <div className="flex justify-between"><dt className="text-muted-foreground">分片总吞吐</dt><dd className="font-medium">{fmtTok(r.clusterAggregateTokPerSec)} tok/s</dd></div>
                                  <div className="flex justify-between"><dt className="text-muted-foreground">单卡占用</dt><dd className="font-medium">{fmtGB(r.vramPerDevice.totalGB)} / {usableVramGB(gpu).toFixed(0)} GB</dd></div>
                                  <div className="flex justify-between"><dt className="text-muted-foreground">最大并发</dt><dd className="font-medium">{Number.isFinite(r.capacity.maxConcurrency) ? `${r.capacity.maxConcurrency.toLocaleString()} 路` : '—'}</dd></div>
                                </dl>
                                {atCap && (
                                  <p className="mt-1.5 text-[11px] text-amber-600 dark:text-amber-400">
                                    并发 {o.entry.concurrency} 路超出份额容量，按满载 {r.capacity.maxConcurrency.toLocaleString()} 路计
                                  </p>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      ) : given ? (
                        <p className="text-xs text-destructive">
                          {deficit > 0
                            ? `集群缺口 ${deficit} 张：该模型保底 ${o.floor ?? '—'} 张，缺口场景下不分配`
                            : `该硬件无法装载此模型（权重装载下界 ${o.floor ?? '—'} 张）`}
                        </p>
                      ) : (
                        <p className="text-xs text-muted-foreground">最少卡数见上方徽章；切「按实际卡数」查看分配</p>
                      )}
                    </div>
                  );
                })}
              </div>

              <div className="flex flex-wrap gap-1.5">
                <Badge variant="outline" className="text-[10px]">屋顶线估算：吞吐随 η 取值浮动，非实测</Badge>
                <Badge variant="outline" className="text-[10px]">各模型独立 KV，无跨模型共享缓存</Badge>
                <Badge variant="outline" className="text-[10px]">PP 跨节点开销未计入带宽</Badge>
                <Badge variant="outline" className="text-[10px]">保底 = 权重装载下界；份额 ≥ 保底</Badge>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
