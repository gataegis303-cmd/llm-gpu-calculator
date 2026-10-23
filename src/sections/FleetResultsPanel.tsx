import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { runFleetAllBands, fleetEntrySummary, fleetKindOf, fleetKindLabel, FLEET_KIND_ORDER, type FleetEntry } from '@/lib/fleet';
import { fmtGB, fmtTok, fmtMs } from '@/lib/search';
import { usableVramGB } from '@/lib/vram';
import type { Optimizations } from '@/lib/optimizations';
import type { GpuSpec } from '@/data/gpus';
import AnimatedNumber from '@/components/result/AnimatedNumber';
import BandCard, { type BandMeta } from '@/components/result/BandCard';
import Callout from '@/components/result/Callout';
import ResultHeader from '@/components/result/ResultHeader';
import { Empty, EmptyDescription, EmptyMedia } from '@/components/ui/empty';
import { Network, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

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
const BAND_LABEL: Record<string, BandMeta> = {
  conservative: { name: '保守', cls: 'text-amber-600 dark:text-amber-400', tip: '带宽效率 0.7、开销 4GB + 权重 10%。', dot: 'bg-amber-500' },
  typical: { name: '典型', cls: 'text-primary', tip: '带宽效率 0.8、开销 3GB + 权重 7.5%，对应 vLLM / SGLang 常见实测水平。', dot: 'bg-primary' },
  aggressive: { name: '激进', cls: 'text-emerald-600 dark:text-emerald-400', tip: '带宽效率 0.9、开销 2GB + 权重 5%，理论接近上限。', dot: 'bg-emerald-500' },
};
const SEG_CLS = ['bg-primary', 'bg-sky-500', 'bg-emerald-500', 'bg-amber-500', 'bg-fuchsia-500', 'bg-cyan-600'];

function clampDevices(n: number): number {
  return Math.min(4096, Math.max(1, Number.isFinite(n) ? Math.round(n) : 1));
}

function fmtParams(b: number): string {
  return b >= 1000 ? `${(b / 1000).toFixed(2)}T` : `${b.toFixed(1)}B`;
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
  const clusterTotal = typical.outcomes.reduce((s, o) => s + (o.result?.clusterAggregateTokPerSec ?? 0), 0);
  const summaries = new Map(entries.map((e) => [e.id, fleetEntrySummary(e, gpu, opts)]));

  const undeployed = typical.outcomes.filter((o) => o.floor === null);

  // 摘要表按模型种类分组
  const tableGroups = FLEET_KIND_ORDER
    .map((kind) => ({ kind, rows: typical.outcomes.filter((o) => fleetKindOf(o.entry.model) === kind) }))
    .filter((g) => g.rows.length > 0);

  return (
    <div className="flex h-full flex-col gap-5">
      <Card className="section-card">
        <ResultHeader
          icon={<Network className="h-4 w-4 text-primary" />}
          title={`推演结果 — ${gpu.name} · ${entries.length} 个模型`}
          infoTip="多模型共享集群：按「权重保底 + 需求比例」把 N 张卡分给各模型，再逐模型推演。摘要表与合计均为典型档口径；三档明细见下方逐模型卡片。均为估算值。"
          subtitle={
            given
              ? `把 ${n} 张卡分给 ${entries.length} 个模型：保底合计 ${typical.outcomes.reduce((s, o) => s + (o.floor ?? 0), 0)} 张，需求合计 ${typical.totalNeed} 张`
              : `分别满足 ${entries.length} 个模型的并发与上下文所需的最少设备数（三档区间）`
          }
          deployMode={deployMode}
          onDeployMode={onDeployMode}
          devices={devices}
          onDevices={(v) => onDevices(clampDevices(v))}
          gpuName={gpu.name}
          actualLabel="集群持有"
        />
        <CardContent className="space-y-5">
          {entries.length === 0 ? (
            <Empty className="min-h-[180px] rounded-2xl border border-dashed border-border/60 bg-secondary/30">
              <EmptyMedia variant="icon">
                <Network className="text-muted-foreground/60" />
              </EmptyMedia>
              <EmptyDescription>先在 ① 中添加模型（可多选，最多 {FLEET_CAP} 个）</EmptyDescription>
            </Empty>
          ) : (
            <>
              {undeployed.length > 0 && (
                <Callout variant="error" icon={<AlertCircle className="h-4 w-4" />}>
                  {undeployed.map((o) => o.entry.model.name).join('、')} 在 {gpu.name} 上无法装载（权重超出 TP×PP 枚举上限），已排除出分配
                </Callout>
              )}
              {given && deficit > 0 && (
                <Callout variant="error" icon={<AlertCircle className="h-4 w-4" />}>
                  {n} 张卡不足以放下全部 {entries.length} 个模型：保底合计 {typical.outcomes.reduce((s, o) => s + (o.floor ?? 0), 0)} 张
                  （{typical.outcomes.filter((o) => o.floor !== null).map((o) => `${o.entry.model.name} 需 ${o.floor}`).join(' · ')}），差 {deficit} 张。请增加卡数或移除模型。
                </Callout>
              )}

              <div className="rounded-2xl bg-secondary/30 p-5">
                {given ? (
                  deficit === 0 ? (
                    <>
                      <div className="metric-value">
                        <AnimatedNumber value={Math.round(clusterTotal)} formatter={(v) => v.toLocaleString()} />
                        <span className="metric-label ml-2"> tok/s · 全集群合计（{entries.length} 个模型分片之和，典型档）</span>
                      </div>
                      <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                        已分配 {sumShare.toLocaleString()} 张 · 闲置 {idle} 张 · 分卡规则：每模型保底（权重装载下界）直配，剩余按需求比例 need / Σ need 分配
                      </p>
                    </>
                  ) : (
                    <>
                      <div className="metric-value text-destructive">—</div>
                      <p className="mt-2 text-xs leading-relaxed text-muted-foreground">缺口 {deficit} 张，无法形成完整分配；各模型保底见下方摘要表</p>
                    </>
                  )
                ) : (
                  <>
                    <div className="metric-value">
                      {minNeed === maxNeed ? minNeed : `${minNeed}–${maxNeed}`}{' '}
                      <span className="metric-label">张 {gpu.name}（全部模型合计）</span>
                    </div>
                    <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                      逐模型最少卡数见下方摘要表；切换「按实际卡数」可查看 N 张卡如何分配、合计能跑多少 tok/s
                    </p>
                  </>
                )}
              </div>

              {given && deficit === 0 && (
                <div>
                  <div className="mb-2 text-xs font-medium text-muted-foreground">分卡比例（典型档）</div>
                  <div className="flex h-4 w-full overflow-hidden rounded-full bg-muted" role="img" aria-label={`分卡比例：共 ${n} 张，闲置 ${idle} 张`}>
                    {typical.outcomes.map((o, i) => (
                      <div
                        key={o.entry.id}
                        className={cn('transition-all duration-slow', SEG_CLS[i % SEG_CLS.length])}
                        style={{ width: `${(o.share / n) * 100}%` }}
                        title={`${o.entry.model.name}：${o.share} 张`}
                      />
                    ))}
                    {idle > 0 && <div className="bg-muted-foreground/25 transition-all duration-slow" style={{ width: `${(idle / n) * 100}%` }} title={`闲置 ${idle} 张`} />}
                  </div>
                  <div className="mt-2 flex flex-col gap-2 text-[11px] text-muted-foreground sm:flex-row sm:flex-wrap sm:gap-3">
                    {typical.outcomes.map((o, i) => (
                      <span key={o.entry.id} className="flex items-center gap-1">
                        <span className={cn('inline-block h-2 w-2 rounded-full', SEG_CLS[i % SEG_CLS.length])} />
                        {o.entry.model.name} <span className="tabular-nums">{o.share}</span> 张
                      </span>
                    ))}
                    {idle > 0 && <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full bg-muted-foreground/25" /> 闲置 <span className="tabular-nums">{idle}</span> 张</span>}
                  </div>
                </div>
              )}

              <div>
                <div className="mb-2 text-xs font-medium text-muted-foreground">
                  模型清单与吞吐摘要（典型档{given ? ' · 按当前分配' : ''}）
                </div>

                {/* md+ 摘要表 */}
                <div className="hidden md:block overflow-x-auto rounded-2xl border border-border/60">
                  <table className="w-full table-auto text-left text-xs">
                    <thead>
                      <tr className="border-b border-border/60 bg-secondary/50 text-[11px] text-muted-foreground">
                        <th className="px-3 py-2 font-medium">模型</th>
                        <th className="px-3 py-2 font-medium">显存占用</th>
                        <th className="px-3 py-2 font-medium">{given ? '卡数' : '最少卡数'}</th>
                        {given && <th className="px-3 py-2 font-medium">单用户</th>}
                        {given && <th className="px-3 py-2 font-medium">分片吞吐</th>}
                        {given && <th className="px-3 py-2 font-medium">最大并发</th>}
                      </tr>
                    </thead>
                    <tbody>
                      {tableGroups.map((g) => (
                        [
                          <tr key={`${g.kind}-head`} className="border-b border-border/60 bg-secondary/30">
                            <td colSpan={given ? 6 : 3} className="px-3 py-1.5 text-[11px] font-semibold text-muted-foreground">
                              {fleetKindLabel(g.kind)} · {g.rows.length} 个
                            </td>
                          </tr>,
                          ...g.rows.map((o) => {
                            const sum = summaries.get(o.entry.id)!;
                            const r = o.result;
                            return (
                              <tr key={o.entry.id} className="border-b border-border/60 last:border-b-0">
                                <td className="px-3 py-2">
                                  <span className="font-medium">{o.entry.model.name}</span>
                                  <span className="ml-1 text-[10px] text-muted-foreground">{fmtParams(o.entry.model.paramsB)}</span>
                                  {o.entry.model.isMoE && <span className="ml-1 text-[10px] font-medium text-amber-600 dark:text-amber-400">MoE</span>}
                                </td>
                                <td className="px-3 py-2 tabular-nums">{fmtGB(sum.totalGB)} GB</td>
                                {given ? (
                                  deficit > 0 ? (
                                    <td className="px-3 py-2 text-destructive">保底 {o.floor ?? '—'}</td>
                                  ) : (
                                    <td className="px-3 py-2 tabular-nums">
                                      {o.share}
                                      {o.idleDevices > 0 && <span className="text-[10px] text-muted-foreground">（闲置 {o.idleDevices}）</span>}
                                    </td>
                                  )
                                ) : (
                                  <td className="px-3 py-2">{o.floor === null ? <span className="text-destructive">无法装载</span> : (<span className="tabular-nums">{o.need ?? o.floor}</span>)}</td>
                                )}
                                {given && <td className="px-3 py-2 tabular-nums">{r ? `${fmtTok(r.throughput.perUserTokPerSec)}` : '—'}</td>}
                                {given && <td className="px-3 py-2 font-medium tabular-nums text-primary">{r ? `${fmtTok(r.clusterAggregateTokPerSec)}` : '—'}</td>}
                                {given && <td className="px-3 py-2 tabular-nums">{r && Number.isFinite(r.capacity.maxConcurrency) ? r.capacity.maxConcurrency.toLocaleString() : '—'}</td>}
                              </tr>
                            );
                          }),
                        ]
                      ))}
                      {given && deficit === 0 && (
                        <tr className="border-t border-border/60 bg-secondary/30 font-semibold">
                          <td className="px-3 py-2">全集群合计</td>
                          <td className="px-3 py-2 tabular-nums">{fmtGB(typical.outcomes.reduce((s, o) => s + (summaries.get(o.entry.id)?.totalGB ?? 0), 0))} GB</td>
                          <td className="px-3 py-2 tabular-nums">{sumShare}{idle > 0 ? ` + 闲置 ${idle}` : ''}</td>
                          <td className="px-3 py-2">—</td>
                          <td className="px-3 py-2 text-primary tabular-nums">{fmtTok(clusterTotal)}</td>
                          <td className="px-3 py-2">—</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>

                {/* <md 按模型分组的行卡片列表 */}
                <div className="md:hidden space-y-4">
                  {tableGroups.map((g) => (
                    <div key={g.kind} className="space-y-3">
                      <div className="text-xs font-semibold text-muted-foreground">
                        {fleetKindLabel(g.kind)} · {g.rows.length} 个
                      </div>
                      {g.rows.map((o) => {
                        const sum = summaries.get(o.entry.id)!;
                        const r = o.result;
                        return (
                          <div key={o.entry.id} className="rounded-2xl bg-secondary/30 p-4 text-xs">
                            <div className="mb-2 flex flex-wrap items-center gap-1.5">
                              <span className="font-medium text-foreground">{o.entry.model.name}</span>
                              <span className="text-[10px] text-muted-foreground">{fmtParams(o.entry.model.paramsB)}</span>
                              {o.entry.model.isMoE && <span className="text-[10px] font-medium text-amber-600 dark:text-amber-400">MoE</span>}
                            </div>
                            <dl className="grid grid-cols-2 gap-x-4 gap-y-2">
                              <div>
                                <dt className="text-muted-foreground">显存占用</dt>
                                <dd className="font-medium tabular-nums">{fmtGB(sum.totalGB)} GB</dd>
                              </div>
                              <div>
                                <dt className="text-muted-foreground">{given ? '卡数' : '最少卡数'}</dt>
                                <dd className="font-medium tabular-nums">
                                  {given ? (
                                    deficit > 0 ? (
                                      <span className="text-destructive">保底 {o.floor ?? '—'}</span>
                                    ) : (
                                      <>
                                        {o.share}
                                        {o.idleDevices > 0 && <span className="text-[10px] text-muted-foreground">（闲置 {o.idleDevices}）</span>}
                                      </>
                                    )
                                  ) : (
                                    o.floor === null ? <span className="text-destructive">无法装载</span> : <span className="tabular-nums">{o.need ?? o.floor}</span>
                                  )}
                                </dd>
                              </div>
                              {given && (
                                <div>
                                  <dt className="text-muted-foreground">单用户</dt>
                                  <dd className="font-medium tabular-nums">{r ? `${fmtTok(r.throughput.perUserTokPerSec)}` : '—'}</dd>
                                </div>
                              )}
                              {given && (
                                <div>
                                  <dt className="text-muted-foreground">分片吞吐</dt>
                                  <dd className="font-medium tabular-nums text-primary">{r ? `${fmtTok(r.clusterAggregateTokPerSec)}` : '—'}</dd>
                                </div>
                              )}
                              {given && (
                                <div>
                                  <dt className="text-muted-foreground">最大并发</dt>
                                  <dd className="font-medium tabular-nums">{r && Number.isFinite(r.capacity.maxConcurrency) ? r.capacity.maxConcurrency.toLocaleString() : '—'}</dd>
                                </div>
                              )}
                            </dl>
                          </div>
                        );
                      })}
                    </div>
                  ))}
                  {given && deficit === 0 && (
                    <div className="rounded-2xl bg-secondary/30 p-4 text-xs">
                      <div className="mb-2 font-semibold text-foreground">全集群合计</div>
                      <dl className="grid grid-cols-2 gap-x-4 gap-y-2">
                        <div>
                          <dt className="text-muted-foreground">显存占用</dt>
                          <dd className="font-medium tabular-nums">{fmtGB(typical.outcomes.reduce((s, o) => s + (summaries.get(o.entry.id)?.totalGB ?? 0), 0))} GB</dd>
                        </div>
                        <div>
                          <dt className="text-muted-foreground">卡数</dt>
                          <dd className="font-medium tabular-nums">{sumShare}{idle > 0 ? ` + 闲置 ${idle}` : ''}</dd>
                        </div>
                        <div>
                          <dt className="text-muted-foreground">分片吞吐</dt>
                          <dd className="font-medium tabular-nums text-primary">{fmtTok(clusterTotal)}</dd>
                        </div>
                      </dl>
                    </div>
                  )}
                </div>

                <p className="mt-1.5 text-[11px] leading-relaxed text-muted-foreground/80">
                  显存 = 权重 + KV（当前并发 × 上下文）+ 开销；吞吐为该模型分到的卡数下的估算值，三档明细见下方卡片
                </p>
              </div>

              <div className="space-y-3">
                {typical.outcomes.map((o, i) => {
                  const seg = SEG_CLS[i % SEG_CLS.length];
                  return (
                    <div key={o.entry.id} className="rounded-2xl bg-secondary/30 p-4 transition-shadow duration-fast hover:shadow-card">
                      <div className="mb-3 flex flex-wrap items-center gap-2">
                        <span className={cn('inline-block h-2.5 w-2.5 rounded-full', o.floor === null ? 'bg-muted-foreground/30' : seg)} />
                        <span className="text-sm font-semibold text-foreground">{o.entry.model.name}</span>
                        <Badge variant="outline" className="rounded-md text-[10px] font-normal">
                          {o.entry.precisionId.toUpperCase()} · C={o.entry.concurrency} · ctx {o.entry.avgCtxTokens.toLocaleString()}
                        </Badge>
                        {o.floor !== null && given && deficit === 0 && (
                          <Badge variant="secondary" className="rounded-md text-[10px] font-normal">分到 {o.share} 张</Badge>
                        )}
                        {o.floor !== null && !given && (
                          <Badge variant="secondary" className="rounded-md text-[10px] font-normal">最少 {o.need ?? o.floor} 张</Badge>
                        )}
                        {o.floor === null && <Badge variant="destructive" className="rounded-md text-[10px] font-normal">无法装载</Badge>}
                      </div>
                      {o.result ? (
                        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                          {BAND_KEYS.map((band) => {
                            const r = bands[band].outcomes[i].result;
                            const meta = BAND_LABEL[band];
                            const atCap = !!r && !r.feasibleAtC;
                            const isTypical = band === 'typical';
                            if (!r) {
                              return (
                                <BandCard key={band} meta={meta} highlighted={isTypical} className={`h-full ${band === 'aggressive' ? 'md:col-span-2 xl:col-span-1' : ''}`}>
                                  <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">该档位下无法承载（保底或并发超限）</p>
                                </BandCard>
                              );
                            }
                            return (
                              <BandCard key={band} meta={meta} highlighted={isTypical} className={`h-full ${band === 'aggressive' ? 'md:col-span-2 xl:col-span-1' : ''}`}>
                                <dl className="mt-2 space-y-1 text-xs">
                                  <div className="flex justify-between"><dt className="text-muted-foreground">方案</dt><dd className="font-medium tabular-nums">{r.plan.tp}×TP · {r.plan.pp}×PP · {r.plan.dp}×DP</dd></div>
                                  <div className="flex justify-between"><dt className="text-muted-foreground">用卡</dt><dd className="font-medium tabular-nums">{r.plan.totalDevices}{r.plan.idleDevices > 0 ? `（闲置 ${r.plan.idleDevices}）` : ''}</dd></div>
                                  <div className="flex justify-between"><dt className="text-muted-foreground">TPOT</dt><dd className="font-medium tabular-nums">{fmtMs(r.throughput.tpotMs)}</dd></div>
                                  <div className="flex justify-between"><dt className="text-muted-foreground">TTFT</dt><dd className="font-medium tabular-nums">{fmtMs(r.throughput.ttftMs)}</dd></div>
                                  <div className="flex justify-between"><dt className="text-muted-foreground">单用户</dt><dd className="font-medium tabular-nums">{fmtTok(r.throughput.perUserTokPerSec)} tok/s</dd></div>
                                  <div className="flex justify-between"><dt className="text-muted-foreground">分片总吞吐</dt><dd className="font-medium tabular-nums">{fmtTok(r.clusterAggregateTokPerSec)} tok/s</dd></div>
                                  <div className="flex justify-between"><dt className="text-muted-foreground">单卡占用</dt><dd className="font-medium tabular-nums">{fmtGB(r.vramPerDevice.totalGB)} / {usableVramGB(gpu).toFixed(0)} GB</dd></div>
                                  <div className="flex justify-between"><dt className="text-muted-foreground">最大并发</dt><dd className="font-medium tabular-nums">{Number.isFinite(r.capacity.maxConcurrency) ? `${r.capacity.maxConcurrency.toLocaleString()} 路` : '—'}</dd></div>
                                </dl>
                                {atCap && (
                                  <p className="mt-2 text-[11px] font-medium text-amber-600 dark:text-amber-400">
                                    并发 {o.entry.concurrency} 路超出份额容量，按满载 {r.capacity.maxConcurrency.toLocaleString()} 路计
                                  </p>
                                )}
                              </BandCard>
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
                        <p className="text-xs text-muted-foreground">最少卡数见上方摘要表；切「按实际卡数」查看分配</p>
                      )}
                    </div>
                  );
                })}
              </div>

              <div className="flex flex-wrap gap-1.5">
                <Badge variant="outline" className="rounded-md text-[10px] font-normal">屋顶线估算：吞吐随 η 取值浮动，非实测</Badge>
                <Badge variant="outline" className="rounded-md text-[10px] font-normal">各模型独立 KV，无跨模型共享缓存</Badge>
                <Badge variant="outline" className="rounded-md text-[10px] font-normal">PP 跨节点开销未计入带宽</Badge>
                <Badge variant="outline" className="rounded-md text-[10px] font-normal">保底 = 权重装载下界；份额 ≥ 保底</Badge>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
