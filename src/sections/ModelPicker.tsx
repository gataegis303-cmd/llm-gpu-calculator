import { useEffect, useMemo, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command';
import { MODELS, CATEGORY_FILTERS, modelCategory, type ModelCategory, type ModelSpec } from '@/data/models';
import { searchOnlineModels, resolveOnlineModel, pipelineCategory, MODEL_TYPE_LABELS, type OnlineModelHit } from '@/lib/modelRepo';
import { ChevronsUpDown, Loader2, CloudDownload, SearchX } from 'lucide-react';
import { cn } from '@/lib/utils';
import InfoTip from '@/components/info-tip';

interface Props {
  selected: ModelSpec | null;
  onSelect: (m: ModelSpec) => void;
  /** multi：可多选形态（选中不关面板、chips 可移除、上限 6）；默认 single，行为与旧版一致 */
  mode?: 'single' | 'multi';
  selectedList?: ModelSpec[];
  onAdd?: (m: ModelSpec) => void;
  onRemove?: (id: string) => void;
  /** 头部右侧插槽（如 单模型/多模型 切换） */
  headerExtra?: React.ReactNode;
}

const FLEET_CAP = 6;

function fmtParams(b: number): string {
  return b >= 1000 ? `${(b / 1000).toFixed(2)}T` : `${b.toFixed(1)}B`;
}

export default function ModelPicker({ selected, onSelect, mode = 'single', selectedList, onAdd, onRemove, headerExtra }: Props) {
  const [open, setOpen] = useState(false);
  const isMulti = mode === 'multi';
  const picked = selectedList ?? [];
  const atCap = isMulti && picked.length >= FLEET_CAP;
  const pick = (m: ModelSpec) => {
    if (isMulti) {
      if (!atCap && !picked.some((x) => x.id === m.id)) onAdd?.(m);
      return; // 多选：保持面板打开
    }
    onSelect(m);
    setOpen(false);
  };
  const [query, setQuery] = useState('');
  const [cat, setCat] = useState<ModelCategory | 'all'>('all');
  const [modality, setModality] = useState<string>('all');
  const [moeOnly, setMoeOnly] = useState(false);
  const [hits, setHits] = useState<OnlineModelHit[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [resolvingId, setResolvingId] = useState<string | null>(null);
  const [resolveError, setResolveError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // 自动查询：输入防抖后实时搜索 HuggingFace
  useEffect(() => {
    if (!open) return;
    const q = query.trim();
    if (q.length < 2) {
      setHits([]);
      setSearchError(null);
      return;
    }
    const t = setTimeout(async () => {
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;
      setSearching(true);
      setSearchError(null);
      try {
        const r = await searchOnlineModels(q, ac.signal);
        setHits(r);
      } catch (e) {
        if (!ac.signal.aborted) {
          setHits([]);
          setSearchError('在线搜索不可用（检查网络或代理），内置库仍可离线使用');
        }
      } finally {
        if (!ac.signal.aborted) setSearching(false);
      }
    }, 350);
    return () => clearTimeout(t);
  }, [query, open]);

  const matchText = (m: ModelSpec) =>
    !query.trim() ||
    m.name.toLowerCase().includes(query.toLowerCase()) ||
    m.family.toLowerCase().includes(query.toLowerCase()) ||
    m.org.toLowerCase().includes(query.toLowerCase());

  // 分类 + MoE + 关键词过滤，并按家族分组
  const { families, counts } = useMemo(() => {
    const pass = (m: ModelSpec) =>
      (cat === 'all' || modelCategory(m) === cat) && (!moeOnly || m.isMoE) && matchText(m) &&
      (modality === 'all' || modality === 'llm');
    const filtered = MODELS.filter(pass);
    const counts: Record<string, number> = { all: MODELS.filter(matchText).length };
    for (const f of CATEGORY_FILTERS) {
      if (f.id !== 'all') counts[f.id] = MODELS.filter((m) => modelCategory(m) === f.id && matchText(m)).length;
    }
    const byFamily = new Map<string, ModelSpec[]>();
    for (const m of filtered) {
      const list = byFamily.get(m.family) ?? [];
      list.push(m);
      byFamily.set(m.family, list);
    }
    return { families: [...byFamily.entries()], counts };
  }, [query, cat, moeOnly, modality]);

  const builtinIds = new Set(MODELS.map((m) => m.id));
  const onlineOnly = hits.filter((h) => !builtinIds.has(h.repoId) && (modality === 'all' || pipelineCategory(h.pipelineTag) === modality));

  async function pickOnline(repoId: string) {
    setResolveError(null);
    setResolvingId(repoId);
    try {
      const spec = await resolveOnlineModel(repoId, new Date().toISOString().slice(0, 10));
      pick(spec); // single：选中并关面板；multi：加入列表、面板保持打开
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.startsWith('NOT_LLM:')) {
        setResolveError(msg.replace(/^NOT_LLM:[^:]*:/, ''));
      } else {
        setResolveError(`拉取 ${repoId} 的 config.json 失败（仓库可能私有、已改名或不存在）`);
      }
    } finally {
      setResolvingId(null);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-1.5 text-base">
          ① {isMulti ? '选择模型（多模型共享集群）' : '选择模型'} <InfoTip text={isMulti ? '同一 GPU 类型下可多选多个模型组成集群（最多 6 个）；每个模型的精度与负载在下方 ② 逐个配置。选中在线模型需网络，内置库可离线使用。' : '支持搜索内置模型库与 HuggingFace 在线模型（实时查询 config.json 自动推导架构）；选中在线模型需网络，内置库可离线使用。可用规模档位与 MoE 过滤快速定位。'} /></CardTitle>
        {headerExtra && <div className="pt-1">{headerExtra}</div>}
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <button
              type="button"
              role="combobox"
              aria-expanded={open}
              className="mt-2 flex h-10 w-full items-center justify-between rounded-md border border-border bg-background px-3 py-2 text-sm hover:bg-muted"
            >
              <span className={cn('truncate', !selected && !isMulti && 'text-muted-foreground')}>
                {isMulti
                  ? picked.length > 0
                    ? `已选 ${picked.length} 个模型 · 点击继续添加`
                    : '搜索或选择模型（可多选）…'
                  : selected ? `${selected.name} · ${fmtParams(selected.paramsB)}` : '搜索或选择模型…'}
              </span>
              <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
            </button>
          </PopoverTrigger>
          <PopoverContent className="w-[440px] p-0" align="start">
            <div className="border-b px-2 py-2">
              <div className="flex flex-wrap gap-1" role="group" aria-label="按规模分类">
                {CATEGORY_FILTERS.map((f) => (
                  <button
                    key={f.id}
                    type="button"
                    onClick={() => setCat(f.id)}
                    className={cn(
                      'rounded-full border px-2 py-0.5 text-[11px] transition-colors',
                      cat === f.id
                        ? 'border-primary bg-primary text-primary-foreground'
                        : 'border-border text-muted-foreground hover:bg-muted',
                    )}
                  >
                    {f.label}
                    <span className="ml-1 opacity-70">{counts[f.id] ?? 0}</span>
                  </button>
                ))}
                {hits.length > 0 && (
                  Object.entries(MODEL_TYPE_LABELS).map(([key, label]) => {
                    const n = hits.filter((h) => pipelineCategory(h.pipelineTag) === key).length;
                    if (n === 0 && key !== 'llm') return null;
                    return (
                      <button
                        key={'t-' + key}
                        type="button"
                        onClick={() => setModality(modality === key ? 'all' : key)}
                        className={cn(
                          'rounded-full border px-2 py-0.5 text-[11px] transition-colors',
                          modality === key
                            ? 'border-sky-500 bg-sky-500/15 text-sky-600 dark:text-sky-400'
                            : 'border-border text-muted-foreground hover:bg-muted',
                        )}
                      >
                        {label}
                        <span className="ml-1 opacity-70">{n}</span>
                      </button>
                    );
                  })
                )}
                <button
                  type="button"
                  onClick={() => setMoeOnly(!moeOnly)}
                  aria-pressed={moeOnly}
                  className={cn(
                    'rounded-full border px-2 py-0.5 text-[11px] transition-colors',
                    moeOnly
                      ? 'border-amber-500 bg-amber-500/15 text-amber-600 dark:text-amber-400'
                      : 'border-border text-muted-foreground hover:bg-muted',
                  )}
                >
                  仅看 MoE
                </button>
              </div>
            </div>
            <Command shouldFilter={false}>
              <CommandInput placeholder="输入关键词，实时搜索 HuggingFace…" value={query} onValueChange={setQuery} />
              <CommandList className="max-h-80">
                <CommandEmpty>
                  <div className="flex flex-col items-center gap-1 py-6 text-sm text-muted-foreground">
                    <SearchX className="h-4 w-4" />
                    没有匹配结果
                  </div>
                </CommandEmpty>

                {families.map(([family, models]) => (
                  <CommandGroup key={family} heading={family}>
                    {models.map((m) => (
                      <CommandItem
                        key={m.id}
                        value={m.id}
                        onSelect={() => {
                          pick(m);
                          setResolveError(null);
                        }}
                      >
                        <div className="flex w-full items-center justify-between gap-2">
                          <span className="truncate">
                            {m.name}
                            <span className="ml-2 text-xs text-muted-foreground">{fmtParams(m.paramsB)}</span>
                          </span>
                          <span className="flex shrink-0 gap-1">
                            {m.isMoE && <span className="text-[10px] text-amber-600 dark:text-amber-400">MoE</span>}
                            {selected?.id === m.id && <span className="text-[10px] text-primary">已选</span>}
                          </span>
                        </div>
                      </CommandItem>
                    ))}
                  </CommandGroup>
                ))}
                {families.length === 0 && (
                  <div className="px-3 py-2 text-xs text-muted-foreground">内置库中无匹配（切换档位或清除过滤）</div>
                )}

                {(searching || onlineOnly.length > 0 || searchError) && (
                  <CommandGroup heading="HuggingFace 在线搜索">
                    {query.includes('/') && (
                                            <CommandItem value={`direct-fetch-${query}`} onSelect={() => pickOnline(query.trim())}>
                        <div className="flex w-full items-center justify-between gap-2">
                          <span className="truncate">
                            直接拉取 <span className="font-medium">{query.trim()}</span>
                            <span className="ml-2 text-xs text-muted-foreground">按完整 repo id 获取 config</span>
                          </span>
                          <CloudDownload className="h-3.5 w-3.5 shrink-0 opacity-50" />
                        </div>
                      </CommandItem>
                    )}
                    {searching && (
                      <div className="flex items-center gap-2 px-2 py-1.5 text-sm text-muted-foreground">
                        <Loader2 className="h-3.5 w-3.5 animate-spin" /> 正在查询…
                      </div>
                    )}
                    {!searching && searchError && (
                      <div className="px-2 py-1.5 text-xs text-muted-foreground">{searchError}</div>
                    )}
                    {onlineOnly.map((h) => {
                      const notLlm = h.pipelineTag && h.pipelineTag !== 'text-generation' && h.pipelineTag !== 'text2text-generation';
                      return (
                        <CommandItem key={h.repoId} value={h.repoId} onSelect={() => pickOnline(h.repoId)}>
                          <div className="flex w-full items-center justify-between gap-2">
                            <span className="truncate">
                              {h.repoId}
                              <span className="ml-2 text-xs text-muted-foreground">{h.downloads.toLocaleString()} 下载</span>
                            </span>
                            <span className="flex shrink-0 items-center gap-1.5">
                              {notLlm && (
                                <span
                                  className="rounded border border-border px-1 py-px text-[10px] text-muted-foreground"
                                  title="非文本生成模型，选中后会校验是否可计算"
                                >
                                  {h.pipelineTag}
                                </span>
                              )}
                              {resolvingId === h.repoId ? (
                                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                              ) : (
                                <CloudDownload className="h-3.5 w-3.5 opacity-50" />
                              )}
                            </span>
                          </div>
                        </CommandItem>
                      );
                    })}
                  </CommandGroup>
                )}
              </CommandList>
            </Command>
            {resolveError && (
              <div className="border-t px-3 py-2 text-xs text-destructive">{resolveError}</div>
            )}
          </PopoverContent>
        </Popover>
        {isMulti && picked.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {picked.map((m) => (
              <span key={m.id} className="inline-flex items-center gap-1 rounded-full border border-border bg-muted/60 py-0.5 pl-2.5 pr-1 text-xs">
                {m.name}
                <button
                  type="button"
                  aria-label={`移除 ${m.name}`}
                  onClick={() => onRemove?.(m.id)}
                  className="rounded-full px-1 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                >
                  ×
                </button>
              </span>
            ))}
            {atCap && <span className="self-center text-[11px] text-amber-600 dark:text-amber-400">已达 {FLEET_CAP} 个上限</span>}
          </div>
        )}
      </CardHeader>
      <CardContent className="border-t bg-muted/40">
        {isMulti ? (
          picked.length > 0 ? (
            <div className="flex flex-wrap items-center gap-2">
              {picked.map((m) => (
                <Badge key={m.id} variant="secondary">{m.name} · {fmtParams(m.paramsB)}</Badge>
              ))}
              <span className="text-xs text-muted-foreground">共 {picked.length} 个模型，在下方 ② 逐个配置负载</span>
            </div>
          ) : (
            <div className="py-1 text-sm text-muted-foreground">尚未添加模型：在上方搜索并选择（可多选，最多 {FLEET_CAP} 个）</div>
          )
        ) : selected ? (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="secondary">{selected.org}</Badge>
              <Badge variant="outline">{fmtParams(selected.paramsB)} 参数</Badge>
              {selected.isMoE && <Badge variant="outline">MoE 激活 {selected.activeParamsB.toFixed(1)}B</Badge>}
              {selected.paramsEstimated && <Badge variant="outline">参数量估算</Badge>}
              {selected.fp8Native && <Badge variant="outline">{selected.nativeDtype}</Badge>}
              <Badge variant="outline">{selected.numHiddenLayers} 层</Badge>
              {selected.isMLA && <Badge variant="outline">MLA</Badge>}
              {selected.kind === 'vlm' && <Badge variant="outline">VLM</Badge>}
              {selected.kind === 'generic' && <Badge variant="outline">非自回归</Badge>}
            </div>
            <div className="mt-2 text-xs text-muted-foreground">
              来源：
              <a href={selected.sourceUrl.startsWith('http') ? selected.sourceUrl : '#'} target="_blank" rel="noreferrer" className="underline">
                {selected.sourceUrl.split(' (')[0]}
              </a>
              <span className="ml-2">抓取于 {selected.fetchedAt}</span>
            </div>
          </>
        ) : (
          <div className="py-1 text-sm text-muted-foreground">尚未选择模型</div>
        )}
      </CardContent>
    </Card>
  );
}
