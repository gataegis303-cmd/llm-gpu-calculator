import { useEffect, useMemo, useRef, useState } from 'react';
import { Card, CardAction, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MODELS, CATEGORY_FILTERS, modelCategory, type ModelCategory, type ModelSpec } from '@/data/models';
import { searchOnlineModels, resolveOnlineModel, pipelineCategory, MODEL_TYPE_LABELS, type OnlineModelHit } from '@/lib/modelRepo';
import { ChevronsUpDown, Loader2, CloudDownload, SearchX, Cpu, ExternalLink, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import InfoTip from '@/components/info-tip';

interface Props {
  className?: string;
  selected: ModelSpec | null;
  onSelect: (m: ModelSpec) => void;
  /** multi：可多选形态（选中不关面板、上限 6）；默认 single，行为与旧版一致 */
  mode?: 'single' | 'multi';
  selectedList?: ModelSpec[];
  onAdd?: (m: ModelSpec) => void;
  onModeChange?: (mode: 'single' | 'multi') => void;
  onRemove?: (modelId: string) => void;
}

const FLEET_CAP = 6;

function fmtParams(b: number): string {
  return b >= 1000 ? `${(b / 1000).toFixed(2)}T` : `${b.toFixed(1)}B`;
}

export default function ModelPicker({
  className,
  selected,
  onSelect,
  mode = 'single',
  selectedList,
  onAdd,
  onModeChange,
  onRemove,
}: Props) {
  const [open, setOpen] = useState(false);
  const isMulti = mode === 'multi';
  const picked = selectedList ?? [];
  const pick = (m: ModelSpec) => {
    if (isMulti) {
      if (picked.length < FLEET_CAP && !picked.some((x) => x.id === m.id)) onAdd?.(m);
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
      } catch {
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

  // 分类 + MoE + 关键词过滤，并按家族分组
  const { families, counts } = useMemo(() => {
    const q = query.toLowerCase().trim();
    const matchText = (m: ModelSpec) =>
      !q ||
      m.name.toLowerCase().includes(q) ||
      m.family.toLowerCase().includes(q) ||
      m.org.toLowerCase().includes(q);
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
    <Card className={cn('section-card', className)}>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-1.5 text-base font-semibold tracking-tight">
          <Cpu className="h-4 w-4 text-primary" />
          选择模型
          <InfoTip text={isMulti ? '同一 GPU 类型下可多选多个模型组成集群（最多 6 个）；选中在线模型需网络，内置库可离线使用。' : '支持搜索内置模型库与 HuggingFace 在线模型（实时查询 config.json 自动推导架构）；选中在线模型需网络，内置库可离线使用。'} />
        </CardTitle>
        <CardAction>
          <Tabs value={mode} onValueChange={(v) => onModeChange?.(v as 'single' | 'multi')}>
            <TabsList className="h-8 rounded-full bg-secondary p-0.5">
              <TabsTrigger
                value="single"
                className="h-7 rounded-full px-3 text-xs transition-all data-[state=active]:bg-card data-[state=active]:shadow-card"
              >
                单模型
              </TabsTrigger>
              <TabsTrigger
                value="multi"
                className="h-7 rounded-full px-3 text-xs transition-all data-[state=active]:bg-card data-[state=active]:shadow-card"
              >
                多模型
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </CardAction>
        <p className="mt-1 text-xs text-muted-foreground">
          {isMulti
            ? `已选 ${picked.length}/${FLEET_CAP} 个模型，可在下方配置区逐个调整精度与负载。`
            : '按规模档位、MoE 过滤或关键词搜索，内置库与 HuggingFace 在线模型均可选用。'}
        </p>

        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <button
              type="button"
              role="combobox"
              aria-expanded={open}
              className="group col-span-2 mt-3 flex h-11 w-full items-center justify-between rounded-xl border border-border/80 bg-secondary/50 px-3 py-2 text-sm transition-all duration-fast hover:border-primary/30 hover:bg-primary/5 focus-visible:border-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
            >
              <span className={cn('truncate', !selected && !isMulti && 'text-muted-foreground')}>
                {isMulti
                  ? picked.length > 0
                    ? `已选 ${picked.length} 个模型`
                    : '选择模型（可多选）'
                  : selected ? `${selected.name} · ${fmtParams(selected.paramsB)}` : '搜索或选择模型…'}
              </span>
              <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 text-muted-foreground/50 transition-colors group-hover:text-muted-foreground" />
            </button>
          </PopoverTrigger>
          <PopoverContent className="w-[min(28rem,calc(100vw-2rem))] origin-top-left animate-scale-in rounded-xl border-border/80 p-0 shadow-elevated" align="start">
            <div className="border-b border-border/60 px-3 py-2.5">
              <div className="flex flex-wrap gap-1.5" role="group" aria-label="按规模分类">
                {CATEGORY_FILTERS.map((f) => (
                  <button
                    key={f.id}
                    type="button"
                    onClick={() => setCat(f.id)}
                    className={cn(
                      'rounded-full px-2.5 py-1 text-[11px] font-medium transition-all duration-fast',
                      cat === f.id
                        ? 'bg-primary text-primary-foreground shadow-sm'
                        : 'bg-secondary text-muted-foreground hover:bg-secondary/80 hover:text-foreground',
                    )}
                  >
                    {f.label}
                    <span className={cn('ml-1 opacity-80', cat === f.id && 'text-primary-foreground/80')}>{counts[f.id] ?? 0}</span>
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
                          'rounded-full px-2.5 py-1 text-[11px] font-medium transition-all duration-fast',
                          modality === key
                            ? 'bg-sky-500 text-white shadow-sm'
                            : 'bg-secondary text-muted-foreground hover:bg-secondary/80 hover:text-foreground',
                        )}
                      >
                        {label}
                        <span className="ml-1 opacity-80">{n}</span>
                      </button>
                    );
                  })
                )}
                <button
                  type="button"
                  onClick={() => setMoeOnly(!moeOnly)}
                  aria-pressed={moeOnly}
                  className={cn(
                    'rounded-full px-2.5 py-1 text-[11px] font-medium transition-all duration-fast',
                    moeOnly
                      ? 'bg-amber-500 text-white shadow-sm'
                      : 'bg-secondary text-muted-foreground hover:bg-secondary/80 hover:text-foreground',
                  )}
                >
                  仅看 MoE
                </button>
              </div>
            </div>
            <Command shouldFilter={false}>
              <CommandInput placeholder="输入关键词，实时搜索 HuggingFace…" value={query} onValueChange={setQuery} className="h-10" />
              <CommandList className="max-h-80">
                <CommandEmpty>
                  <div className="flex flex-col items-center gap-1.5 py-8 text-sm text-muted-foreground">
                    <SearchX className="h-5 w-5 opacity-50" />
                    没有匹配结果
                  </div>
                </CommandEmpty>

                {families.map(([family, models]) => (
                  <CommandGroup key={family} heading={family} className="text-xs font-medium text-muted-foreground">
                    {models.map((m) => (
                      <CommandItem
                        key={m.id}
                        value={m.id}
                        onSelect={() => {
                          pick(m);
                          setResolveError(null);
                        }}
                        className="cursor-pointer rounded-lg transition-colors"
                      >
                        <div className="flex w-full items-center justify-between gap-2">
                          <span className="truncate">
                            {m.name}
                            <span className="ml-2 text-xs text-muted-foreground">{fmtParams(m.paramsB)}</span>
                          </span>
                          <span className="flex shrink-0 gap-1.5">
                            {m.isMoE && <span className="rounded-full bg-amber-500/10 px-1.5 py-px text-[10px] font-medium text-amber-600 dark:text-amber-400">MoE</span>}
                            {selected?.id === m.id && <span className="text-[10px] font-medium text-primary">已选</span>}
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
                  <CommandGroup heading="HuggingFace 在线搜索" className="text-xs font-medium text-muted-foreground">
                    {query.includes('/') && (
                      <CommandItem value={`direct-fetch-${query}`} onSelect={() => pickOnline(query.trim())} className="cursor-pointer rounded-lg transition-colors">
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
                        <CommandItem key={h.repoId} value={h.repoId} onSelect={() => pickOnline(h.repoId)} className="cursor-pointer rounded-lg transition-colors">
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
              <div className="border-t border-border/60 px-3 py-2 text-xs text-destructive">{resolveError}</div>
            )}
          </PopoverContent>
        </Popover>

      </CardHeader>
      <CardContent className="border-t border-border/60 bg-secondary/20 py-3">
        {isMulti ? (
          picked.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {picked.map((m) => (
                <Badge key={m.id} variant="secondary" className="flex max-w-full items-center gap-1 rounded-md pl-2 pr-1 font-normal">
                  <span className="truncate">{m.name}</span>
                  <button
                    type="button"
                    onClick={() => onRemove?.(m.id)}
                    className="ml-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-secondary-foreground/10 hover:text-foreground"
                    aria-label={`移除 ${m.name}`}
                  >
                    <X className="h-3 w-3" />
                  </button>
                </Badge>
              ))}
            </div>
          )
        ) : selected ? (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="secondary" className="rounded-md font-normal">{selected.org}</Badge>
              <Badge variant="outline" className="rounded-md font-normal">{fmtParams(selected.paramsB)} 参数</Badge>
              {selected.isMoE && <Badge variant="outline" className="rounded-md border-amber-500/30 font-normal text-amber-600 dark:text-amber-400">MoE 激活 {selected.activeParamsB.toFixed(1)}B</Badge>}
              {selected.fp8Native && <Badge variant="outline" className="rounded-md font-normal">{selected.nativeDtype}</Badge>}
            </div>
            <div className="mt-2.5 flex items-center gap-1 text-xs text-muted-foreground">
              <span>来源：</span>
              <a
                href={selected.sourceUrl.startsWith('http') ? selected.sourceUrl : '#'}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-0.5 text-primary transition-colors hover:underline"
              >
                {selected.sourceUrl.split(' (')[0]}
                <ExternalLink className="h-3 w-3" />
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
