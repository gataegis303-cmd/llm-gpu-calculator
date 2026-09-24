import { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command';
import { MODELS, type ModelSpec } from '@/data/models';
import { searchOnlineModels, resolveOnlineModel, type OnlineModelHit } from '@/lib/modelRepo';
import { ChevronsUpDown, Loader2, CloudDownload, SearchX } from 'lucide-react';
import { cn } from '@/lib/utils';
import InfoTip from '@/components/info-tip';

interface Props {
  selected: ModelSpec | null;
  onSelect: (m: ModelSpec) => void;
}

function fmtParams(b: number): string {
  return b >= 1000 ? `${(b / 1000).toFixed(2)}T` : `${b.toFixed(1)}B`;
}

export default function ModelPicker({ selected, onSelect }: Props) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
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

  const builtin = MODELS.filter(
    (m) =>
      !query.trim() ||
      m.name.toLowerCase().includes(query.toLowerCase()) ||
      m.family.toLowerCase().includes(query.toLowerCase()) ||
      m.org.toLowerCase().includes(query.toLowerCase()),
  );
  const builtinIds = new Set(MODELS.map((m) => m.id));
  const onlineOnly = hits.filter((h) => !builtinIds.has(h.repoId));

  async function pickOnline(repoId: string) {
    setResolveError(null);
    setResolvingId(repoId);
    try {
      const spec = await resolveOnlineModel(repoId, new Date().toISOString().slice(0, 10));
      onSelect(spec);
      setOpen(false);
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
        <CardTitle className="flex items-center gap-1.5 text-base">① 选择模型 <InfoTip text="支持搜索内置模型库与 HuggingFace 在线模型（实时查询 config.json 自动推导架构）；选中在线模型需网络，内置库可离线使用。" /></CardTitle>
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <button
              type="button"
              role="combobox"
              aria-expanded={open}
              className="mt-2 flex h-10 w-full items-center justify-between rounded-md border border-border bg-background px-3 py-2 text-sm hover:bg-muted"
            >
              <span className={cn('truncate', !selected && 'text-muted-foreground')}>
                {selected ? `${selected.name} · ${fmtParams(selected.paramsB)}` : '搜索或选择模型…'}
              </span>
              <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
            </button>
          </PopoverTrigger>
          <PopoverContent className="w-[420px] p-0" align="start">
            <Command shouldFilter={false}>
              <CommandInput placeholder="输入关键词，实时搜索 HuggingFace…" value={query} onValueChange={setQuery} />
              <CommandList className="max-h-80">
                <CommandEmpty>
                  <div className="flex flex-col items-center gap-1 py-6 text-sm text-muted-foreground">
                    <SearchX className="h-4 w-4" />
                    没有匹配结果
                  </div>
                </CommandEmpty>

                {builtin.length > 0 && (
                  <CommandGroup heading="内置模型库（离线快照）">
                    {builtin.map((m) => (
                      <CommandItem
                        key={m.id}
                        value={m.id}
                        onSelect={() => {
                          onSelect(m);
                          setResolveError(null);
                          setOpen(false);
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
      </CardHeader>
      <CardContent className="border-t bg-muted/40">
        {selected ? (
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
