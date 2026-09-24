// 性能优化选项：分块预填充、KV FP8、热缓存（内存）、冷缓存（SSD）。
// 各选项的估算口径见 throughput.ts / parallelism.ts 的实现注释。

export interface Optimizations {
  /** KV 缓存 FP8 量化：KV 字节数减半，解码读取 KV 变快（有精度代价） */
  kvFp8: boolean;
  /** 分块预填充：预填充块与解码交错，TPOT 轻微上浮、并发下 TTFT 显著下降 */
  chunkedPrefill: boolean;
  chunkSize: number; // tokens per chunk
  /** 热缓存（内存/CPU RAM）：频繁访问的 KV/前缀块驻留内存，命中时免重复预填充 */
  hotCache: { enabled: boolean; sizeGB: number; hitRate: number }; // hitRate: 0.2/0.4/0.6
  /** 冷缓存（SSD）：被驱逐的 KV 块持久化到 NVMe，扩展 KV 容量；命中检索按 SSD 带宽计延迟 */
  coldCache: { enabled: boolean; sizeGB: number; ssdGBs: number };
}

export const OPTS_OFF: Optimizations = {
  kvFp8: false,
  chunkedPrefill: false,
  chunkSize: 1024,
  hotCache: { enabled: false, sizeGB: 256, hitRate: 0 },
  coldCache: { enabled: false, sizeGB: 1024, ssdGBs: 7 },
};

export function anyOptEnabled(o: Optimizations): boolean {
  return (
    o.kvFp8 ||
    o.chunkedPrefill ||
    o.hotCache.enabled ||
    o.coldCache.enabled
  );
}

/** 热缓存 + 冷缓存合计可卸载的 KV 字节数（GB）。 */
export function tierOffloadGB(o: Optimizations): number {
  return (o.hotCache.enabled ? o.hotCache.sizeGB : 0) + (o.coldCache.enabled ? o.coldCache.sizeGB : 0);
}
