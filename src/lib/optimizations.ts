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

export type OptSupport = 'yes' | 'int8' | 'limited' | 'no';

export interface HardwareOptSupport {
  /** KV FP8：fp8Tflops 有值 → 原生；仅 INT8 → int8；都没有 → no */
  kvFp8: OptSupport;
  /** 分块预填充：各平台推理栈均支持（vLLM/SGLang 默认开启，llama.cpp 由 ubatch 承担） */
  chunked: OptSupport;
  /** 热缓存（内存卸载层）：需要独立于显存的主机内存；统一内存设备为同一池 → no */
  hotTier: OptSupport;
  /** 冷缓存（SSD 持久层）：所有平台物理可行；统一内存设备工具链支持有限 → limited */
  coldTier: OptSupport;
}

/** 按所选硬件判定各优化项的真实可用性。gpu 为 null 时按通用服务器给出。 */
export function hardwareOptSupport(gpu: { fp8Tflops?: number; int8Tflops?: number; usableFraction?: number; vendor: string } | null): HardwareOptSupport {
  const kvFp8: OptSupport = !gpu ? 'yes' : gpu.fp8Tflops ? 'yes' : gpu.int8Tflops ? 'int8' : 'no';
  const unified = !!gpu?.usableFraction; // Apple / GB10 / AI Max：统一内存，无独立主机内存池
  return {
    kvFp8,
    chunked: 'yes',
    hotTier: unified ? 'no' : 'yes',
    coldTier: unified ? 'limited' : 'yes',
  };
}
