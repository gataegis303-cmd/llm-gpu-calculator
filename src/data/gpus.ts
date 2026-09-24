// GPU / accelerator specs — vendor datasheet values (official numbers),
// each entry carries a sourceUrl. Fetched/compiled 2026-09-23.
// bf16Tflops / fp8Tflops / int8Tflops are DENSE (non-sparse) numbers.

export type Vendor = 'nvidia' | 'amd' | 'ascend' | 'apple';

export interface GpuSpec {
  id: string;
  vendor: Vendor;
  name: string;
  vramGB: number;
  memType: string;
  memBandwidthGBs: number;
  bf16Tflops: number; // dense
  fp8Tflops?: number; // dense; undefined = not supported / unknown
  int8Tflops?: number; // dense
  interconnect: string;
  gpusPerNode: number; // typical serving node topology
  /** Apple unified memory: fraction of RAM usable by the GPU (llama.cpp/MLX heuristics) */
  usableFraction?: number;
  sourceUrl: string;
}

export const VENDOR_LABELS: Record<Vendor, string> = {
  nvidia: 'NVIDIA',
  amd: 'AMD',
  ascend: '华为昇腾',
  apple: 'Apple M 系列',
};

export const GPUS: GpuSpec[] = [
  // ── NVIDIA datacenter ────────────────────────────────────────────────────
  {
    id: 'h100-sxm', vendor: 'nvidia', name: 'H100 SXM',
    vramGB: 80, memType: 'HBM3', memBandwidthGBs: 3350,
    bf16Tflops: 989, fp8Tflops: 1979, int8Tflops: 1979,
    interconnect: 'NVLink 4 (900GB/s)', gpusPerNode: 8,
    sourceUrl: 'https://www.nvidia.com/en-us/data-center/h100/',
  },
  {
    id: 'h100-pcie', vendor: 'nvidia', name: 'H100 PCIe',
    vramGB: 80, memType: 'HBM3', memBandwidthGBs: 2000,
    bf16Tflops: 756, fp8Tflops: 1513, int8Tflops: 1513,
    interconnect: 'PCIe 5.0', gpusPerNode: 8,
    sourceUrl: 'https://www.nvidia.com/en-us/data-center/h100/',
  },
  {
    id: 'h200', vendor: 'nvidia', name: 'H200',
    vramGB: 141, memType: 'HBM3e', memBandwidthGBs: 4800,
    bf16Tflops: 989, fp8Tflops: 1979, int8Tflops: 1979,
    interconnect: 'NVLink 4 (900GB/s)', gpusPerNode: 8,
    sourceUrl: 'https://www.nvidia.com/en-us/data-center/h200/',
  },
  {
    id: 'h800', vendor: 'nvidia', name: 'H800 (SXM)',
    vramGB: 80, memType: 'HBM3', memBandwidthGBs: 3350,
    bf16Tflops: 989, fp8Tflops: 1979, int8Tflops: 1979,
    interconnect: 'NVLink 4 (受限 400GB/s)', gpusPerNode: 8,
    sourceUrl: 'https://www.nvidia.com/en-us/data-center/h100/ (H800 为中国区衍生型号)',
  },
  {
    id: 'h20', vendor: 'nvidia', name: 'H20',
    vramGB: 96, memType: 'HBM3', memBandwidthGBs: 4000,
    bf16Tflops: 148, fp8Tflops: 296, int8Tflops: 296,
    interconnect: 'NVLink (900GB/s)', gpusPerNode: 8,
    sourceUrl: 'https://www.nvidia.com/en-us/data-center/ (H20 规格页)',
  },
  {
    id: 'b200', vendor: 'nvidia', name: 'B200',
    vramGB: 192, memType: 'HBM3e', memBandwidthGBs: 8000,
    bf16Tflops: 2250, fp8Tflops: 4500, int8Tflops: 4500,
    interconnect: 'NVLink 5 (1.8TB/s)', gpusPerNode: 8,
    sourceUrl: 'https://www.nvidia.com/en-us/data-center/blackwell-architecture/',
  },
  {
    // DGX Spark 桌面超芯片。官方 datasheet: 128GB LPDDR5x / 273GB/s / 1 PFLOP (FP4 sparse)
    // / ConnectX-7 200GbE；BF16/FP8 dense 为按 Blackwell 张量核比例推算值。
    id: 'gb10-dgx-spark', vendor: 'nvidia', name: 'DGX Spark (GB10)',
    vramGB: 128, memType: 'LPDDR5x 统一内存', memBandwidthGBs: 273,
    bf16Tflops: 50, fp8Tflops: 100, int8Tflops: 100,
    interconnect: 'ConnectX-7 (200GbE)', gpusPerNode: 1, usableFraction: 0.93,
    sourceUrl: 'https://www.nvidia.com/en-us/products/dgx-spark/',
  },
  {
    // GB10 Grace Blackwell Superchip（DGX Spark 桌面整机）：
    // 128GB LPDDR5x 统一内存；官方标称 1 PFLOP FP4（稀疏），
    // BF16/FP8 稠密值按 Blackwell 比例换算并在此标注，官方未单列。
    id: 'gb10-spark', vendor: 'nvidia', name: 'GB10 (DGX Spark)',
    vramGB: 128, memType: 'LPDDR5x 统一内存', memBandwidthGBs: 273,
    bf16Tflops: 125, fp8Tflops: 250, int8Tflops: 250,
    interconnect: 'NVLink 5 (200GB/s，可双机互联)', gpusPerNode: 1, usableFraction: 0.9,
    sourceUrl: 'https://www.nvidia.com/en-us/products/dgx-spark/',
  },
  {
    id: 'a100-40', vendor: 'nvidia', name: 'A100 40GB SXM',
    vramGB: 40, memType: 'HBM2e', memBandwidthGBs: 1555,
    bf16Tflops: 312, int8Tflops: 624,
    interconnect: 'NVLink 3 (600GB/s)', gpusPerNode: 8,
    sourceUrl: 'https://www.nvidia.com/en-us/data-center/a100/',
  },
  {
    id: 'a100-80', vendor: 'nvidia', name: 'A100 80GB SXM',
    vramGB: 80, memType: 'HBM2e', memBandwidthGBs: 2039,
    bf16Tflops: 312, int8Tflops: 624,
    interconnect: 'NVLink 3 (600GB/s)', gpusPerNode: 8,
    sourceUrl: 'https://www.nvidia.com/en-us/data-center/a100/',
  },
  {
    id: 'a800-80', vendor: 'nvidia', name: 'A800 80GB SXM',
    vramGB: 80, memType: 'HBM2e', memBandwidthGBs: 2039,
    bf16Tflops: 312, int8Tflops: 624,
    interconnect: 'NVLink 3 (受限 400GB/s)', gpusPerNode: 8,
    sourceUrl: 'https://www.nvidia.com/en-us/data-center/a100/ (A800 为中国区衍生型号)',
  },
  {
    id: 'a30', vendor: 'nvidia', name: 'A30',
    vramGB: 24, memType: 'HBM2', memBandwidthGBs: 933,
    bf16Tflops: 165, int8Tflops: 330,
    interconnect: 'NVLink 3 (600GB/s)', gpusPerNode: 8,
    sourceUrl: 'https://www.nvidia.com/en-us/data-center/products/a30/',
  },
  {
    id: 'a10', vendor: 'nvidia', name: 'A10',
    vramGB: 24, memType: 'GDDR6', memBandwidthGBs: 600,
    bf16Tflops: 62.5, int8Tflops: 125,
    interconnect: 'PCIe 4.0', gpusPerNode: 8,
    sourceUrl: 'https://www.nvidia.com/en-us/data-center/products/a10-gpu/',
  },
  {
    id: 'a10g', vendor: 'nvidia', name: 'A10G',
    vramGB: 24, memType: 'GDDR6', memBandwidthGBs: 600,
    bf16Tflops: 62.5, int8Tflops: 125,
    interconnect: 'PCIe 4.0', gpusPerNode: 8,
    sourceUrl: 'https://www.nvidia.com/en-us/data-center/products/a10-gpu/',
  },
  {
    id: 'l40s', vendor: 'nvidia', name: 'L40S',
    vramGB: 48, memType: 'GDDR6', memBandwidthGBs: 864,
    bf16Tflops: 181, fp8Tflops: 362, int8Tflops: 362,
    interconnect: 'PCIe 4.0', gpusPerNode: 8,
    sourceUrl: 'https://www.nvidia.com/en-us/data-center/l40s/',
  },
  {
    id: 'l20', vendor: 'nvidia', name: 'L20',
    vramGB: 48, memType: 'GDDR6', memBandwidthGBs: 864,
    bf16Tflops: 119, int8Tflops: 239,
    interconnect: 'PCIe 4.0', gpusPerNode: 8,
    sourceUrl: 'https://www.nvidia.com/en-us/data-center/ (L20 规格页)',
  },
  {
    id: 't4', vendor: 'nvidia', name: 'T4',
    vramGB: 16, memType: 'GDDR6', memBandwidthGBs: 300,
    bf16Tflops: 32.5, int8Tflops: 130,
    interconnect: 'PCIe 3.0', gpusPerNode: 8,
    sourceUrl: 'https://www.nvidia.com/en-us/data-center/tesla-t4/',
  },
  {
    id: 'v100-32', vendor: 'nvidia', name: 'V100 32GB',
    vramGB: 32, memType: 'HBM2', memBandwidthGBs: 900,
    bf16Tflops: 15.7, int8Tflops: 62.5,
    interconnect: 'NVLink 2 (300GB/s)', gpusPerNode: 8,
    sourceUrl: 'https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r1.pdf',
  },
  // ── NVIDIA workstation / consumer ────────────────────────────────────────
  {
    id: 'rtx-6000-ada', vendor: 'nvidia', name: 'RTX 6000 Ada',
    vramGB: 48, memType: 'GDDR6', memBandwidthGBs: 960,
    bf16Tflops: 145.7, int8Tflops: 291.5,
    interconnect: 'PCIe 4.0', gpusPerNode: 4,
    sourceUrl: 'https://www.nvidia.com/en-us/design-visualization/rtx-6000-ada/',
  },
  {
    id: 'rtx-pro-6000-blackwell', vendor: 'nvidia', name: 'RTX PRO 6000 Blackwell',
    vramGB: 96, memType: 'GDDR7', memBandwidthGBs: 1792,
    bf16Tflops: 238, fp8Tflops: 476, int8Tflops: 476,
    interconnect: 'PCIe 5.0', gpusPerNode: 4,
    sourceUrl: 'https://www.nvidia.com/en-us/products/workstations/rtx-pro-6000/',
  },
  {
    id: 'rtx-5090', vendor: 'nvidia', name: 'RTX 5090',
    vramGB: 32, memType: 'GDDR7', memBandwidthGBs: 1792,
    bf16Tflops: 104.8, fp8Tflops: 209.5, int8Tflops: 209.5,
    interconnect: 'PCIe 5.0', gpusPerNode: 4,
    sourceUrl: 'https://www.nvidia.com/en-us/geforce/rtx-50-series/rtx-5090/',
  },
  {
    id: 'rtx-4090', vendor: 'nvidia', name: 'RTX 4090',
    vramGB: 24, memType: 'GDDR6X', memBandwidthGBs: 1008,
    bf16Tflops: 82.6, fp8Tflops: 165.2, int8Tflops: 165.2,
    interconnect: 'PCIe 4.0', gpusPerNode: 4,
    sourceUrl: 'https://www.nvidia.com/en-us/geforce/rtx-40-series/rtx-4090/',
  },
  {
    id: 'rtx-4090d', vendor: 'nvidia', name: 'RTX 4090D',
    vramGB: 24, memType: 'GDDR6X', memBandwidthGBs: 1008,
    bf16Tflops: 74.3, int8Tflops: 148.7,
    interconnect: 'PCIe 4.0', gpusPerNode: 4,
    sourceUrl: 'https://www.nvidia.cn/geforce/rtx-40-series/rtx-4090/ (4090D 规格页)',
  },
  {
    id: 'rtx-3090', vendor: 'nvidia', name: 'RTX 3090',
    vramGB: 24, memType: 'GDDR6X', memBandwidthGBs: 936,
    bf16Tflops: 35.6, int8Tflops: 142,
    interconnect: 'PCIe 4.0', gpusPerNode: 4,
    sourceUrl: 'https://www.nvidia.com/en-us/geforce/rtx-30-series/rtx-3090/',
  },
  // ── AMD Instinct ─────────────────────────────────────────────────────────
  {
    id: 'mi325x', vendor: 'amd', name: 'Instinct MI325X',
    vramGB: 256, memType: 'HBM3e', memBandwidthGBs: 6000,
    bf16Tflops: 1307, fp8Tflops: 2614,
    interconnect: 'Infinity Fabric', gpusPerNode: 8,
    sourceUrl: 'https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html',
  },
  {
    id: 'mi300x', vendor: 'amd', name: 'Instinct MI300X',
    vramGB: 192, memType: 'HBM3', memBandwidthGBs: 5300,
    bf16Tflops: 1307, fp8Tflops: 2614,
    interconnect: 'Infinity Fabric (8×236GB/s)', gpusPerNode: 8,
    sourceUrl: 'https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html',
  },
  {
    id: 'mi250x', vendor: 'amd', name: 'Instinct MI250X',
    vramGB: 128, memType: 'HBM2e', memBandwidthGBs: 3200,
    bf16Tflops: 383,
    interconnect: 'Infinity Fabric', gpusPerNode: 8,
    sourceUrl: 'https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html',
  },
  {
    id: 'mi210', vendor: 'amd', name: 'Instinct MI210',
    vramGB: 64, memType: 'HBM2e', memBandwidthGBs: 1600,
    bf16Tflops: 181,
    interconnect: 'PCIe 4.0', gpusPerNode: 8,
    sourceUrl: 'https://www.amd.com/en/products/accelerators/instinct/mi200/mi210.html',
  },
  {
    id: 'rx-7900xtx', vendor: 'amd', name: 'Radeon RX 7900 XTX',
    vramGB: 24, memType: 'GDDR6', memBandwidthGBs: 960,
    bf16Tflops: 122.9, int8Tflops: 245.8,
    interconnect: 'PCIe 4.0', gpusPerNode: 4,
    sourceUrl: 'https://www.amd.com/en/products/graphics/radeon-rx-7900xtx.html',
  },
  // ── 华为昇腾 ─────────────────────────────────────────────────────────────
  {
    id: 'ascend-910b', vendor: 'ascend', name: '昇腾 910B (64GB)',
    vramGB: 64, memType: 'HBM2e', memBandwidthGBs: 800,
    bf16Tflops: 313, int8Tflops: 626,
    interconnect: 'HCCS (392GB/s)', gpusPerNode: 8,
    sourceUrl: 'https://e.huawei.com/cn/products/computing/ascend/atlas-800t-a2 (910B 系列规格)',
  },
  {
    id: 'ascend-910c', vendor: 'ascend', name: '昇腾 910C',
    vramGB: 128, memType: 'HBM2e', memBandwidthGBs: 3200,
    bf16Tflops: 789, int8Tflops: 1578,
    interconnect: 'HCCS', gpusPerNode: 8,
    sourceUrl: 'https://e.huawei.com/cn/ (910C 公开报道规格，标注为估算)',
  },
  {
    id: 'ascend-910a', vendor: 'ascend', name: '昇腾 910A (32GB)',
    vramGB: 32, memType: 'HBM2e', memBandwidthGBs: 1200,
    bf16Tflops: 320, int8Tflops: 640,
    interconnect: 'HCCS', gpusPerNode: 8,
    sourceUrl: 'https://e.huawei.com/cn/products/computing/ascend/ (910A 规格页)',
  },
  {
    id: 'atlas-300i-duo', vendor: 'ascend', name: 'Atlas 300I Duo (96GB)',
    vramGB: 96, memType: 'LPDDR4X', memBandwidthGBs: 408,
    bf16Tflops: 140, int8Tflops: 280,
    interconnect: 'PCIe 4.0', gpusPerNode: 8,
    sourceUrl: 'https://e.huawei.com/cn/products/computing/ascend/atlas-300i-duo',
  },
  {
    id: 'atlas-300v', vendor: 'ascend', name: 'Atlas 300V Pro (48GB)',
    vramGB: 48, memType: 'LPDDR4X', memBandwidthGBs: 204,
    bf16Tflops: 70, int8Tflops: 140,
    interconnect: 'PCIe 4.0', gpusPerNode: 8,
    sourceUrl: 'https://e.huawei.com/cn/products/computing/ascend/atlas-300v-pro',
  },
  {
    id: 'atlas-800t-a2', vendor: 'ascend', name: 'Atlas 800T A2 (整机 8×64GB)',
    vramGB: 64, memType: 'HBM2e', memBandwidthGBs: 800,
    bf16Tflops: 313, int8Tflops: 626,
    interconnect: 'HCCS', gpusPerNode: 8,
    sourceUrl: 'https://e.huawei.com/cn/products/computing/ascend/atlas-800t-a2',
  },
  // ── Apple M 系列 ─────────────────────────────────────────────────────────
  {
    id: 'm4-max', vendor: 'apple', name: 'M4 Max (128GB)',
    vramGB: 128, memType: 'LPDDR5X 统一内存', memBandwidthGBs: 546,
    bf16Tflops: 30,
    interconnect: '统一内存架构', gpusPerNode: 1, usableFraction: 0.75,
    sourceUrl: 'https://www.apple.com/macbook-pro/specs/',
  },
  {
    id: 'm4-pro', vendor: 'apple', name: 'M4 Pro (64GB)',
    vramGB: 64, memType: 'LPDDR5X 统一内存', memBandwidthGBs: 273,
    bf16Tflops: 17,
    interconnect: '统一内存架构', gpusPerNode: 1, usableFraction: 0.75,
    sourceUrl: 'https://www.apple.com/mac-mini/specs/',
  },
  {
    id: 'm3-ultra-512', vendor: 'apple', name: 'M3 Ultra (512GB)',
    vramGB: 512, memType: 'LPDDR5X 统一内存', memBandwidthGBs: 819,
    bf16Tflops: 34,
    interconnect: '统一内存架构', gpusPerNode: 1, usableFraction: 0.75,
    sourceUrl: 'https://www.apple.com/mac-studio/specs/',
  },
  {
    id: 'm3-ultra-283', vendor: 'apple', name: 'M3 Ultra (283GB)',
    vramGB: 283, memType: 'LPDDR5X 统一内存', memBandwidthGBs: 819,
    bf16Tflops: 34,
    interconnect: '统一内存架构', gpusPerNode: 1, usableFraction: 0.75,
    sourceUrl: 'https://www.apple.com/mac-studio/specs/',
  },
  {
    id: 'm3-max-128', vendor: 'apple', name: 'M3 Max (128GB)',
    vramGB: 128, memType: 'LPDDR5X 统一内存', memBandwidthGBs: 400,
    bf16Tflops: 29,
    interconnect: '统一内存架构', gpusPerNode: 1, usableFraction: 0.75,
    sourceUrl: 'https://www.apple.com/macbook-pro/specs/',
  },
  {
    id: 'm3-max-96', vendor: 'apple', name: 'M3 Max (96GB)',
    vramGB: 96, memType: 'LPDDR5X 统一内存', memBandwidthGBs: 400,
    bf16Tflops: 29,
    interconnect: '统一内存架构', gpusPerNode: 1, usableFraction: 0.75,
    sourceUrl: 'https://www.apple.com/macbook-pro/specs/',
  },
  {
    id: 'm3-pro-36', vendor: 'apple', name: 'M3 Pro (36GB)',
    vramGB: 36, memType: 'LPDDR5 统一内存', memBandwidthGBs: 150,
    bf16Tflops: 12,
    interconnect: '统一内存架构', gpusPerNode: 1, usableFraction: 0.75,
    sourceUrl: 'https://www.apple.com/macbook-pro/specs/',
  },
  {
    id: 'm2-ultra-192', vendor: 'apple', name: 'M2 Ultra (192GB)',
    vramGB: 192, memType: 'LPDDR5 统一内存', memBandwidthGBs: 800,
    bf16Tflops: 27,
    interconnect: '统一内存架构', gpusPerNode: 1, usableFraction: 0.75,
    sourceUrl: 'https://www.apple.com/mac-studio/specs/',
  },
  {
    id: 'm2-max-96', vendor: 'apple', name: 'M2 Max (96GB)',
    vramGB: 96, memType: 'LPDDR5 统一内存', memBandwidthGBs: 400,
    bf16Tflops: 27,
    interconnect: '统一内存架构', gpusPerNode: 1, usableFraction: 0.75,
    sourceUrl: 'https://www.apple.com/macbook-pro/specs/',
  },
  {
    id: 'm2-pro-32', vendor: 'apple', name: 'M2 Pro (32GB)',
    vramGB: 32, memType: 'LPDDR5 统一内存', memBandwidthGBs: 200,
    bf16Tflops: 15.4,
    interconnect: '统一内存架构', gpusPerNode: 1, usableFraction: 0.75,
    sourceUrl: 'https://www.apple.com/mac-mini/specs/',
  },
  {
    id: 'm1-ultra-128', vendor: 'apple', name: 'M1 Ultra (128GB)',
    vramGB: 128, memType: 'LPDDR5 统一内存', memBandwidthGBs: 800,
    bf16Tflops: 21,
    interconnect: '统一内存架构', gpusPerNode: 1, usableFraction: 0.75,
    sourceUrl: 'https://www.apple.com/mac-studio/specs/',
  },
  {
    id: 'm1-max-64', vendor: 'apple', name: 'M1 Max (64GB)',
    vramGB: 64, memType: 'LPDDR5 统一内存', memBandwidthGBs: 400,
    bf16Tflops: 10.4,
    interconnect: '统一内存架构', gpusPerNode: 1, usableFraction: 0.75,
    sourceUrl: 'https://www.apple.com/macbook-pro/specs/',
  },
];

export function gpuById(id: string): GpuSpec | undefined {
  return GPUS.find((g) => g.id === id);
}
