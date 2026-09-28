// 多模型共享集群：把 N 张卡分给若干模型，再逐模型复用 runCapacity 反向推演。
// 分配规则（可解释优先，不做全局最优求解）：
//   1. 保底 floor_i = 权重装载下界（含开销、不含 KV）；floor 为 null = 本硬件放不下该模型
//   2. need_i = 服务其声明并发的最小卡数（findMinimalPlan 可行时取之，否则退回 floor_i）
//   3. Σ floor > N → 报缺口（deficit），不做分配；否则保底直配
//   4. 剩余 (N − Σ floor) 按需求比例 need_i / Σ need 分配（最大余数法），
//      零头贪心给"每加一张卡最大并发增益最大"的模型，仍余则计为闲置
// 口径与 capacity.ts 逐字一致：三档各自独立跑一遍分配与推演。
import type { ModelSpec } from '@/data/models';
import { PRECISIONS } from '@/data/models';
import type { GpuSpec } from '@/data/gpus';
import { kvBytesPerToken, usableVramGB } from './vram';
import { findMinimalPlan, perDeviceFixedGB, type SearchInput } from './parallelism';
import { runCapacity, type CapacityResult } from './capacity';
import { BANDS, type Band } from './search';
import { OPTS_OFF, tierOffloadGB, type Optimizations } from './optimizations';

export interface FleetEntry {
  /** 稳定 id（UI 层生成，用于增删与回贴） */
  id: string;
  model: ModelSpec;
  precisionId: string;
  concurrency: number;
  avgCtxTokens: number;
  inputTokens: number;
}

export interface FleetInput {
  gpu: GpuSpec;
  /** 集群实际设备数 N（≥1） */
  devices: number;
  entries: FleetEntry[];
  opts?: Optimizations;
}

export interface FleetModelOutcome {
  entry: FleetEntry;
  /** 权重装载下界；null = 本硬件 TP×PP 枚举内放不下 */
  floor: number | null;
  /** 服务其声明并发的最小卡数（floor 的上界估计；floor 为 null 时 = null） */
  need: number | null;
  /** 实际分配到的卡数；缺口或不可部署时 = 0 */
  share: number;
  /** 该份额内未用满的卡 = share − result.plan.totalDevices */
  idleDevices: number;
  /** 该模型在其份额内的推演；未分到卡时为 null */
  result: CapacityResult | null;
}

export interface FleetBandResult {
  band: Band;
  devices: number;
  outcomes: FleetModelOutcome[];
  /** 全部可部署条目的最少卡数之和 */
  totalNeed: number;
  /** Σ floor > N 时的缺口 = Σ floor − N；否则 0 */
  deficit: number;
  undeployableIds: string[];
  /** 分配后仍未发出的卡数 */
  idleDevices: number;
}

function searchInputOfEntry(entry: FleetEntry, band: Band, gpu: GpuSpec, opts: Optimizations): SearchInput {
  const config = BANDS[band];
  const prec = PRECISIONS[entry.precisionId] ?? PRECISIONS.bf16;
  return {
    model: entry.model,
    gpu,
    precisionId: entry.precisionId,
    bytesPerParam: prec.bytesPerParam,
    kvBytesPerToken: kvBytesPerToken(entry.model, entry.precisionId),
    kvOffloadGB: tierOffloadGB(opts),
    concurrency: entry.concurrency,
    avgCtxTokens: entry.avgCtxTokens,
    overheadFactor: config.overheadFactor,
    overheadBaseGB: config.overheadBaseGB,
  };
}

/** 权重装载下界（含开销、不含 KV）；TP×PP 枚举内放不下为 null。与 capacity.ts 同口径。 */
function weightsFloorOf(si: SearchInput): number | null {
  const maxTp = si.gpu.vendor === 'apple' ? 1 : si.gpu.gpusPerNode;
  const usable = usableVramGB(si.gpu);
  for (let dpr = 1; dpr <= maxTp * 8; dpr++) {
    const decomposable = Array.from({ length: dpr }, (_, i) => i + 1).some(
      (tp) => tp <= maxTp && dpr % tp === 0 && dpr / tp <= 8,
    );
    if (decomposable && perDeviceFixedGB(1, dpr, si) <= usable) return dpr;
  }
  return null;
}

export function runFleet(input: FleetInput, band: Band): FleetBandResult {
  const opts = input.opts ?? OPTS_OFF;
  const n = Math.max(1, Math.floor(input.devices));

  // 1. 逐条目保底与需求
  const floors = input.entries.map((e) => weightsFloorOf(searchInputOfEntry(e, band, input.gpu, opts)));
  const needs = input.entries.map((e, i) => {
    if (floors[i] === null) return null;
    const minimal = findMinimalPlan(searchInputOfEntry(e, band, input.gpu, opts));
    return minimal.feasible ? minimal.totalDevices : (floors[i] as number);
  });

  const undeployableIds = input.entries.filter((_, i) => floors[i] === null).map((e) => e.id);
  const deployableIdx = input.entries.map((_, i) => i).filter((i) => floors[i] !== null);
  const sumFloor = deployableIdx.reduce((s, i) => s + (floors[i] as number), 0);
  const totalNeed = deployableIdx.reduce((s, i) => s + (needs[i] as number), 0);

  const shares = input.entries.map(() => 0);
  let idle = 0;

  if (deployableIdx.length === 0) {
    idle = n; // 没有可部署模型，整池闲置
  } else if (sumFloor > n) {
    // 缺口：不做软分配，交由 UI 报告各模型保底
    idle = n;
  } else {
    // 2. 保底直配
    for (const i of deployableIdx) shares[i] = floors[i] as number;
    let spare = n - sumFloor;

    // 3. 剩余按 need_i / Σ need 比例分配（按比例取整）
    if (spare > 0 && totalNeed > 0) {
      const quota = deployableIdx.map((i) => ((needs[i] as number) / totalNeed) * spare);
      const base = quota.map((q) => Math.floor(q));
      deployableIdx.forEach((i, k) => { shares[i] += base[k]; });
      let leftover = spare - base.reduce((s, b) => s + b, 0);
      spare = 0;

      // 4. 零头贪心：每加一张卡给"最大并发增益"最大的模型（平手取 need 更大者）。
      //    全部增益 ≤ 0（加卡在各自分解下只能闲置）→ 剩余计入池闲置。
      const concCache = new Map<string, number>();
      const maxConcOf = (i: number, share: number): number => {
        const key = i + ':' + share;
        let v = concCache.get(key);
        if (v === undefined) {
          v = runCapacity(capacityInputOf(input.entries[i], input.gpu, share), band, opts).capacity.maxConcurrency;
          concCache.set(key, v);
        }
        return v;
      };
      const gainOf = (i: number) => maxConcOf(i, shares[i] + 1) - maxConcOf(i, shares[i]);
      while (leftover > 0) {
        let best = -1;
        let bestGain = 0;
        let bestNeed = -1;
        for (const i of deployableIdx) {
          const g = gainOf(i);
          if (g <= 0) continue;
          if (g > bestGain || (g === bestGain && (needs[i] as number) > bestNeed)) {
            best = i;
            bestGain = g;
            bestNeed = needs[i] as number;
          }
        }
        if (best < 0) break;
        shares[best] += 1;
        leftover -= 1;
      }
    }
    idle = n - shares.reduce((s, v) => s + v, 0);
  }

  // 5. 逐模型推演
  const outcomes: FleetModelOutcome[] = input.entries.map((e, i) => {
    const floor = floors[i];
    const share = shares[i];
    if (floor === null || share < 1) {
      return { entry: e, floor, need: needs[i], share, idleDevices: 0, result: null };
    }
    const result = runCapacity(
      {
        model: e.model,
        gpu: input.gpu,
        precisionId: e.precisionId,
        concurrency: e.concurrency,
        avgCtxTokens: e.avgCtxTokens,
        inputTokens: e.inputTokens,
        devices: share,
      },
      band,
      opts,
    );
    return { entry: e, floor, need: needs[i], share, idleDevices: share - result.plan.totalDevices, result };
  });

  return {
    band,
    devices: n,
    outcomes,
    totalNeed,
    deficit: Math.max(0, sumFloor - n),
    undeployableIds,
    idleDevices: idle,
  };
}

function capacityInputOf(entry: FleetEntry, gpu: GpuSpec, devices: number) {
  return {
    model: entry.model,
    gpu,
    precisionId: entry.precisionId,
    concurrency: entry.concurrency,
    avgCtxTokens: entry.avgCtxTokens,
    inputTokens: entry.inputTokens,
    devices,
  };
}

export function runFleetAllBands(input: FleetInput): Record<Band, FleetBandResult> {
  return {
    conservative: runFleet(input, 'conservative'),
    typical: runFleet(input, 'typical'),
    aggressive: runFleet(input, 'aggressive'),
  };
}

/** 模型种类分组（按 kind，展示顺序即数组顺序） */
export const FLEET_KIND_ORDER = ['llm', 'vlm', 'generic'] as const;
export type FleetKind = (typeof FLEET_KIND_ORDER)[number];

export function fleetKindOf(model: ModelSpec): FleetKind {
  return model.kind === 'vlm' || model.kind === 'generic' ? model.kind : 'llm';
}

export function fleetKindLabel(kind: FleetKind): string {
  return kind === 'vlm' ? 'VLM（多模态）' : kind === 'generic' ? '非自回归' : 'LLM（文本生成）';
}

export interface FleetEntrySummary {
  kind: FleetKind;
  kindLabel: string;
  /** 权重显存（当前精度） */
  weightsGB: number;
  /** KV 常驻显存（当前并发 × 上下文，未折减口径） */
  kvGB: number;
  /** 单副本服务声明负载的总显存 = 权重 + KV + 开销（典型档系数） */
  totalGB: number;
  /** 权重装载下界；null = 本硬件 TP×PP 枚举内放不下 */
  floor: number | null;
  /** 服务声明并发的最少卡数 */
  need: number | null;
}

/** 单条目的清单摘要（类型 / 显存 / 最少卡数），供 ①② 行展示；典型档口径。 */
export function fleetEntrySummary(entry: FleetEntry, gpu: GpuSpec, opts: Optimizations = OPTS_OFF): FleetEntrySummary {
  const si = searchInputOfEntry(entry, 'typical', gpu, opts);
  const weightsGB = (entry.model.paramsB * 1e9 * si.bytesPerParam) / 1e9;
  const kvGB = (entry.concurrency * entry.avgCtxTokens * si.kvBytesPerToken) / 1e9;
  const overheadGB = si.overheadBaseGB + weightsGB * si.overheadFactor;
  const floor = weightsFloorOf(si);
  const need = floor === null
    ? null
    : (() => {
        const minimal = findMinimalPlan(si);
        return minimal.feasible ? minimal.totalDevices : floor;
      })();
  const kind = fleetKindOf(entry.model);
  return {
    kind,
    kindLabel: fleetKindLabel(kind),
    weightsGB,
    kvGB,
    totalGB: weightsGB + kvGB + overheadGB,
    floor,
    need,
  };
}
