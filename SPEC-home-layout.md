# Spec: Home 页面布局重设计（单列纵向）

## Objective

把 LLM GPU 计算器的 Home 页面从「多列栅格」（模型+GPU 并排、配置+结果分栏）重构为「单列纵向」布局：每个功能模块独占一整行、全宽展示，从上到下依次排列，形成清晰的单向信息流，从根本上消除并排导致的卡片高度失衡与窄列横向溢出。

视觉 token（颜色 / 字体 / 圆角 / 阴影）沿用已定稿的 Apple 极简风格，本次只动布局结构，不改视觉样式与业务逻辑。

## 布局结构

页面自上而下的模块顺序（与 Stepper「模型 → 硬件 → 负载 → 结果」流程一致）：

1. Header（标题 + 副标题）
2. Stepper（步骤指示）
3. 样例场景 toolbar
4. 选择模型（ModelPicker）
5. 选择硬件（GpuPicker）
6. 配置负载（ConfigPicker / 多模型时 FleetConfigPicker）
7. 性能优化选项（OptPanel）
8. 推演结果（ResultsPanel / 多模型时 FleetResultsPanel）
9. Footer

约束：
- 每个模块独占一整行、全宽，模块之间不并排
- 模块间距用 `section-gap`（`mb-8 md:mb-10 xl:mb-12`）
- 各模块保留 `animate-fade-in-up` 依次入场

## Commands

- Dev: `npm run dev`
- Build: `npm run build`
- （本项目无测试套件，验证以构建 + 浏览器运行时检查为准）

## Project Structure

- `src/pages/Home.tsx` — 页面骨架与模块编排（本次主要改动）
- `src/sections/*.tsx` — 各功能模块卡片（内部字段排布适配全宽）
- `src/index.css` — 布局工具类（`.page-container` / `.section-gap` / `.section-card`，本次不变）

## 范围与边界

- **Always do**：模块独占一行全宽；模块顺序与 Stepper 一致；单/多模型两态都验证
- **Ask first**：调整模块顺序、合并/拆分模块、改动间距 token 数值
- **Never do**：改业务计算逻辑（`src/lib/*`、`src/data/*`）、改视觉 token（颜色/字体/圆角/阴影）、改单/多模型交互流程、新增依赖

## Success Criteria

- [ ] 单模型与多模型两种模式下，5 个功能模块均独占一行、全宽
- [ ] 页面无横向滚动，无元素横向溢出（检测：任意元素 `scrollWidth <= clientWidth`）
- [ ] 模块顺序 = 模型 → 硬件 → 负载 → 优化 → 结果
- [ ] `npm run build` 通过，无 TypeScript 错误
- [ ] 视觉 token 与改动前一致（颜色 / 字体 / 圆角无变化）

## Open Questions

（当前无未决问题）