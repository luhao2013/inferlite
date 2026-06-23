# inferlite 文档目录

> 从零手写 LLM 推理引擎 · 代码全手敲 · AI 辅助规划/复盘/文档
> GitHub: [luhao-lab/inferlite](https://github.com/luhao-lab/inferlite)

---

## 目录结构

```
docs/
├── README.md          ← 本文件，目录总览
├── setup.md           ← 环境安装、make 命令速查、大坑预警（新人必读）
│
├── plan/              ← 「规划层」记录做什么、为什么、做到哪了
│   ├── PLAN.md            里程碑路线图（14 个 M，项目全貌）
│   ├── PROGRESS.md        每个 M 的进度状态 + 变更日志
│   ├── M1.md              M1 作战地图：架构图、任务总表、测试金字塔
│   └── m2-kv-cache-design.md   M2 技术设计文档：方案调研、ADR、数据流
│
├── tasks/             ← 「执行层」每张任务卡是一次 PR 粒度的作战单元
│   ├── _TEMPLATE.md       任务卡 7 字段模板（AI /work 命令自动填充）
│   ├── M2-T*.md（5张）    当前活跃任务卡（M2 KV Cache 阶段）
│   └── M1-archive/        已完成的 M1 任务卡（历史档案，可查学习记录）
│       └── M1-T*.md（12张）
│
└── kb/                ← 「知识层」沉淀可复用的知识，防止经验流失
    ├── knowledge.md       知识卡片库：Papers / Libraries / Concepts / Tools / ADR / 参考资料
    ├── lessons.md         踩坑教训（L1~L4，叙事性，按时间追加）
    └── blueprints.md      模块契约：每个核心模块的接口、踩坑、跨 M 依赖
```

---

## 阅读指引

**新开一个里程碑**（如 M2）：
1. `plan/PROGRESS.md` — 确认当前进度状态
2. `plan/m2-kv-cache-design.md` — 读技术设计文档，理解方案
3. `tasks/M2-T*.md` — 逐卡推进

**接手一张任务卡**（如 M2-T1）：
1. 读对应 `tasks/M2-T1-*.md` — 算法核心、测试清单、DoD
2. `kb/knowledge.md` — 查前置知识章节
3. `kb/blueprints.md` — 查涉及模块的接口契约

**踩坑 / 复盘**：
- `kb/lessons.md` — 查已有的坑，避免重蹈
- 完成任务卡后，往 `kb/lessons.md` 和 `kb/knowledge.md` 追加一条

**项目全貌**：
- `plan/PLAN.md` — 14 个里程碑，从 M1 单序列推理到 M14 VLM 工程化

---

## 当前进度

M1 Qwen3 单序列推理 ✅ → **M2 KV Cache 进行中** → M3 Continuous Batching ⬜

详见 [plan/PROGRESS.md](./plan/PROGRESS.md)
