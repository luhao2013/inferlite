# M1-TX <任务名>

> 任务卡模板。复制改名为 `M1-TX-<short-name>.md`。

## 元信息
- **任务 ID**: TX
- **里程碑**: M1·P1 / M1·P2
- **状态**: ⬜ pending / 🟡 in-progress / ✅ done
- **前置**: TY, TZ
- **估时**: Nh

## 目标

**要解决什么问题**：
（本卡在解决什么瓶颈，之前是什么状态，做完改善什么）

**做完是什么效果**：
（可以用一行命令或一段代码表达：做完之后能跑什么、看到什么结果）

**不做什么**（边界）：
（明确排除范围，防止范围蔓延）

**在推理链路中的位置**：
```
（ASCII 图：本卡的模块在 generate() 调用链中哪个位置，上游给它什么，下游拿走什么）
```

## 产出文件
- `inferlite/.../xxx.py::ClassName`
- `tests/unit/test_xxx.py`

## 算法核心
公式 + 代码骨架（TODO 留给作者）：

```python
class ClassName(nn.Module):
    def __init__(self, ...):
        # TODO
        ...
    def forward(self, x):
        # TODO
        ...
```

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
| --- | --- | --- | --- |
| 1 | ... | `transformers.Qwen3X` | 1e-5 / 5e-3 |

## DoD
- [ ] 测试 N/N 全绿
- [ ] commit `feat(...): ... (TX done)`
- [ ] PROGRESS.md 更新
- [ ] `README.md` 当前进度同步（若本任务改变首页可见进度）
- [ ] docs/tasks/README.md 状态改 ✅

## 坑（按概率排序）
1. ...
2. ...
