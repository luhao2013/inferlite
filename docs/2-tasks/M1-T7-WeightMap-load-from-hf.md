# M1-T7 WeightMap + load_from_hf

## 元信息
- **任务 ID**: T7
- **里程碑**: M1·P1（数值对齐）
- **状态**: 🟡 in-progress
- **前置**: T6 `Qwen3Model`
- **估时**: 3h

## 目标
实现从本地 HuggingFace / ModelScope Qwen3-0.6B 权重目录加载 `config.json` 与 `model.safetensors`，并把 HF state_dict key 精确映射到 inferlite `Qwen3Model`。T7 完成后，`Qwen3Model` 应能加载真实 Qwen3-0.6B 权重，但还不负责 lm_head logits 对齐和文本生成。

## 产出文件
- `inferlite/model/weights.py::load_from_hf`
- `inferlite/model/weights.py::map_hf_key_to_inferlite_key` 或等价 WeightMap 逻辑
- `tests/unit/test_weights.py`
- 如有必要：`inferlite/model/__init__.py` 导出

## 参考代码

### 主参考
- HF safetensors key（来自 Qwen3-0.6B `model.safetensors`）
- Transformers 固定版 `Qwen3Model` / `Qwen3ForCausalLM` 命名：
  https://github.com/huggingface/transformers/blob/0dad7b822255a0ae261ec45ae937371e859ffd1a/src/transformers/models/qwen3/modeling_qwen3.py

### 辅助参考
- vLLM Qwen3 权重加载：
  https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py

只借鉴 key 映射思路；不要引入 vLLM 的 fused qkv、gate/up packing、TP、quant、cache engine。

## 权重物理位置

ModelScope 常见缓存路径：

```text
~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

T7 单元测试不应强依赖真实 0.6B 文件存在；真实文件可作为可选集成测试 / 手动 smoke test。

## key 映射核心

### 1. 主体映射

HF key 示例：

```text
model.embed_tokens.weight
model.layers.0.input_layernorm.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.self_attn.q_norm.weight
model.layers.0.self_attn.k_norm.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.post_attention_layernorm.weight
model.norm.weight
```

inferlite 当前命名应尽量保持同构：

```text
embed_tokens.weight
layers.0.input_layernorm.weight
layers.0.self_attn.q_proj.weight
layers.0.self_attn.k_proj.weight
layers.0.self_attn.v_proj.weight
layers.0.self_attn.o_proj.weight
layers.0.self_attn.q_norm.weight
layers.0.self_attn.k_norm.weight
layers.0.mlp.gate_proj.weight
layers.0.mlp.up_proj.weight
layers.0.mlp.down_proj.weight
layers.0.post_attention_layernorm.weight
norm.weight
```

因此最小映射规则可以是：

```python
if hf_key.startswith("model."):
    inferlite_key = hf_key.removeprefix("model.")
```

### 2. lm_head 暂不加载

HF 可能有：

```text
lm_head.weight
```

T6 `Qwen3Model` 没有 `lm_head`，所以 T7 对它应显式跳过或记录为 unused。T8 引入 logits/lm_head 时再处理。

### 3. tied embedding 风险

Qwen3-0.6B `tie_word_embeddings=True`。在 HF CausalLM 中，`lm_head.weight` 可能与 `embed_tokens.weight` 共享语义。T7 当前只加载 backbone，先保证：

```text
model.embed_tokens.weight -> embed_tokens.weight
```

不要提前创造 lm_head。

## API 草案

```python
def load_from_hf(
    model: Qwen3Model,
    model_dir: str | Path,
    *,
    strict: bool = True,
) -> None:
    """从 HF/ModelScope 本地目录加载 Qwen3Model backbone 权重。"""
```

建议行为：

1. 读取 `model.safetensors`
2. 遍历 HF key
3. 映射到 inferlite key
4. 跳过 T6 不存在的 key（如 `lm_head.weight`）
5. 校验 shape
6. `model.load_state_dict(mapped_state_dict, strict=strict)`

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
| --- | --- | --- | --- |
| 1 | key 映射 `model.` 前缀去除 | 手工 key 表 | exact |
| 2 | `lm_head.weight` 在 backbone 阶段被跳过 | 手工 state_dict | exact |
| 3 | 小尺寸 fake safetensors 可加载 | 自造 `Qwen3Model.state_dict()` | exact |
| 4 | shape mismatch 能早失败 | 人造错误 shape | raise |
| 5 | missing/unexpected key 报告清晰 | `load_state_dict` 结果 | exact |
| 6 | 可选 smoke：真实 Qwen3-0.6B safetensors key 能被覆盖到 backbone | 本地真实模型文件 | no error |

## DoD
- [ ] `tests/unit/test_weights.py` 全绿
- [ ] fake safetensors 加载后模型参数与源 state_dict 完全一致
- [ ] `lm_head.weight` 在 T7 被明确 skip，不静默误加载
- [ ] shape mismatch 有清晰错误
- [ ] `uv run pytest tests/unit/test_config.py tests/unit/test_rmsnorm.py tests/unit/test_mlp.py tests/unit/test_rope.py tests/unit/test_attention.py tests/unit/test_decoder_layer.py tests/unit/test_qwen3_model.py tests/unit/test_weights.py -q` 全绿
- [ ] `docs/3-kb/knowledge.md` 补权重加载 / safetensors / HF key 映射知识卡（如缺）
- [ ] `docs/2-tasks/README.md` 与 `docs/1-plan/M1.md` 状态更新
- [ ] `README.md` 当前进度同步（若本任务完成）
- [ ] commit `feat(model): add HF weight loading for Qwen3Model`

## 坑（按概率排序）
1. **误加载 lm_head**：T6 backbone 没有 lm_head，T7 不要为了消 unexpected key 加模块。
2. **key 前缀处理不一致**：HF 是 `model.layers...`，inferlite 是 `layers...`。
3. **shape mismatch 静默跳过**：必须早失败，否则 T8 logits 对齐会很难定位。
4. **真实文件依赖导致 CI 不稳定**：unit test 用 fake safetensors；真实 Qwen3-0.6B 只做可选 smoke。
5. **dtype 问题**：safetensors 可能是 bf16/fp16；加载后模型参数 dtype 要与目标模型一致或明确转换策略。
6. **tied weight 提前复杂化**：tie embedding/lm_head 留到 T8 CausalLM/logits 阶段。
7. **vLLM fused 权重映射误抄**：inferlite 当前模块未 fused，保持 HF 同构 key 最简单。
