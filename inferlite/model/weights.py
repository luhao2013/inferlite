"""Qwen3Model 的 HuggingFace / ModelScope 权重加载工具。

T7 解决的是“模型结构已经有了，如何把训练好的参数灌进去”的问题。

完整链路分成两层：

1. 高层入口 `load_from_hf(model_dir)`：

   ```text
   model_dir/config.json
     -> ModelConfig.from_json
     -> Qwen3Model(config)          # 先搭出结构，此时参数仍是随机初始化
     -> load_weights_into_model
     -> 返回已加载权重的 Qwen3Model
   ```

2. 底层入口 `load_weights_into_model(model, model_dir)`：

   ```text
   model_dir/model.safetensors
     -> safetensors.torch.load_file
     -> HF state_dict
     -> key 映射成 inferlite state_dict
     -> model.load_state_dict
   ```

术语说明：
- `state_dict` 是 PyTorch 模型的“参数字典”：key 是模块路径，value 是 Tensor。
- `model.safetensors` 本质上是保存到磁盘上的 HF state_dict。
- HF 的 Qwen3 CausalLM key 通常带外层 `model.` 前缀；inferlite 当前 `Qwen3Model`
  是裸 backbone，没有这层外壳，所以需要做 key 映射。
"""

from pathlib import Path

import torch
from safetensors.torch import load_file

from inferlite.config import ModelConfig
from inferlite.model.qwen3 import Qwen3Model

# T7 只实现 Qwen3Model backbone 加载，不实现 Qwen3ForCausalLM。
# HF CausalLM checkpoint 里可能带 `lm_head.weight`，但当前模型没有 lm_head 模块。
# 这里把它列成“显式跳过”的 key，避免两种错误：
# 1. strict=True 时因为 unexpected key 失败；
# 2. 为了消除 unexpected key 而提前在 T7 创建 lm_head，破坏任务边界。
_SKIPPED_HF_KEYS = {"lm_head.weight"}


def map_hf_key_to_inferlite_key(hf_key: str) -> str | None:
    """把 HF checkpoint key 映射成 inferlite Qwen3Model 的 state_dict key。

    例子：

    ```text
    HF / Transformers:
      model.layers.0.self_attn.q_proj.weight

    inferlite Qwen3Model:
      layers.0.self_attn.q_proj.weight
    ```

    为什么 HF 多了 `model.`？
    - HF 的 `Qwen3ForCausalLM` 一般长这样：`self.model = Qwen3Model(...)`。
    - 所以 backbone 权重会放在 `model.xxx` 下面。
    - inferlite T6/T7 当前直接暴露裸 `Qwen3Model`，没有 `self.model` 外壳。

    Returns:
        - `str`: 映射后的 inferlite key。
        - `None`: 表示这个 HF key 在 T7 阶段应跳过，例如 `lm_head.weight`。
    """
    if hf_key in _SKIPPED_HF_KEYS:
        return None
    if hf_key.startswith("model."):
        return hf_key.removeprefix("model.")
    return hf_key


def _model_safetensors_path(model_dir: str | Path) -> Path:
    """定位单文件 `model.safetensors`，并在缺失时早失败。

    Qwen3-0.6B 通常可以是单个：

    ```text
    model.safetensors
    ```

    更大的模型可能会分片：

    ```text
    model-00001-of-000xx.safetensors
    model-00002-of-000xx.safetensors
    model.safetensors.index.json
    ```

    T7 先只支持单文件权重；如果文件不存在，直接抛 `FileNotFoundError`，
    比后面 `load_file` 报一个更底层的错误更清楚。
    """
    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        msg = f"model.safetensors not found: {path}"
        raise FileNotFoundError(msg)
    return path


def _build_mapped_state_dict(
    hf_state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
    strict: bool,
) -> dict[str, torch.Tensor]:
    """把 HF state_dict 转成 inferlite state_dict，并做加载前校验。

    参数关系：
    - `hf_state_dict`: 从 `model.safetensors` 读出来的 checkpoint 参数。
      key 形如 `model.layers.0.self_attn.q_proj.weight`。
    - `model_state_dict`: 当前 inferlite `Qwen3Model` 自己的参数字典。
      key 形如 `layers.0.self_attn.q_proj.weight`。
    - `strict`: 是否严格处理 checkpoint 里无法映射到当前模型的 key。

    为什么不直接把 HF state_dict 喂给 `load_state_dict`？
    - key 前缀不同：HF 有 `model.`，inferlite 没有。
    - T7 要跳过 `lm_head.weight`。
    - shape mismatch 最好在这里给出带 HF key 和 inferlite key 的清晰错误。
    """
    mapped_state_dict: dict[str, torch.Tensor] = {}

    for hf_key, tensor in hf_state_dict.items():
        # Step 1: key 映射。
        # - 普通 backbone key：`model.xxx` -> `xxx`
        # - `lm_head.weight`：返回 None，T7 显式跳过
        inferlite_key = map_hf_key_to_inferlite_key(hf_key)
        if inferlite_key is None:
            continue

        # Step 2: 检查映射后的 key 是否真存在于当前模型。
        # 如果 strict=True，未知 key 要早失败；否则跳过未知 key。
        # 这样可以区分：
        # - `lm_head.weight`：任务边界内明确允许 skip
        # - `model.not_a_real_weight`：很可能是映射规则或模型结构错了
        if inferlite_key not in model_state_dict:
            if strict:
                msg = f"Unexpected HF weight key after mapping: {hf_key!r} -> {inferlite_key!r}"
                raise KeyError(msg)
            continue

        # Step 3: shape 校验。
        # PyTorch load_state_dict 自己也会检查 shape，但这里提前检查可以报出
        # “HF 原始 key -> inferlite key”的完整映射关系，定位更快。
        expected_shape = model_state_dict[inferlite_key].shape
        if tensor.shape != expected_shape:
            msg = (
                f"Shape mismatch for {hf_key!r} -> {inferlite_key!r}: "
                f"checkpoint {tuple(tensor.shape)} vs model {tuple(expected_shape)}"
            )
            raise ValueError(msg)

        # Step 4: 放入最终要传给 load_state_dict 的字典。
        # 注意这里不主动 cast dtype；PyTorch 默认会把 checkpoint tensor copy 到
        # 目标 Parameter 里，最终参数 dtype 跟目标模型参数保持一致。
        mapped_state_dict[inferlite_key] = tensor

    return mapped_state_dict


def load_weights_into_model(
    model: Qwen3Model,
    model_dir: str | Path,
    strict: bool = True,
) -> None:
    """从本地 HF/ModelScope 目录读取 `model.safetensors` 并灌入已有模型。

    这是 T7 的底层 API，适合测试和手动控制：

    ```python
    cfg = ModelConfig.from_json(model_dir / "config.json")
    model = Qwen3Model(cfg)
    load_weights_into_model(model, model_dir)
    ```

    Args:
        model: 已经按 config 构造好的 inferlite `Qwen3Model`。
        model_dir: 包含 `model.safetensors` 的本地模型目录。
        strict: 传递给 key 映射校验和 PyTorch `load_state_dict` 的严格模式。

    Raises:
        FileNotFoundError: `model.safetensors` 不存在。
        KeyError: strict=True 且出现非跳过的未知 checkpoint key。
        ValueError: checkpoint tensor shape 与当前模型参数 shape 不一致。
        RuntimeError: `load_state_dict` 发现 missing key 等严格加载错误。
    """
    # 1. 找到权重文件。
    checkpoint_path = _model_safetensors_path(model_dir)

    # 2. 读取 safetensors。
    # `load_file` 返回 dict[str, Tensor]，即 HF checkpoint state_dict。
    hf_state_dict = load_file(checkpoint_path)

    # 3. 取当前模型的 state_dict 作为“目标 key/shape 清单”。
    # 后续映射时会用它判断 key 是否存在、shape 是否匹配。
    model_state_dict = model.state_dict()

    # 4. HF key -> inferlite key，并做 skip/unknown/shape 校验。
    mapped_state_dict = _build_mapped_state_dict(
        hf_state_dict,
        model_state_dict,
        strict,
    )

    # 5. 真正把 tensor copy 到模型参数里。
    #
    # `load_state_dict` 是 PyTorch `nn.Module` 内置的权重加载接口，作用是：
    #
    #   mapped_state_dict["layers.0.self_attn.q_proj.weight"]
    #       -> model.layers[0].self_attn.q_proj.weight
    #
    # 它不是按“对象引用”加载，而是按 state_dict 的字符串 key 逐个查找目标参数：
    # - key 对得上：把 checkpoint tensor 拷贝进对应 Parameter / buffer。
    # - key 少了：记为 missing_keys。
    # - key 多了：记为 unexpected_keys。
    # - shape 不一致：直接报错，不能加载。
    #
    # `strict` 控制 missing/unexpected key 的处理方式：
    # - strict=True：checkpoint 和 model 的 key 集合必须严格一致，否则抛 RuntimeError。
    # - strict=False：允许 missing/unexpected key，但返回值里会记录它们。
    #
    # 返回值类型是 PyTorch 的 `_IncompatibleKeys`，里面主要有：
    # - missing_keys：模型需要但 checkpoint 没提供的 key。
    # - unexpected_keys：checkpoint 里有但模型不认识的 key。
    #
    # T7 当前 API 只需要“加载成功或失败”的语义，不对外暴露 missing/unexpected 明细；
    # 所以赋给 `_`，表示我们知道有返回值，但这里有意忽略。
    _ = model.load_state_dict(mapped_state_dict, strict=strict)


def load_from_hf(model_dir: str | Path, strict: bool = True) -> Qwen3Model:
    """完整 from_pretrained 风格入口：读 config、建模型、加载权重、返回模型。

    这是 T7 给调用方最方便的高级 API：

    ```python
    model = load_from_hf("~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B")
    ```

    它完成：

    ```text
    config.json
      -> ModelConfig.from_json
      -> Qwen3Model(config)
      -> load_weights_into_model(model, model_dir)
      -> return model
    ```

    注意：T7 返回的是 backbone `Qwen3Model`，不是 CausalLM；因此不会输出 logits，
    也不会处理 tokenizer/generation_config/chat template。
    """
    model_dir = Path(model_dir)
    config = ModelConfig.from_json(model_dir / "config.json")
    model = Qwen3Model(config)
    load_weights_into_model(model, model_dir, strict=strict)
    return model
