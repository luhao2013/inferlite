"""Qwen3Model / Qwen3ForCausalLM 的 HuggingFace / ModelScope 权重加载工具。

T7 解决 backbone 权重加载，T8 在此基础上补 CausalLM/lm_head 权重加载。

完整链路分成两层：

1. 高层入口：

   ```text
   load_from_hf(model_dir)
     -> config.json -> Qwen3Model(config) -> model.safetensors -> backbone 权重

   load_causal_lm_from_hf(model_dir)
     -> config.json -> Qwen3ForCausalLM(config) -> model.safetensors -> backbone + lm_head 权重
   ```

2. 底层入口 `load_weights_into_model(model, model_dir, target=...)`：

   ```text
   model_dir/model.safetensors
     -> safetensors.torch.load_file
     -> HF state_dict
     -> 按 target 做 key 映射
     -> model.load_state_dict
   ```

术语说明：
- `state_dict` 是 PyTorch 模型的“参数字典”：key 是模块路径，value 是 Tensor。
- `model.safetensors` 本质上是保存到磁盘上的 HF state_dict。
- backbone 模式加载裸 `Qwen3Model`，需要把 HF 的 `model.xxx` 映射为 `xxx`。
- causal_lm 模式加载 `Qwen3ForCausalLM`，其 state_dict key 与 HF CausalLM 基本一致。
"""

from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file

from inferlite.config import ModelConfig
from inferlite.model.qwen3 import Qwen3ForCausalLM, Qwen3Model

# backbone 模式加载裸 Qwen3Model，没有 lm_head，所以要跳过 lm_head.weight。
# causal_lm 模式加载完整 Qwen3ForCausalLM，lm_head.weight 会原样加载。
_SKIPPED_HF_KEYS = {"lm_head.weight"}


def map_hf_key_to_inferlite_key(hf_key: str, target: str) -> str | None:
    """把 HF checkpoint key 映射成 inferlite state_dict key。

    两种 target 对应两种模型结构：

    1. `target="backbone"`：加载裸 `Qwen3Model`

       ```text
       HF key:        model.layers.0.self_attn.q_proj.weight
       inferlite key: layers.0.self_attn.q_proj.weight
       ```

       因为裸 `Qwen3Model` 没有外层 `self.model`，所以要去掉 `model.` 前缀。
       同时裸 backbone 没有 `lm_head`，所以 `lm_head.weight` 返回 None，表示跳过。

    2. `target="causal_lm"`：加载完整 `Qwen3ForCausalLM`

       ```text
       HF key:        model.layers.0.self_attn.q_proj.weight
       inferlite key: model.layers.0.self_attn.q_proj.weight
       HF key:        lm_head.weight
       inferlite key: lm_head.weight
       ```

       因为 `Qwen3ForCausalLM` 内部字段也叫 `self.model` 和 `self.lm_head`，
       它的 state_dict key 和 HF CausalLM checkpoint 基本天然一致，所以原样返回。
    """
    # codeflicker-fix: LOGIC-Issue-001/dwv03qen2tgtzojek3hz
    if target == "backbone":
        if hf_key in _SKIPPED_HF_KEYS:
            return None
        if hf_key.startswith("model."):
            return hf_key.removeprefix("model.")
        # backbone checkpoint 里不应该出现既不属于 _SKIPPED_HF_KEYS、
        # 也不以 "model." 开头的 key（Qwen3 checkpoint 不存在这种 key）。
        # 原来会 fall-through 到末尾 return hf_key，静默地把 HF key 原样返回；
        # 现在改为明确报错，方便在加载其他 checkpoint 时快速定位问题。
        msg = (
            f"Unexpected HF key in backbone mode: {hf_key!r}. "
            "Expected keys to start with 'model.' or be in _SKIPPED_HF_KEYS. "
            "Check the checkpoint format or update _SKIPPED_HF_KEYS."
        )
        raise ValueError(msg)
    elif target == "causal_lm":
        return hf_key
    else:
        msg = f"Unknown weight loading target: {target!r}"
        raise ValueError(msg)


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
    target: Literal["backbone", "causal_lm"],
) -> dict[str, torch.Tensor]:
    """把 HF state_dict 转成 inferlite state_dict，并做加载前校验。

    参数关系：
    - `hf_state_dict`: 从 `model.safetensors` 读出来的 checkpoint 参数。
      backbone / CausalLM checkpoint 里常见 key 包括：
        - `model.layers.0.self_attn.q_proj.weight`
        - `model.norm.weight`
        - `lm_head.weight`
    - `model_state_dict`: 当前 inferlite 模型自己的参数字典。
      如果当前模型是裸 `Qwen3Model`，key 形如 `layers.0...`；
      如果当前模型是 `Qwen3ForCausalLM`，key 形如 `model.layers.0...` 和 `lm_head.weight`。
    - `strict`: 是否严格处理 checkpoint 里无法映射到当前模型的 key。
    - `target`: 控制 HF key 如何映射到当前模型：
        - `backbone`: 给裸 `Qwen3Model` 加载，去掉 `model.`，跳过 `lm_head.weight`。
        - `causal_lm`: 给完整 `Qwen3ForCausalLM` 加载，key 原样保留。

    为什么不直接把 HF state_dict 喂给 `load_state_dict`？
    - backbone 模式下 key 前缀不同：HF 有 `model.`，裸 Qwen3Model 没有。
    - backbone 模式下要跳过 `lm_head.weight`，因为 T7 的裸模型没有 lm_head。
    - CausalLM 模式下不应跳过 `lm_head.weight`，因为 T8 正是要加载它。
    - shape mismatch 最好在这里给出带 HF key 和 inferlite key 的清晰错误。
    """
    mapped_state_dict: dict[str, torch.Tensor] = {}

    for hf_key, tensor in hf_state_dict.items():
        # Step 1: 根据 target 做 key 映射。
        # - target="backbone":
        #     `model.xxx` -> `xxx`
        #     `lm_head.weight` -> None，表示跳过
        # - target="causal_lm":
        #     `model.xxx` -> `model.xxx`
        #     `lm_head.weight` -> `lm_head.weight`
        #
        # 这一步是 T7/T8 权重加载最核心的差异点。
        inferlite_key = map_hf_key_to_inferlite_key(hf_key, target=target)
        if inferlite_key is None:
            continue

        # Step 2: 检查映射后的 key 是否真存在于当前模型。
        # 如果 strict=True，未知 key 要早失败；否则跳过未知 key。
        # 这样可以区分：
        # - backbone 模式下的 `lm_head.weight`：任务边界内明确允许 skip。
        # - causal_lm 模式下的 `lm_head.weight`：应该能在模型里找到并加载。
        # - `model.not_a_real_weight`：很可能是映射规则或模型结构错了。
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
    model: torch.nn.Module,
    model_dir: str | Path,
    target: Literal["backbone", "causal_lm"] = "backbone",
    strict: bool = True,
) -> None:
    """从本地 HF/ModelScope 目录读取 `model.safetensors` 并灌入已有模型。

    这是 T7/T8 共用的底层 API，适合测试和手动控制：

    ```python
    # T7: 加载裸 backbone
    cfg = ModelConfig.from_json(model_dir / "config.json")
    model = Qwen3Model(cfg)
    load_weights_into_model(model, model_dir, target="backbone")

    # T8: 加载完整 CausalLM
    causal_lm = Qwen3ForCausalLM(cfg)
    load_weights_into_model(causal_lm, model_dir, target="causal_lm")
    ```

    Args:
        model: 已经按 config 构造好的 PyTorch 模型，可以是 `Qwen3Model` 或 `Qwen3ForCausalLM`。
        model_dir: 包含 `model.safetensors` 的本地模型目录。
        target: key 映射目标。`backbone` 用于裸 `Qwen3Model`；`causal_lm` 用于完整 CausalLM。
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
    # 注意 target 决定了这里的 key 映射方式：
    # - backbone: HF `model.xxx` 对应裸模型 `xxx`
    # - causal_lm: HF `model.xxx` 对应外壳模型 `model.xxx`
    mapped_state_dict = _build_mapped_state_dict(
        hf_state_dict,
        model_state_dict,
        strict=strict,
        target=target,
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
    # T7/T8 当前 API 只需要“加载成功或失败”的语义，不对外暴露 missing/unexpected 明细；
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
    load_weights_into_model(
        model,
        model_dir,
        target="backbone",
        strict=strict,
    )
    return model


def load_causal_lm_from_hf(model_dir: str | Path, strict: bool = True) -> Qwen3ForCausalLM:
    """完整 CausalLM 加载入口：读 config、建 Qwen3ForCausalLM、加载 backbone + lm_head。

    这是 T8 推荐的路线 A：不要把 backbone 和 lm_head 拆开两次加载，
    而是构造完整外壳后一次性调用 `load_weights_into_model(..., target="causal_lm")`。

    原因是 `Qwen3ForCausalLM` 内部字段名刻意与 HF 对齐：

    ```text
    self.model    -> state_dict key: model.layers.0...
    self.lm_head  -> state_dict key: lm_head.weight
    ```

    因此 HF CausalLM checkpoint key 可以原样加载：

    ```text
    model.embed_tokens.weight  -> model.embed_tokens.weight
    model.layers.0...          -> model.layers.0...
    lm_head.weight             -> lm_head.weight
    ```
    """
    model_dir = Path(model_dir)
    config = ModelConfig.from_json(model_dir / "config.json")
    model = Qwen3ForCausalLM(config)
    load_weights_into_model(
        model,
        model_dir,
        strict=strict,
        target="causal_lm",
    )
    return model
