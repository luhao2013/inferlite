"""Unit tests for inferlite.model.weights.

T7 目标：从 HF/ModelScope 本地目录加载 config.json + model.safetensors，
构造 Qwen3Model backbone，并把 HF key 映射到 inferlite key 后灌入权重。

为什么这些测试不用真实 Qwen3-0.6B 文件？
- 单测应稳定、快速、可离线运行。
- 真实 0.6B 权重很大，并且依赖本机缓存路径。
- 所以这里用 tiny Qwen3Model 自造一个 fake HF 目录：

  ```text
  tmp/fake-qwen3/
  ├── config.json
  └── model.safetensors
  ```

fake `model.safetensors` 的内容来自一个 source Qwen3Model 的 state_dict，
只是把 key 加上 HF 的 `model.` 前缀，用来模拟真实 HF checkpoint。

运行：
  uv run pytest tests/unit/test_weights.py -q
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from inferlite.config import ModelConfig
from inferlite.model.qwen3 import Qwen3Model
from inferlite.model.weights import (
    load_from_hf,
    load_weights_into_model,
    map_hf_key_to_inferlite_key,
)


def _tiny_model_config(num_hidden_layers: int = 1) -> ModelConfig:
    """构造一个小尺寸 Qwen3 config，避免单测实例化真实 0.6B。"""
    return ModelConfig(
        hidden_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=100,
        max_position_embeddings=32,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )


def _write_config_json(path: Path, cfg: ModelConfig) -> None:
    """写一个 HF 风格 `config.json`，供 `ModelConfig.from_json` 读取。

    这里手写 JSON，而不是直接 pickle/dataclass，是为了覆盖 T7 高层入口真实会走的路径：

    ```text
    load_from_hf(model_dir)
      -> ModelConfig.from_json(model_dir / "config.json")
    ```
    """
    data = {
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "intermediate_size": cfg.intermediate_size,
        "vocab_size": cfg.vocab_size,
        "max_position_embeddings": cfg.max_position_embeddings,
        "rope_theta": cfg.rope_theta,
        "rms_norm_eps": cfg.rms_norm_eps,
        "tie_word_embeddings": cfg.tie_word_embeddings,
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _hf_state_dict_from_model(model: Qwen3Model) -> dict[str, torch.Tensor]:
    """把 inferlite state_dict 伪装成 HF backbone state_dict。

    inferlite key：

    ```text
    layers.0.self_attn.q_proj.weight
    norm.weight
    ```

    fake HF key：

    ```text
    model.layers.0.self_attn.q_proj.weight
    model.norm.weight
    ```

    `detach().clone()` 的作用：
    - detach：测试数据不需要梯度图。
    - clone：避免 source/target 参数共享同一块内存，保证测试是在验证“加载复制”。
    """
    return {f"model.{key}": tensor.detach().clone() for key, tensor in model.state_dict().items()}


def _write_fake_hf_dir(model_dir: Path, model: Qwen3Model, cfg: ModelConfig) -> None:
    """写出一个最小 fake HF 模型目录。

    目录内容：

    ```text
    fake-qwen3/
    ├── config.json
    └── model.safetensors
    ```

    `save_file` 是 `safetensors.torch` 的写入函数，对应实现里的 `load_file`。
    这样单测覆盖的是和真实权重相同的文件格式，而不是临时 Python dict。
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_config_json(model_dir / "config.json", cfg)
    hf_state_dict = _hf_state_dict_from_model(model)
    # T7 backbone 没有 lm_head；这里故意加入该 key，验证加载时会显式跳过。
    # 如果没有这个测试，未来很容易不小心把 lm_head 当作 unexpected key 报错。
    hf_state_dict["lm_head.weight"] = model.embed_tokens.weight.detach().clone()
    save_file(hf_state_dict, model_dir / "model.safetensors")


def test_map_hf_key_to_inferlite_key_removes_model_prefix():
    """HF backbone key 应去掉外层 `model.` 前缀。"""
    assert (
        map_hf_key_to_inferlite_key("model.layers.0.self_attn.q_proj.weight")
        == "layers.0.self_attn.q_proj.weight"
    )
    assert map_hf_key_to_inferlite_key("model.norm.weight") == "norm.weight"


def test_map_hf_key_to_inferlite_key_skips_lm_head():
    """T7 只加载 Qwen3Model backbone，因此 lm_head.weight 应显式跳过。"""
    assert map_hf_key_to_inferlite_key("lm_head.weight") is None


def test_load_weights_into_model_loads_fake_safetensors(tmp_path: Path):
    """底层 API 应能把 fake safetensors 权重灌入已有模型。

    测试方法：
    1. source 模型提供“期望权重”。
    2. target 模型随机初始化。
    3. 把 source 权重写成 fake HF safetensors。
    4. 调用 load_weights_into_model(target, model_dir)。
    5. target 每个 state_dict tensor 都应等于 source。
    """
    torch.manual_seed(0)
    cfg = _tiny_model_config()
    source = Qwen3Model(cfg)
    target = Qwen3Model(cfg)
    model_dir = tmp_path / "fake-qwen3"
    _write_fake_hf_dir(model_dir, source, cfg)

    load_weights_into_model(target, model_dir)

    for key, expected in source.state_dict().items():
        actual = target.state_dict()[key]
        assert torch.equal(actual, expected), key


def test_load_from_hf_reads_config_constructs_model_and_loads_weights(tmp_path: Path):
    """高层 API 应覆盖 config -> model -> weights -> return model 全链路。"""
    torch.manual_seed(0)
    cfg = _tiny_model_config(num_hidden_layers=2)
    source = Qwen3Model(cfg)
    model_dir = tmp_path / "fake-qwen3"
    _write_fake_hf_dir(model_dir, source, cfg)

    loaded = load_from_hf(model_dir)

    # 先验证“结构来自 config.json”。
    assert isinstance(loaded, Qwen3Model)
    assert loaded.config == cfg
    assert len(loaded.layers) == cfg.num_hidden_layers

    # 再验证“参数来自 model.safetensors”。
    for key, expected in source.state_dict().items():
        actual = loaded.state_dict()[key]
        assert torch.equal(actual, expected), key


def test_load_weights_into_model_raises_on_shape_mismatch(tmp_path: Path):
    """checkpoint tensor shape 和模型参数 shape 不一致时应早失败。"""
    cfg = _tiny_model_config()
    model = Qwen3Model(cfg)
    model_dir = tmp_path / "fake-qwen3"
    model_dir.mkdir()

    hf_state_dict = _hf_state_dict_from_model(model)
    # embed_tokens.weight 正确 shape 应是 [vocab_size, hidden_size]。
    # 这里故意写成 [1, 1]，验证实现会抛出清晰的 ValueError。
    hf_state_dict["model.embed_tokens.weight"] = torch.zeros(1, 1)
    save_file(hf_state_dict, model_dir / "model.safetensors")

    with pytest.raises(ValueError, match="Shape mismatch"):
        load_weights_into_model(model, model_dir)


def test_load_weights_into_model_raises_on_unexpected_key_when_strict(tmp_path: Path):
    """strict=True 时，非 skip 列表里的未知 key 应报错。"""
    cfg = _tiny_model_config()
    model = Qwen3Model(cfg)
    model_dir = tmp_path / "fake-qwen3"
    model_dir.mkdir()

    hf_state_dict = _hf_state_dict_from_model(model)
    # 这个 key 映射后会变成 `not_a_real_weight`，当前 Qwen3Model 没有这个参数。
    # 它不是 lm_head.weight 这种任务边界内允许跳过的 key，所以 strict=True 应失败。
    hf_state_dict["model.not_a_real_weight"] = torch.zeros(1)
    save_file(hf_state_dict, model_dir / "model.safetensors")

    with pytest.raises(KeyError, match="Unexpected HF weight key"):
        load_weights_into_model(model, model_dir, strict=True)


def test_load_weights_into_model_raises_when_safetensors_missing(tmp_path: Path):
    """模型目录缺少 model.safetensors 时，应抛 FileNotFoundError。"""
    cfg = _tiny_model_config()
    model = Qwen3Model(cfg)

    with pytest.raises(FileNotFoundError, match="model.safetensors not found"):
        load_weights_into_model(model, tmp_path)
