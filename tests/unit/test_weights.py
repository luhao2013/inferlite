"""Unit tests for inferlite.model.weights.

T7/T8 的权重加载目标：
- backbone 模式：HF `model.xxx` -> inferlite `Qwen3Model` 的 `xxx`，跳过 `lm_head.weight`。
- causal_lm 模式：HF key 原样加载到 `Qwen3ForCausalLM`，包括 `lm_head.weight`。

这些测试使用 tiny fake HF 目录，避免依赖真实 Qwen3-0.6B 权重缓存。

运行：
  uv run pytest tests/unit/test_weights.py -q
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from inferlite.config import ModelConfig
from inferlite.model.qwen3 import Qwen3ForCausalLM, Qwen3Model
from inferlite.model.weights import (
    load_causal_lm_from_hf,
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
    """写一个 HF 风格 `config.json`，供 `ModelConfig.from_json` 读取。"""
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


def _hf_backbone_state_dict_from_model(model: Qwen3Model) -> dict[str, torch.Tensor]:
    """把裸 Qwen3Model state_dict 伪装成 HF backbone state_dict。"""
    return {f"model.{key}": tensor.detach().clone() for key, tensor in model.state_dict().items()}


def _hf_causal_lm_state_dict_from_model(model: Qwen3ForCausalLM) -> dict[str, torch.Tensor]:
    """Qwen3ForCausalLM 的 key 与 HF CausalLM key 基本一致，直接 clone 即可。"""
    return {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}


def _write_fake_backbone_hf_dir(model_dir: Path, model: Qwen3Model, cfg: ModelConfig) -> None:
    """写出 fake backbone HF 目录：config.json + model.safetensors。"""
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_config_json(model_dir / "config.json", cfg)
    hf_state_dict = _hf_backbone_state_dict_from_model(model)
    # backbone 模式应跳过 lm_head.weight。
    hf_state_dict["lm_head.weight"] = model.embed_tokens.weight.detach().clone()
    save_file(hf_state_dict, model_dir / "model.safetensors")


def _write_fake_causal_lm_hf_dir(
    model_dir: Path,
    model: Qwen3ForCausalLM,
    cfg: ModelConfig,
) -> None:
    """写出 fake CausalLM HF 目录，包含 backbone 权重和 lm_head.weight。"""
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_config_json(model_dir / "config.json", cfg)
    hf_state_dict = _hf_causal_lm_state_dict_from_model(model)
    save_file(hf_state_dict, model_dir / "model.safetensors")


def test_map_hf_key_to_inferlite_key_backbone_removes_model_prefix():
    """backbone 模式：HF `model.xxx` 应映射到裸 Qwen3Model 的 `xxx`。"""
    assert (
        map_hf_key_to_inferlite_key(
            "model.layers.0.self_attn.q_proj.weight",
            target="backbone",
        )
        == "layers.0.self_attn.q_proj.weight"
    )
    assert map_hf_key_to_inferlite_key("model.norm.weight", target="backbone") == "norm.weight"


def test_map_hf_key_to_inferlite_key_backbone_skips_lm_head():
    """backbone 模式：裸 Qwen3Model 没有 lm_head，因此跳过 lm_head.weight。"""
    assert map_hf_key_to_inferlite_key("lm_head.weight", target="backbone") is None


def test_map_hf_key_to_inferlite_key_causal_lm_keeps_keys():
    """causal_lm 模式：完整 Qwen3ForCausalLM key 与 HF key 一致，应原样保留。"""
    assert (
        map_hf_key_to_inferlite_key(
            "model.layers.0.self_attn.q_proj.weight",
            target="causal_lm",
        )
        == "model.layers.0.self_attn.q_proj.weight"
    )
    assert map_hf_key_to_inferlite_key("lm_head.weight", target="causal_lm") == "lm_head.weight"


def test_map_hf_key_to_inferlite_key_rejects_unknown_target():
    """非法 target 应早失败，避免拼写错误静默走错映射逻辑。"""
    with pytest.raises(ValueError, match="Unknown weight loading target"):
        map_hf_key_to_inferlite_key("model.norm.weight", target="bad")


def test_load_weights_into_model_loads_fake_backbone_safetensors(tmp_path: Path):
    """底层 API 应能把 fake backbone safetensors 权重灌入已有 Qwen3Model。"""
    torch.manual_seed(0)
    cfg = _tiny_model_config()
    source = Qwen3Model(cfg)
    target = Qwen3Model(cfg)
    model_dir = tmp_path / "fake-qwen3"
    _write_fake_backbone_hf_dir(model_dir, source, cfg)

    load_weights_into_model(target, model_dir, target="backbone")

    for key, expected in source.state_dict().items():
        actual = target.state_dict()[key]
        assert torch.equal(actual, expected), key


def test_load_from_hf_reads_config_constructs_model_and_loads_weights(tmp_path: Path):
    """backbone 高层 API 应覆盖 config -> Qwen3Model -> weights -> return model。"""
    torch.manual_seed(0)
    cfg = _tiny_model_config(num_hidden_layers=2)
    source = Qwen3Model(cfg)
    model_dir = tmp_path / "fake-qwen3"
    _write_fake_backbone_hf_dir(model_dir, source, cfg)

    loaded = load_from_hf(model_dir)

    assert isinstance(loaded, Qwen3Model)
    assert loaded.config == cfg
    assert len(loaded.layers) == cfg.num_hidden_layers
    for key, expected in source.state_dict().items():
        actual = loaded.state_dict()[key]
        assert torch.equal(actual, expected), key


def test_load_causal_lm_from_hf_loads_backbone_and_lm_head(tmp_path: Path):
    """CausalLM 高层 API 应一次性加载 `model.xxx` 和 `lm_head.weight`。"""
    torch.manual_seed(0)
    cfg = _tiny_model_config(num_hidden_layers=2)
    source = Qwen3ForCausalLM(cfg)
    model_dir = tmp_path / "fake-qwen3-causal-lm"
    _write_fake_causal_lm_hf_dir(model_dir, source, cfg)

    loaded = load_causal_lm_from_hf(model_dir)

    assert isinstance(loaded, Qwen3ForCausalLM)
    assert loaded.config == cfg
    assert tuple(loaded.lm_head.weight.shape) == (cfg.vocab_size, cfg.hidden_size)
    for key, expected in source.state_dict().items():
        actual = loaded.state_dict()[key]
        assert torch.equal(actual, expected), key


def test_load_weights_into_model_raises_on_shape_mismatch(tmp_path: Path):
    """checkpoint tensor shape 和模型参数 shape 不一致时应早失败。"""
    cfg = _tiny_model_config()
    model = Qwen3Model(cfg)
    model_dir = tmp_path / "fake-qwen3"
    model_dir.mkdir()

    hf_state_dict = _hf_backbone_state_dict_from_model(model)
    hf_state_dict["model.embed_tokens.weight"] = torch.zeros(1, 1)
    save_file(hf_state_dict, model_dir / "model.safetensors")

    with pytest.raises(ValueError, match="Shape mismatch"):
        load_weights_into_model(model, model_dir, target="backbone")


def test_load_weights_into_model_raises_on_unexpected_key_when_strict(tmp_path: Path):
    """strict=True 时，非 skip 列表里的未知 key 应报错。"""
    cfg = _tiny_model_config()
    model = Qwen3Model(cfg)
    model_dir = tmp_path / "fake-qwen3"
    model_dir.mkdir()

    hf_state_dict = _hf_backbone_state_dict_from_model(model)
    hf_state_dict["model.not_a_real_weight"] = torch.zeros(1)
    save_file(hf_state_dict, model_dir / "model.safetensors")

    with pytest.raises(KeyError, match="Unexpected HF weight key"):
        load_weights_into_model(model, model_dir, target="backbone", strict=True)


def test_load_weights_into_model_raises_when_safetensors_missing(tmp_path: Path):
    """模型目录缺少 model.safetensors 时，应抛 FileNotFoundError。"""
    cfg = _tiny_model_config()
    model = Qwen3Model(cfg)

    with pytest.raises(FileNotFoundError, match="model.safetensors not found"):
        load_weights_into_model(model, tmp_path)
