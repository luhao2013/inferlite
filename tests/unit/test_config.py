"""ModelConfig L0 单测：5 测试覆盖工厂 / JSON / 兜底 / 不可变 / GQA 校验。"""

import json
from dataclasses import FrozenInstanceError

import pytest

from inferlite.config import ModelConfig


def test_qwen3_0_6b_factory():
    """T1: 硬编码工厂方法 11 字段全对（值来自 modelscope config.json）。"""
    c = ModelConfig.qwen3_0_6b()
    assert c.hidden_size == 1024
    assert c.num_hidden_layers == 28
    assert c.num_attention_heads == 16
    assert c.num_key_value_heads == 8
    assert c.head_dim == 128  # ⚠️ 不等于 H/n_q=64
    assert c.intermediate_size == 3072
    assert c.vocab_size == 151936
    assert c.max_position_embeddings == 40960
    assert c.rope_theta == 1_000_000.0
    assert c.rms_norm_eps == pytest.approx(1e-6)
    assert c.tie_word_embeddings is True


def test_from_json_matches_factory(tmp_path):
    """T2: from_json(...) 与 qwen3_0_6b() 完全一致（JSON round-trip + 白名单过滤）。"""
    p = tmp_path / "config.json"
    # 模拟 HF config.json，含一堆白名单外的字段（应被忽略）
    p.write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],  # 应被过滤
                "torch_dtype": "bfloat16",  # 应被过滤
                "attention_dropout": 0.0,  # 应被过滤
                # 白名单 11 项
                "hidden_size": 1024,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 3072,
                "vocab_size": 151936,
                "max_position_embeddings": 40960,
                "rope_theta": 1000000,  # JSON int, 应被 cast 成 float
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": True,
            }
        )
    )
    c1 = ModelConfig.from_json(p)
    c2 = ModelConfig.qwen3_0_6b()
    assert c1 == c2
    assert isinstance(c1.rope_theta, float)


def test_from_json_head_dim_fallback(tmp_path):
    """T3: 当 config.json 缺 head_dim 时用 H // n_q 兜底（部分老模型 config 没这字段）。"""
    p = tmp_path / "config.json"
    p.write_text(
        json.dumps(
            {
                "hidden_size": 512,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                # 注意：故意不写 head_dim
                "intermediate_size": 1024,
                "vocab_size": 100,
                "max_position_embeddings": 128,
                "rope_theta": 10000,
                "rms_norm_eps": 1e-5,
                "tie_word_embeddings": False,
            }
        )
    )
    c = ModelConfig.from_json(p)
    assert c.head_dim == 64  # 512 // 8


def test_frozen_immutable():
    """T4: frozen=True 不允许字段修改（推理时 config 是只读契约）。"""
    c = ModelConfig.qwen3_0_6b()
    with pytest.raises(FrozenInstanceError):
        c.hidden_size = 9999  # type: ignore[misc]


def test_gqa_divisibility_validation():
    """T5: __post_init__ 校验 GQA 合法性 (n_q % n_kv == 0)。"""
    with pytest.raises(AssertionError, match="GQA"):
        ModelConfig(
            hidden_size=1024,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=7,  # ❌ 16 % 7 != 0
            head_dim=128,
            intermediate_size=3072,
            vocab_size=151936,
            max_position_embeddings=40960,
            rope_theta=1e6,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
        )
