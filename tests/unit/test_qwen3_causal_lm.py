"""Unit tests for inferlite.model.qwen3.Qwen3ForCausalLM.

T8 目标：在 Qwen3Model backbone 后接 lm_head，输出 next-token logits，
并在小尺寸 fp32 下与 transformers.Qwen3ForCausalLM 对齐。

运行：
  uv run pytest tests/unit/test_qwen3_causal_lm.py -q
"""

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM as RefQwen3ForCausalLM

from inferlite.config import ModelConfig
from inferlite.model.qwen3 import Qwen3ForCausalLM, Qwen3Model


def _tiny_model_config(num_hidden_layers: int = 2) -> ModelConfig:
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


def _tiny_qwen3_config(num_hidden_layers: int = 2) -> Qwen3Config:
    return Qwen3Config(
        hidden_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=100,
        max_position_embeddings=32,
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=False,
        tie_word_embeddings=False,
    )


def _copy_causal_lm_weights(mine: Qwen3ForCausalLM, ref: RefQwen3ForCausalLM) -> None:
    """复制 transformers.Qwen3ForCausalLM 权重到 inferlite.Qwen3ForCausalLM。"""
    mine.model.embed_tokens.weight.data.copy_(ref.model.embed_tokens.weight.data)
    mine.model.norm.weight.data.copy_(ref.model.norm.weight.data)
    mine.lm_head.weight.data.copy_(ref.lm_head.weight.data)

    for mine_layer, ref_layer in zip(mine.model.layers, ref.model.layers, strict=True):
        mine_layer.self_attn.q_proj.weight.data.copy_(ref_layer.self_attn.q_proj.weight.data)
        mine_layer.self_attn.k_proj.weight.data.copy_(ref_layer.self_attn.k_proj.weight.data)
        mine_layer.self_attn.v_proj.weight.data.copy_(ref_layer.self_attn.v_proj.weight.data)
        mine_layer.self_attn.o_proj.weight.data.copy_(ref_layer.self_attn.o_proj.weight.data)
        mine_layer.self_attn.q_norm.weight.data.copy_(ref_layer.self_attn.q_norm.weight.data)
        mine_layer.self_attn.k_norm.weight.data.copy_(ref_layer.self_attn.k_norm.weight.data)

        mine_layer.mlp.gate_proj.weight.data.copy_(ref_layer.mlp.gate_proj.weight.data)
        mine_layer.mlp.up_proj.weight.data.copy_(ref_layer.mlp.up_proj.weight.data)
        mine_layer.mlp.down_proj.weight.data.copy_(ref_layer.mlp.down_proj.weight.data)

        mine_layer.input_layernorm.weight.data.copy_(ref_layer.input_layernorm.weight.data)
        mine_layer.post_attention_layernorm.weight.data.copy_(
            ref_layer.post_attention_layernorm.weight.data
        )


def test_qwen3_causal_lm_structure_and_shapes():
    """CausalLM 外壳应包含 backbone model 和 lm_head。"""
    cfg = _tiny_model_config(num_hidden_layers=2)
    model = Qwen3ForCausalLM(cfg)

    assert isinstance(model.model, Qwen3Model)
    assert model.vocab_size == cfg.vocab_size
    assert tuple(model.lm_head.weight.shape) == (cfg.vocab_size, cfg.hidden_size)


def test_qwen3_causal_lm_forward_logits_shape():
    """input_ids [B, T] 应输出 logits [B, T, vocab_size]。"""
    torch.manual_seed(0)
    cfg = _tiny_model_config(num_hidden_layers=2)
    model = Qwen3ForCausalLM(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (2, 5, cfg.vocab_size)


def test_qwen3_causal_lm_auto_position_ids_matches_explicit_position_ids():
    """CausalLM 不传 position_ids 时，应透传 backbone 的自动 position_ids 能力。"""
    torch.manual_seed(0)
    cfg = _tiny_model_config(num_hidden_layers=2)
    model = Qwen3ForCausalLM(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
    position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)

    with torch.no_grad():
        auto_logits = model(input_ids)
        explicit_logits = model(input_ids, position_ids=position_ids)

    assert torch.allclose(auto_logits, explicit_logits, atol=0, rtol=0)


def test_qwen3_causal_lm_vs_transformers_logits_fp32():
    """小尺寸 fp32 下，inferlite logits 应与 transformers.Qwen3ForCausalLM 对齐。"""
    torch.manual_seed(0)
    cfg = _tiny_model_config(num_hidden_layers=2)
    ref_cfg = _tiny_qwen3_config(num_hidden_layers=2)
    mine = Qwen3ForCausalLM(cfg).eval()
    ref = RefQwen3ForCausalLM(ref_cfg).eval()
    _copy_causal_lm_weights(mine, ref)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
    position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)

    with torch.no_grad():
        logits_mine = mine(input_ids, position_ids=position_ids)
        logits_ref = ref(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
        ).logits

    assert logits_mine.shape == logits_ref.shape
    assert torch.allclose(logits_mine, logits_ref, atol=1e-5, rtol=1e-5)


def test_tie_word_embeddings_shares_weight_object():
    """tie_word_embeddings=True 时，lm_head.weight 应与 embed_tokens.weight 是同一对象。"""
    cfg = ModelConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=100,
        max_position_embeddings=32,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(cfg)

    # `is` 检查：两者必须是同一个 Parameter 对象，而不仅仅是值相等。
    assert (
        model.lm_head.weight is model.model.embed_tokens.weight
    ), "tie_word_embeddings=True 时，lm_head.weight 应与 embed_tokens.weight 共享同一对象"


def test_tie_word_embeddings_false_has_independent_weights():
    """tie_word_embeddings=False 时，lm_head.weight 与 embed_tokens.weight 应独立。"""
    cfg = _tiny_model_config()  # tie_word_embeddings=False
    model = Qwen3ForCausalLM(cfg)

    assert (
        model.lm_head.weight is not model.model.embed_tokens.weight
    ), "tie_word_embeddings=False 时，lm_head.weight 与 embed_tokens.weight 应为独立对象"


def test_tie_word_embeddings_update_propagates():
    """tie 生效时，修改 embed_tokens.weight 应立即反映到 lm_head.weight。"""
    cfg = ModelConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=100,
        max_position_embeddings=32,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(cfg)

    # 修改 embed_tokens.weight 数据，lm_head.weight 应同步变化（同一 tensor）。
    with torch.no_grad():
        model.model.embed_tokens.weight.fill_(3.14)

    assert torch.all(
        model.lm_head.weight == 3.14
    ), "tie 生效时修改 embed_tokens.weight 应同步到 lm_head.weight"
