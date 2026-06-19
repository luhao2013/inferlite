"""ModelConfig: Qwen3 11 个核心超参的 dataclass + JSON 反序列化。

为什么需要 ModelConfig？参考 docs/3-kb/knowledge.md §Qwen3 Tech Report §4。
一句话：所有 layer 共享同一份事实源，避免 magic number 散落在 N 个文件里。

设计要点：
1. frozen=True   → 推理时 config 是只读契约，防误改
2. __post_init__ → 早 fail（构造期）vs 晚 fail（forward 时 shape 错位）
3. from_json     → 白名单过滤，HF config.json 30+ 字段只取我们用的 11 个
4. qwen3_0_6b()  → 工厂方法，便于测试不依赖磁盘

固定参考链接（对齐当前本地 transformers==5.10.2，避免 main/tag 漂移）：
- Qwen3Config 源码：
  https://github.com/huggingface/transformers/blob/0dad7b822255a0ae261ec45ae937371e859ffd1a/src/transformers/models/qwen3/configuration_qwen3.py
- Qwen3-0.6B config.json：
  https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json
"""

import json
from dataclasses import dataclass
from pathlib import Path

# 白名单：T0 阶段只读这 11 个字段。
# HF 的 Qwen3Config 是工业级全集；这里是 M1 前向对齐的最小超参合同。
# 未来 M11 (YaRN) 加 rope_scaling；M2 prefix cache 复杂场景加 sliding_window。
_WHITELIST = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "rope_theta",
    "rms_norm_eps",
    "tie_word_embeddings",
)


@dataclass(frozen=True)
class ModelConfig:
    # 11 个字段；含义/值参见 knowledge.md §Qwen3 §4
    # decoder layer参数；维度和层数
    hidden_size: int  # H = 1024
    num_hidden_layers: int  # N=28
    # attention 参数
    num_attention_heads: int  # n_q = 16
    num_key_value_heads: int  # n_kv = 8 (GQA)
    head_dim: int  # d = 128
    # FFN 参数
    intermediate_size: int  # I = 3072 (SwiGLU中间维度)
    # 词表参数
    vocab_size: int  # V = 151936
    # 位置编码参数
    max_position_embeddings: int  # seq_len = 40960
    rope_theta: float  # RoPE 基频 base； 1e6
    # Norm 参数
    rms_norm_eps: float  # 1e-6
    # lm_head.weight 是否复用 embed_tokens.weight（权重共享）。
    # 来自各模型 config.json：Qwen3-0.6B=True，Qwen3-1.7B 及以上版本=False。
    # codeflicker-fix: LOGIC-Issue-002/dwv03qen2tgtzojek3hz
    tie_word_embeddings: bool

    def __post_init__(self):
        # 关键合法性校验：只校验"算法层不能违反"的 invariant
        # 不写可推导的关系（如 head_dim 与 H/n_q 的关系，因为它们独立）
        assert self.hidden_size > 0, f"hidden_size must be > 0, got {self.hidden_size}"
        assert self.num_attention_heads > 0
        assert self.num_key_value_heads > 0
        assert (
            self.num_attention_heads % self.num_key_value_heads == 0
        ), f"GQA: n_q={self.num_attention_heads} 必须能被 n_kv={self.num_key_value_heads} 整除"
        assert self.head_dim > 0
        assert self.vocab_size > 0
        assert 0 < self.rms_norm_eps < 1, f"rms_norm_eps 应在 (0, 1) 区间, got {self.rms_norm_eps}"
        assert self.rope_theta > 0

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        """从 HF config.json 反序列化。

        过滤掉白名单外的字段（architectures / torch_dtype / attention_dropout 等）。
        head_dim 缺失时用 hidden_size // num_attention_heads 兜底。
        rope_theta JSON 里是 int (e.g. 1000000)，cast 成 float。
        """
        with open(path) as f:
            raw = json.load(f)
        # 兜底只服务“老 config 没有 head_dim 字段”的兼容场景。
        # 标准 attention 的 per-head dim 由 Query 头数定义：H / n_q；
        # GQA 只减少 KV 头的“组数”，不改变每个 KV head 的维度。
        # Qwen3-0.6B 的 config.json 明确给了 head_dim=128，不会走这个分支。
        if "head_dim" not in raw:
            raw["head_dim"] = raw["hidden_size"] // raw["num_attention_heads"]
        kwargs = {k: raw[k] for k in _WHITELIST}
        kwargs["rope_theta"] = float(kwargs["rope_theta"])
        return cls(**kwargs)

    @classmethod
    def qwen3_0_6b(cls) -> "ModelConfig":
        """Qwen3-0.6B 硬编码工厂，返回 Qwen3-0.6B 的 ModelConfig。

        这里的值来自 Qwen3-0.6B config.json；它是测试用 ground truth，
        让单测不依赖本地磁盘是否存在 modelscope/huggingface 缓存。
        """
        return cls(
            hidden_size=1024,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            intermediate_size=3072,
            vocab_size=151936,
            max_position_embeddings=40960,
            rope_theta=1000000.0,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,  # Qwen3-0.6B config.json 明确为 True；>0.6B 版本为 False
        )
