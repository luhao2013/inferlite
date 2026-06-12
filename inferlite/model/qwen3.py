"""Qwen3 decoder layer 的最小数值对齐实现。

T5 的目标是把前面已经手写并测试过的叶子模块串起来：

    RMSNorm -> GQAAttention -> residual add -> RMSNorm -> SwiGLUMLP -> residual add

Qwen3 采用 pre-norm 结构：先归一化，再进入 attention/MLP，最后与 residual 相加。

pre-norm 与 post-norm 对比：

1. Pre-norm（现代 decoder-only LLM 主流）

       x -> norm -> sublayer -> + residual

   写成代码就是：

       residual = x
       x = norm(x)
       x = sublayer(x)
       x = residual + x

   特点：
   - residual 主干更像一条“干净高速通道”，梯度可以更稳定地穿过很多层；
   - 深层 Transformer 更容易训练，因此 GPT-2 之后的大量 decoder-only LLM 都采用 pre-norm；
   - LLaMA / Qwen / Mistral / Gemma 等现代开源主流大模型基本都是 pre-norm 变体，通常配 RMSNorm。

2. Post-norm（原始 Transformer 论文结构）

       x -> sublayer -> + residual -> norm

   写成代码就是：

       residual = x
       x = sublayer(x)
       x = residual + x
       x = norm(x)

   特点：
   - 原始 Transformer 使用 post-norm；
   - 浅层或 encoder 场景仍可见；
   - 很深的 decoder-only LLM 中通常更难稳定训练，往往需要更谨慎的初始化、学习率 warmup 或其他稳定化技巧。

结论：最新主流 decoder-only 大模型以 pre-norm 为主；本项目为了对齐 Qwen3，也实现 pre-norm。
"""

from typing import override

import torch
import torch.nn as nn

from inferlite.config import ModelConfig
from inferlite.model.attention import GQAAttention
from inferlite.model.layers import RMSNorm, SwiGLUMLP


class DecoderLayer(nn.Module):
    """Qwen3 单个 decoder block: self-attention + MLP。

    一个 decoder layer 有两段子层：
    1. self-attention 子层：负责 token 间信息交互。
    2. MLP 子层：负责每个 token 位置上的非线性特征变换。

    两段都采用相同的 pre-norm + residual 模式。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # hidden_size 是 residual stream 的宽度，DecoderLayer 输入输出都保持这个维度。
        self.hidden_size: int = config.hidden_size

        # self_attn 对应 transformers.Qwen3DecoderLayer.self_attn。
        # 命名对齐很重要：T7 加载 HF 权重时会依赖 state_dict key 的模块路径。
        self.self_attn: GQAAttention = GQAAttention(config)

        # MLP 是 attention 后的前馈网络，对每个 token 独立作用，不做 token 间通信。
        self.mlp: SwiGLUMLP = SwiGLUMLP(config.hidden_size, config.intermediate_size)

        # 两个 RMSNorm 都归一化 hidden_size 维，而不是 attention 的 head_dim。
        # input_layernorm: attention 前的 pre-norm。
        self.input_layernorm: RMSNorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # post_attention_layernorm: MLP 前的 pre-norm。
        # 名字里的 post_attention 表示“attention 之后”，不是 post-norm 结构。
        self.post_attention_layernorm: RMSNorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

    @override
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """执行一个 Qwen3 decoder layer。

        Args:
            hidden_states: [B, T, hidden_size]
                当前层输入的 residual stream。
            position_ids: [B, T]
                传给 attention 内部 RoPE，用于给 q/k 注入位置信息。

        Returns:
            [B, T, hidden_size]
        """
        # 1. Attention 子层：pre-norm -> self-attn -> residual add。
        #
        # 为什么先保存 residual？
        # residual 是“未经本子层变换”的主干信号，最后要和 attention 的增量相加。
        # pre-norm 中，norm 只作用在送进子层的分支上，不修改 residual 主干。
        residual = hidden_states
        # input_layernorm: [B, T, H] -> [B, T, H]
        # 这里归一化每个 token 的 hidden_size 维，让 attention 输入尺度更稳定。
        hidden_states = self.input_layernorm(hidden_states)
        # self_attn 内部会执行：q/k/v projection -> q/k norm -> RoPE -> GQA attention -> o_proj。
        # 输出仍是 [B, T, H]，表示 attention 子层计算出的“增量”。
        hidden_states = self.self_attn(hidden_states, position_ids)
        # residual add：把 attention 增量加回主干。
        hidden_states = residual + hidden_states

        # 2. MLP 子层：重新记录 residual，再 pre-norm -> MLP -> residual add。
        #
        # 注意这里必须重新设置 residual。
        # 第二段 residual 的起点是“已经经过 attention 子层后的 hidden_states”，
        # 不能复用 attention 前的 residual，否则会跳过 attention 子层的贡献。
        residual = hidden_states
        # post_attention_layernorm 虽然名字里有 post_attention，但结构上仍然是 MLP 前的 pre-norm。
        hidden_states = self.post_attention_layernorm(hidden_states)
        # MLP 对每个 token 独立做 SwiGLU 变换，不改变 [B, T, H] 形状。
        hidden_states = self.mlp(hidden_states)
        # 第二次 residual add：把 MLP 增量加回主干。
        hidden_states = residual + hidden_states
        return hidden_states
