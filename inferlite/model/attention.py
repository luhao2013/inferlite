"""Qwen3 GQA Attention 的最小数值对齐实现。

写 Attention 可以按下面这份伪代码拆：

1. 先从 config 固化结构超参
   - hidden_size: H，残差流宽度，也就是每个 token 在主干网络里的向量宽度
   - num_heads: n_q，Query head 数
   - num_key_value_heads: n_kv，Key/Value head 数；GQA 中它通常小于 n_q
   - head_dim: D，单 head 宽度，Qwen3-0.6B 显式给出 128，不能用 H / n_q 推导
   - num_key_value_groups: n_q / n_kv，GQA 中每个 KV head 被多少个 Q head 共享
   - scaling: D ** -0.5，attention score 缩放因子，避免 q·k 随 D 变大而过大

2. 再定义 Attention 子结构
   - q_proj / k_proj / v_proj: hidden_states -> q/k/v
   - o_proj: 多头 attention output -> hidden_size，回到 residual stream
   - q_norm / k_norm: Qwen3 特有，RoPE 前只在 head_dim 上做 RMSNorm
   - rotary_emb: 根据 position_ids 生成 cos/sin，真正旋转由 apply_rotary_pos_emb 完成

3. forward 按数据流写
   hidden_states [B, T, H]
     -> q/k/v projection
     -> reshape to [B, heads, T, D]
     -> q_norm / k_norm
     -> RoPE(q, k)
     -> repeat_kv(k, v)
     -> q @ k^T * scaling
     -> causal mask
     -> softmax
     -> attn @ v
     -> o_proj

T4 只做 full causal attention，不做 KV cache；KV cache 留到 M2。
"""

from typing import override

import torch
import torch.nn as nn

from inferlite.config import ModelConfig
from inferlite.model.layers import RMSNorm, RotaryEmbedding, apply_rotary_pos_emb


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把 GQA 的 KV heads repeat 到 Query heads 数量。

    GQA（Grouped Query Attention）的核心是：
    - Query 头更多：`num_heads = n_q`
    - Key/Value 头更少：`num_key_value_heads = n_kv`
    - 多个 Query head 共享同一个 Key/Value head

    因此 attention 真正做 `q @ k^T` 前，需要把 k/v 从 n_kv 个 head 复制到 n_q 个 head：

        [B, n_kv, T, D] -> [B, n_kv, n_rep, T, D] -> [B, n_q, T, D]

    以 Qwen3-0.6B 为例：
    - n_q = 16
    - n_kv = 8
    - n_rep = 2

    所以第 0 个 KV head 会服务第 0/1 个 Query head，第 1 个 KV head 会服务
    第 2/3 个 Query head，以此类推。

    Args:
        hidden_states: [B, num_key_value_heads, T, head_dim]
        n_rep: 每个 KV head 复制给多少个 Query head 使用。

    Returns:
        [B, num_key_value_heads * n_rep, T, head_dim]
    """
    if n_rep == 1:
        # MHA 退化情况：如果 n_q == n_kv，就不需要 repeat。
        return hidden_states

    batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape

    # 先在 KV head 后面插入一个 group 维度：
    #   [B, n_kv, T, D] -> [B, n_kv, 1, T, D]
    # 再用 expand 逻辑复制 n_rep 份。expand 不立刻拷贝数据，只创建广播视图。
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size,
        num_key_value_heads,
        n_rep,
        seq_len,
        head_dim,
    )

    # 最后把 [n_kv, n_rep] 合并成 n_q。reshape 会在需要时 materialize。
    return hidden_states.reshape(
        batch_size,
        num_key_value_heads * n_rep,
        seq_len,
        head_dim,
    )


class GQAAttention(nn.Module):
    """Qwen3 decoder self-attention: QK-norm + RoPE + GQA。

    这层只实现“单段 prefill/full attention”：输入一整段 token，输出同长度 hidden states。
    暂不处理 KV cache、sliding window、attention dropout、返回 attention weights 等工程功能。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # 这些结构超参会在 forward 的 reshape/repeat/scaling 中反复使用，必须挂到 self。
        # 注意：head_dim 直接来自 config.head_dim。Qwen3-0.6B 中 H/n_q=64，但 head_dim=128，
        # 不能用 hidden_size // num_heads 代替。
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.num_key_value_heads: int = config.num_key_value_heads
        self.head_dim: int = config.head_dim
        self.num_key_value_groups: int = self.num_heads // self.num_key_value_heads
        self.scaling: float = self.head_dim**-0.5
        self.rms_norm_eps: float = config.rms_norm_eps
        self.rope_theta: float = config.rope_theta

        # Qwen3 attention projection 都没有 bias。
        # q_proj 输出 n_q * D；k/v_proj 输出 n_kv * D。
        # GQA 减少的是 K/V head 数，不减少 Query head 数。
        self.q_proj: nn.Linear = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj: nn.Linear = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj: nn.Linear = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        # o_proj 把拼回来的所有 Query heads 输出重新映射回 hidden_size，供 residual add 使用。
        self.o_proj: nn.Linear = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

        # Qwen3 在 q/k 做 RoPE 前分别做 RMSNorm，归一化维度是 head_dim。
        # 这里的 eps 必须来自 config.rms_norm_eps，保证和 transformers/Qwen3Config 对齐。
        self.q_norm: RMSNorm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)
        self.k_norm: RMSNorm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)
        self.rotary_emb: RotaryEmbedding = RotaryEmbedding(self.head_dim, self.rope_theta)

    # @override 是给类型检查器看的声明：forward 是在重写 nn.Module.forward。
    # 它不改变运行逻辑，但能防止把 forward 误拼成 foward 这类错误。
    @override
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """执行单段 causal self-attention。

        Args:
            hidden_states: [B, T, hidden_size]
                来自上一层 decoder block 或 embedding 的残差流张量。
            position_ids: [B, T]
                每个 token 的绝对位置，用于生成 RoPE 的 cos/sin。

        Returns:
            attention output: [B, T, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. q/k/v projection，并把最后一维拆成 heads × head_dim。
        # projection 后的中间形状：
        #   q: [B, T, n_q * D]
        #   k: [B, T, n_kv * D]
        #   v: [B, T, n_kv * D]
        # view 后变成：
        #   q: [B, T, n_q, D]
        #   k/v: [B, T, n_kv, D]
        q = self.q_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        k = self.k_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        v = self.v_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        )

        # [B, T, heads, D] -> [B, heads, T, D]。
        # 这样最后两维就是 [T, D]，方便后面做：
        #   q @ k.transpose(-2, -1) => [T, D] @ [D, T] = [T, T]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 2. Qwen3 特有：RoPE 前做 q/k RMSNorm。
        # RMSNorm 的 normalized_shape 是 head_dim，所以它会在每个 head 的 D 维上归一化。
        # 注意：v 不做 RMSNorm；QK-norm 只服务 q/k 相似度计算稳定性。
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 3. 根据 position_ids 生成 cos/sin，再应用到 q/k。v 不做 RoPE。
        # rotary_emb 传入 q 只是借用 q 的 device/dtype；角度由 position_ids + inv_freq 决定。
        # apply_rotary_pos_emb 后，位置信息进入 q/k，随后通过 q·k 影响 attention score。
        cos, sin = self.rotary_emb(q, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        # 4. GQA：把 KV heads repeat 到 Query heads 数量。
        # repeat 后：
        #   k/v: [B, n_kv, T, D] -> [B, n_q, T, D]
        # 这样 q 和 k/v 的 head 维度才能一一对应做 attention。
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # 5. scaled dot-product attention。
        # attn_weights: [B, n_q, T, D] @ [B, n_q, D, T] -> [B, n_q, T, T]
        # 第 i 行表示第 i 个 query token 对所有 key token 的打分。
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling

        # causal mask 禁止当前位置看未来 token。
        # torch.triu(..., diagonal=1) 得到上三角 True：
        #   row i, col j 为 True 表示 j > i，也就是未来 token。
        # M1 每步 forward 都重建这个 tensor，随 seq_len 增长开销增大。
        # M2 引入 KV cache 后：
        #   - prefill 阶段仍需完整 causal mask；
        #   - decode 步单 token 管看到所有历史，不需要 causal mask，届时可以删採。
        # TODO(M2): KV cache 后删採 decode 步的 causal mask 重建逻辑。
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device),
            diagonal=1,
        )
        # 显式扩到 [1, 1, T, T]，广播到 [B, n_q, T, T]。
        causal_mask = causal_mask[None, None, :, :]
        # 被 mask 的位置填成 dtype 最小值，softmax 后概率接近 0。
        attn_weights = attn_weights.masked_fill(
            causal_mask,
            torch.finfo(attn_weights.dtype).min,
        )
        # 与 transformers eager attention 对齐：softmax 用 fp32 做，再 cast 回 q dtype。
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        # 6. 用 attention 概率聚合 value，再回到 [B, T, hidden_size]。
        # [B, n_q, T, T] @ [B, n_q, T, D] -> [B, n_q, T, D]
        attn_output = torch.matmul(attn_weights, v)
        # [B, n_q, T, D] -> [B, T, n_q, D] -> [B, T, n_q * D]
        # contiguous 是为了保证 transpose 后内存布局可被 view 安全解释。
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                seq_len,
                self.num_heads * self.head_dim,
            )
        )
        # o_proj 回到 hidden_size，输出将进入 DecoderLayer 的 residual add。
        return self.o_proj(attn_output)
