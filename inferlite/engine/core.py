"""Minimal single-step inference engine and greedy generate loop.

`EngineCore` 是推理流程调度层，不直接实现神经网络计算，也不直接实现采样策略。
它负责把已经完成的各组件串起来：

    model(input_ids, logits_to_keep=1) -> logits [B, 1, V]
    logits[:, -1, :] -> next_token_logits [B, V]
    sampler(next_token_logits) -> next_token [B, 1]

`generate()` 在 `EngineCore.step()` 之上实现 greedy generate loop：

    for _ in range(max_new_tokens):
        next_token = engine.step(input_ids)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if eos_token_id is not None and next_token == eos_token_id:
            break

注意：
- 当前实现没有 KV cache，每一步都会 full forward，正确但慢（M2 再优化）。
- `logits_to_keep=1` 优化已生效：模型只计算最后一个位置的 lm_head 输出。
- 调用 `generate()` 的上层（如 CLI）应保证在 `torch.no_grad()` 上下文里运行，
  避免构建不必要的梯度计算图。
"""

import torch

from inferlite.engine.protocol import LLMModel
from inferlite.sampler.greedy import GreedySampler


class EngineCore:
    """最小单步推理引擎。

    EngineCore 只依赖 `LLMModel` 协议，不绑定具体模型类，例如 `Qwen3ForCausalLM`。
    因此只要一个对象能 `model(input_ids) -> logits`，就可以被 EngineCore 使用。
    """

    def __init__(self, model: LLMModel, sampler: GreedySampler) -> None:
        self.model: LLMModel = model
        self.sampler: GreedySampler = sampler

    def step(self, input_ids: torch.Tensor) -> torch.Tensor:
        """执行一步 greedy decode。

        Args:
            input_ids: token ids，shape 为 [B, T]。

        Returns:
            next_token: 下一 token ids，shape 为 [B, 1]。
        """
        # logits_to_keep=1：模型只计算最后一个 token 位置的 lm_head 输出，
        # 省去前 T-1 个位置的投影，节约内存和计算量（T12-pre 优化）。
        logits = self.model(input_ids, logits_to_keep=1)
        # logits 形状为 [B, 1, V]，取 [:, -1, :] 得到 [B, V] 交给 sampler。
        next_token_logits = logits[:, -1, :]

        # sampler 只负责 [B, V] -> [B, 1]，不关心 logits 来自哪个模型或哪个位置。
        next_token = self.sampler(next_token_logits)
        return next_token


def generate(
    engine: EngineCore,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """用 `EngineCore.step` 做最小 greedy generate loop，支持 EOS 提前停止。

    Args:
        engine: 已经组装好 model + sampler 的单步推理引擎。
        input_ids: prompt token ids，shape 为 [B, T]。
        max_new_tokens: 最多生成多少个新 token（硬上限）。
        eos_token_id: EOS token 的 id。当生成的 token 等于 eos_token_id 时提前停止。
            设为 None 时不做 EOS 检查，严格跑满 max_new_tokens 步（向后兼容）。

    Returns:
        output_ids: prompt + generated token ids，shape 为 [B, T + n]，
            其中 n <= max_new_tokens。若提前遇到 EOS，n 可能小于 max_new_tokens。

    调用方应在 `torch.no_grad()` 上下文里使用此函数，避免构建不必要的梯度图。
    这个函数只负责 token id 级别的循环，不负责 tokenizer encode/decode。
    CLI 会在外层完成文本与 token id 的转换。
    """
    for _ in range(max_new_tokens):
        next_token = engine.step(input_ids)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        # EOS 停止：当前 batch 所有序列都生成了 EOS token 时退出。
        # `(next_token == eos_token_id).all()` 保证 batch 里每条序列都已到达 EOS，
        # 避免 batch 中个别序列先到 EOS 就截断其他序列。
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break
    return input_ids
