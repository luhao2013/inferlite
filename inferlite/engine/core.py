"""Minimal single-step inference engine and greedy generate loop.

`EngineCore` 是推理流程调度层，不直接实现神经网络计算，也不直接实现采样策略。
它负责把 T8/T9/T10 已经完成的组件串起来：

    model(input_ids) -> logits [B, T, V]
    logits[:, -1, :] -> next_token_logits [B, V]
    sampler(next_token_logits) -> next_token [B, 1]

T11 在 `EngineCore.step()` 之上增加最小 generate loop：

    for _ in range(max_new_tokens):
        next_token = engine.step(input_ids)
        input_ids = torch.cat([input_ids, next_token], dim=1)

注意：
- 当前实现没有 KV cache，每一步都会 full forward，正确但慢。
- 当前实现会先拿到完整 logits [B, T, V]，再取最后位置。
- T12-pre 会优化成只计算最后 token 的 lm_head logits。
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
        # 当前 LLMModel 协议返回完整 logits [B, T, V]。
        # 单步生成只需要最后一个位置的 logits，因为它代表“基于当前完整上下文预测下一个 token”。
        logits = self.model(input_ids, logits_to_keep=1)
        next_token_logits = logits[:, -1, :]

        # sampler 只负责 [B, V] -> [B, 1]，不关心 logits 来自哪个模型或哪个位置。
        next_token = self.sampler(next_token_logits)
        return next_token


def generate(engine: EngineCore, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """用 `EngineCore.step` 做最小 greedy generate loop。

    Args:
        engine: 已经组装好 model + sampler 的单步推理引擎。
        input_ids: prompt token ids，shape 为 [B, T]。
        max_new_tokens: 固定生成多少个新 token。T11 暂不处理 EOS，所以会严格循环该次数。

    Returns:
        output_ids: prompt + generated token ids，shape 为 [B, T + max_new_tokens]。

    这个函数只负责 token id 级别的循环，不负责 tokenizer encode/decode。
    CLI 会在外层完成文本与 token id 的转换。
    """
    for _ in range(max_new_tokens):
        next_token = engine.step(input_ids)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids
