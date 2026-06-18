"""Engine-facing model protocol.

`engine` 层不应该直接绑定某一个具体模型类，比如 `Qwen3ForCausalLM`。
它真正需要的能力很小：

    input_ids [B, T] -> logits [B, T, V]

因此这里用 `Protocol` 定义一个结构化类型：只要某个对象支持
`model(input_ids)` 并返回 logits Tensor，它就可以被 EngineCore 当作 LLMModel 使用。

T12-pre（本任务卡）在原有基础上扩展了 `logits_to_keep` 参数，用来优化只计算最后几个 token 位置的
logits，例如 `model(input_ids, logits_to_keep=1)` 可以节省前 T-1 个位置的 lm_head 计算和内存。

注意：
- `LLMModel` 不是模型实现，不会被实例化。
- `__call__` 里的 `...` 不是 TODO，而是“只声明接口，不实现逻辑”。
- 调用方可以传入 `logits_to_keep` 来控制 logits 范围；None 表示不截断。
- 真实逻辑由具体模型提供，例如 `Qwen3ForCausalLM.forward`。
- FakeModel 只要实现 `__call__`，也能在单测里满足这个协议。
"""

from typing import Protocol

import torch


class LLMModel(Protocol):
    """最小 LLM 推理协议：input_ids -> logits，支持可选的 logits_to_keep。

    这个协议描述的是 EngineCore 对模型的最低要求：

        logits = model(input_ids)             # 默认返回完整 logits [B, T, V]
        logits = model(input_ids, logits_to_keep=1)  # 仅返回最后一个位置的 logits

    其中：
    - input_ids: [B, T]，token id tensor。
    - logits: [B, T, vocab_size]，每个位置对词表的未归一化分数。
    - 若传入 logits_to_keep，只保留该数量的最后位置。

    为什么定义 `__call__` 而不是 `forward`？
    - EngineCore 实际会调用 `model(input_ids)`。
    - PyTorch 的 `nn.Module.__call__` 会转发到具体模型的 `forward`。
    - 普通 FakeModel 也可以直接实现 `__call__`，不必继承 `nn.Module`。
    """

    def __call__(
        self, input_ids: torch.Tensor, *, logits_to_keep: int | None = None
    ) -> torch.Tensor:
        """返回 logits。

        Args:
            input_ids: [B, T] 形状的 token ids。
            logits_to_keep: 若为非 None，只返回最后 ``logits_to_keep`` 个位置的 logits。

        Returns:
            logits: [B, T, V] 或 [B, logits_to_keep, V]。
        """
        ...
