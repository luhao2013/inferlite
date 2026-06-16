"""Unit tests for inferlite.engine.core.EngineCore.

T10 目标：实现最小单步推理引擎：

    input_ids -> model -> logits[:, -1, :] -> sampler -> next_token

运行：
  uv run pytest tests/unit/test_engine_core.py -q
"""

import torch

from inferlite.engine import EngineCore, LLMModel
from inferlite.sampler import GreedySampler


class FakeModel:
    """满足 LLMModel Protocol 的 fake model，用固定 logits 测 EngineCore 流程。"""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits
        self.calls: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.calls.append(input_ids)
        batch_size, seq_len = input_ids.shape
        assert self.logits.shape[:2] == (batch_size, seq_len)
        return self.logits


def _accept_llm_model(model: LLMModel, input_ids: torch.Tensor) -> torch.Tensor:
    """测试 EngineCore 依赖的是 LLMModel 协议，而不是具体 Qwen3 类。"""
    return model(input_ids)


def test_engine_core_step_returns_next_token_shape():
    """EngineCore.step 应返回 next_token [B, 1]。"""
    input_ids = torch.tensor([[1, 2]])
    logits = torch.tensor([[[0.0, 1.0, 2.0], [0.1, 0.9, 0.2]]])
    model = FakeModel(logits)
    engine = EngineCore(model=model, sampler=GreedySampler())

    next_token = engine.step(input_ids)

    assert torch.equal(next_token, torch.tensor([[1]]))
    assert next_token.shape == (1, 1)
    assert next_token.dtype == torch.long


def test_engine_core_step_uses_last_position_logits_only():
    """EngineCore.step 必须取 logits[:, -1, :]，不能误用第 0 个位置。"""
    input_ids = torch.tensor([[1, 2]])
    logits = torch.tensor(
        [
            [
                [100.0, 0.0, 0.0],
                [0.0, 0.0, 100.0],
            ]
        ]
    )
    model = FakeModel(logits)
    engine = EngineCore(model=model, sampler=GreedySampler())

    next_token = engine.step(input_ids)

    # 如果错误地用第 0 个位置，会选 token 0；正确使用最后位置应选 token 2。
    assert torch.equal(next_token, torch.tensor([[2]]))


def test_engine_core_step_handles_batch_independently():
    """batch 中每条序列都应根据自己的最后位置 logits 独立选择 next token。"""
    input_ids = torch.tensor(
        [
            [1, 2],
            [3, 4],
        ]
    )
    logits = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 3.0]],
            [[0.0, 0.0, 0.0], [5.0, 1.0, 0.0]],
        ]
    )
    model = FakeModel(logits)
    engine = EngineCore(model=model, sampler=GreedySampler())

    next_token = engine.step(input_ids)

    assert torch.equal(next_token, torch.tensor([[2], [0]]))
    assert next_token.shape == (2, 1)


def test_engine_core_passes_input_ids_to_model_unchanged():
    """EngineCore.step 应把 input_ids 原样传给 model。"""
    input_ids = torch.tensor([[1, 2]])
    logits = torch.zeros(1, 2, 3)
    model = FakeModel(logits)
    engine = EngineCore(model=model, sampler=GreedySampler())

    _ = engine.step(input_ids)

    assert len(model.calls) == 1
    assert torch.equal(model.calls[0], input_ids)


def test_engine_package_exports_engine_core_and_protocol():
    """包门面应支持 from inferlite.engine import EngineCore, LLMModel。"""
    input_ids = torch.tensor([[1, 2]])
    logits = torch.zeros(1, 2, 3)
    model = FakeModel(logits)

    output = _accept_llm_model(model, input_ids)

    assert torch.equal(output, logits)
    assert EngineCore.__name__ == "EngineCore"
