"""Unit tests for T11 greedy generate loop."""

import torch

from inferlite.engine import generate


class FakeEngine:
    """Fake EngineCore：根据当前序列长度生成确定性 next token。"""

    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def step(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.calls.append(input_ids.clone())
        batch_size = input_ids.shape[0]
        next_id = input_ids.shape[1]
        return torch.full((batch_size, 1), next_id, dtype=torch.long)


def test_generate_appends_max_new_tokens():
    """generate 应追加固定数量的新 token，并返回 prompt + generated 的完整序列。"""
    engine = FakeEngine()
    input_ids = torch.tensor([[10, 20]])

    output_ids = generate(engine, input_ids, max_new_tokens=3)

    assert torch.equal(output_ids, torch.tensor([[10, 20, 2, 3, 4]]))
    assert output_ids.shape == (1, 5)


def test_generate_updates_input_ids_each_step():
    """每一轮 step 都应该看到上一轮 append 后的 input_ids。"""
    engine = FakeEngine()
    input_ids = torch.tensor([[10, 20]])

    _ = generate(engine, input_ids, max_new_tokens=3)

    assert len(engine.calls) == 3
    assert torch.equal(engine.calls[0], torch.tensor([[10, 20]]))
    assert torch.equal(engine.calls[1], torch.tensor([[10, 20, 2]]))
    assert torch.equal(engine.calls[2], torch.tensor([[10, 20, 2, 3]]))


def test_generate_handles_batch():
    """batch 场景下，generate 追加的 next token shape 应保持 [B, 1]。"""
    engine = FakeEngine()
    input_ids = torch.tensor(
        [
            [10, 20],
            [30, 40],
        ]
    )

    output_ids = generate(engine, input_ids, max_new_tokens=2)

    assert torch.equal(
        output_ids,
        torch.tensor(
            [
                [10, 20, 2, 3],
                [30, 40, 2, 3],
            ]
        ),
    )
    assert output_ids.shape == (2, 4)
