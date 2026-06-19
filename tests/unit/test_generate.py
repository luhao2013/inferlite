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


class EosEngine:
    """FakeEngine 变体：在指定 step 数后返回 eos_token_id。"""

    def __init__(self, eos_after: int, eos_token_id: int) -> None:
        self.step_count = 0
        self.eos_after = eos_after
        self.eos_token_id = eos_token_id

    def step(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.step_count += 1
        batch_size = input_ids.shape[0]
        if self.step_count >= self.eos_after:
            return torch.full((batch_size, 1), self.eos_token_id, dtype=torch.long)
        return torch.full((batch_size, 1), 99, dtype=torch.long)


def test_generate_stops_at_eos():
    """生成到 EOS token 时，应提前退出，输出长度 < prompt_len + max_new_tokens。"""
    eos_token_id = 2
    engine = EosEngine(eos_after=3, eos_token_id=eos_token_id)
    input_ids = torch.tensor([[10, 20]])  # prompt len = 2

    # max_new_tokens=10，但第 3 步就会生成 EOS，应提前停止
    output_ids = generate(engine, input_ids, max_new_tokens=10, eos_token_id=eos_token_id)

    # 应该跑 3 步：两步 99，第 3 步 EOS -> 停止
    assert output_ids.shape[1] == 5  # prompt 2 + 2 个 99 + 1 个 EOS
    assert output_ids[0, -1].item() == eos_token_id


def test_generate_eos_none_runs_full_max_new_tokens():
    """eos_token_id=None 时，应跑满 max_new_tokens 步，不提前退出。"""
    engine = FakeEngine()
    input_ids = torch.tensor([[10, 20]])

    output_ids = generate(engine, input_ids, max_new_tokens=5, eos_token_id=None)

    assert output_ids.shape == (1, 7)  # prompt 2 + 5 new tokens


def test_generate_default_eos_none_backward_compat():
    """不传 eos_token_id 时行为应与旧接口一致（向后兼容）。"""
    engine = FakeEngine()
    input_ids = torch.tensor([[10, 20]])

    # 不传 eos_token_id 参数
    output_ids = generate(engine, input_ids, max_new_tokens=3)

    assert output_ids.shape == (1, 5)
    assert torch.equal(output_ids, torch.tensor([[10, 20, 2, 3, 4]]))
