"""Unit tests for inferlite CLI.

CLI tests must not load real Qwen3-0.6B weights. We monkeypatch tokenizer/model/generate
so the test only verifies wiring: args -> tokenizer/model/engine/generate/decode/print.
"""

import torch

from inferlite import cli


class FakeTokenizer:
    def __init__(self) -> None:
        self.prompt_seen: str | None = None
        self.decoded_ids: list[int] | None = None

    def encode(self, prompt: str, return_tensors: str) -> torch.Tensor:
        self.prompt_seen = prompt
        assert return_tensors == "pt"
        return torch.tensor([[10, 20]])

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        self.decoded_ids = token_ids.tolist()
        return "decoded text"


class FakeAutoTokenizer:
    tokenizer = FakeTokenizer()
    model_dir_seen: str | None = None
    trust_remote_code_seen: bool | None = None

    @classmethod
    def from_pretrained(cls, model_dir: str, trust_remote_code: bool) -> FakeTokenizer:
        cls.model_dir_seen = model_dir
        cls.trust_remote_code_seen = trust_remote_code
        return cls.tokenizer


class FakeModel:
    pass


def test_parse_args_reads_cli_options():
    args = cli.parse_args(
        [
            "--model-dir",
            "fake-model",
            "--prompt",
            "你好",
            "--max-new-tokens",
            "3",
        ]
    )

    assert args.model_dir == "fake-model"
    assert args.prompt == "你好"
    assert args.max_new_tokens == 3


def test_cli_main_wires_components_and_prints_text(monkeypatch, capsys):
    calls: dict[str, object] = {}

    def fake_load_causal_lm_from_hf(model_dir: str) -> FakeModel:
        calls["model_dir"] = model_dir
        return FakeModel()

    def fake_generate(engine, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        calls["engine"] = engine
        calls["input_ids"] = input_ids
        calls["max_new_tokens"] = max_new_tokens
        return torch.tensor([[10, 20, 30]])

    monkeypatch.setattr(cli, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(cli, "load_causal_lm_from_hf", fake_load_causal_lm_from_hf)
    monkeypatch.setattr(cli, "generate", fake_generate)

    cli.main(
        [
            "--model-dir",
            "fake-model",
            "--prompt",
            "你好",
            "--max-new-tokens",
            "3",
        ]
    )

    captured = capsys.readouterr()
    assert captured.out == "decoded text\n"
    assert FakeAutoTokenizer.model_dir_seen == "fake-model"
    assert FakeAutoTokenizer.trust_remote_code_seen is True
    assert FakeAutoTokenizer.tokenizer.prompt_seen == "你好"
    assert FakeAutoTokenizer.tokenizer.decoded_ids == [10, 20, 30]
    assert calls["model_dir"] == "fake-model"
    assert calls["max_new_tokens"] == 3
    assert torch.equal(calls["input_ids"], torch.tensor([[10, 20]]))
