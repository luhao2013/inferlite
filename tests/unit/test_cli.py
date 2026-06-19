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
        # Qwen3 EOS token id
        self.eos_token_id: int = 151645

    def encode(self, prompt: str, return_tensors: str) -> torch.Tensor:
        self.prompt_seen = prompt
        assert return_tensors == "pt"
        return torch.tensor([[10, 20]])

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        self.decoded_ids = token_ids.tolist()
        return "decoded text"

    def apply_chat_template(
        self,
        messages: list,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        """返回格式化后的 prompt，便于单测验证 chat template 路径是否被触发。"""
        assert tokenize is False
        assert add_generation_prompt is True
        # 简单模拟：用 role+content 包装
        content = messages[0]["content"]
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


class FakeAutoTokenizer:
    tokenizer = FakeTokenizer()
    model_dir_seen: str | None = None
    trust_remote_code_seen: bool | None = None

    @classmethod
    def from_pretrained(
        cls, model_dir: str, trust_remote_code: bool, local_files_only: bool = False
    ) -> FakeTokenizer:
        cls.model_dir_seen = model_dir
        cls.trust_remote_code_seen = trust_remote_code
        return cls.tokenizer


class FakeModel:
    eval_called: bool = False

    def eval(self) -> "FakeModel":
        FakeModel.eval_called = True
        return self


def _make_fake_generate(calls: dict):
    """创建 fake generate 函数，捕获调用参数。"""

    def fake_generate(
        engine,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        calls["engine"] = engine
        calls["input_ids"] = input_ids
        calls["max_new_tokens"] = max_new_tokens
        calls["eos_token_id"] = eos_token_id
        return torch.tensor([[10, 20, 30]])

    return fake_generate


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
    assert args.chat_template is False


def test_parse_args_chat_template_flag():
    args = cli.parse_args(
        [
            "--model-dir",
            "fake-model",
            "--prompt",
            "你好",
            "--chat-template",
        ]
    )

    assert args.chat_template is True


def test_cli_main_wires_components_and_prints_text(monkeypatch, capsys):
    calls: dict = {}
    FakeModel.eval_called = False

    def fake_load_causal_lm_from_hf(model_dir: str) -> FakeModel:
        calls["model_dir"] = model_dir
        return FakeModel()

    monkeypatch.setattr(cli, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(cli, "load_causal_lm_from_hf", fake_load_causal_lm_from_hf)
    monkeypatch.setattr(cli, "generate", _make_fake_generate(calls))

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
    assert FakeAutoTokenizer.model_dir_seen.endswith("fake-model")
    assert FakeAutoTokenizer.trust_remote_code_seen is True
    # 不加 --chat-template：prompt 应直接传入 tokenizer.encode
    assert FakeAutoTokenizer.tokenizer.prompt_seen == "你好"
    assert FakeAutoTokenizer.tokenizer.decoded_ids == [10, 20, 30]
    assert calls["model_dir"].endswith("fake-model")
    assert calls["max_new_tokens"] == 3
    assert torch.equal(calls["input_ids"], torch.tensor([[10, 20]]))
    # Fix 1: eval() 应被调用
    assert FakeModel.eval_called is True
    # Fix 4: eos_token_id 应被传入 generate
    assert calls["eos_token_id"] == FakeAutoTokenizer.tokenizer.eos_token_id


def test_cli_main_with_chat_template(monkeypatch, capsys):
    """--chat-template 开启时，prompt 应经过 apply_chat_template 包装后再 encode。"""
    calls: dict = {}
    FakeAutoTokenizer.tokenizer = FakeTokenizer()  # reset state

    def fake_load_causal_lm_from_hf(model_dir: str) -> FakeModel:
        return FakeModel()

    monkeypatch.setattr(cli, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(cli, "load_causal_lm_from_hf", fake_load_causal_lm_from_hf)
    monkeypatch.setattr(cli, "generate", _make_fake_generate(calls))

    cli.main(
        [
            "--model-dir",
            "fake-model",
            "--prompt",
            "你好",
            "--chat-template",
        ]
    )

    # chat template 包装后的 prompt 应包含 <|im_start|> 标记
    encoded_prompt = FakeAutoTokenizer.tokenizer.prompt_seen
    assert encoded_prompt is not None
    assert "<|im_start|>" in encoded_prompt
    assert "你好" in encoded_prompt

    captured = capsys.readouterr()
    assert captured.out == "decoded text\n"
