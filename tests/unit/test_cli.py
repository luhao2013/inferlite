"""Unit tests for inferlite CLI.

CLI tests must not load real Qwen3-0.6B weights. We monkeypatch tokenizer/model/generate
so the test only verifies wiring: args -> tokenizer/model/engine/generate/decode/print.
"""

import torch

from inferlite import cli
from inferlite.config import ModelConfig


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
    """Fake model，仅记录关键方法调用，不加载真实权重。

    T5 新增 model.to(device, dtype=dtype) 和 model.config 的访问，均在此实现。
    """

    eval_called: bool = False
    to_called_with: tuple | None = None  # (device, dtype)

    def to(self, device, *, dtype=None) -> "FakeModel":
        FakeModel.to_called_with = (device, dtype)
        return self

    def eval(self) -> "FakeModel":
        FakeModel.eval_called = True
        return self

    @property
    def config(self) -> ModelConfig:
        # 使用 qwen3_0_6b 工厂；KVCache.from_config 只读层数、KV head 数、head_dim
        return ModelConfig.qwen3_0_6b()


def _make_fake_generate(calls: dict):
    """创建 fake generate 函数，捕获调用参数。

    T5 新增 kv_cache 参数，需要同步到签名里，否则 generate() 传入 kv_cache=... 时报错。
    """

    def fake_generate(
        engine,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        kv_cache=None,
    ) -> torch.Tensor:
        calls["engine"] = engine
        calls["input_ids"] = input_ids
        calls["max_new_tokens"] = max_new_tokens
        calls["eos_token_id"] = eos_token_id
        calls["kv_cache"] = kv_cache
        return torch.tensor([[10, 20, 30]])

    return fake_generate


# ---------------------------------------------------------------------------
# parse_args 测试
# ---------------------------------------------------------------------------


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


def test_parse_args_device_dtype_max_seq_len_defaults():
    """T5 新参数：默认值验证。"""
    args = cli.parse_args(["--model-dir", "m", "--prompt", "p"])

    assert args.device == "auto"
    assert args.dtype == "auto"
    assert args.max_seq_len == 1024


def test_parse_args_device_dtype_max_seq_len_explicit():
    """T5 新参数：显式传入值验证。"""
    args = cli.parse_args(
        [
            "--model-dir",
            "m",
            "--prompt",
            "p",
            "--device",
            "cpu",
            "--dtype",
            "fp32",
            "--max-seq-len",
            "512",
        ]
    )

    assert args.device == "cpu"
    assert args.dtype == "fp32"
    assert args.max_seq_len == 512


# ---------------------------------------------------------------------------
# resolve_device_dtype 测试
# ---------------------------------------------------------------------------


def test_resolve_device_dtype_explicit_cpu_fp32():
    """显式指定 cpu + fp32，不依赖硬件环境。"""
    device, dtype = cli.resolve_device_dtype("cpu", "fp32")

    assert device == "cpu"
    assert dtype == torch.float32


def test_resolve_device_dtype_explicit_bf16():
    """显式指定 bf16，dtype 映射正确。"""
    _, dtype = cli.resolve_device_dtype("cpu", "bf16")

    assert dtype == torch.bfloat16


def test_resolve_device_dtype_auto_falls_back_to_cpu(monkeypatch):
    """auto 模式：mps/cuda 均不可用时回退到 cpu，dtype 自动选 float32。"""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    device, dtype = cli.resolve_device_dtype("auto", "auto")

    assert device == "cpu"
    assert dtype == torch.float32


def test_resolve_device_dtype_auto_selects_mps(monkeypatch):
    """auto 模式：mps 可用时选 mps，dtype 自动选 bfloat16。"""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    device, dtype = cli.resolve_device_dtype("auto", "auto")

    assert device == "mps"
    assert dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# main 集成测试（monkeypatch 全部 IO）
# ---------------------------------------------------------------------------


def test_cli_main_wires_components_and_prints_text(monkeypatch, capsys):
    calls: dict = {}
    FakeModel.eval_called = False
    FakeModel.to_called_with = None

    def fake_load_causal_lm_from_hf(model_dir: str) -> FakeModel:
        calls["model_dir"] = model_dir
        return FakeModel()

    monkeypatch.setattr(cli, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(cli, "load_causal_lm_from_hf", fake_load_causal_lm_from_hf)
    monkeypatch.setattr(cli, "generate", _make_fake_generate(calls))
    # 固定 device 为 cpu，让测试不依赖硬件
    monkeypatch.setattr(cli, "resolve_device_dtype", lambda d, t: ("cpu", torch.float32))

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
    # eval() 应被调用
    assert FakeModel.eval_called is True
    # eos_token_id 应被传入 generate
    assert calls["eos_token_id"] == FakeAutoTokenizer.tokenizer.eos_token_id
    # T5: kv_cache 应被创建并传入 generate
    assert calls["kv_cache"] is not None
    # T5: model.to() 应被调用，device=cpu，dtype=float32
    assert FakeModel.to_called_with == ("cpu", torch.float32)


def test_cli_main_with_chat_template(monkeypatch, capsys):
    """--chat-template 开启时，prompt 应经过 apply_chat_template 包装后再 encode。"""
    calls: dict = {}
    FakeAutoTokenizer.tokenizer = FakeTokenizer()  # reset state
    FakeModel.eval_called = False
    FakeModel.to_called_with = None

    def fake_load_causal_lm_from_hf(model_dir: str) -> FakeModel:
        return FakeModel()

    monkeypatch.setattr(cli, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(cli, "load_causal_lm_from_hf", fake_load_causal_lm_from_hf)
    monkeypatch.setattr(cli, "generate", _make_fake_generate(calls))
    monkeypatch.setattr(cli, "resolve_device_dtype", lambda d, t: ("cpu", torch.float32))

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
    # T5: kv_cache 也应被传入
    assert calls["kv_cache"] is not None
