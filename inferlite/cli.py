"""Command line entrypoint for minimal greedy generation.

CLI 负责把用户输入的文本参数转换成一次完整推理调用：

    parse args
      -> load tokenizer
      -> (可选) apply_chat_template 包装 prompt
      -> load Qwen3ForCausalLM weights
      -> model.eval()
      -> build GreedySampler + EngineCore
      -> tokenizer.encode(prompt)
      -> torch.no_grad() 上下文里 generate(engine, input_ids)
      -> tokenizer.decode(output_ids)
      -> print text

注意：
- `model.eval()` 关掉训练模式（Dropout 等），是推理的标准做法。
- `torch.no_grad()` 禁止构建梯度计算图，降低内存占用、加快推理速度。
- `--chat-template` 开启后用 apply_chat_template 包装 prompt，
  让模型按 instruction-tuning 格式理解输入，输出质量明显优于裸 prompt。
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from inferlite.engine import EngineCore, generate
from inferlite.model.weights import load_causal_lm_from_hf
from inferlite.sampler import GreedySampler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 inferlite-generate 命令行参数。

    `argv=None` 时使用真实命令行参数，供 console script / python -m 调用。
    单测可以传入显式 list，避免修改全局 `sys.argv`。
    """
    parser = argparse.ArgumentParser(description="Run minimal greedy generation with inferlite.")
    parser.add_argument("--model-dir", required=True, help="Local HF/ModelScope model directory.")
    parser.add_argument("--prompt", required=True, help="Prompt text to generate from.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Maximum number of new tokens to generate. Stops early at EOS if encountered.",
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help=(
            "Wrap prompt with the tokenizer's chat template. "
            "Recommended for instruction-tuned models like Qwen3; "
            "produces much better output than bare prompts."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI 主入口。"""
    args = parse_args(argv)

    # tokenizer 负责 text <-> token ids。这里使用真实模型目录里的 tokenizer 配置。
    # local_files_only=True：显式告诉 transformers 这是本地目录，跳过 HuggingFace Hub
    # 的 repo-id 格式校验（新版 transformers 5.x 会对本地路径触发 hub 验证导致 OSError）。
    model_dir = str(Path(args.model_dir).expanduser().resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )

    # --chat-template: 用 apply_chat_template 包装 prompt。
    # Qwen3 在 <|im_start|>user...<|im_end|><|im_start|>assistant 格式下训练，
    # 裸 prompt 会导致输出重复、不稳定；开启 chat template 质量明显更好。
    if args.chat_template:
        messages = [{"role": "user", "content": args.prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = args.prompt

    # 模型加载只负责 config + safetensors -> Qwen3ForCausalLM。
    model = load_causal_lm_from_hf(model_dir)
    # eval() 把模型设置为推理模式，关掉 Dropout / BatchNorm 的训练行为。
    # 对当前 Qwen3 实现（无 Dropout）影响不大，但这是推理代码的标准做法，不能省。
    model.eval()

    # CLI 只负责组装依赖；generate 不自己创建 sampler/engine，便于后续替换采样策略。
    sampler = GreedySampler()
    engine = EngineCore(model, sampler)

    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

    # torch.no_grad() 在推理时禁止 PyTorch 构建梯度计算图，减少内存占用并加快速度。
    # generate() 本身不会调用 loss.backward()，但不加 no_grad 的话每个 tensor 操作
    # 都会在后台维护 autograd 元数据，造成不必要的开销。
    with torch.no_grad():
        output_ids = generate(
            engine,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )

    # output_ids 包含 prompt + generated tokens；默认打印完整文本，最直观。
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
