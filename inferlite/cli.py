"""Command line entrypoint for minimal greedy generation.

CLI 负责把用户输入的文本参数转换成一次完整推理调用：

    parse args
      -> load tokenizer
      -> load Qwen3ForCausalLM weights
      -> build GreedySampler + EngineCore
      -> tokenizer.encode(prompt)
      -> generate(engine, input_ids)
      -> tokenizer.decode(output_ids)
      -> print text

注意：
- T11 CLI 默认打印完整序列，也就是 prompt + generated text。
- 当前没有 EOS 停止、KV cache、top-p/temperature，只是最小 greedy 闭环。
"""

import argparse

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
        help="Number of new tokens to generate. EOS stopping is not implemented in T11.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI 主入口。"""
    args = parse_args(argv)

    # tokenizer 负责 text <-> token ids。这里使用真实模型目录里的 tokenizer 配置。
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # 模型加载只负责 config + safetensors -> Qwen3ForCausalLM。
    model = load_causal_lm_from_hf(args.model_dir)

    # CLI 只负责组装依赖；generate 不自己创建 sampler/engine，便于后续替换采样策略。
    sampler = GreedySampler()
    engine = EngineCore(model, sampler)

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    output_ids = generate(engine, input_ids, max_new_tokens=args.max_new_tokens)

    # output_ids 包含 prompt + generated tokens；T11 默认打印完整文本，最直观。
    # 后续可以增加 --only-new-text 只输出新增部分。
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
