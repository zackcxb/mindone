import argparse
from pathlib import Path
from typing import List
import time
import mindspore as ms
import numpy as np

from mindone.transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def _build_image_payloads(image_paths: List[str]):
    payloads = []
    for path_str in image_paths:
        path = Path(path_str).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"找不到图片文件: {path}")
        payloads.append(
            {
                "type": "image",
                "url": path_str,
            }
        )
    return payloads


def _numpy_to_ms(inputs):
    for key, value in inputs.items():
        if isinstance(value, np.ndarray):
            inputs[key] = ms.Tensor(value)
        elif isinstance(value, list):
            inputs[key] = ms.Tensor(value)
    return inputs


def generate(args):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        mindspore_dtype=ms.bfloat16,
        attn_implementation=args.attn_implementation,
    )

    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=False)

    image_payloads = _build_image_payloads(args.images)
    messages = [
        {
            "role": "user",
            "content": [
                *image_payloads,
                {
                    "type": "text",
                    "text": args.prompt,
                },
            ],
        }
    ]

    model_inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="np",
    )

    model_inputs = _numpy_to_ms(model_inputs)
    if args.profile:
        profiler = ms.profiler.Profiler(start_profile=False, output_path="/home/cxb/profiler_data")
        profiler.start()
    total_time = 0
    for i in range(args.num_prefill_only):
        start_time = time.time()
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1,
        )
        if i > 0:
            total_time += time.time() - start_time
    print(f"Average time per prefill: {total_time / args.num_prefill_only}")
    if args.profile:
        profiler.stop()
        profiler.analyse()

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    outputs = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(outputs[0])


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3VL 多图多模态推理示例")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="文本提示词",
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        required=True,
        help="本地图片路径列表，支持多个",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="预训练权重名称或路径，例如 Qwen/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "eager"],
        help="注意力实现方式",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="生成的最大 token 数",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="是否使用采样解码",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="nucleus sampling 参数",
    )
    parser.add_argument(
        "--num_prefill_only",
        type=int,
        default=10,
        help="只进行 prefill 的次数",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="是否进行 profiling",
    )
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())

