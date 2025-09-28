import functools
import unittest

import mindspore as ms
from mindspore import ops
from mindone.diffusers import DiffusionPipeline, FluxTransformer2DModel

from FBCache_ms import utils
import json

def apply_cache_on_transformer(
    transformer: FluxTransformer2DModel,
):
    if getattr(transformer, "_is_cached", False):
        return transformer

    cached_transformer_blocks = ms.nn.CellList(
        [
            utils.CachedTransformerBlocks(
                transformer.transformer_blocks,
                transformer.single_transformer_blocks,
                transformer=transformer,
                return_hidden_states_first=False,
            )
        ]
    )
    dummy_single_transformer_blocks = ms.nn.CellList()

    original_construct = transformer.construct

    @functools.wraps(original_construct)
    def new_forward(
        self,
        *args,
        **kwargs,
    ):
        with unittest.mock.patch.object(
            self,
            "transformer_blocks",
            cached_transformer_blocks,
        ), unittest.mock.patch.object(
            self,
            "single_transformer_blocks",
            dummy_single_transformer_blocks,
        ):
            return original_construct(
                *args,
                **kwargs,
            )

    transformer.construct = new_forward.__get__(transformer)

    transformer._is_cached = True

    return transformer


def apply_cache_on_pipe(
    pipe: DiffusionPipeline,
    *,
    shallow_patch: bool = False,
    save_params: bool = False,
    save_params_path: str = "cache_usage_history.json",
    **kwargs,
):
    if not getattr(pipe, "_is_cached", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs_):
            # 先创建缓存上下文对象并保存引用
            cache_ctx = utils.create_cache_context(save_params=save_params, **kwargs)
            # 在 with 语句中使用这个对象
            with utils.cache_context(cache_ctx):
                result = original_call(self, *args, **kwargs_)
            # 在 with 语句结束后打印缓存上下文的数据
                if save_params:
                    cache_usage_history = utils.get_cache_usage_history()
                    print(f"Cache scheme: {cache_usage_history}")
                    with open(save_params_path, "w") as f:
                        json.dump(cache_usage_history, f, indent=4)
            return result

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_cached = True

    if not shallow_patch:
        apply_cache_on_transformer(pipe.transformer)

    return pipe