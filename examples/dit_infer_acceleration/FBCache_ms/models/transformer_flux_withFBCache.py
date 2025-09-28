import mindspore as ms
from mindspore import mint
from typing import Any, Callable, Dict, List, Optional, Union     
from mindone.diffusers.models.modeling_outputs import Transformer2DModelOutput
from mindone.diffusers.utils import logging
logger = logging.get_logger(__name__)


class CacheContext(ms.nn.Cell):
    def __init__(self, 
                batch_size: int,
                seq_len: int,
                inner_dim: int,
                dtype: ms.dtype = ms.bfloat16,
                do_true_cfg: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.inner_dim = inner_dim
        self.residual = ms.Parameter(mint.ones((batch_size, seq_len, inner_dim), dtype = dtype), name = "residual")
        self.first_residual = ms.Parameter(mint.ones((batch_size, seq_len, inner_dim), dtype = dtype), name = "first_residual")
        self.do_true_cfg = do_true_cfg
        self.dtype = dtype
        if do_true_cfg:
            self.neg_first_residual = ms.Parameter(mint.zeros((batch_size, seq_len, inner_dim), dtype = dtype), name = "neg_first_residual")
            self.neg_residual = ms.Parameter(mint.zeros((batch_size, seq_len, inner_dim), dtype = dtype), name = "neg_residual")
    def construct(self, new_residual: ms.Tensor):
        self.update_residual(new_residual)
    def update_residual(self, new_residual: ms.Tensor):
        self.residual = new_residual
    def update_first_residual(self, new_first_residual: ms.Tensor):
        self.first_residual = new_first_residual
    def update_neg_first_residual(self, new_neg_first_residual: ms.Tensor):
        self.neg_first_residual = new_neg_first_residual
    def update_neg_residual(self, new_neg_residual: ms.Tensor):
        self.neg_residual = new_neg_residual
        
def are_two_tensors_similar(t1, t2, *, threshold):
    if threshold <= 0.0:
        return False

    if t1.shape != t2.shape:
        return False

    mean_diff = (t1 - t2).abs().mean()
    mean_t1 = t1.abs().mean()
    # if parallelized:
    #     mean_diff = DP.all_reduce_sync(mean_diff, "avg")
    #     mean_t1 = DP.all_reduce_sync(mean_t1, "avg")
    diff = mean_diff / mean_t1
    return diff < threshold

def FBCache_transformer_construct(
    self,
    hidden_states: ms.Tensor,
    encoder_hidden_states: ms.Tensor = None,
    pooled_projections: ms.Tensor = None,
    timestep: ms.Tensor = None,
    img_ids: ms.Tensor = None,
    txt_ids: ms.Tensor = None,
    guidance: ms.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = False,
    controlnet_blocks_repeat: bool = False,
    return_hidden_states_first: bool = False,
    residual_diff_threshold: float = 0.12,
) -> Union[ms.Tensor, Transformer2DModelOutput]:
    """
    The [`FluxTransformer2DModel`] forward method.

    Args:
        hidden_states (`ms.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
            Input `hidden_states`.
        encoder_hidden_states (`ms.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`ms.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
            from the embeddings of input conditions.
        timestep ( `ms.Tensor`):
            Used to indicate denoising step.
        block_controlnet_hidden_states: (`list` of `ms.Tensor`):
            A list of tensors that if specified are added to the residuals of transformer blocks.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """

    if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
        logger.warning(
            "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
        )

    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d ms.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d mindspore Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d ms.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d mindspore Tensor"
        )
        img_ids = img_ids[0]

    ids = mint.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})
    ############FB Cache starts here###############
    original_hidden_states = hidden_states
    first_block = self.transformer_blocks[0]
    hidden_states = first_block(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        image_rotary_emb=image_rotary_emb,
        joint_attention_kwargs=joint_attention_kwargs,
    )
    if not isinstance(hidden_states, ms.Tensor): # distinguish transformer blocks and single
        hidden_states, encoder_hidden_states = hidden_states
        if not return_hidden_states_first:
            hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
    first_hidden_states_residual = hidden_states - original_hidden_states
    can_use_cache = are_two_tensors_similar(
        self.cache_context.first_residual,
        first_hidden_states_residual,
        threshold=residual_diff_threshold,
    )
    if can_use_cache:
        hidden_states = hidden_states + self.cache_context.residual
    else:
        self.cache_context.update_first_residual(first_hidden_states_residual)
        original_hidden_states = hidden_states
        for index_block, block in enumerate(self.transformer_blocks[1:]):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = (len(self.transformer_blocks) + len(controlnet_block_samples) - 1) // len(
                    controlnet_block_samples
                )  # not supporting numpy
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
        hidden_states = mint.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = (
                    len(self.single_transformer_blocks) + len(controlnet_single_block_samples) - 1
                ) // len(
                    controlnet_single_block_samples
                )  # not supporting numpy
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        # hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        hidden_states = mint.split(
            hidden_states,
            [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],
            dim=1,
        )[1]
        self.cache_context.update_residual(hidden_states - original_hidden_states)
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output)

    return Transformer2DModelOutput(sample=output)
