from typing import Literal
import os
import mindspore as ms
from PIL import Image, ImageDraw
from mindone.transformers import AutoProcessor

SIMILARITY_SRC = Literal["hidden", "query", "key"]
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
# ------------ 可视化工具 ------------
def _idx_to_hw(idx: int, h_tokens: int, w_tokens: int):
    h = idx // w_tokens
    w = idx % w_tokens
    return h, w

def visualize_pruned_tokens(
    image_path_1: str,
    image_path_2: str,
    removed_indices: ms.Tensor,
    grid_h: int,
    grid_w: int,
    save_path: str,
    removed_color=(255, 0, 0, 90),
):
    """在第二张图上叠加色块，红色=被裁剪，绿色=保留"""
    img = Image.open(image_path_1).convert("RGB")
    draw = ImageDraw.Draw(img, mode="RGBA")
    # 以真实图像尺寸均匀切分网格，避免 patch/merge 推算误差导致溢出
    token_w = img.width / grid_w
    token_h = img.height / grid_h
    for idx in removed_indices.asnumpy().tolist():
        h, w = _idx_to_hw(idx, grid_h, grid_w)
        x0, y0 = w * token_w, h * token_h
        x1, y1 = x0 + token_w, y0 + token_h
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2, fill=removed_color)
    img.save(f"{save_path}_anchor.png")

    img = Image.open(image_path_2).convert("RGB")
    draw = ImageDraw.Draw(img, mode="RGBA")
    # 以真实图像尺寸均匀切分网格，避免 patch/merge 推算误差导致溢出
    token_w = img.width / grid_w
    token_h = img.height / grid_h
    for idx in removed_indices.asnumpy().tolist():
        h, w = _idx_to_hw(idx, grid_h, grid_w)
        x0, y0 = w * token_w, h * token_h
        x1, y1 = x0 + token_w, y0 + token_h
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2, fill=removed_color)
    img.save(f"{save_path}_pruned.png")
    print(f"[viz] saved -> {save_path}")


# ------------ 裁剪信息捕获（monkey-patch，不改模型文件） ------------
def enable_prune_debug(vision_model, similarity_src: SIMILARITY_SRC = "hidden"):
    """
    在 Qwen3VLVisionModel 上打补丁：
    - 复用原始裁剪逻辑，但允许选择相似度来源(hidden/query/key)
    - 将 prune_info 存到 vision_model.last_prune_debug，供可视化使用
    """
    # 若未曾设置，先占位，避免首次访问属性不存在
    vision_model.last_prune_debug = None
    attn_for_qk = vision_model.blocks[1].attn  # 裁剪发生在 layer_num == 1 时

    def compute_qk_states(hidden_states):
        seq_length = hidden_states.shape[0]
        q_states, k_states, _ = (
            attn_for_qk.qkv(hidden_states)
            .reshape(seq_length, 3, attn_for_qk.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        # head 维求均值，得到 per-token 表征
        q_states = q_states.mean(axis=1)
        k_states = k_states.mean(axis=1)
        return q_states, k_states

    def patched(hidden_states, cu_seqlens, position_embeddings):
        seq1_len = int(cu_seqlens[1].asnumpy().item())
        seq2_len = int((cu_seqlens[2] - cu_seqlens[1]).asnumpy().item())

        # 选择相似度来源
        if similarity_src == "hidden":
            src_states = hidden_states
        else:
            q_states, k_states = compute_qk_states(hidden_states)
            src_states = q_states if similarity_src == "query" else k_states

        seq1_states = src_states[:seq1_len]
        seq2_states = src_states[seq1_len : seq1_len + seq2_len]
        similarities = vision_model._compute_token_similarity(seq1_states, seq2_states)

        # --- 以下逻辑复制自原函数，仅把 similarities 换成上面计算的 ---
        remove_mask = (similarities > vision_model.token_prune_threshold).astype(ms.bool_)
        keep_mask = ms.mint.logical_not(remove_mask)
        keep_count = int(keep_mask.astype(ms.int32).sum().asnumpy().item())
        if keep_count == seq2_len or keep_count == 0:
            vision_model.last_prune_debug = None
            return hidden_states, cu_seqlens, position_embeddings, None

        keep_indices = ms.mint.nonzero(keep_mask.astype(ms.int32)).flatten().astype(ms.int32)
        removed_indices = ms.mint.nonzero(remove_mask.astype(ms.int32)).flatten().astype(ms.int32)
        seq2_kept = ms.mint.index_select(hidden_states[seq1_len:], 0, keep_indices)
        new_hidden_states = ms.mint.cat([hidden_states[:seq1_len], seq2_kept], dim=0)

        new_cu = cu_seqlens.copy()
        new_cu[2] = seq1_len + keep_count

        cos, sin = position_embeddings
        cos = vision_model._prune_second_sequence(cos, cu_seqlens, keep_indices)
        sin = vision_model._prune_second_sequence(sin, cu_seqlens, keep_indices)
        new_position_embeddings = (cos, sin)

        prune_info = {
            "removed_mask": remove_mask,
            "keep_mask": keep_mask,
            "original_cu_seqlens": cu_seqlens,
            "keep_indices": keep_indices,
            "removed_indices": removed_indices,
            "original_position_embeddings": position_embeddings,
        }
        # 记录调试信息
        vision_model.last_prune_debug = {
            "similarity_src": similarity_src,
            "seq1_len": seq1_len,
            "seq2_len": seq2_len,
            "removed_indices": removed_indices,
            "keep_indices": keep_indices,
            "grid_hw": (seq2_len // int(vision_model.spatial_merge_size ** 2), None),  # 占位，下面运行时再填
        }
        print(f"keep_count: {keep_count}")
        return new_hidden_states, new_cu, new_position_embeddings, prune_info

    vision_model._prune_hidden_states_position_embeddings = patched
    print(f"[patch] prune debug enabled, similarity_src={similarity_src}")


# ------------ 运行与可视化示例 ------------
def run_and_visualize(
    model,
    image_path_1: str,
    image_path_2: str,
    similarity_src: SIMILARITY_SRC = "hidden",
    out_path: str = "prune_viz.png",
    token_prune_threshold: float = 0.99,
):
    # 启用调试/替换相似度来源
    enable_prune_debug(model.visual, similarity_src=similarity_src)

    # 读取图片 & 预处理（示意，替换为真实 processor）
    processor = AutoProcessor.from_pretrained(MODEL)  # 示例
    images = [Image.open(image_path_1).convert("RGB"), Image.open(image_path_2).convert("RGB")]
    # 传入占位 text，避免 text=None 时处理器在插入占位符时对 None 迭代报错
    inputs = processor(images=images, text=[""], return_tensors="ms")
    grid_thw = inputs["image_grid_thw"]

    # 前向（只需要视觉部分即可，也可以走全模型）
    ms.context.set_context(mode=0)  # pynative
    _ = model.visual(inputs["pixel_values"], grid_thw=grid_thw,
        token_prune_enabled=True, token_prune_threshold=token_prune_threshold)

    debug = model.visual.last_prune_debug
    if debug is None:
        print("[viz] 没有发生裁剪或 keep_count 为 0/全部")
        return

    # grid_h, grid_w 针对第二张图：grid_thw[1, 1], grid_thw[1, 2]
    grid_h = int(grid_thw[1, 1].asnumpy().item())
    grid_w = int(grid_thw[1, 2].asnumpy().item())
    debug["grid_hw"] = (grid_h, grid_w)

    visualize_pruned_tokens(
        image_path_1=image_path_1,
        image_path_2=image_path_2,
        removed_indices=debug["removed_indices"],
        keep_indices=debug["keep_indices"],
        grid_h=grid_h,
        grid_w=grid_w,
        patch_size=model.config.vision_config.patch_size,
        merge_size=model.config.vision_config.spatial_merge_size,
        save_path=out_path,
    )


if __name__ == "__main__":
    # 使用方式示例
    from mindone.transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, trust_remote_code=True, mindspore_dtype=ms.bfloat16
    )
    for (image1_idx, image2_idx) in [(1, 2), (2, 3), (3, 4), (4, 5)]:
        save_path = f"prune_visualization/image{image1_idx}_{image2_idx}"
        os.makedirs(save_path, exist_ok=True)
        for similarity_src in ["hidden", "query", "key"]:
            for token_prune_threshold in [0.99, 0.995]:
                run_and_visualize(
                    model,  
                    image_path_1=f"images/image{image1_idx}.jpg",
                    image_path_2=f"images/image{image2_idx}.jpg",
                    similarity_src=similarity_src,  # 可选 "hidden" / "query" / "key"
                    out_path=f"{save_path}/{similarity_src}_{token_prune_threshold}",
                    token_prune_threshold=token_prune_threshold,
            )