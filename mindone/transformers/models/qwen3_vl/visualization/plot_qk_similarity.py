import argparse
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def load_layer_arrays(folder: Path, pattern: str):
    """返回 [(layer_idx, arr)]，按层号排序"""
    files = sorted(folder.glob(pattern))
    out = []
    for f in files:
        m = re.search(r"(\d+)", f.stem)
        if not m:
            continue
        layer_idx = int(m.group(1))
        arr = np.load(f)
        arr = arr.squeeze()  # 支持 [N] 或 [1,N]
        out.append((layer_idx, arr))
    out.sort(key=lambda x: x[0])
    return out

def plot_2d(layers, title, save_path):
    plt.figure(figsize=(10, 4))
    for layer_idx, arr in layers:
        x = np.arange(len(arr))
        plt.plot(x, arr, label=f"layer {layer_idx}")
    plt.xlabel("token position")
    plt.ylabel("cosine similarity")
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[save] {save_path}")

def plot_3d(layers, title, save_path):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection="3d")
    colors = cm.viridis(np.linspace(0, 1, len(layers)))
    for (layer_idx, arr), color in zip(layers, colors):
        x = np.arange(len(arr))
        y = np.full_like(x, layer_idx)
        ax.plot(x, y, arr, color=color, label=f"layer {layer_idx}")
    ax.set_xlabel("token position")
    ax.set_ylabel("layer")
    ax.set_zlabel("cosine similarity")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[save] {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="存放 npy 的目录")
    parser.add_argument("--pattern", type=str, default="*q_sim_layer*.npy", help="glob 匹配模式")
    parser.add_argument("--title", type=str, default="Q/K cosine similarity per layer")
    parser.add_argument("--prefix", type=str, default="q_sim")
    parser.add_argument("--no3d", action="store_true", help="不输出 3D 视图")
    args = parser.parse_args()

    folder = Path(args.folder)
    layers = load_layer_arrays(folder, args.pattern)
    if not layers:
        print("未找到匹配的 npy 文件")
        return

    plot_2d(layers, args.title, folder / f"{args.prefix}_2d.png")
    if not args.no3d:
        plot_3d(layers, args.title, folder / f"{args.prefix}_3d.png")

if __name__ == "__main__":
    main()