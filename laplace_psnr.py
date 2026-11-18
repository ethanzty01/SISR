import argparse
import csv
import json
from copy import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models import DPSR, FSRCNN
from utils.dataloader import rgb_to_y_uint8 as rgb_to_model_y
from utils.laplace import laplacian_map as shared_laplacian_map, rgb_to_studio_y


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VAL_DIR = Path("/home/tyzheng/Datasets_pt/test/Test4k")
DEFAULT_DPSR_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "DPSR_x4_0815_1611.pth"
DEFAULT_FSRCNN_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "FSRCNN_x4_0806_1601.pth"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "spatial_redundancy_plots" / "x4_test4k_fp32_patch16"
LAPLACIAN_STEP = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate FP32 SR models on blocks grouped by LR Laplacian magnitude."
    )
    parser.add_argument("--model", choices=("all", "dpsr", "fsrcnn"), default="all")
    parser.add_argument("--val_dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dpsr_checkpoint", type=Path, default=DEFAULT_DPSR_CHECKPOINT)
    parser.add_argument("--fsrcnn_checkpoint", type=Path, default=DEFAULT_FSRCNN_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scale", type=int, default=4, choices=(2, 3, 4))
    parser.add_argument("--in_channels", type=int, default=3, choices=(1, 3))
    parser.add_argument("--channel_nums", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=5)
    parser.add_argument("--subnet_channels", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=16, help="LR block width/height (16 LR = 64 HR at x4).")
    parser.add_argument("--max_images", type=int, default=0, help="0 uses the full dataset.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--trend_degree", type=int, default=3, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--plot_range", nargs=2, type=float, default=None,
                        help="Optional display range only; defaults to all bins and never filters statistics.")
    return parser.parse_args()


def to_y_channel(image):
    """Convert an NCHW tensor in [0, 255] to one studio-range Y channel."""
    return rgb_to_studio_y(image)


def integer_psnr_pixels(image):
    """Round RGB and then studio Y half-up; subtract in float to avoid uint8 overflow."""
    pixels = torch.floor(image.clamp(0, 255) + 0.5)
    return torch.floor(to_y_channel(pixels).clamp(0, 255) + 0.5)


def iter_validation_pairs(val_dir, scale, in_channels):
    shard_files = sorted(val_dir.glob("*.pt"))
    if not shard_files:
        raise ValueError(f"No validation .pt shards found in {val_dir}")
    lr_key = f"lr_x{scale}"
    for shard_path in shard_files:
        packed = torch.load(shard_path, weights_only=False)
        if not isinstance(packed, dict) or "hr" not in packed or lr_key not in packed:
            raise ValueError(f"{shard_path} is not a supported validation shard")
        if len(packed["hr"]) != len(packed[lr_key]):
            raise ValueError(f"{shard_path} has mismatched HR/LR sample counts")
        for lr, hr in zip(packed[lr_key], packed["hr"]):
            if in_channels == 1:
                lr = rgb_to_model_y(lr)
                hr = rgb_to_model_y(hr)
            yield lr.unsqueeze(0), hr.unsqueeze(0)


def block_means(image, block_size):
    blocks = F.unfold(image, kernel_size=block_size, stride=block_size)
    return blocks.mean(dim=1).reshape(-1)


def block_mse(prediction, target, block_size):
    return block_means((prediction - target).square(), block_size)


def load_model(args, device):
    checkpoint_path = args.checkpoint or getattr(args, f"{args.model}_checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if args.model == "dpsr":
        model = DPSR(
            scale=args.scale,
            in_dim=args.in_channels,
            fea_dim=args.channel_nums,
            num_blocks=args.num_blocks,
            bias=False,
            subnet_channels=args.subnet_channels,
            use_bilinear=checkpoint.get("model_config", {}).get("is_residual", True),
        )
    else:
        model = FSRCNN(scale_factor=args.scale, num_channels=args.in_channels)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    return model.to(device=device, dtype=torch.float32).eval(), checkpoint_path


def model_outputs(model, model_input, bilinear, args):
    if args.model == "all":
        outputs = {}
        for name, child in model.items():
            child_args = copy(args)
            child_args.model = name
            outputs.update(model_outputs(child, model_input, bilinear, child_args))
        return outputs
    if args.model == "dpsr":
        return {
            "dpsr_full": model(model_input, channels=args.channel_nums) * 255.0,
            "dpsr_subnet": model(model_input, channels=args.subnet_channels) * 255.0,
        }
    return {"fsrcnn": model(model_input) * 255.0}


def collect_block_statistics(model, loader, args, device):
    score_parts = []
    model_mse_parts = {name: [] for name in ("dpsr_full", "dpsr_subnet") if args.model in ("all", "dpsr")}
    if args.model in ("all", "fsrcnn"):
        model_mse_parts["fsrcnn"] = []
    bilinear_mse_parts = []

    with torch.no_grad():
        for image_index, (lr, hr) in enumerate(loader):
            if args.max_images and image_index >= args.max_images:
                break
            lr = lr.to(device=device, dtype=torch.float32)
            hr = hr.to(device=device, dtype=torch.float32)
            lr_height = min(lr.shape[-2], hr.shape[-2] // args.scale)
            lr_width = min(lr.shape[-1], hr.shape[-1] // args.scale)
            lr_height = (lr_height // args.block_size) * args.block_size
            lr_width = (lr_width // args.block_size) * args.block_size
            if lr_height == 0 or lr_width == 0:
                continue

            lr = lr[..., :lr_height, :lr_width]
            hr = hr[..., : lr_height * args.scale, : lr_width * args.scale]
            bilinear = F.interpolate(
                lr, scale_factor=args.scale, mode="bilinear", align_corners=False
            ).round().clamp(0, 255)
            predictions = model_outputs(model, lr / 255.0, bilinear, args)
            hr_y = integer_psnr_pixels(hr)
            bilinear_y = integer_psnr_pixels(bilinear)
            hr_block_size = args.block_size * args.scale
            score_parts.append(
                block_means(shared_laplacian_map(lr), args.block_size).cpu()
            )
            bilinear_mse_parts.append(block_mse(bilinear_y, hr_y, hr_block_size).cpu())
            for name, prediction in predictions.items():
                prediction_y = integer_psnr_pixels(prediction)
                model_mse_parts[name].append(
                    block_mse(prediction_y, hr_y, hr_block_size).cpu()
                )

            print(f"Processed image {image_index + 1}: {lr_height // args.block_size * (lr_width // args.block_size):,} patches", flush=True)

    if not score_parts:
        raise ValueError(f"No complete {args.block_size}x{args.block_size} LR blocks found in {args.val_dir}")
    return (
        torch.cat(score_parts).numpy(),
        {name: torch.cat(parts).numpy() for name, parts in model_mse_parts.items()},
        torch.cat(bilinear_mse_parts).numpy(),
    )


def bin_statistics(scores, model_mses, bilinear_mse):
    """Use unit-width [min, max) bins covering every observed Laplacian score."""
    if scores.size == 0 or not np.all(np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("Laplacian scores must be nonempty, finite and nonnegative")
    bins = int(np.floor(float(scores.max()) / LAPLACIAN_STEP)) + 1
    edges = np.arange(bins + 1, dtype=np.float64) * LAPLACIAN_STEP
    bin_ids = np.digitize(scores, edges[1:-1])
    counts = np.bincount(bin_ids, minlength=bins)

    def mean_psnr_by_bin(mse):
        with np.errstate(divide="ignore"):
            psnr = 10.0 * np.log10((255.0**2) / mse.astype(np.float64))
        psnr_sums = np.bincount(bin_ids, weights=psnr, minlength=bins)
        return np.divide(psnr_sums, counts, out=np.full(bins, np.nan), where=counts > 0)

    return {
        "edges": edges,
        "centers": 0.5 * (edges[:-1] + edges[1:]),
        "counts": counts,
        "model_psnr": {name: mean_psnr_by_bin(mse) for name, mse in model_mses.items()},
        "bilinear_psnr": mean_psnr_by_bin(bilinear_mse),
    }


def save_csv(stats, output_path):
    model_columns = [f"{name}_psnr_db" for name in stats["model_psnr"]]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow((
            "laplacian_min", "laplacian_max", "laplacian_center", "block_count",
            *model_columns, "bilinear_psnr_db",
        ))
        for index, center in enumerate(stats["centers"]):
            writer.writerow((
                stats["edges"][index], stats["edges"][index + 1], center,
                int(stats["counts"][index]),
                *(values[index] for values in stats["model_psnr"].values()),
                stats["bilinear_psnr"][index],
            ))


def fit_trend_curve(centers, values, counts, degree, points=256):
    valid = np.isfinite(centers) & np.isfinite(values) & (counts > 0)
    x = centers[valid]
    y = values[valid]
    if x.size < 2:
        raise ValueError("At least two populated Laplacian intervals are required.")
    trend = np.polynomial.Chebyshev.fit(
        x, y, min(degree, x.size - 1), w=np.sqrt(counts[valid])
    )
    trend_x = np.linspace(x.min(), x.max(), points)
    return trend_x, trend(trend_x)


def save_line_plot(stats, output_path, plot_range, trend_degree):
    centers = stats["centers"]
    counts = stats["counts"]
    plot_min, plot_max = plot_range
    curves = [(stats["bilinear_psnr"], "#5a89e6", "Bilinear")]
    styles = {
        "fsrcnn": ("#8064a2", "FSRCNN"),
        "dpsr_subnet": ("#d9534f", "DPSR 16ch + Bilinear"),
        "dpsr_full": ("#8ddb51", "DPSR 32ch + Bilinear"),
    }
    curves.extend((values, *styles[name]) for name, values in stats["model_psnr"].items())
    fig, ax = plt.subplots(figsize=(5.0, 3.6), constrained_layout=True)
    for values, color, label in curves:
        visible = (centers >= plot_min) & (centers <= plot_max)
        trend_x, trend_y = fit_trend_curve(
            centers[visible], values[visible], counts[visible], trend_degree
        )
        ax.plot(trend_x, trend_y, color=color, linewidth=3, label=label)
    ax.set_xlim(plot_min, plot_max)
    ax.set_xlabel("Laplacian Magnitude", fontsize=22, labelpad=4, fontweight="bold")
    ax.set_ylabel("PSNR (dB)", fontsize=22, labelpad=4, fontweight="bold")
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(color="gray", linewidth=1.0, alpha=0.65, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, prop={"size": 13, "weight": "bold"})
    fig.savefig(output_path, dpi=300, bbox_inches="tight", format="svg", transparent=True)
    plt.close(fig)


def main():
    args = parse_args()
    if args.model in ("all", "dpsr") and not 0 < args.subnet_channels < args.channel_nums:
        raise ValueError("--subnet_channels must be in [1, channel_nums - 1] for DPSR")
    if args.block_size <= 0 or args.max_images < 0:
        raise ValueError("block_size must be positive and max_images nonnegative")
    if args.model == "all" and args.checkpoint is not None:
        raise ValueError("Use --dpsr_checkpoint and --fsrcnn_checkpoint with --model all")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.plot_range is not None:
        args.plot_range = tuple(args.plot_range)
        plot_min, plot_max = args.plot_range
        if not np.all(np.isfinite(args.plot_range)) or not 0 <= plot_min < plot_max:
            raise ValueError("--plot_range must satisfy 0 <= MIN < MAX with finite bounds")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint_paths = {}, {}
    for name in (("dpsr", "fsrcnn") if args.model == "all" else (args.model,)):
        child_args = copy(args)
        child_args.model = name
        model[name], checkpoint_paths[name] = load_model(child_args, device)
    if args.model != "all":
        model = model[args.model]
    loader = iter_validation_pairs(args.val_dir, args.scale, args.in_channels)
    scores, model_mses, bilinear_mse = collect_block_statistics(model, loader, args, device)
    stats = bin_statistics(scores, model_mses, bilinear_mse)
    if args.plot_range is None:
        args.plot_range = (float(stats["edges"][0]), float(stats["edges"][-1]))

    output_stem = args.output_dir / f"laplace_psnr_{args.model}"
    csv_path = output_stem.with_suffix(".csv")
    plot_path = output_stem.with_suffix(".svg")
    save_csv(stats, csv_path)
    save_line_plot(stats, plot_path, args.plot_range, args.trend_degree)

    metadata = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "checkpoints": {key: str(value) for key, value in checkpoint_paths.items()},
        "device": str(device), "precision": "FP32, TF32 disabled",
        "patch": "non-overlapping LR patches after full-image inference; incomplete edges cropped",
        "laplacian": "patch mean of 0.299*abs(L(R))+0.587*abs(L(G))+0.114*abs(L(B)); grayscale input: abs(L(Y)); normalized eight-neighbor kernel, replicate padding, no intermediate rounding",
        "psnr": "RGB clamp and half-up rounding, studio Y conversion and half-up rounding; peak 255; arithmetic mean of patch PSNR, zero MSE gives infinity",
        "bilinear": "align_corners=False, rounded as in training before adding DPSR residual",
        "total_patches": len(scores), "binned_patches": int(stats["counts"].sum()),
        "laplacian_observed_max": float(scores.max()),
        "laplacian_bin_range": [float(stats["edges"][0]), float(stats["edges"][-1])],
        "laplacian_bin_width": LAPLACIAN_STEP,
    }
    output_stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    included_blocks = int(stats["counts"].sum())
    print(f"Evaluated {args.model.upper()} from {checkpoint_paths} on {device}.")
    print(f"Analyzed {len(scores):,} blocks; all {included_blocks:,} are in [0.0, {stats['edges'][-1]:.1f}).")
    print(f"Saved statistics to {csv_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
