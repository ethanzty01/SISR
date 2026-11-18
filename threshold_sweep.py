"""Threshold sweeps using cached, overlapping DPSR patch predictions."""
import csv
import json
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from laplace_psnr import integer_psnr_pixels, iter_validation_pairs
from utils.laplace import laplacian_map


def threshold_configs():
    return [("bilinear_to_16ch", low, 20) for low in range(2, 11)] + [
        ("16ch_to_32ch", 5, high) for high in range(15, 26)
    ]


class CachedPatches:
    """Same patch positions, branch rounding and overlap averaging as DPSRFrame."""

    def __init__(self, frame, lr, batch_size):
        self.frame = frame
        self.height, self.width = lr.shape[-2:]
        self.rows, self.cols = frame._starts(self.height), frame._starts(self.width)
        p = frame.patch_size
        laplace = laplacian_map(lr)
        patches, scores = [], []
        for top in self.rows:
            for left in self.cols:
                patch = lr[..., top:top + p, left:left + p]
                scores.append(laplace[..., top:top + p, left:left + p].mean())
                patches.append(F.pad(patch, (0, p - patch.shape[-1], 0, p - patch.shape[-2]), mode="replicate"))
        patches = torch.cat(patches)
        self.scores = torch.stack(scores)
        outputs = [[], [], []]
        for batch in patches.split(batch_size):
            base = F.interpolate(batch, scale_factor=frame.scale, mode="bilinear", align_corners=False).round().clamp(0, 255)
            outputs[0].append(base.to(torch.uint8))
            for branch, channels in ((1, frame.subnet_channels), (2, frame.residual_model.fea_dim)):
                sr = frame._prediction(batch, channels).round().clamp(0, 255)
                outputs[branch].append(sr.to(torch.uint8))
        # Storing rounded pixels as uint8 is lossless; all model inference is FP32.
        self.outputs = torch.stack([torch.cat(parts) for parts in outputs])
        self.patch_ids = torch.arange(len(scores), device=lr.device)
        self.weight = self.stitch(torch.ones_like(self.outputs[0, :, :1], dtype=torch.float32))
        if not torch.all(self.weight > 0):
            raise ValueError("Patch layout leaves uncovered pixels")

    def _groups(self, starts):
        # The final patch can have an irregular stride to cover the image edge.
        if len(starts) > 1 and starts[-1] - starts[-2] != self.frame.stride:
            return [list(range(len(starts) - 1)), [len(starts) - 1]]
        return [list(range(len(starts)))]

    def stitch(self, patches):
        scale = self.frame.scale
        p, stride = self.frame.patch_size * scale, self.frame.stride * scale
        channels = patches.shape[1]
        grid = patches.reshape(len(self.rows), len(self.cols), channels, p, p)
        canvas = patches.new_zeros((1, channels, max(self.height * scale, p), max(self.width * scale, p)), dtype=torch.float32)
        for rows in self._groups(self.rows):
            for cols in self._groups(self.cols):
                group = grid[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
                blocks = group.reshape(-1, channels * p * p).T.unsqueeze(0).float()
                h, w = (len(rows) - 1) * stride + p, (len(cols) - 1) * stride + p
                folded = F.fold(blocks, (h, w), kernel_size=p, stride=stride)
                top, left = self.rows[rows[0]] * scale, self.cols[cols[0]] * scale
                canvas[..., top:top + h, left:left + w] += folded
        return canvas[..., :self.height * scale, :self.width * scale]

    def infer(self, low, high):
        branches = torch.where(self.scores >= high, 2, torch.where(self.scores >= low, 1, 0))
        image = self.stitch(self.outputs[branches, self.patch_ids]) / self.weight
        return image, torch.bincount(branches, minlength=3).cpu().tolist()


def image_psnr(prediction, target, border):
    prediction = integer_psnr_pixels(prediction)[..., border:-border, border:-border]
    target = integer_psnr_pixels(target)[..., border:-border, border:-border]
    if prediction.numel() == 0 or prediction.shape != target.shape:
        raise ValueError("Invalid prediction/target size after border crop")
    mse = (prediction - target).square().double().mean().item()
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_threshold_sweeps(args):
    from final_test import DPSRFrame, build_residual_model

    if (args.scale, args.arm_patch_size, args.arm_overlap) != (4, 16, 2):
        raise ValueError("These sweeps require --scale 4 --arm_patch_size 16 --arm_overlap 2")
    if args.clip_ratio != 1:
        raise ValueError("Sweeps require all images (clip_ratio=1)")
    if args.patch_batch_size < 1 or args.max_images < 0:
        raise ValueError("Invalid patch batch size or image limit")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_residual_model(args, device).float()
    frame = DPSRFrame(model, args.scale, args.arm_patch_size, args.arm_overlap, 20, 5, device, args.subnet_channels)
    configs = threshold_configs()
    dataset = Path(args.val_root) / "Test4k"
    output = Path(args.sweep_output_dir) if args.sweep_output_dir else Path(__file__).resolve().parent / "threshold_sweeps" / "x4_test4k_patch16_overlap2"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    loader = iter_validation_pairs(dataset, args.scale, 3)
    if args.max_images:
        loader = islice(loader, args.max_images)
    print(f"Running {len(configs)} threshold configurations on {device}, FP32, patch=16, overlap=2", flush=True)
    with torch.inference_mode():
        for index, (lr, hr) in enumerate(loader, 1):
            lr, hr = lr.to(device).float(), hr.to(device).float()
            if hr.shape[-2:] != (lr.shape[-2] * args.scale, lr.shape[-1] * args.scale):
                raise ValueError("HR dimensions must match LR dimensions times scale")
            cached = CachedPatches(frame, lr, args.patch_batch_size)
            # (5,20) appears in both experiments; compute it once per image.
            results = {}
            for experiment, low, high in configs:
                if (low, high) not in results:
                    sr, counts = cached.infer(low, high)
                    results[low, high] = (image_psnr(sr, hr, args.scale), counts)
                psnr, counts = results[low, high]
                records.append(dict(experiment=experiment, image_index=index, bilinear_to_16ch=low,
                                    to_32ch=high, psnr_db=psnr, bilinear_patches=counts[0],
                                    dpsr16_patches=counts[1], dpsr32_patches=counts[2]))
            del cached
            print(f"Processed image {index}: baseline (5,20) PSNR={results[5,20][0]:.6f} dB", flush=True)
    if not records:
        raise ValueError("No images evaluated")
    summary = []
    for experiment, low, high in configs:
        rows = [r for r in records if (r['experiment'], r['bilinear_to_16ch'], r['to_32ch']) == (experiment, low, high)]
        counts = np.array([[r['bilinear_patches'], r['dpsr16_patches'], r['dpsr32_patches']] for r in rows]).sum(axis=0)
        summary.append(dict(experiment=experiment, bilinear_to_16ch=low, to_32ch=high,
                            mean_psnr_db=float(np.mean([r['psnr_db'] for r in rows])), image_count=len(rows),
                            total_patches=int(counts.sum()), bilinear_ratio=counts[0]/counts.sum(),
                            dpsr16_ratio=counts[1]/counts.sum(), dpsr32_ratio=counts[2]/counts.sum()))
    baseline = next(r['mean_psnr_db'] for r in summary if (r['bilinear_to_16ch'], r['to_32ch']) == (5,20))
    for row in summary:
        row['delta_vs_5_20_db'] = row['mean_psnr_db'] - baseline
    write_csv(output / 'per_image.csv', records)
    write_csv(output / 'summary.csv', summary)
    for experiment in ('bilinear_to_16ch', '16ch_to_32ch'):
        write_csv(output / f'{experiment}.csv', [r for r in summary if r['experiment'] == experiment])
    metadata = dict(arguments=vars(args), device=str(device), precision='FP32; TF32 disabled',
                    dataset=str(dataset), checkpoint=str(Path(args.dpsr_checkpoint).resolve()),
                    routing='score < low: bilinear; low <= score < high: 16ch; score >= high: 32ch',
                    laplacian='patch mean of 0.299*abs(L(R))+0.587*abs(L(G))+0.114*abs(L(B)) on full LR image; normalized eight-neighbor kernel, replicate padding',
                    stitching='DPSRFrame patch starts, replicate-pad small images, average overlapping rounded RGB outputs',
                    psnr='round/clamp RGB and studio Y half-up after overlap averaging; peak=255; crop 4 HR pixels; FP64 MSE; arithmetic mean of image PSNR',
                    image_count=summary[0]['image_count'])
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    save_sweep_plot(summary, output / 'threshold_sweeps.png')
    for row in summary:
        print(f"{row['experiment']}: ({row['bilinear_to_16ch']}, {row['to_32ch']}) {row['mean_psnr_db']:.6f} dB ({row['delta_vs_5_20_db']:+.6f})")
    print(f"Saved results to {output}", flush=True)


def save_sweep_plot(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, experiment, key, title in zip(axes, ('bilinear_to_16ch', '16ch_to_32ch'),
                                         ('bilinear_to_16ch', 'to_32ch'), ('32ch threshold fixed at 20', 'Bilinear threshold fixed at 5')):
        rows = [r for r in summary if r['experiment'] == experiment]
        ax.plot([r[key] for r in rows], [r['mean_psnr_db'] for r in rows], marker='o')
        ax.set(xlabel='Laplacian threshold', ylabel='Mean image PSNR (dB)', title=title)
        ax.grid(alpha=0.3)
    fig.savefig(path, dpi=200)
    plt.close(fig)
