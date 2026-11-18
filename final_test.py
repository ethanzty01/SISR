import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import DPSR, QDPSR, channel_label
from utils import create_val_loader, final_test_parser, metrics
from utils.laplace import laplacian_map


class DPSRFrame:
    def __init__(
        self,
        residual_model,
        scale,
        patch_size,
        overlap,
        full_threshold,
        subnet_threshold,
        device,
        subnet_channels=16,
    ):
        self.residual_model = residual_model
        self.scale = scale
        self.patch_size = patch_size
        self.overlap = overlap
        self.full_threshold = float(full_threshold)
        self.subnet_threshold = float(subnet_threshold)
        self.subnet_channels = int(subnet_channels)
        self.device = device
        if overlap < 0 or overlap % 2:
            raise ValueError("中心拼接要求 arm_overlap 为非负偶数")
        self.halo = overlap // 2
        self.stride = patch_size - overlap
        if self.stride <= 0:
            raise ValueError("arm_overlap 必须小于 arm_patch_size")
        if not self.subnet_threshold < self.full_threshold:
            raise ValueError("必须满足 arm_subnet_threshold < arm_threshold")

    def _starts(self, length):
        return range(0, length, self.stride)

    def _patch_smooth_score(self, laplace_scores, top, bottom, left, right):
        return laplace_scores[:, :, top:bottom, left:right].mean().item()

    def _prediction(self, lr_patch, channels):
        return self.residual_model(lr_patch / 255.0, channels=channels) * 255.0

    def infer(self, lr_img):
        batch_size, channels, lr_h, lr_w = lr_img.shape
        if batch_size != 1 or channels != 3:
            raise ValueError(
                f"ARMSR expects a single RGB image, got shape {tuple(lr_img.shape)}"
            )

        sr_img = torch.zeros_like(
            F.interpolate(lr_img, scale_factor=self.scale, mode="nearest")
        )
        total_patches = enhanced_patches = 0
        score_sum = 0.0
        branch_usage = {"bilinear": 0, "subnet": 0, "full": 0}

        with torch.no_grad():
            laplace_scores = laplacian_map(lr_img)
            padded = F.pad(lr_img, (self.halo,) * 4, mode="replicate")
            for top in self._starts(lr_h):
                for left in self._starts(lr_w):
                    # Coordinates describe non-overlapping output cores.
                    bottom = min(top + self.stride, lr_h)
                    right = min(left + self.stride, lr_w)
                    actual_h, actual_w = bottom - top, right - left
                    # Shift the last input tile back, but only write its unused tail.
                    patch_top = min(top, max(0, lr_h - self.stride))
                    patch_left = min(left, max(0, lr_w - self.stride))
                    lr_patch = padded[
                        :, :, patch_top:patch_top + self.patch_size,
                        patch_left:patch_left + self.patch_size
                    ]
                    # Only images smaller than one core need extra local padding.
                    pad_h = self.patch_size - lr_patch.shape[-2]
                    pad_w = self.patch_size - lr_patch.shape[-1]
                    if pad_h or pad_w:
                        lr_patch = F.pad(lr_patch, (0, pad_w, 0, pad_h), mode="replicate")

                    score = self._patch_smooth_score(
                        laplace_scores, top, bottom, left, right
                    )
                    score_sum += score
                    total_patches += 1
                    base = (
                        F.interpolate(
                            lr_patch,
                            scale_factor=self.scale,
                            mode="bilinear",
                            align_corners=False,
                        )
                        .round()
                        .clamp(0, 255)
                    )
                    if score >= self.full_threshold:
                        branch_usage["full"] += 1
                        prediction = self._prediction(
                            lr_patch, channels=self.residual_model.fea_dim
                        )
                    elif score >= self.subnet_threshold:
                        branch_usage["subnet"] += 1
                        prediction = self._prediction(
                            lr_patch, channels=self.subnet_channels
                        )
                    else:
                        branch_usage["bilinear"] += 1
                        prediction = None

                    if prediction is None:
                        sr_patch = base
                    else:
                        sr_patch = prediction.round().clamp(0, 255)
                        enhanced_patches += 1
                    hr_top, hr_left = top * self.scale, left * self.scale
                    hr_h_valid, hr_w_valid = (
                        actual_h * self.scale,
                        actual_w * self.scale,
                    )
                    crop_top = (self.halo + top - patch_top) * self.scale
                    crop_left = (self.halo + left - patch_left) * self.scale
                    sr_valid = sr_patch[
                        :, :, crop_top:crop_top + hr_h_valid,
                        crop_left:crop_left + hr_w_valid
                    ]
                    sr_img[
                        :, :, hr_top:hr_top + hr_h_valid, hr_left:hr_left + hr_w_valid
                    ] = sr_valid

        branch_ratios = {
            branch: count / max(1, total_patches)
            for branch, count in branch_usage.items()
        }
        return sr_img, {
            "total_patches": total_patches,
            "enhanced_patches": enhanced_patches,
            "enhanced_ratio": enhanced_patches / max(1, total_patches),
            "avg_laplace_score": score_sum / max(1, total_patches),
            "branch_usage": branch_usage,
            "branch_ratios": branch_ratios,
        }


def build_residual_model(args, device):
    checkpoint = torch.load(args.dpsr_checkpoint, map_location=device, weights_only=False)
    model_kwargs = dict(
        scale=args.scale,
        in_dim=3,
        fea_dim=args.channel_nums,
        num_blocks=args.num_blocks,
        bias=False,
        subnet_channels=args.subnet_channels,
        use_bilinear=checkpoint.get("model_config", {}).get("is_residual", True),
    )
    model_class = DPSR
    if args.model_name == "QDPSR":
        model_class = QDPSR
        config = checkpoint.get("model_config", {})
        model_kwargs.update(
            weight_bitwidth=args.wbits if args.wbits is not None else config.get("wbits", 4),
            activation_bitwidth=args.abits if args.abits is not None else config.get("abits", 4),
        )
    model = model_class(**model_kwargs).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()
    return model


def validate_arm_metrics(frame, val_loader, scale, clip_ratio=1.0):
    metrics_list = []
    total_patches = 0
    enhanced_patches = 0
    score_sum = 0.0
    branch_usage = {}

    with torch.no_grad():
        for image_index, (lr_img, hr_img) in enumerate(
            tqdm(val_loader, desc="Test4k", unit="image")
        ):
            lr_img, hr_img = (
                lr_img.to(frame.device).float(),
                hr_img.to(frame.device).float(),
            )
            sr_img, stat = frame.infer(lr_img)
            crop_border = scale
            sr_img = sr_img[:, :, crop_border:-crop_border, crop_border:-crop_border]
            hr_img = hr_img[:, :, crop_border:-crop_border, crop_border:-crop_border]
            metrics_list.append(
                (
                    metrics.calculate_psnr(sr_img.squeeze(0), hr_img.squeeze(0)),
                    metrics.calculate_ssim(sr_img.squeeze(0), hr_img.squeeze(0)),
                )
            )
            image_name = val_loader.dataset.names[image_index]
            tqdm.write(
                f"[{image_index + 1}/{len(val_loader)}] {image_name}: "
                f"PSNR: {metrics_list[-1][0]:.2f} dB"
            )
            total_patches += stat["total_patches"]
            enhanced_patches += stat["enhanced_patches"]
            score_sum += stat["avg_laplace_score"] * stat["total_patches"]
            for branch, count in stat["branch_usage"].items():
                branch_usage[branch] = branch_usage.get(branch, 0) + count

    if clip_ratio < 1.0:
        metrics_list.sort(key=lambda item: item[0], reverse=True)
        metrics_list = metrics_list[: max(1, int(len(metrics_list) * clip_ratio))]
    result = {
        "psnr": sum(item[0] for item in metrics_list) / len(metrics_list),
        "ssim": sum(item[1] for item in metrics_list) / len(metrics_list),
    }
    branch_ratios = {
        branch: count / max(1, total_patches) for branch, count in branch_usage.items()
    }
    return result, {
        "enhanced_ratio": enhanced_patches / max(1, total_patches),
        "avg_laplace_score": score_sum / max(1, total_patches),
        "enhanced_patches": enhanced_patches,
        "total_patches": total_patches,
        "branch_usage": branch_usage,
        "branch_ratios": branch_ratios,
    }


def main():
    args = final_test_parser()
    if args.threshold_sweep:
        from threshold_sweep import run_threshold_sweeps

        run_threshold_sweeps(args)
        return
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    residual_model = build_residual_model(args, device)
    frame = DPSRFrame(
        residual_model=residual_model,
        scale=args.scale,
        patch_size=args.arm_patch_size,
        overlap=args.arm_overlap,
        full_threshold=args.arm_threshold,
        subnet_threshold=args.arm_subnet_threshold,
        device=device,
        subnet_channels=args.subnet_channels,
    )
    print(
        f"Routing: bilinear < {args.arm_subnet_threshold:g} <= "
        f"{channel_label(args.subnet_channels)} < {args.arm_threshold:g} <= "
        f"{channel_label(args.channel_nums)}"
    )
    val_loaders = {
        dataset_name: create_val_loader(
            os.path.join(args.val_root, dataset_name), args.scale, in_channels=3
        )
        for dataset_name in ("Test4k",)
    }
    for dataset_name, loader in val_loaders.items():
        dpsr_result, stat = validate_arm_metrics(
            frame, loader, args.scale, args.clip_ratio
        )
        print(f"{dataset_name}:")
        print(
            f"  Dynamic {args.model_name}: PSNR: {dpsr_result['psnr']:.2f}, "
            f"SSIM: {dpsr_result['ssim']:.4f}, "
            f"NNPatch: {stat['enhanced_ratio'] * 100:.2f}% "
            f"({stat['enhanced_patches']}/{stat['total_patches']}), "
            f"LapMean: {stat['avg_laplace_score']:.4f}"
        )
        usage_text = ", ".join(
            f"{branch}:{count}"
            for branch, count in sorted(stat["branch_usage"].items())
        )
        ratio_text = ", ".join(
            f"{branch}:{ratio * 100:.2f}%"
            for branch, ratio in sorted(stat["branch_ratios"].items())
        )
        print(f"  BranchUsage: {usage_text}")
        print(f"  BranchRatio: {ratio_text}")


if __name__ == "__main__":
    main()
