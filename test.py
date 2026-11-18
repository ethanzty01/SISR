from pathlib import Path

import torch

from utils import create_val_loader, test_parser, validate_metrics
from models import DPSR, channel_label


if __name__ == "__main__":
    args = test_parser()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    net = DPSR(
        scale=args.scale,
        in_dim=args.in_channels,
        fea_dim=args.channel_nums,
        num_blocks=args.num_blocks,
        bias=False,
        subnet_channels=args.subnet_channels,
        use_bilinear=checkpoint.get("model_config", {}).get("is_residual", True),
    ).to(device)
    model_state_dict = checkpoint.get("model_state_dict", checkpoint)
    net.load_state_dict(model_state_dict, strict=True)
    net.eval()
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    loader = create_val_loader(
        str(Path(args.val_root) / "Test4k"), args.scale, in_channels=args.in_channels
    )
    for name, channels in (("full", args.channel_nums), ("subnet", args.subnet_channels)):
        result = validate_metrics(
            net, loader, args.scale, device, 1.0,
            channels=channels
        )
        print(
            f"Test4k | {name} {channel_label(channels)}: "
            f"PSNR={result['psnr']:.4f} dB, SSIM={result['ssim']:.6f}"
        )
