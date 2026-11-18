import copy
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch_ema import ExponentialMovingAverage

from utils.trainer import validate_loss_and_metrics
from models import FSRCNN, DPSR, channel_label
from utils import (
    MixedLoss,
    WarmupCosineScheduler,
    bilinear_metrics,
    create_logger,
    create_train_loader,
    create_val_loader,
    train_epoch,
    train_parser,
    validate_metrics,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def _build_model(args, device):
    if args.is_residual or args.model_name == "DPSR" or args.no_bilinear:
        model = DPSR(
            scale=args.scale,
            in_dim=args.in_channels,
            fea_dim=args.channel_nums,
            num_blocks=args.num_blocks,
            bias=False,
            subnet_channels=args.subnet_channels,
            use_bilinear=args.is_residual,
        ).to(device)
        print(f"Using DPSR model for scale {args.scale}.")
        model_name = "DPSR"
    else:
        model = FSRCNN(
            scale_factor=args.scale,
            num_channels=args.in_channels
        ).to(device)
        print(f"Using FSRCNN model for scale {args.scale}.")
        model_name = "FSRCNN"
    return model, model_name

def _load_pretrained_model(model, checkpoint_path, device, logger):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(model_state, strict=True)
    logger.info(f"Loaded full-width checkpoint: {checkpoint_path}")


def _model_configurations(args):
    """Return the full-channel path and optional explicit subnet path."""
    configurations = [(channel_label(args.channel_nums), args.channel_nums)]
    if args.joint_width_training:
        configurations.append(
            (channel_label(args.subnet_channels), args.subnet_channels)
        )
    return configurations


def _validate_configurations(model, val_loaders, loss_func, args, device, logger):
    configuration_losses = {}
    for label, channels in _model_configurations(args):
        weighted_loss, total_samples = 0.0, 0
        for dataset_name, loader in val_loaders.items():
            loss, result = validate_loss_and_metrics(
                model, loader, loss_func, device,
                channels=channels,
            )
            count = len(loader.dataset)
            weighted_loss += loss * count
            total_samples += count
            logger.log_validation_results(f'{dataset_name}/{label}', result)
        configuration_losses[label] = weighted_loss / total_samples
    combined = configuration_losses[channel_label(args.channel_nums)]
    if args.joint_width_training:
        combined = (combined + args.subnet_loss_weight * configuration_losses[channel_label(args.subnet_channels)]) / (1.0 + args.subnet_loss_weight)
    return {"combined": combined, "configurations": configuration_losses}


def _log_model_metrics(logger, model, val_loaders, args, device):
    for config_name, channels in _model_configurations(args):
        for dataset_name, loader in val_loaders.items():
            val_metrics = validate_metrics(
                model,
                loader,
                args.scale,
                device,
                1.0,
                channels=channels,
            )
            logger.log_validation_results(f"{dataset_name}/{config_name}", val_metrics)


def main():
    args = train_parser()
    if args.no_bilinear:
        if args.model_name not in ("", "DPSR"):
            raise ValueError('--no-bilinear requires DPSR')
        args.model_name = "DPSR"
        args.is_residual = False
    if args.model_name == "DPSR":
        args.is_residual = not args.no_bilinear
    if args.joint_width_training and not (args.is_residual or args.model_name == "DPSR"):
        raise ValueError("--joint_width_training is only supported by DPSR")
    if args.subnet_loss_weight < 0.0:
        raise ValueError("--subnet_loss_weight must be non-negative")
    if args.distill_loss_weight < 0.0:
        raise ValueError("--distill_loss_weight must be non-negative")

    if args.val_interval < 1:
        raise ValueError('--val-interval must be positive')

    print(f"Training configuration: {args}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, model_name = _build_model(args, device)

    logger = create_logger(log_dir="./logs", model_name=model_name, scale=args.scale)
    logger.info(f"使用设备: {device}")
    logger.info(
        f"Full path: {channel_label(args.channel_nums)}; subnet path: "
        f"{channel_label(args.subnet_channels)}; blocks: {args.num_blocks}"
    )

    if args.pretrained_fp:
        _load_pretrained_model(model, args.pretrained_fp, device, logger)

    datasets_root = Path(args.datasets_root)
    train_loader = create_train_loader(
        datasets_root / "train",
        scale=args.scale,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
    )
    val_loaders = {
        dataset_name: create_val_loader(
            datasets_root / "val" / dataset_name,
            args.scale,
            in_channels=args.in_channels,
        )
        for dataset_name in ("Set5", "Set14", "B100", "U100", "M109")
    }

    time_stamp = datetime.now().strftime("%m%d_%H%M")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{model_name}_x{args.scale}_{time_stamp}.pth"

    total_params = model.param_num()
    logger.info(f"模型总参数量: {total_params:,}")
    loss_func = MixedLoss(eps=1e-8, gamma=0.25)
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=args.lr)
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        eta_min=args.minlr,
        warmup_start_lr=args.warmup_start_lr,
    )

    train_model = model
    if args.compile:
        if not hasattr(torch, 'compile'):
            raise RuntimeError('--compile requires torch.compile support')
        train_model = torch.compile(model)
        logger.info('torch.compile enabled for training; initial compilation may be slow')
    logger.info(f'Validation interval: {args.val_interval}')

    logger.log_training_start(args, total_params, len(train_loader))
    best_val_loss = float("inf")
    logger.info("Begin Training")

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            train_model,
            train_loader,
            loss_func,
            optimizer,
            device,
            epoch,
            ema=ema,
            joint_width_training=args.joint_width_training,
            subnet_channels=args.subnet_channels,
            subnet_loss_weight=args.subnet_loss_weight,
            distill_loss_weight=args.distill_loss_weight,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        logger.log_epoch_train(epoch, args.epochs, train_loss, current_lr)

        best_candidate = None
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            with ema.average_parameters():
                val_losses = _validate_configurations(
                    model, val_loaders, loss_func, args, device, logger
                )
                val_loss = val_losses["combined"]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_candidate = copy.deepcopy(model)
            logger.log_epoch_val(epoch, args.epochs, val_loss)
            logger.info(f'Validation configuration losses: {val_losses["configurations"]}')

        if best_candidate is not None:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "iteration": (epoch + 1) * len(train_loader),
                    "model_state_dict": best_candidate.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "validation_losses": val_losses["configurations"],
                    "model_config": {
                        "model_name": model_name,
                        "input_transform": "rgb",
                        "scale": args.scale,
                        "in_channels": args.in_channels,
                        "is_residual": args.is_residual,
                        "num_blocks": args.num_blocks,
                        "full_channels": args.channel_nums,
                        "subnet_channels": args.subnet_channels,
                    },
                },
                model_path,
            )
            logger.log_best_model(val_loss)

        scheduler.step()

    logger.log_training_finished()
    logger.log_testing_start("Best Model")
    net, _ = _build_model(args, device)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    net.load_state_dict(state_dict["model_state_dict"])
    net.eval()
    _log_model_metrics(logger, net, val_loaders, args, device)

    logger.log_testing_start("Bilinear Interpolation")
    for dataset_name, loader in val_loaders.items():
        logger.log_validation_results(
            dataset_name, bilinear_metrics(loader, args.scale, device)
        )
    logger.close()


if __name__ == "__main__":
    main()
