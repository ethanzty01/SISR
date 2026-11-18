import os
import sys
import torch
import torch.optim as optim
import copy
from datetime import datetime
from pathlib import Path
from torch_ema import ExponentialMovingAverage

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import QConv2dLSQP, QConvTranspose2dLSQP, build_qdpsr, build_qfsrcnn
from utils.trainer import validate_loss_and_metrics
from utils import train_parser, train_epoch, validate_metrics, bilinear_metrics, \
create_logger, create_train_loader, create_val_loader, WarmupCosineScheduler, MixedLoss

def _build_q_model(args, device):
    if args.model_name == 'QFSRCNN':
        return build_qfsrcnn(
            scale=args.scale,
            in_dim=args.in_channels,
            weight_bitwidth=args.wbits,
            activation_bitwidth=args.abits,
        ).to(device)
    return build_qdpsr(
        scale=args.scale,
        in_dim=args.in_channels,
        fea_dim=args.channel_nums,
        num_blocks=args.num_blocks,
        bias=False,
        weight_bitwidth=args.wbits,
        activation_bitwidth=args.abits,
        subnet_channels=args.subnet_channels,
        use_bilinear=args.is_residual,
    ).to(device)

def _load_fp_checkpoint_into_q_model(model, checkpoint_path, device, logger):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    fp_state = checkpoint.get('model_state_dict', checkpoint)
    q_state = model.state_dict()
    mapped_state = {}
    block_conv_names = {'filter1', 'filter2', 'projection1', 'projection2'}

    for key, value in fp_state.items():
        q_key = key

        if key.startswith('first_part.0.'):
            q_key = 'first_part.0.conv.' + key[len('first_part.0.'):]
        elif key.startswith('mid_part.') and key.split('.')[-1] in {'weight', 'bias'}:
            parts = key.split('.')
            if len(parts) >= 3 and parts[1].isdigit() and int(parts[1]) % 2 == 0:
                q_key = f"mid_part.{parts[1]}.conv.{parts[2]}"
        elif key.startswith('last_part.'):
            q_key = 'last_part.conv.' + key[len('last_part.'):]
        elif key.startswith('head.'):
            q_key = 'head.conv.' + key[len('head.'):]
        elif key.startswith('tail.'):
            q_key = 'tail.conv.' + key[len('tail.'):]
        else:
            parts = key.split('.')
            if (
                len(parts) >= 4
                and parts[0] == 'body'
                and parts[2] in block_conv_names
                and parts[3] in {'weight', 'bias'}
            ):
                q_key = f'body.{parts[1]}.{parts[2]}.conv.{parts[3]}'

        if q_key in q_state and q_state[q_key].shape == value.shape:
            mapped_state[q_key] = value
        else:
            logger.info(f'Skip pretrained key: {key} -> {q_key}')

    q_state.update(mapped_state)
    model.load_state_dict(q_state)
    if checkpoint.get('model_config', {}).get('activation_groups'):
        logger.info('Ignoring FP activation groups; activation quantization is per-channel')
    for module in model.modules():
        if hasattr(module, 'initialized'):
            module.initialized = False
    logger.info(f'Loaded {len(mapped_state)} tensors from full-precision checkpoint: {checkpoint_path}')

def _model_configurations(args):
    if args.model_name == 'QFSRCNN':
        return [('56ch', 56)]
    configurations = [(f'{args.channel_nums}ch', args.channel_nums)]
    if args.joint_width_training:
        configurations.append((f'{args.subnet_channels}ch', args.subnet_channels))
    return configurations


def _calibrate_lsqplus_activations(model, train_loader, args, device, logger):
    """Initialize each activation quantizer from full-precision feature maps."""
    if args.quant_calibration_batches <= 0:
        return

    quantized_layers = {
        name: module for name, module in model.named_modules()
        if isinstance(module, (QConv2dLSQP, QConvTranspose2dLSQP)) and not module.act_quant.disabled and module.quantize_input
    }
    samples = {name: [] for name in quantized_layers}
    sample_counts = {name: 0 for name in quantized_layers}

    def make_hook(name):
        def hook(_module, inputs):
            feature_map = inputs[0].detach()
            channels = feature_map.shape[1]
            values = feature_map.movedim(1, 0).reshape(channels, -1)
            per_channel_limit = max(1, args.quant_calibration_samples // channels)
            remaining = per_channel_limit - sample_counts[name]
            if remaining <= 0 or values.shape[1] == 0:
                return
            take = min(remaining, max(1, 8192 // channels), values.shape[1])
            indices = torch.linspace(0, values.shape[1] - 1, take, device=values.device).long()
            samples[name].append(values[:, indices].cpu())
            sample_counts[name] += take
        return hook

    handles = [module.register_forward_pre_hook(make_hook(name)) for name, module in quantized_layers.items()]
    was_training = model.training
    model.set_quantization_enabled(False)
    model.eval()
    try:
        with torch.no_grad():
            for batch_index, (lr_img, _) in enumerate(train_loader):
                if batch_index >= args.quant_calibration_batches:
                    break
                model(lr_img.to(device, non_blocking=True).float() / 255.0, channels=args.channel_nums)
    finally:
        for handle in handles:
            handle.remove()
        model.set_quantization_enabled(True)
        model.train(was_training)

    for name, module in quantized_layers.items():
        if not samples[name]:
            raise RuntimeError(f'No calibration samples collected for {name}')
        module.act_quant.initialize_from_samples(torch.cat(samples[name], dim=1))
    logger.info(
        f'Initialized {len(quantized_layers)} LSQ+ activation quantizers with MSE calibration '
        f'from {args.quant_calibration_batches} full-precision batches.'
    )


def _optimizer_parameter_groups(model, args):
    quant_parameters = []
    quant_parameter_ids = set()
    for module in model.modules():
        if isinstance(module, (QConv2dLSQP, QConvTranspose2dLSQP)):
            for parameter in (*module.act_quant.parameters(), *module.weight_quant.parameters()):
                if id(parameter) not in quant_parameter_ids:
                    quant_parameters.append(parameter)
                    quant_parameter_ids.add(id(parameter))
    network_parameters = [
        parameter for parameter in model.parameters()
        if id(parameter) not in quant_parameter_ids
    ]
    return [
        {'params': network_parameters, 'lr': args.lr, 'name': 'network'},
        {'params': quant_parameters, 'lr': args.quant_lr, 'name': 'quantizer'},
    ]


def _validate_configurations(model, val_loaders, loss_func, args, device, logger):
    losses = {}
    for label, channels in _model_configurations(args):
        weighted_loss, total_samples = 0.0, 0
        for dataset_name, loader in val_loaders.items():
            loader_loss, val_metrics = validate_loss_and_metrics(
                model, loader, loss_func, device,
                channels=channels,
            )
            count = len(loader.dataset)
            weighted_loss += loader_loss * count
            total_samples += count
            logger.log_validation_results(f'{dataset_name}/{label}', val_metrics)
        losses[label] = weighted_loss / total_samples
    full_loss = next(iter(losses.values()))
    if args.joint_width_training:
        subnet_loss = losses[f'{args.subnet_channels}ch']
        full_loss = (full_loss + args.subnet_loss_weight * subnet_loss) / (1.0 + args.subnet_loss_weight)
    return full_loss, losses


def _should_validate(epoch, epochs, interval):
    return (epoch + 1) % interval == 0 or epoch == epochs - 1


def _log_model_metrics(logger, model, val_loaders, args, device):
    for label, channels in _model_configurations(args):
        for dataset_name, loader in val_loaders.items():
            val_metrics = validate_metrics(
                model, loader, args.scale, device, 1.0, channels=channels
            )
            logger.log_validation_results(f'{dataset_name}/{label}', val_metrics)


def main():

    args = train_parser()
    if not args.model_name:
        args.model_name = "QDPSR"
    if args.model_name == "FSRCNN":
        args.model_name = "QFSRCNN"
    if args.model_name == "DPSR":
        args.model_name = "QDPSR"
    if args.no_bilinear and args.model_name != "QDPSR":
        raise ValueError('--no-bilinear requires QDPSR')
    args.is_residual = args.model_name == "QDPSR" and not args.no_bilinear
    if args.joint_width_training and args.model_name != "QDPSR":
        raise ValueError('--joint_width_training is only supported by QDPSR')
    if args.subnet_loss_weight < 0.0 or args.distill_loss_weight < 0.0:
        raise ValueError('subnet and distillation loss weights must be non-negative')
    if args.quant_calibration_batches < 0 or args.quant_calibration_samples < 1:
        raise ValueError('quant calibration batches must be non-negative and samples must be positive')
    
    if args.val_interval < 1:
        raise ValueError('--val-interval must be positive')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 创建logger
    logger = create_logger(log_dir="./logs", model_name=args.model_name, scale=args.scale)
    logger.info(f"使用设备: {device}")
    if args.joint_width_training:
        logger.info(f'QAT model: {args.model_name}; jointly train {args.channel_nums}ch and '
                    f'{args.subnet_channels}ch in every batch; validate both widths')
    else:
        logger.info(f'QAT model: {args.model_name}; width: {_model_configurations(args)[0][0]} only')

    datasets_root = Path(args.datasets_root)
    train_loader = create_train_loader(
        datasets_root / 'train',
        scale=args.scale,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        in_channels=args.in_channels
    )
                                   
    val_loaders = {
        name: create_val_loader(
            datasets_root / 'val' / name,
            args.scale,
            in_channels=args.in_channels,
        )
        for name in ('Set5', 'Set14', 'B100', 'U100', 'M109')
    }
    
    time_stamp = datetime.now().strftime("%m%d_%H%M")
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{args.model_name}_x{args.scale}_{time_stamp}.pth"
    model = _build_q_model(args, device)

    if args.pretrained_fp:
        _load_fp_checkpoint_into_q_model(model, args.pretrained_fp, device, logger)

    _calibrate_lsqplus_activations(model, train_loader, args, device, logger)

    # Initialize before EMA snapshots the scales, and outside compiled graphs.
    for module in model.modules():
        if isinstance(module, (QConv2dLSQP, QConvTranspose2dLSQP)):
            module.weight_quant._init_from_tensor(module.conv.weight)

    # 统计模型参数量
    total_params = model.param_num()
    logger.info(f"模型总参数量: {total_params:,}")

    # 损失函数
    loss_func = MixedLoss(eps=1e-8, gamma=0.25)

    # 优化器
    optimizer = optim.Adam(_optimizer_parameter_groups(model, args), betas=(0.9, 0.999))
    logger.info(f'Optimizer learning rates: network={args.lr:.3e}, quantizer={args.quant_lr:.3e}')
    
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
    
    # 学习率调度器：warmup + cosine annealing
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        eta_min=[args.minlr, args.quant_min_lr],
        warmup_start_lr=[args.warmup_start_lr, args.quant_initial_lr]
    )
   
    train_model = model
    if args.compile:
        if not hasattr(torch, 'compile'):
            raise RuntimeError('--compile requires a torch build with torch.compile')
        train_model = torch.compile(model)
        logger.info('torch.compile enabled for training; initial compilation may be slow')
    logger.info(f'Validate every {args.val_interval} epochs and on the last epoch')

    # 记录训练开始信息
    logger.log_training_start(args, total_params, len(train_loader))

    # 训练循环
    best_val_loss = float('inf')

    logger.info("Begin Training")
    for epoch in range(args.epochs):
        # 训练
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

        current_lr = optimizer.param_groups[0]['lr']
        
        logger.log_epoch_train(epoch, args.epochs, train_loss, current_lr)
        logger.info(f'Quantizer learning rate: {optimizer.param_groups[1]["lr"]:.6e}')

        best_candidate = None
        if _should_validate(epoch, args.epochs, args.val_interval):
            with ema.average_parameters():
                val_loss, validation_losses = _validate_configurations(
                    model, val_loaders, loss_func, args, device, logger
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_candidate = copy.deepcopy(model)
            logger.log_epoch_val(epoch, args.epochs, val_loss)
            logger.info(f'Validation losses by width: {validation_losses}')

        if best_candidate is not None:
            torch.save({
                'epoch': epoch + 1,
                'iteration': (epoch + 1) * len(train_loader),
                'model_state_dict': best_candidate.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'validation_losses': validation_losses,
                'model_config': {
                    'quantizer': 'lsqplus',
                    'quantizer_version': 'per_channel_projected_step_v3',
                    'quant_lr': args.quant_lr,
                    'model_name': args.model_name,
                    'is_residual': args.is_residual,
                    'input_transform': 'rgb',
                    'input_quantization': None,
                    'width_training': 'joint' if args.joint_width_training else 'full_only',
                    'scale': args.scale,
                    'in_channels': args.in_channels,
                    'num_blocks': args.num_blocks,
                    'full_channels': args.channel_nums,
                    'subnet_channels': args.subnet_channels,
                    'wbits': args.wbits,
                    'abits': args.abits,
                },
            }, model_path)
            
            logger.log_best_model(val_loss)

        scheduler.step()
    
    logger.log_training_finished()

    logger.log_testing_start("Best Model")
    net = _build_q_model(args, device)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    net.load_state_dict(state_dict['model_state_dict'])
    net.eval()
    _log_model_metrics(logger, net, val_loaders, args, device)

    logger.log_testing_start("Bilinear Interpolation")
    for dataset_name, loader in val_loaders.items():
        val_metrics = bilinear_metrics(loader, args.scale, device)
        logger.log_validation_results(dataset_name, val_metrics)

    # 关闭logger
    logger.close()

if __name__ == "__main__":
    main()
