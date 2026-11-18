"""Train x4 DPSR without its bilinear skip, QAT W8A8/W4A4, then Test4k.

Run with the SISR Python environment. Outputs stay under a unique run directory.
Test4k is used only after training, never for checkpoint selection.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def evaluate(checkpoints, output, device, datasets_root):
    import torch
    from models import DPSR, build_qdpsr
    from utils import create_val_loader, metrics

    loader = create_val_loader(datasets_root / 'test' / 'Test4k', 4, in_channels=3)
    rows = []
    for precision, path in checkpoints.items():
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint['model_config']
        if config.get('is_residual') is not False:
            raise ValueError(f'{path} does not declare direct HR prediction')
        kwargs = dict(scale=4, in_dim=3, fea_dim=32, num_blocks=5,
                      bias=False, subnet_channels=16, use_bilinear=False)
        if precision == 'FP32':
            model = DPSR(**kwargs)
        else:
            model = build_qdpsr(**kwargs, weight_bitwidth=config['wbits'],
                               activation_bitwidth=config['abits'])
        model.to(device).eval()
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        with torch.inference_mode():
            for channels in (32, 16):
                for index, (lr, hr) in enumerate(loader):
                    prediction = (model(lr.to(device).float() / 255,
                                        channels=channels) * 255).round().clamp(0, 255)
                    prediction = prediction[0, :, 4:-4, 4:-4]
                    target = hr.to(device).float()[0, :, 4:-4, 4:-4]
                    psnr = float(metrics.calculate_psnr(prediction, target))
                    row = dict(precision=precision, channels=channels,
                               image=loader.dataset.names[index], psnr_db=psnr)
                    rows.append(row)
                    print(row, flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    with (output / 'test4k_per_image.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for precision in checkpoints:
        for channels in (32, 16):
            values = [row['psnr_db'] for row in rows
                      if row['precision'] == precision and row['channels'] == channels]
            summary.append(dict(precision=precision, channels=channels, images=len(values),
                                psnr_db=sum(values) / len(values)))
    (output / 'test4k_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--datasets-root', type=Path, default=Path('/home/tyzheng/Datasets_pt'))
    parser.add_argument('--fp-epochs', type=int, default=200)
    parser.add_argument('--qat-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: full training was not started. Run on a CUDA host.')
    output = (args.run_dir or ROOT / 'experiments' / ('no_bilinear_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if args.evaluate_only:
        manifest = json.loads(manifest_path.read_text())
        evaluate(manifest['checkpoints'], output, torch.device('cuda'), args.datasets_root)
        return
    if manifest_path.exists():
        raise ValueError('Run directory already contains an experiment; choose a new directory')
    common = ['--scale', '4', '--model_name', 'DPSR', '--no-bilinear',
              '--channel_nums', '32', '--subnet-channels', '16', '--num_blocks', '5',
              '--batch_size', str(args.batch_size), '--num_workers', str(args.workers),
              '--datasets-root', str(args.datasets_root), '--device', 'cuda',
              '--in_channels', '3', '--patch_size', '128', '--warmup_epochs', '10',
              '--ema_decay', '0.999', '--joint_width_training',
              '--subnet_loss_weight', '1.0', '--distill_loss_weight', '0.1',
              '--val-interval', '5']
    if args.compile:
        common.append('--compile')
    manifest = dict(seed=args.seed, common_args=common, checkpoints={},
                    metric='mean per-image Y PSNR, rounded/clamped RGB, 4-pixel border crop',
                    torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0), stages={})

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    def stage(name, script, extra):
        directory = output / name
        directory.mkdir()
        command_args = common + extra + ['--save_dir', str(directory)]
        # Set all RNG seeds in the training process before importing the entry point.
        bootstrap = ('import random, numpy as np, torch, runpy, sys; '
                     f'random.seed({args.seed}); np.random.seed({args.seed}); torch.manual_seed({args.seed}); '
                     f'torch.cuda.manual_seed_all({args.seed}); sys.path.insert(0, {str(ROOT)!r}); '
                     f'sys.argv={[script] + command_args!r}; '
                     f'runpy.run_path({str(ROOT / script)!r}, run_name="__main__")')
        manifest['stages'][name] = dict(script=script, args=command_args, status='running')
        save()
        with (directory / 'console.log').open('w') as log:
            result = subprocess.run([sys.executable, '-u', '-c', bootstrap], cwd=output,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'PYTHONHASHSEED': str(args.seed)})
        manifest['stages'][name]['status'] = 'complete' if result.returncode == 0 else 'failed'
        save()
        result.check_returncode()
        paths = list(directory.glob('*.pth'))
        if len(paths) != 1:
            raise RuntimeError(f'Expected exactly one best checkpoint in {directory}')
        manifest['checkpoints'][name] = str(paths[0])
        save()
        return paths[0]

    fp = stage('FP32', 'train.py', ['--epochs', str(args.fp_epochs), '--lr', '2.5e-3',
                                   '--warmup_start_lr', '1e-3', '--minlr', '5e-5'])
    for bits in (8, 4):
        stage(f'W{bits}A{bits}', 'qtrain.py', ['--epochs', str(args.qat_epochs),
              '--network-initial-lr', '3e-4', '--network-lr', '1e-3', '--network-min-lr', '5e-5',
              '--quant-initial-lr', '1e-5', '--quant-lr', '6e-4', '--quant-min-lr', '4e-5',
              '--wbits', str(bits), '--abits', str(bits), '--pretrained_fp', str(fp)])
    evaluate(manifest['checkpoints'], output, torch.device('cuda'), args.datasets_root)


if __name__ == '__main__':
    main()
