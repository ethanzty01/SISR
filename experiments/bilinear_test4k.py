"""Evaluate x4 bilinear/bicubic Test4k using the existing Y-PSNR protocol."""
import argparse
import csv
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import create_val_loader, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('bilinear', 'bicubic'), default='bilinear')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = create_val_loader('/home/tyzheng/Datasets_pt/test/Test4k', 4, in_channels=3)
    output = ROOT / 'experiments' / f'{args.mode}_test4k_x4'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for index, (lr, hr) in enumerate(loader):
            sr = F.interpolate(lr.to(device).float(), scale_factor=4,
                               mode=args.mode, align_corners=False).round().clamp(0, 255)
            psnr = float(metrics.calculate_psnr(sr[0, :, 4:-4, 4:-4],
                                               hr.to(device).float()[0, :, 4:-4, 4:-4]))
            rows.append(dict(image=loader.dataset.names[index], psnr_db=psnr))
            print(f'{index+1}/{len(loader)} {rows[-1]}', flush=True)
    with (output / 'per_image.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image', 'psnr_db'])
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(method=args.mode, scale=4, dataset='Test4k', images=len(rows),
                   psnr_db=sum(r['psnr_db'] for r in rows) / len(rows),
                   align_corners=False, rgb_round_clamp=True, crop_border=4,
                   metric='mean per-image Y PSNR', device=str(device))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
