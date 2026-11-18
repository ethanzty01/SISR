import argparse


def test_parser():
    parser = argparse.ArgumentParser(description="Test4k testing")
    parser.add_argument("--checkpoint", type=str, required=True, help="DPSR/QDPSR checkpoint")
    parser.add_argument(
        "--scale", type=int, default=2, choices=[2, 3, 4], help="super-resolution scale"
    )
    parser.add_argument(
        "--channel_nums",
        type=int,
        default=32,
        choices=[32],
        help="full-path feature channels",
    )
    parser.add_argument(
        "--subnet-channels",
        type=int,
        default=16,
        help="explicit feature channels used by the subnet path",
    )
    parser.add_argument("--num_blocks", type=int, default=5, help="number of blocks")
    parser.add_argument("--device", type=str, default="cuda", help="test device")
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        choices=[1, 3],
        help="input channels, 1 for Y and 3 for RGB",
    )
    parser.add_argument(
        "--val_root",
        type=str,
        default="/home/tyzheng/Datasets_pt/test",
        help="validation set root",
    )
    return parser.parse_args()
