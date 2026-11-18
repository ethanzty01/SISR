import argparse


def final_test_parser():
    parser = argparse.ArgumentParser(description="Dynamic DPSR/QDPSR testing")
    parser.add_argument("--model_name", "--model-name", choices=["DPSR", "QDPSR"], default="DPSR", help="残差网络类型")
    parser.add_argument("--wbits", type=int, default=None, help="QDPSR权重量化位宽，默认读取checkpoint配置，无配置时为4")
    parser.add_argument("--abits", type=int, default=None, help="QDPSR激活量化位宽，默认读取checkpoint配置，无配置时为4")
    parser.add_argument(
        "--dpsr-checkpoint",
        "--dpsr_checkpoint",
        "--checkpoint",
        dest="dpsr_checkpoint",
        type=str,
        required=True,
        help="dynamic DPSR/QDPSR checkpoint path; --checkpoint is a compatibility alias",
    )
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4], help="超分倍数")
    parser.add_argument("--channel_nums", type=int, default=32, choices=[32], help="完整路径通道数")
    parser.add_argument("--subnet_channels", type=int, default=16, help="子网显式通道数")
    parser.add_argument("--num_blocks", type=int, default=5, help="DPSR网络的Block数")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    parser.add_argument("--arm_patch_size", type=int, default=16, help="含外围邻域的LR输入块边长；末块回退取满，只写入剩余区域")
    parser.add_argument("--arm_overlap", type=int, default=2, help="输入重叠像素，须为非负偶数；每边裁去 overlap/2 后直接拼接")
    parser.add_argument("--arm_threshold", type=float, default=12.5, help="进入完整通道NN的Laplace高阈值")
    parser.add_argument("--arm_subnet_threshold", type=float, default=2.5, help="从bilinear进入子网NN的Laplace阈值")
    parser.add_argument("--val_root", type=str, default="/home/tyzheng/Datasets_pt/test", help="测试集根目录")
    parser.add_argument("--clip_ratio", type=float, default=1, help="指标裁剪比例")
    parser.add_argument("--threshold-sweep", action="store_true", help="Run both Test4k threshold sweeps")
    parser.add_argument("--sweep-output-dir", type=str, default=None)
    parser.add_argument("--patch-batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=0, help="0 evaluates all images")
    return parser.parse_args()
