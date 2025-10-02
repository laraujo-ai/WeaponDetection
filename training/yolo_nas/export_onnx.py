import argparse
import sys
from pathlib import Path
from typing import Tuple, Sequence

import torch
from super_gradients.training import models
from super_gradients.common.object_names import Models
from super_gradients.conversion.onnx.export_to_onnx import export_to_onnx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export SuperGradients model to ONNX.")
    p.add_argument("--checkpoint", "-c", required=True, type=str, help="Path to checkpoint (.pth).")
    p.add_argument("--num-classes", "-n", required=True, type=int, help="Number of classes the model predicts.")
    p.add_argument(
        "--model-arch",
        "-m",
        required=True,
        type=str,
        help="Model architecture. Either a Models enum name (e.g. YOLO_NAS_S) or a string accepted by models.get().",
    )
    p.add_argument("--output", "-o", required=True, type=str, help="Output ONNX filename.")
    p.add_argument(
        "--input-shape",
        nargs=4,
        type=int,
        default=(1, 3, 640, 640),
        help="Input tensor shape: N C H W (default: 1 3 640 640).",
    )
    p.add_argument("--opset", type=int, default=12, help="ONNX opset version (default: 12).")
    p.add_argument("--fp16", action="store_true", help="Export model and dummy input in FP16 (requires CUDA).")
    p.add_argument("--device", type=str, default="cuda", help="Device to use for exporting (default: cuda).")
    return p.parse_args()


def resolve_model_arch(name: str):
    """
    Try to resolve a Models enum attribute from name. If not found, return the raw name.
    This lets users pass either 'YOLO_NAS_S' (enum) or a custom string the registry accepts.
    """
    try:
        return getattr(Models, name)
    except Exception:
        return name


def make_dummy_input(shape: Sequence[int], fp16: bool, device: torch.device) -> torch.Tensor:
    dtype = torch.float16 if fp16 else torch.float32
    return torch.randn(tuple(shape), dtype=dtype, device=device)


def validate_args(args: argparse.Namespace) -> None:
    if args.fp16:
        if args.device.lower() != "cuda" and not torch.cuda.is_available():
            raise SystemExit("FP16 export requires CUDA (use --device cuda).")
        if not torch.cuda.is_available():
            raise SystemExit("FP16 export requires CUDA available on the machine.")


def main() -> int:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output)
    input_shape: Tuple[int, int, int, int] = tuple(args.input_shape)

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2

    validate_args(args)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.lower() == "cuda" else "cpu")
    arch = resolve_model_arch(args.model_arch)
    model = models.get(arch, num_classes=args.num_classes, checkpoint_path=str(ckpt_path))
    model.eval()

    if args.fp16:
        if device.type != "cuda":
            raise SystemExit("FP16 export requires CUDA device.")
        model.to(device)
        model.half()
        dummy = make_dummy_input(input_shape, fp16=True, device=device)
    else:
        model.to(device)
        dummy = make_dummy_input(input_shape, fp16=False, device=device)

    try:
        export_to_onnx(
            model=model,
            model_input=dummy,
            onnx_filename=str(out_path),
            input_names=["images"],
            output_names=["output1", "output2"],
            onnx_opset=args.opset,
            do_constant_folding=True,
        )
    except Exception as e:
        print("ONNX export failed:", e, file=sys.stderr)
        return 3

    print("ONNX export completed:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
