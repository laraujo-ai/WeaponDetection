import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser("Prepare checkpoint for ONNX / deployment")
    parser.add_argument(
        "--model_ckpt",
        type=str,
        required=True,
        help="Path to the checkpoint to prepare.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Convert floating-point tensors to float16 in the saved state_dict.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="cleaned_up_detector.pth",
        help="Output filename (placed next to the input checkpoint if not an absolute path).",
    )
    parser.add_argument(
        "--strip_module",
        action="store_true",
        help="Strip leading 'module.' prefixes from state_dict keys (useful after DataParallel training).",
    )
    return parser.parse_args()


def try_load_ckpt(path: Path) -> Any:
    """Load a checkpoint to CPU.

    Args:
        path: Path to the checkpoint file.

    Returns:
        The unpickled checkpoint object (usually a dict).

    Raises:
        Exception: Any exception raised by ``torch.load`` is propagated.
    """
    return torch.load(path.as_posix(), map_location="cpu")


def _is_tensor_like(x: Any) -> bool:
    """Return True if ``x`` is a torch tensor or a numpy array.

    Args:
        x: Object to test.

    Returns:
        True if the object is torch tensor-like, False otherwise.
    """
    return torch.is_tensor(x) or isinstance(x, np.ndarray)


def extract_state_dict(ckpt: Any) -> Optional[Dict[str, Any]]:
    """Extract a state_dict-like mapping from a checkpoint.

    The function looks for common checkpoint shapes in this order:
    1. Common keys: 'state_dict', 'model', 'model_state_dict', 'state', 'weights'.
    2. Top-level dictionary whose values are tensors/arrays.
    3. Nested dictionary where one sub-dictionary contains tensor/array values.

    Args:
        ckpt: Loaded checkpoint (often a dict containing model and metadata).

    Returns:
        A mapping from parameter name to tensor/array if found, otherwise None.
    """
    if not isinstance(ckpt, dict) or not ckpt:
        return None

    common_keys = ("state_dict", "model", "model_state_dict", "state", "weights")

    def candidate_iter():
        for k in common_keys:
            yield ckpt.get(k)
        yield ckpt
        for v in ckpt.values():
            yield v

    for cand in candidate_iter():
        if isinstance(cand, dict) and cand and all(_is_tensor_like(v) for v in cand.values()):
            return cand

    return None



def strip_module_prefix(state_dict: Dict[str, Any], prefix: str = "module.") -> Dict[str, Any]:
    """Remove a common prefix from state_dict keys if present.

    Args:
        state_dict: State dict mapping (name -> tensor).
        prefix: Prefix to remove from any key that starts with it.

    Returns:
        A new dict with the prefix stripped from keys that contained it. If the
        input state_dict is falsy, it is returned unchanged.
    """
    if not state_dict:
        return state_dict
    if any(k.startswith(prefix) for k in state_dict):
        return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}
    return state_dict


def maybe_convert_to_fp16(state_dict: Dict[str, Any], convert: bool) -> Dict[str, Any]:
    """Convert floating tensors/arrays to float16 when requested.

    Args:
        state_dict: Mapping of parameter names to tensors/arrays/objects.
        convert: If True, floating point tensors and numpy arrays are converted
            to float16. Non-floating values are left unchanged.

    Returns:
        A new mapping with converted tensors/arrays when ``convert`` is True,
        otherwise the original mapping (possibly the same object) is returned.
    """
    if not convert or state_dict is None:
        return state_dict

    out: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            out[k] = v.half() if torch.is_floating_point(v) else v
        elif isinstance(v, np.ndarray):
            out[k] = v.astype(np.float16) if np.issubdtype(v.dtype, np.floating) else v
        else:
            out[k] = v
    return out


def main(args: argparse.Namespace) -> int:
    """Main entry point.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code: 0 on success, non-zero on error.
    """
    ckpt_path = Path(args.model_ckpt)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        return 2

    out_path = Path(args.output_name)
    if not out_path.is_absolute():
        out_path = ckpt_path.with_name(out_path.name)

    try:
        ckpt = try_load_ckpt(ckpt_path)
    except Exception as exc:
        print(f"Failed to load checkpoint: {exc}")
        return 3

    state_dict = extract_state_dict(ckpt)
    if state_dict is None:
        debug_out = ckpt_path.with_name(ckpt_path.stem + "_original_dump.pth")
        torch.save(ckpt, debug_out.as_posix())
        print("No state_dict found. Saved original checkpoint for inspection at:", debug_out)
        return 0

    if args.strip_module:
        state_dict = strip_module_prefix(state_dict, "module.")

    state_dict = maybe_convert_to_fp16(state_dict, args.fp16)
    torch.save(state_dict, out_path.as_posix())
    print(f"Saved extracted state_dict to {out_path} (keys: {len(state_dict)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
