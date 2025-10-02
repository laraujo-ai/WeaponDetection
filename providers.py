from models.base import IObjectDetector
from models.yolox import YOLOXDetector
from models.yolo_nas import YOLONasDetector


providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_max_workspace_size': 2147483648,
        'trt_fp16_enable': True,
        'trt_builder_optimization_level': 0,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': './trt_cache',
    }),
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider'
]


def get_yolo_nas_detector(model_path : str) -> IObjectDetector:
    """Create and return a YOLO-NAS object detector instance.

    Args:
        model_path: Path to the YOLO-NAS ONNX model file.

    Returns:
        An initialized YOLONasDetector instance configured with execution providers.
    """
    return YOLONasDetector(model_path, providers)


def get_yolox_detector(model_path: str) -> IObjectDetector:
    """Create and return a YOLOX object detector instance.

    Args:
        model_path: Path to the YOLOX ONNX model file.

    Returns:
        An initialized YOLOXDetector instance configured with execution providers.
    """
    return YOLOXDetector(model_path, providers)
    


