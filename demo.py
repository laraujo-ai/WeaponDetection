import argparse
import os

import cv2

import providers as p
from tools.media_processor import Visualizer
from tools.utils import is_rtsp, is_video_file, _process_class_names


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - media_link: Path to image/video or RTSP URL
            - model_path: Path to the pre-trained model
            - model_family: Model family to use (yolox or yolo-nas)
            - class_names: List of class names for the model
            - nms_thr: NMS threshold for detections
            - score_thr: Confidence threshold for detections
            - show: Whether to show live detections with OpenCV
            - out_file: Output video file path
    """
    parser = argparse.ArgumentParser(description="VisionTech Object Detection Demo")
    parser.add_argument(
        "--media_link",
        type=str,
        required=True,
        help="Path to image/video or RTSP URL.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pre-trained model.",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        required=True,
        choices=["yolox", "yolo-nas"],
        help="Model family to use.",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        help="List of class names for the model (comma-separated if single string).",
    )
    parser.add_argument(
        "--nms_thr",
        type=float,
        default=0.45,
        help="NMS threshold for detections.",
    )
    parser.add_argument(
        "--score_thr",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live detections with OpenCV.",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="output.mp4",
        help="Output video file path (if not showing live).",
    )
    return parser.parse_args()



def _get_detector(model_family, model_path):
    """Initialize the object detector based on model family.

    Args:
        model_family: Model family name (yolox or yolo-nas).
        model_path: Path to the pre-trained model weights.

    Returns:
        Detector instance for the specified model family.

    Raises:
        ValueError: If the model family is unsupported.
    """
    if model_family == "yolo-nas":
        return p.get_yolo_nas_detector(model_path=model_path)
    elif model_family == "yolox":
        return p.get_yolox_detector(model_path=model_path)
    else:
        raise ValueError(f"Unsupported model family: {model_family}")


def _process_video_stream(cap, detector, visualizer, args):
    """Process video stream or RTSP feed.

    Args:
        cap: OpenCV VideoCapture object.
        detector: Object detector instance.
        visualizer: Visualizer instance for drawing detections.
        args: Parsed command-line arguments.
    """
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if args.show:
        visualizer._process_video_live(cap, detector, args)
    else:
        visualizer._process_video_to_file(cap, detector, args, fps, in_w, in_h)


def main():
    """Main function to run object detection on images, videos, or RTSP streams."""
    args = parse_args()

    class_names = _process_class_names(args.class_names)
    visualizer = Visualizer(class_names)
    detector = _get_detector(args.model_family, args.model_path)

    if is_video_file(args.media_link) or is_rtsp(args.media_link):
        cap = cv2.VideoCapture(args.media_link)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video/stream: {args.media_link}")
        _process_video_stream(cap, detector, visualizer, args)
    else:
        visualizer._process_image(args.media_link, detector, args)


if __name__ == "__main__":
    main()
