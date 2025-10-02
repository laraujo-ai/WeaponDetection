import argparse
import cv2
import os
from tools.utils import is_video_file, is_rtsp
import tools.providers as p
from tools.media_processor import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description="VisionTech Object Detection Demo")
    parser.add_argument("--media_link", type=str, required=True,
                        help="Path to image/video or RTSP URL.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pre-trained model.")
    parser.add_argument("--model_family", type=str, required=True,
                        choices=["yolox", "yolo-nas"], help="Model family to use.")
    parser.add_argument("--class_names", type=str, nargs='+',
                        help="List of class names for the model (comma-separated if single string).")
    parser.add_argument("--nms_thr", type=float, default=0.45, help="NMS threshold for detections.")
    parser.add_argument("--score_thr", type=float, default=0.25, help="Confidence threshold for detections.")
    parser.add_argument("--show", action="store_true", help="Show live detections with OpenCV.")
    parser.add_argument("--out_file", type=str, default="output.mp4",
                        help="Output video file path (if not showing live).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Process class names
    if args.class_names and len(args.class_names) == 1 and ',' in args.class_names[0]:
        class_names = [c.strip() for c in args.class_names[0].split(',') if c.strip()]
    else:
        class_names = args.class_names or []

    visualizer = Visualizer(class_names)

    # Initialize detector
    if args.model_family == "yolo-nas":
        detector = p.get_yolo_nas_detector(model_path=args.model_path)
    elif args.model_family == "yolox":
        detector = p.get_yolox_detector(model_path=args.model_path)
    else:
        raise ValueError(f"Unsupported model family: {args.model_family}")

    # Process video or RTSP
    if is_video_file(args.media_link) or is_rtsp(args.media_link):
        cap = cv2.VideoCapture(args.media_link)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video/stream: {args.media_link}")

        in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        if args.show:
            cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                detections = detector.detect(frame, score_thr=args.score_thr, nms_thr=args.nms_thr)
                vis_frame = visualizer.draw_detections(frame, detections)
                cv2.imshow("Detections", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(args.out_file, fourcc, fps, (in_w, in_h))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                detections = detector.detect(frame, score_thr=args.score_thr, nms_thr=args.nms_thr)
                vis_frame = visualizer.draw_detections(frame, detections)
                out.write(vis_frame)
            cap.release()
            out.release()

    # Process single image
    else:
        image = cv2.imread(args.media_link)
        if image is None:
            raise FileNotFoundError(f"Image not found: {args.media_link}")
        detections = detector.detect(image, score_thr=args.score_thr, nms_thr=args.nms_thr)
        vis_image = visualizer.draw_detections(image, detections)
        out_path = os.path.splitext(args.media_link)[0] + "_detected.jpg"
        cv2.imwrite(out_path, vis_image)
        if args.show:
            cv2.imshow("Detections", vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(f"Saved output image to {out_path}")


if __name__ == "__main__":
    main()
