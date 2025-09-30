import argparse
import cv2

import providers as p
from utils import is_video_file
from media_processor import  Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Demo for the VisionTech object detection library.")
    parser.add_argument("--media_link", type=str, required=True, help="Link to the media (image or video) to process.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--model_family", type=str, required=True, choices=["yolox", "yolo-nas"], help="Model family to use.")
    parser.add_argument("--score_thr", type=float, default=0.25, help="Score threshold for detections.")
    parser.add_argument("--nms_thr", type=float, default=0.45, help="NMS threshold for detections.")
    parser.add_argument("--class_names", type=str, nargs='+', help="List of class names for the model.")
    return parser.parse_args()


def main():
    args = parse_args()
    detector = None

    if args.class_names and len(args.class_names) == 1 and ',' in args.class_names[0]:
        class_names = [c.strip() for c in args.class_names[0].split(',') if c.strip()]
    else:
        class_names = args.class_names or []
    visualizer = Visualizer(class_names)

    if args.model_family == "yolo-nas":
        detector = p.get_yolo_nas_detector(
            model_path=args.model_path) 
    elif args.model_family == "yolox":
        detector = p.get_yolox_detector(
            model_path=args.model_path)
    else:
        raise ValueError(f"Unsupported model family: {args.model_family}")
    
    if is_video_file(args.media_link):
        cap = cv2.VideoCapture(args.media_link)
        cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {args.media_link}")
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
        image = cv2.imread(args.media_link)
        detections = detector.detect(image, score_thr=args.score_thr, nms_thr=args.nms_thr)
        vis_image = visualizer.draw_detections(image, detections)
        cv2.imwrite("out.jpg", vis_image)


if __name__ == "__main__":
    main()
