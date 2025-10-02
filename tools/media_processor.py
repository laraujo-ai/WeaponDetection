import cv2
import numpy as np
from typing import List

class Visualizer:
    def __init__(self, class_names: List[str]):
        if len(class_names) == 1 and isinstance(class_names[0], str) and ',' in class_names[0]:
            class_names = [c.strip() for c in class_names[0].split(',') if c.strip()]
        self.class_names = class_names

        self.colors = {
            name: tuple(int(x) for x in np.random.randint(0, 256, size=3).tolist())
            for name in self.class_names
        }

    def draw_detections(self, image: np.ndarray, detections: np.ndarray, alpha: float = 0.25) -> np.ndarray:
        """
        Draw bounding boxes with translucent fill and labels.

        Args:
            image: BGR image (np.ndarray) to draw on — modified in-place and returned.
            detections: iterable of [x1, y1, x2, y2, score, class_id] or [x, y, w, h, score, class_id].
            alpha: float in [0,1] for translucency of the filled box.
        """
        if detections is None or len(detections) == 0:
            return image

        img_h, img_w = image.shape[:2]

        for det in detections:
            det = np.asarray(det, dtype=float)
            if det.size < 6:
                continue

            bbox = self._parse_bbox(det, img_w, img_h)
            if bbox is None:
                continue

            x1, y1, x2, y2, score, class_id = bbox
            class_name = self._get_class_name(class_id)
            color = self._get_color(class_name)

            self._draw_translucent_box(image, x1, y1, x2, y2, color, alpha)
            self._draw_box_outline(image, x1, y1, x2, y2, color)
            self._draw_label(image, x1, y1, x2, y2, class_name, score, color, img_w, img_h)

        return image

    def _parse_bbox(self, det: np.ndarray, img_w: int, img_h: int):
        """Parse and normalize bounding box coordinates."""
        x1, y1, x2, y2, score, class_id = det[:6]

        # Convert from (x, y, w, h) to (x1, y1, x2, y2) if needed
        if x2 <= x1 or (x2 - x1) < 1:
            x2 = x1 + x2  # x2 is width
            y2 = y1 + y2  # y2 is height

        # Clip coordinates to image bounds
        x1 = int(np.clip(round(x1), 0, img_w))
        y1 = int(np.clip(round(y1), 0, img_h))
        x2 = int(np.clip(round(x2), 0, img_w))
        y2 = int(np.clip(round(y2), 0, img_h))

        # Skip degenerate boxes
        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2, score, class_id

    def _get_class_name(self, class_id: float) -> str:
        """Get class name from class ID."""
        class_id_i = int(class_id)
        if 0 <= class_id_i < len(self.class_names):
            return self.class_names[class_id_i]
        return f"class_{class_id_i}"

    def _get_color(self, class_name: str) -> tuple:
        """Get BGR color tuple for a class."""
        color = self.colors.get(class_name, (0, 255, 0))
        return tuple(int(c) for c in color)

    def _draw_translucent_box(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                              color: tuple, alpha: float):
        """Draw translucent filled box."""
        try:
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                color_overlay = np.full_like(crop, color, dtype=np.uint8)
                cv2.addWeighted(crop, 1.0 - alpha, color_overlay, alpha, 0, crop)
        except Exception:
            pass  # Continue even if blending fails

    def _draw_box_outline(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple):
        """Draw bounding box outline."""
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    def _draw_label(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                    class_name: str, score: float, color: tuple, img_w: int, img_h: int):
        """Draw label with background."""
        label_text = self._format_label(class_name, score)
        font_scale, thickness = self._calculate_font_params(img_w, img_h)

        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        label_rect, text_pos = self._calculate_label_position(
            x1, y1, x2, y2, text_w, text_h, baseline, img_w, img_h
        )

        cv2.rectangle(image, label_rect[:2], label_rect[2:], color, cv2.FILLED)
        cv2.putText(image, label_text, text_pos, font, font_scale,
                   (255, 255, 255), thickness, cv2.LINE_AA)

    def _format_label(self, class_name: str, score: float) -> str:
        """Format label text with class name and score."""
        try:
            score_f = float(score)
            score_text = f"{score_f * 100:.0f}%" if 0.0 <= score_f <= 1.0 else f"{score_f:.2f}"
        except (ValueError, TypeError):
            score_text = "0%"
        return f"{class_name} - {score_text}"

    def _calculate_font_params(self, img_w: int, img_h: int) -> tuple:
        """Calculate adaptive font scale and thickness based on image size."""
        font_scale = max(0.35, min(img_w, img_h) / 1000.0)
        thickness = max(1, int(round(font_scale * 2)))
        return font_scale, thickness

    def _calculate_label_position(self, x1: int, y1: int, x2: int, y2: int,
                                  text_w: int, text_h: int, baseline: int,
                                  img_w: int, img_h: int) -> tuple:
        """Calculate label background rectangle and text position."""
        pad_x, pad_y = 6, 4

        # Try placing label above the box
        rect_x1 = x1 - 1
        rect_x2 = x1 + text_w + pad_x
        rect_y1 = y1 - text_h - baseline - pad_y
        rect_y2 = y1

        # If label goes above image, place it inside the box at the top
        if rect_y1 < 0:
            rect_y1 = y1
            rect_y2 = y1 + text_h + baseline + pad_y
            text_pos = (x1 + 3, rect_y1 + text_h + baseline - 3)
        else:
            text_pos = (x1 + 3, y1 - 4)

        # Clamp rectangle to image bounds
        rect_x1 = int(np.clip(rect_x1, 0, img_w))
        rect_x2 = int(np.clip(rect_x2, 0, img_w))
        rect_y1 = int(np.clip(rect_y1, 0, img_h))
        rect_y2 = int(np.clip(rect_y2, 0, img_h))

        return (rect_x1, rect_y1, rect_x2, rect_y2), text_pos


    def _process_video_live(self, cap, detector, args):
        """Process video stream and display live detections.

        Args:
            cap: OpenCV VideoCapture object.
            detector: Object detector instance.
            args: Parsed command-line arguments.
        """
        cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = detector.detect(
                frame, score_thr=args.score_thr, nms_thr=args.nms_thr
            )
            vis_frame = self.draw_detections(frame, detections)
            cv2.imshow("Detections", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


    def _process_video_to_file(self, cap, detector, args, fps, width, height):
        """Process video stream and save detections to file.

        Args:
            cap: OpenCV VideoCapture object.
            detector: Object detector instance.
            args: Parsed command-line arguments.
            fps: Frames per second of the input video.
            width: Width of the video frames.
            height: Height of the video frames.
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.out_file, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = detector.detect(
                frame, score_thr=args.score_thr, nms_thr=args.nms_thr
            )
            vis_frame = self.draw_detections(frame, detections)
            out.write(vis_frame)

        cap.release()
        out.release()
        print(f"Saved output video to {args.out_file}")
    
    def _process_image(self, image_path, detector, args):
        """Process a single image and save detections.

        Args:
            image_path: Path to the input image.
            detector: Object detector instance.
            args: Parsed command-line arguments.

        Raises:
            FileNotFoundError: If the image file is not found.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        detections = detector.detect(
            image, score_thr=args.score_thr, nms_thr=args.nms_thr
        )
        vis_image = self.draw_detections(image, detections)

        out_path = os.path.splitext(image_path)[0] + "_detected.jpg"
        cv2.imwrite(out_path, vis_image)

        if args.show:
            cv2.imshow("Detections", vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print(f"Saved output image to {out_path}")
