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
        if detections is None:
            return image

        img_h, img_w = image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        for det in detections:
            det = np.asarray(det, dtype=float)
            if det.size < 6:
                continue

            x1, y1, x2, y2, score, class_id = det[:6]

            # Heuristic: if x2 <= x1 or width tiny => treat input as x,y,w,h
            if x2 <= x1 or (x2 - x1) < 1:
                w = x2
                h = y2
                x2 = x1 + w
                y2 = y1 + h

            # Round & clip coordinates to image bounds
            x1_i = int(max(0, round(x1)))
            y1_i = int(max(0, round(y1)))
            x2_i = int(min(img_w, round(x2)))
            y2_i = int(min(img_h, round(y2)))

            # skip degenerate boxes
            if x2_i <= x1_i or y2_i <= y1_i:
                continue

            # safe class lookup
            class_id_i = int(class_id)
            if 0 <= class_id_i < len(self.class_names):
                class_name = self.class_names[class_id_i]
            else:
                class_name = f"class_{class_id_i}"

            # choose color (BGR tuple)
            color = self.colors.get(class_name, (0, 255, 0))
            color = tuple(int(c) for c in color)

            # ---- translucent fill inside box ----
            try:
                crop = image[y1_i:y2_i, x1_i:x2_i]
                if crop.size > 0:
                    # create color overlay same size as crop
                    color_box = np.ones_like(crop, dtype=np.uint8) * np.array(color, dtype=np.uint8).reshape(1, 1, 3)
                    # blend in place
                    cv2.addWeighted(crop, 1.0 - alpha, color_box, alpha, 0, crop)
            except Exception:
                # If anything goes wrong with blending, continue and at least draw rect/label
                pass

            # draw box outline
            cv2.rectangle(image, (x1_i, y1_i), (x2_i, y2_i), color, 2)

            # ---- label text ----
            # choose how to display score: if in [0,1] show percent, else show raw
            try:
                score_f = float(score)
            except Exception:
                score_f = 0.0
            if 0.0 <= score_f <= 1.0:
                score_text = f"{score_f*100:.0f}%"
            else:
                score_text = f"{score_f:.2f}"

            label_text = f"{class_name} - {score_text}"

            # adaptive font scale and thickness based on image size
            # min(img_w,img_h)/1000 -> ~1.0 for 1000px images, 0.5 for 500px
            font_scale = max(0.35, min(img_w, img_h) / 1000.0)
            thickness = max(1, int(round(font_scale * 2)))

            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            pad_x = 6
            pad_y = 4

            # label background rectangle coordinates (try above the box; if out of bounds, draw below)
            rect_x1 = x1_i - 1
            rect_x2 = x1_i + tw + pad_x
            rect_y2 = y1_i
            rect_y1 = y1_i - th - baseline - pad_y

            # if label would be above image, draw it inside the top of box instead
            if rect_y1 < 0:
                rect_y1 = y1_i
                rect_y2 = y1_i + th + baseline + pad_y
                text_origin = (x1_i + 3, rect_y1 + th + baseline - 3)
            else:
                text_origin = (x1_i + 3, y1_i - 4)

            # clamp rect coords
            rect_x1 = int(max(0, rect_x1))
            rect_x2 = int(min(img_w, rect_x2))
            rect_y1 = int(max(0, rect_y1))
            rect_y2 = int(min(img_h, rect_y2))

            # draw filled label background and put text
            cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), color, cv2.FILLED)
            cv2.putText(image, label_text, text_origin, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return image
