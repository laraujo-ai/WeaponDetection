import onnxruntime as ort
import cv2
import numpy as np
from typing import Tuple, List, Any

from models.base import IObjectDetector

class YOLONasDetector(IObjectDetector):
    """YOLO-NAS object detector using ONNX Runtime.

    Implements the SuperGradients YOLO-NAS preprocessing and postprocessing pipeline.
    """

    def __init__(self, model_path: str, providers: List[str] = None):
        """Initialize the YOLO-NAS detector.

        Args:
            model_path: Path to the ONNX model file.
            providers: ONNX Runtime execution providers.
        """
        self.model = self.load_model(model_path, providers)
        inp = self.model.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.in_c, self.in_h, self.in_w = map(int, shape[1:])
        self.expected_dtype = np.float16 if "float16" in str(inp.type) else np.float32
        self.output_names = [o.name for o in self.model.get_outputs()]

    def load_model(self, model_path: str, providers):
        """Load ONNX model with optimized session settings.

        Args:
            model_path: Path to the ONNX model file.
            providers: ONNX Runtime execution providers.

        Returns:
            ONNX Runtime inference session.
        """
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 1
        session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
        return session

    def _det_long_max_rescale(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Rescale image maintaining aspect ratio to fit model input size.

        Args:
            img: Input image as numpy array.

        Returns:
            Tuple of rescaled image and metadata dictionary.
        """
        h, w = img.shape[:2]
        scale_factor = min((self.in_h - 4) / h, (self.in_w - 4) / w)

        if scale_factor != 1.0:
            new_h, new_w = round(h * scale_factor), round(w * scale_factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        metadata = {"scale_factors": (scale_factor, scale_factor)}
        return img, metadata

    def _bot_right_pad(self, img, pad_value):
        """Pad image to model input size on bottom and right sides.

        Args:
            img: Input image to pad.
            pad_value: Pixel value to use for padding.

        Returns:
            Tuple of padded image and padding metadata.
        """
        pad_height, pad_width = self.in_h - img.shape[0], self.in_w - img.shape[1]
        return cv2.copyMakeBorder(
            img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[pad_value] * img.shape[-1]
        ), {"padding": (0, pad_height, 0, pad_width)}

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """Preprocess image for model inference.

        Args:
            image: Input image in BGR format.

        Returns:
            Tuple of preprocessed blob and list of metadata dictionaries.
        """
        metadata: List[dict] = []
        img, meta_rescale = self._det_long_max_rescale(image)
        metadata.append(meta_rescale)
        img, meta_pad = self._bot_right_pad(img, pad_value=114)
        metadata.append(meta_pad)
        blob = cv2.dnn.blobFromImage(img, swapRB=True)
        blob = blob.astype(self.expected_dtype)
        return blob, metadata

    def _postprocess(self, outputs: List[Any], metadata: List[dict], score_thr: float, nms_thr: float) -> np.ndarray:
        """Postprocess model outputs to final detections.

        Args:
            outputs: Raw outputs from ONNX model [boxes, scores].
            metadata: Metadata from preprocessing steps.
            score_thr: Confidence score threshold.
            nms_thr: NMS IoU threshold.

        Returns:
            Array of detections with shape (N, 6): [x, y, w, h, score, class_id].
        """
        if len(outputs) < 2:
            return np.zeros((0, 6), dtype=np.float32)

        raw_boxes = np.squeeze(outputs[0]).astype(np.float32)
        scores = np.squeeze(outputs[1]).astype(np.float32)

        for meta in reversed(metadata):
            if "padding" in meta:
                pad_top, _, pad_left, _ = meta["padding"]
                raw_boxes[:, [0, 2]] -= pad_left
                raw_boxes[:, [1, 3]] -= pad_top
            elif "scale_factors" in meta:
                scale_w, scale_h = meta["scale_factors"]
                raw_boxes[:, [0, 2]] /= scale_w
                raw_boxes[:, [1, 3]] /= scale_h

        raw_boxes[:, 2] -= raw_boxes[:, 0]
        raw_boxes[:, 3] -= raw_boxes[:, 1]

        class_ids = np.argmax(scores, axis=1)
        conf_scores = np.max(scores, axis=1)

        indices = cv2.dnn.NMSBoxes(raw_boxes.tolist(), conf_scores.tolist(), score_thr, nms_thr)
        if len(indices) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        final_boxes = raw_boxes[indices]
        final_scores = conf_scores[indices]
        final_class_ids = class_ids[indices]

        detections = np.hstack([
            final_boxes,
            final_scores[:, np.newaxis],
            final_class_ids[:, np.newaxis]
        ]).astype(np.float32)

        return detections

    def detect(self, image: np.ndarray, score_thr: float = 0.25, nms_thr: float = 0.45) -> np.ndarray:
        """Perform object detection on an image.

        Args:
            image: Input image in BGR format.
            score_thr: Confidence score threshold for filtering detections.
            nms_thr: IoU threshold for Non-Maximum Suppression.

        Returns:
            Array of detections with shape (N, 6): [x, y, w, h, score, class_id].
        """
        inp_blob, metadata = self._preprocess(image)
        outputs = self.model.run(None, {self.input_name: inp_blob})
        detections = self._postprocess(outputs, metadata, score_thr, nms_thr)

        orig_h, orig_w = image.shape[:2]
        detections[:, 0] = np.clip(detections[:, 0], 0, orig_w)
        detections[:, 1] = np.clip(detections[:, 1], 0, orig_h)
        detections[:, 2] = np.clip(detections[:, 2], 0, orig_w - detections[:, 0])
        detections[:, 3] = np.clip(detections[:, 3], 0, orig_h - detections[:, 1])

        return detections