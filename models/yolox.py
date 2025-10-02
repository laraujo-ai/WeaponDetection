import onnxruntime as ort
import numpy as np

from models.base import IObjectDetector
from models.functions import xywh_to_xyxy, multiclass_nms_class_agnostic
import cv2

class YOLOXDetector(IObjectDetector):
    def __init__(self, model_path: str, providers : list):
        self._model_engine = self.load_model(model_path, providers)
        self.input_name = self._model_engine.get_inputs()[0].name
        self.input_shape = self._model_engine.get_inputs()[0].shape[2:]
        
    def load_model(self, model_path: str, providers : list):
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 1 
        session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
        return session

    def _preprocess(self, ori_frame):
        target_h, target_w = self.input_shape
        padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 114
        r = min(target_h / ori_frame.shape[0], target_w / ori_frame.shape[1])
        resized_img = cv2.resize(
            ori_frame,
            (int(ori_frame.shape[1] * r), int(ori_frame.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(ori_frame.shape[0] * r), : int(ori_frame.shape[1] * r)] = resized_img
        padded_img = padded_img.transpose((2, 0, 1))
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

        return padded_img, r

    def _postprocess(self, outputs, ratio, score_threshold=0.25, nms_threshold=0.45):
        grids, expanded_strides = [], []
        strides = [8, 16, 32]
        hsizes = [self.input_shape[0] // stride for stride in strides]
        wsizes = [self.input_shape[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)

        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        predictions = outputs[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = xywh_to_xyxy(boxes, ratio)

        dets = multiclass_nms_class_agnostic(
            boxes_xyxy, scores,
            nms_thr=nms_threshold,
            score_thr=score_threshold,
        )
        return dets

    def detect(self, image, score_thr:float = 0.25 , nms_thr : float = 0.45):
        preprocessed_image, ratio = self._preprocess(image)
        inp = preprocessed_image[None]
        if inp.dtype != np.float32:
            inp = inp.astype(np.float32)

        raw_output = self._model_engine.run(None, {self.input_name: inp.astype(np.float16)})[0].astype(np.float32)
        detections = self._postprocess(raw_output, ratio, score_threshold=score_thr, nms_threshold=nms_thr)
        if detections is None or len(detections) == 0:
            return np.array([])
        return detections