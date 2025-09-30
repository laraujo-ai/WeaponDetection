from models.base import IObjectDetector
from models.yolox import YOLOXDetector
from models.yolo_nas import YOLONasDetector

def get_yolo_nas_detector(model_path : str) -> IObjectDetector:
    return YOLONasDetector(model_path)


def get_yolox_detector(model_path: str) -> IObjectDetector:
    return YOLOXDetector(model_path)
    


