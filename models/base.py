from typing import Protocol, Any


class IObjectDetector(Protocol):
    """Protocol for object detection models.

    Defines the interface that all object detector implementations must follow.
    """

    def detect(self, image: Any, **kwargs) -> Any:
        """Perform object detection on an image.

        Args:
            image: Input image for detection.
            **kwargs: Additional detection parameters.

        Returns:
            Detection results.
        """
        ...

    def load_model(self, model_path: str, **kwargs) -> Any:
        """Load the detection model from file.

        Args:
            model_path: Path to the model file.
            **kwargs: Additional loading parameters.

        Returns:
            Loaded model instance.
        """
        ...

    def _preprocess(self, image: Any) -> Any:
        """Preprocess image for model input.

        Args:
            image: Input image to preprocess.

        Returns:
            Preprocessed image data.
        """
        ...

    def _postprocess(self, outputs: Any, **kwargs) -> Any:
        """Postprocess model outputs to detections.

        Args:
            outputs: Raw model outputs.
            **kwargs: Additional postprocessing parameters.

        Returns:
            Processed detection results.
        """
        ...

