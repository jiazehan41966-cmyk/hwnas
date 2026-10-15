"""Training/validation protocol with a sealed test split."""
from .loss import DetectionLossConfig,DenseAnchorDetectionLoss,decode_detection_predictions
from .protocol import loss_config_for_model
from .evaluation import evaluate,metrics_from_records
__all__=["DetectionLossConfig","DenseAnchorDetectionLoss","decode_detection_predictions",
         "loss_config_for_model","evaluate","metrics_from_records"]
