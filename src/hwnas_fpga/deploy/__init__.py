"""Export, quantization, FPGA backend integration, and inference."""

from .export import export_checkpoint_to_onnx, export_model_to_onnx
from .hls_backend import HLSProjectConfig, attach_hls_report, create_hls_project_stub
from .inference import (
    iter_image_paths,
    load_checkpoint_model,
    load_run_config_from_checkpoint,
    predict_image,
    preprocess_image,
    resolve_class_names,
    resolve_inference_settings,
)
from .quantization import (
    QuantizationConfig,
    build_quantized_weight_package,
    export_checkpoint_quantized_weights,
    quantize_tensor_symmetric,
)
from .reparam import (
    FoldedDenoiseBlock,
    FoldedEdgeBlock,
    fold_denoise_block,
    fold_edge_block,
    fold_sonar_blocks,
)
from .report_parser import parse_backend_report, parse_backend_report_text

__all__ = [
    "FoldedDenoiseBlock",
    "FoldedEdgeBlock",
    "HLSProjectConfig",
    "QuantizationConfig",
    "fold_denoise_block",
    "fold_edge_block",
    "fold_sonar_blocks",
    "attach_hls_report",
    "build_quantized_weight_package",
    "create_hls_project_stub",
    "export_checkpoint_quantized_weights",
    "export_checkpoint_to_onnx",
    "export_model_to_onnx",
    "iter_image_paths",
    "load_checkpoint_model",
    "load_run_config_from_checkpoint",
    "parse_backend_report",
    "parse_backend_report_text",
    "predict_image",
    "preprocess_image",
    "quantize_tensor_symmetric",
    "resolve_class_names",
    "resolve_inference_settings",
]
