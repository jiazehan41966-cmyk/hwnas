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
from .fixed_point import (
    FixedPointContract,
    avg_pool2d_int_reference,
    conv2d_int_reference,
    linear_int_reference,
    max_pool2d_int_reference,
    quantize_bias_int32,
    relu_int_reference,
    requantize_per_output_int8,
    residual_add_int_reference,
)
from .int8_reference import (
    IntegerReferenceClassifier,
    IntegerReferenceResult,
    UnsupportedIntegerOperatorError,
    compare_integer_tensors,
    run_integer_reference,
)
from .ptq_eval import prepare_integer_ptq
from .mixconv_v2 import (
    MIXCONV_V2_BIAS_LAYOUT,
    MIXCONV_V2_INTEGER_SCHEMA,
    MIXCONV_V2_WEIGHT_LAYOUT,
    build_mixconv_v2_integer_package,
    quantize_mixconv_v2_input,
    simulate_mixconv_v2_int8,
)
from .reparam import (
    FoldedDenoiseBlock,
    FoldedEdgeBlock,
    FoldedMixConvV2Block,
    fold_denoise_block,
    fold_edge_block,
    fold_mixconv_v2_block,
    fold_sonar_blocks,
)
from .report_parser import parse_backend_report, parse_backend_report_text

__all__ = [
    "FoldedDenoiseBlock",
    "FoldedEdgeBlock",
    "FoldedMixConvV2Block",
    "HLSProjectConfig",
    "MIXCONV_V2_BIAS_LAYOUT",
    "MIXCONV_V2_INTEGER_SCHEMA",
    "MIXCONV_V2_WEIGHT_LAYOUT",
    "QuantizationConfig",
    "FixedPointContract",
    "IntegerReferenceResult",
    "IntegerReferenceClassifier",
    "UnsupportedIntegerOperatorError",
    "fold_denoise_block",
    "fold_edge_block",
    "fold_mixconv_v2_block",
    "fold_sonar_blocks",
    "attach_hls_report",
    "build_quantized_weight_package",
    "build_mixconv_v2_integer_package",
    "avg_pool2d_int_reference",
    "compare_integer_tensors",
    "conv2d_int_reference",
    "create_hls_project_stub",
    "export_checkpoint_quantized_weights",
    "linear_int_reference",
    "max_pool2d_int_reference",
    "export_checkpoint_to_onnx",
    "export_model_to_onnx",
    "iter_image_paths",
    "load_checkpoint_model",
    "load_run_config_from_checkpoint",
    "parse_backend_report",
    "parse_backend_report_text",
    "predict_image",
    "preprocess_image",
    "prepare_integer_ptq",
    "quantize_tensor_symmetric",
    "quantize_mixconv_v2_input",
    "quantize_bias_int32",
    "relu_int_reference",
    "requantize_per_output_int8",
    "residual_add_int_reference",
    "run_integer_reference",
    "simulate_mixconv_v2_int8",
    "resolve_class_names",
    "resolve_inference_settings",
]
