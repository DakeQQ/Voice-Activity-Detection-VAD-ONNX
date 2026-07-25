"""Optimize the glued SmartTurn v3 CPU and GPU ONNX models."""

import gc
import sys
from pathlib import Path

import onnx


SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
    if (_candidate / "Optimize_ONNX_Common.py").exists() and (
        _candidate / "vad_onnx_metadata.py"
    ).exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

from Optimize_ONNX_Common import OptimizerConfig, Plan, run_optimizer


ORIGINAL_FOLDER_PATH = str(SCRIPT_DIR / "SmartTurn_Glued_ONNX")
OPTIMIZED_FOLDER_PATH = str(SCRIPT_DIR / "SmartTurn_Glued_Optimized")

ENABLE_FP16 = False  # Applied only to float models; existing int8 graphs stay unchanged.

UPGRADE_OPSET = 20

MODEL_NAMES = (
    "smart-turn-v3.2-cpu-glued",
    "smart-turn-v3.2-gpu-glued",
)


def is_quantized(path):
    """Return whether a source graph already contains int8 quantization nodes."""
    model = onnx.load(str(path), load_external_data=False)
    try:
        return any(
            node.op_type
            in {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"}
            for node in model.graph.node
        )
    finally:
        del model
        gc.collect()


def optimization_method(model_name):
    if not ENABLE_FP16:
        return "F32"
    model_path = Path(ORIGINAL_FOLDER_PATH) / f"{model_name}.onnx"
    return "F32" if model_path.is_file() and is_quantized(model_path) else "F16"


MODEL_PLANS = {
    model_name: Plan(
        method=optimization_method(model_name),
        fp16_symbolic_shape_infer=False,
        f16_op_block_list=[
            "DynamicQuantizeLinear",
            "DequantizeLinear",
            "DynamicQuantizeMatMul",
            "Range",
            "MatMulIntegerToFloat",
            "ReduceMean",
            "ReduceSum",
        ],
    )
    for model_name in MODEL_NAMES
}

CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    upgrade_opset=UPGRADE_OPSET,
)


if __name__ == "__main__":
    run_optimizer(CONFIG)