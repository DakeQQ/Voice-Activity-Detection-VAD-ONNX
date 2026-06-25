import gc
import os

import onnx
import onnx.version_converter
from onnxruntime.transformers.optimizer import optimize_model
from onnxslim import slim

# Path Setting
# Anchored to this script's folder so the glued models are found regardless of the
# working directory (matches Export_SmartTurn_Glued.py, which writes them here).
script_dir = os.path.dirname(os.path.abspath(__file__))
original_folder_path = os.path.join(script_dir, "SmartTurn_Glued_ONNX")                  # The glued end-to-end models (int16 PCM -> logits).
optimized_folder_path = os.path.join(script_dir, "SmartTurn_Glued_Optimized")            # The optimized folder.
model_names = ["smart-turn-v3.2-cpu-glued.onnx", "smart-turn-v3.2-gpu-glued.onnx"]        # The glued CPU (int8) and GPU (fp32) models.
use_fp16 = False                                                                        # Set True for fp16 quant (applied to fp32 models only; int8 models are skipped).
opset = 0                                                                               # Upgrade the Opset version. Set opset <= 0 for disable. (optional process)

os.makedirs(optimized_folder_path, exist_ok=True)


def is_quantized(path):
    """True if the model already carries int8 QDQ nodes — those must not be fp16-converted."""
    return any(
        node.op_type in ("QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear")
        for node in onnx.load(path).graph.node
    )


for model_name in model_names:
    model_path = os.path.join(original_folder_path, model_name)
    optimized_model_path = os.path.join(optimized_folder_path, model_name)

    if not os.path.isfile(model_path):
        print(f"Skip: {model_path} not found.")
        continue

    print(f"\nOptimizing: {model_name}")

    # ONNX Model Optimizer
    slim(
        model=model_path,
        output_model=optimized_model_path,
        no_shape_infer=False,   # False for more optimize but may get errors.
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=False,
        verbose=False
    )

    # transformers.optimizer
    model = optimize_model(optimized_model_path,
                           use_gpu=False,
                           opt_level=2,
                           num_heads=0,
                           hidden_size=0,
                           verbose=False,
                           model_type='bert')
    if use_fp16 and not is_quantized(model_path):
        model.convert_float_to_float16(
            keep_io_types=False,
            force_fp16_initializers=True,
            use_symbolic_shape_infer=False,  # True for more optimize but may get errors.
            op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'Range', 'MatMulIntegerToFloat', 'ReduceMean', 'ReduceSum']
        )
    model.save_model_to_file(optimized_model_path, use_external_data_format=False)
    del model
    gc.collect()

    # onnxslim 2nd
    slim(
        model=optimized_model_path,
        output_model=optimized_model_path,
        no_shape_infer=False,               # False for more optimize but may get errors.
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=False,
        verbose=False
    )

    if opset > 0:
        model = onnx.load(optimized_model_path)
        model = onnx.version_converter.convert_version(model, opset)
        onnx.save(model, optimized_model_path, save_as_external_data=False)
        del model
        gc.collect()

    print(f"Done: {optimized_model_path}")