import gc
import os

import onnx.version_converter
from onnxruntime.transformers.optimizer import optimize_model
from onnxslim import slim

# Path Setting
original_folder_path = "/home/DakeQQ/Downloads/NVIDIA_MarbleNet_ONNX"                   # The fp32 saved folder.
optimized_folder_path = "/home/DakeQQ/Downloads/NVIDIA_MarbleNet_Optimized"             # The optimized folder.
model_path = os.path.join(original_folder_path, "NVIDIA_MarbleNet.onnx")                # The original fp32 model name.
optimized_model_path = os.path.join(optimized_folder_path, "NVIDIA_MarbleNet.onnx")     # The optimized model name.
use_fp16 = False                                                                        # Set True for fp16 quant.
opset = 0                                                                               # Upgrade the Opset version. Set opset <= 0 for disable. (optional process)


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
if use_fp16:
    model.convert_float_to_float16(
        keep_io_types=False,
        force_fp16_initializers=True,
        use_symbolic_shape_infer=False,  # True for more optimize but may get errors.
        op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'Range', 'MatMulIntegerToFloat']
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