import gc
import os
import subprocess

import onnx.version_converter
from onnxruntime.transformers.optimizer import optimize_model
from onnxslim import slim

# Path Setting
original_folder_path = "/home/DakeQQ/Downloads/DFSMN_VAD_ONNX"                      # The fp32 saved folder.
optimized_folder_path = "/home/DakeQQ/Downloads/DFSMN_VAD_Optimized"                # The optimized folder.
model_path = os.path.join(original_folder_path, "DFSMN_VAD.onnx")                   # The original fp32 model name.
optimized_model_path = os.path.join(optimized_folder_path, "DFSMN_VAD.onnx")        # The optimized model name.
use_gpu_fp16 = False                                                                # CUDA + DFSMN_VAD-FP16 will cause bad VAD results.
provider = 'CPUExecutionProvider'                                                   # ['CPUExecutionProvider', 'CUDAExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider']
target_platform = "amd64"                                                           # ['arm', 'amd64']; The 'amd64' means x86_64 desktop, not means the AMD chip.
target_opset = 17

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
                       opt_level=1,
                       num_heads=0,
                       hidden_size=0,
                       provider=provider,
                       verbose=False,
                       model_type='bert')
if use_gpu_fp16:
    model.convert_float_to_float16(
        keep_io_types=False,
        force_fp16_initializers=True,
        use_symbolic_shape_infer=True,  # True for more optimize but may get errors.
        op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'Range', 'MatMulIntegerToFloat']
    )
model.save_model_to_file(optimized_model_path, use_external_data_format=False)
del model
gc.collect()


# onnxslim 2nd
slim(
    model=optimized_model_path,
    output_model=optimized_model_path,
    no_shape_infer=False,   # False for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=False,
    verbose=False
)


# Upgrade the Opset version. (optional process)
if target_opset != 0:
    model = onnx.load(optimized_model_path)
    model = onnx.version_converter.convert_version(model, target_opset)
    onnx.save(model, optimized_model_path, save_as_external_data=False)
    del model
    gc.collect()


if not use_gpu_fp16:
    # Convert the simplified model to ORT format.
    if provider == 'CPUExecutionProvider':
        optimization_style = "Fixed"
    else:
        optimization_style = "Runtime"  # ['Runtime', 'Fixed']; Runtime for XNNPACK/NNAPI/QNN/CoreML..., Fixed for CPU provider
    # Call subprocess may get permission failed on Windows system.
    subprocess.run([f'python -m onnxruntime.tools.convert_onnx_models_to_ort --output_dir {optimized_folder_path} --optimization_style {optimization_style} --target_platform {target_platform} --enable_type_reduction {optimized_folder_path}'], shell=True)
