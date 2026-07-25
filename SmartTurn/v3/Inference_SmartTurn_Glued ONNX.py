#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Turn — GLUED end-to-end ONNX inference (waveform -> P(complete))
=====================================================================

Runs the fused Smart Turn v3 model produced by ``Export_SmartTurn_Glued.py`` in a
single ONNX Runtime session. The glued graph already embeds sample-rate conversion
and the Whisper log-mel feature extractor. Decoded mono INT16 PCM is converted to the waveform
dtype declared by model metadata, then scored without external feature extraction.

    audio  [batch, n_samples]  float16/float32/int16  ->  probability  [batch, 1]

``n_samples`` (the fixed 8 s context window at the configured input rate) is read
from the model's own input shape, so it always matches whatever glued model is
loaded. A clip is truncated to its last ``n_samples`` (most recent context) or
front-padded with zeros, then scored; ``probability > THRESHOLD`` => turn complete.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "vad_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from vad_onnx_metadata import (
    REQUIRED_SMART_TURN_METADATA_KEYS,
    load_runtime_metadata,
)
from Example_Audio import model_audio_path


# ═════════════════════════════════════════════════════════════════════════════
# Settings
# ═════════════════════════════════════════════════════════════════════════════
# Fused end-to-end model from Export_SmartTurn_Glued.py (audio -> P(complete)).
default_model_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "SmartTurn_Glued_Optimized",
)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_folder",
        nargs="?",
        default=default_model_folder,
        help="Folder containing the exported Smart Turn glued ONNX models.",
    )
    model_folder = os.path.abspath(os.path.expanduser(parser.parse_args().model_folder))
else:
    model_folder = default_model_folder
onnx_model_A = os.path.join(
    model_folder,
    "smart-turn-v3.2-cpu-glued.onnx",
)
test_audio = model_audio_path("smart_turn_v3")  # Mono audio to score; a dummy clip is used if missing.

COMPLETE_THRESHOLD = 0.5              # Scores above this mark the current turn as complete.
ORT_ACCELERATE_PROVIDERS = []         # Add installed execution providers in priority order; [] uses ORT defaults.
CPU_DISABLE_MATMUL_ADD_FUSION = True  # ORT 1.27 wraps rank-3 MatMul+Add in costly Reshape/Gemm/Reshape chains.
CPU_DISABLE_NCHWC = True              # NCHWc reorders regress mean/tail latency on the target i7-1165G7.
CPU_EXTRA_DISABLED_OPTIMIZERS = [     # Individually benchmarked on the same CPU / ORT build.
    "ConvAddActivationFusion",
    "MatmulTransposeFusion",
]
MAX_THREADS = 0                       # OpenVINO option value; ORT session thread counts below remain automatic.
DEVICE_ID = 0                         # Device index used when allocating accelerator OrtValue buffers.


if "OpenVINOExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
    provider_options = [
        {
            "device_type": "CPU",                         # [CPU, NPU, GPU, GPU.0, GPU.1]
            "precision": "ACCURACY",                      # [FP32, FP16, ACCURACY]
            "num_of_threads": MAX_THREADS,
            "num_streams": 1,
            "enable_opencl_throttling": True,
            "enable_qdq_optimizer": False,                # Enable it carefully.
            "disable_dynamic_shapes": False,
        }
    ]
elif "CUDAExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
    provider_options = [
        {
            "device_id": DEVICE_ID,
            "gpu_mem_limit": 8 * 1024 * 1024 * 1024,      # 8 GB
            "arena_extend_strategy": "kNextPowerOfTwo",  # [kNextPowerOfTwo, kSameAsRequested]
            "cudnn_conv_algo_search": "EXHAUSTIVE",      # [DEFAULT, HEURISTIC, EXHAUSTIVE]
            "sdpa_kernel": "2",                          # [0, 1, 2]
            "use_tf32": "1",
            "fuse_conv_bias": "1",
            "cudnn_conv_use_max_workspace": "1",
            "cudnn_conv1d_pad_to_nc1d": "1",
            "tunable_op_enable": "0",
            "tunable_op_tuning_enable": "0",
            "tunable_op_max_tuning_duration_ms": 10,
            "do_copy_in_default_stream": "0",
            "enable_cuda_graph": "0",                    # Set to 0 to avoid potential errors when enabled.
            "prefer_nhwc": "0",
            "enable_skip_layer_norm_strict_mode": "0",
            "use_ep_level_unified_stream": "0",
        }
    ]
elif "DmlExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
    provider_options = [
        {
            "device_id": DEVICE_ID,
            "performance_preference": "high_performance",  # [high_performance, default, minimum_power]
            "device_filter": "any",                         # [any, npu, gpu]
        }
    ]
else:
    # Configure provider options here for other execution providers.
    provider_options = None

# ═════════════════════════════════════════════════════════════════════════════
# ONNX Runtime session
# ═════════════════════════════════════════════════════════════════════════════
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4  # Fatal level; adjustable.
session_opts.log_verbosity_level = 4  # Fatal level; adjustable.
session_opts.inter_op_num_threads = 0  # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0  # Under the node, execute operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True  # True for execution speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")

run_options = onnxruntime.RunOptions()
run_options.log_severity_level = 4
run_options.log_verbosity_level = 4
run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")

_CPU_EP_ONLY = not ORT_ACCELERATE_PROVIDERS or set(ORT_ACCELERATE_PROVIDERS) == {"CPUExecutionProvider"}
disabled_opts = []
if _CPU_EP_ONLY and CPU_DISABLE_MATMUL_ADD_FUSION:
    disabled_opts.append("MatMulAddFusion")
if _CPU_EP_ONLY and CPU_DISABLE_NCHWC:
    disabled_opts.append("NchwcTransformer")
if _CPU_EP_ONLY:
    disabled_opts.extend(CPU_EXTRA_DISABLED_OPTIMIZERS)
disabled_opts = disabled_opts or None


def make_session(model_path):
    return onnxruntime.InferenceSession(
        str(model_path),
        sess_options=session_opts,
        providers=ORT_ACCELERATE_PROVIDERS,
        provider_options=provider_options,
        disabled_optimizers=disabled_opts,
    )


def numpy_dtype_from_onnx_type(type_name):
    tensor_dtypes = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(int16)": np.int16,
    }
    try:
        return tensor_dtypes[type_name]
    except KeyError as error:
        raise TypeError(f"Unsupported ONNX tensor type: {type_name}") from error


def resolve_tensor_shape(node_arg, dynamic_extents=None):
    dynamic_extents = dynamic_extents or {}
    rank = len(node_arg.shape)
    normalized_extents = {
        axis if axis >= 0 else rank + axis: int(extent)
        for axis, extent in dynamic_extents.items()
    }
    dynamic_axes = {
        axis
        for axis, dimension in enumerate(node_arg.shape)
        if not isinstance(dimension, int)
    }
    if dynamic_axes != set(normalized_extents):
        raise ValueError(
            f"Tensor {node_arg.name!r} has dynamic axes {sorted(dynamic_axes)} "
            f"in shape {node_arg.shape}, but extents were supplied for "
            f"{sorted(normalized_extents)}."
        )
    return tuple(
        dimension if isinstance(dimension, int) else normalized_extents[axis]
        for axis, dimension in enumerate(node_arg.shape)
    )


ort_session_A = make_session(onnx_model_A)
model_metadata = load_runtime_metadata(
    onnx_model_A,
    make_session,
    required_keys=REQUIRED_SMART_TURN_METADATA_KEYS,
)
if model_metadata.string("task", required=True) != "turn_completion_detection":
    raise ValueError("SmartTurn metadata task must be turn_completion_detection.")
INPUT_AUDIO_DTYPES = {
    "float16": ("tensor(float16)", np.float16),
    "float": ("tensor(float)", np.float32),
    "int16": ("tensor(int16)", np.int16),
}
metadata_dtype = model_metadata.string("input_audio_dtype", required=True).lower()
if metadata_dtype not in INPUT_AUDIO_DTYPES:
    raise ValueError(f"Unsupported SmartTurn input_audio_dtype={metadata_dtype!r}.")
expected_onnx_dtype, _ = INPUT_AUDIO_DTYPES[metadata_dtype]
if not model_metadata.required_bool("normalization_embedded"):
    raise ValueError("SmartTurn glued model metadata must declare embedded normalization.")
IN_SAMPLE_RATE = model_metadata.required_int("input_sample_rate")
MODEL_SAMPLE_RATE = model_metadata.required_int("model_sample_rate")
metadata_input_audio_length = model_metadata.required_int("input_audio_length")

inputs = ort_session_A.get_inputs()
outputs = ort_session_A.get_outputs()
if len(inputs) != model_metadata.required_int("num_inputs"):
    raise ValueError("SmartTurn input count does not match metadata.")
if len(outputs) != model_metadata.required_int("num_outputs"):
    raise ValueError("SmartTurn output count does not match metadata.")
if (
    inputs[0].type != expected_onnx_dtype
    or len(inputs[0].shape) != 3
    or inputs[0].shape[1] != 1
    or any(not isinstance(dimension, int) for dimension in inputs[0].shape[1:])
):
    raise ValueError(f"Unexpected SmartTurn input contract: {inputs[0].type} {inputs[0].shape}.")
model_dynamic_batch = not isinstance(inputs[0].shape[0], int)
if model_dynamic_batch != model_metadata.required_bool("dynamic_batch"):
    raise ValueError("SmartTurn dynamic batch state does not match metadata.")
model_window = inputs[0].shape[-1]
if model_window != metadata_input_audio_length:
    raise ValueError(
        f"SmartTurn model window {model_window} does not match metadata "
        f"input_audio_length={metadata_input_audio_length}."
    )
INPUT_AUDIO_LENGTH = model_window
print(f"\nUsable Providers: {ort_session_A.get_providers()}")

in_name_A0 = inputs[0].name       # "audio"
out_name_A0 = outputs[0].name     # P(complete)
input_numpy_dtype = numpy_dtype_from_onnx_type(inputs[0].type)


def ortvalue_device_type(session):
    primary_provider = session.get_providers()[0]
    if primary_provider in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}:
        return "cuda"
    if primary_provider == "DmlExecutionProvider":
        return "dml"
    return "cpu"


def create_ort_buffer(shape, dtype, device_type):
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.zeros(shape, dtype=dtype),
        device_type,
        DEVICE_ID,
    )


def update_ort_buffer(buffer, array):
    array = np.ascontiguousarray(array)
    if hasattr(buffer, "update_inplace"):
        buffer.update_inplace(array)
    else:
        np.copyto(buffer.numpy(), array)


device_type = ortvalue_device_type(ort_session_A)
input_shape = resolve_tensor_shape(
    inputs[0], {0: 1} if not isinstance(inputs[0].shape[0], int) else {}
)
output_shape = resolve_tensor_shape(
    outputs[0], {0: 1} if not isinstance(outputs[0].shape[0], int) else {}
)
input_buffer = create_ort_buffer(
    input_shape, input_numpy_dtype, device_type)
output_buffer = create_ort_buffer(
    output_shape,
    numpy_dtype_from_onnx_type(outputs[0].type),
    device_type,
)
binding_A = ort_session_A.io_binding()
binding_A.bind_ortvalue_input(in_name_A0, input_buffer)
binding_A.bind_ortvalue_output(out_name_A0, output_buffer)

# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════
def truncate_or_pad_int16(pcm, n_samples):
    """
    Keep the last ``n_samples`` (most recent context window) of a long clip, or
    zero-pad a short clip at the *front* — so the result is always ``n_samples``.
    """
    pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
    if len(pcm) > n_samples:
        return pcm[-n_samples:]
    if len(pcm) < n_samples:
        return np.pad(pcm, (n_samples - len(pcm), 0), mode="constant", constant_values=0)
    return pcm


def prepare_audio_input(pcm_int16):
    """Convert decoded int16 PCM to the waveform dtype declared by the model."""
    pcm_int16 = np.asarray(pcm_int16, dtype=np.int16)
    if input_numpy_dtype == np.int16:
        return pcm_int16
    normalized = pcm_int16.astype(np.float32) * np.float32(1.0 / 32768.0)
    return normalized.astype(input_numpy_dtype, copy=False)


def load_audio_int16(path):
    """Load any audio file as mono int16 PCM at the graph input rate."""
    from pydub import AudioSegment  # optional dependency; only needed to decode files

    audio = (
        AudioSegment.from_file(path)
        .set_channels(1)
        .set_frame_rate(IN_SAMPLE_RATE)
        .set_sample_width(2)  # 16-bit
    )
    return np.array(audio.get_array_of_samples(), dtype=np.int16)


def predict_endpoint(pcm_int16):
    """
    Score one clip with the glued Smart Turn model.

    Args:
        pcm_int16: Mono int16 PCM samples at IN_SAMPLE_RATE (any length).

    Returns:
        dict with ``prediction`` (1 complete / 0 incomplete) and ``probability``.
    """
    pcm_window = truncate_or_pad_int16(pcm_int16, INPUT_AUDIO_LENGTH)
    window = prepare_audio_input(pcm_window).reshape(input_shape)
    update_ort_buffer(input_buffer, window)
    ort_session_A.run_with_iobinding(binding_A, run_options=run_options)
    probability = output_buffer.numpy().flat[0]
    return {
        "prediction": 1 if probability > COMPLETE_THRESHOLD else 0,
        "probability": probability,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Run
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if os.path.isfile(test_audio):
        print(f"\nTest Input Audio: {test_audio}")
        pcm = load_audio_int16(test_audio)
    else:
        print(f"\n(test audio '{test_audio}' not found — scoring a 2 s dummy clip)")
        pcm = (np.random.randn(IN_SAMPLE_RATE * 2) * 4000.0).astype(np.int16)

    start_time = time.perf_counter()
    result = predict_endpoint(pcm)
    elapsed_seconds = time.perf_counter() - start_time
    audio_duration_seconds = INPUT_AUDIO_LENGTH / IN_SAMPLE_RATE
    rtf = elapsed_seconds / audio_duration_seconds

    print(f"\nProbability (complete) : {result['probability']:.4f}")
    print(f"Prediction             : {'Complete' if result['prediction'] else 'Incomplete'}")
    print(f"RTF                    : {rtf:.4f}")
