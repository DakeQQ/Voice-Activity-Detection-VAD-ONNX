import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime
from datetime import timedelta
from pydub import AudioSegment

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "vad_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from vad_onnx_metadata import (
    REQUIRED_VAD_METADATA_KEYS,
    load_runtime_metadata,
    validate_vad_metadata,
)
from Example_Audio import model_audio_path

parent_path = Path(__file__).resolve().parent
onnx_model_A = parent_path / "DFSMN_VAD_ONNX" / "DFSMN_VAD.onnx"
test_near_end_audio = model_audio_path("dfsmn", "near_end")
test_far_end_audio = model_audio_path("dfsmn", "far_end")
save_timestamps_second = parent_path / "timestamps_second.txt"
save_timestamps_indices = parent_path / "timestamps_indices.txt"

# User-adjustable inference settings.
NORMALIZE_AUDIO = False                 # True applies RMS normalization before inference.
NORMALIZE_TARGET_RMS = 4096.0           # Target PCM16 RMS; used only when normalization is enabled.
MAX_DYNAMIC_AUDIO_SECONDS = 360         # Maximum window size for a dynamic-length model.
FUSION_THRESHOLD = 0.3                  # Merge speech segments separated by at most this many seconds.
MIN_SPEECH_DURATION = 0.2               # Drop speech segments shorter than this many seconds.
SPEAKING_SCORE = 0.5                    # Minimum speech score required to leave silence.
SILENCE_SCORE = 0.5                     # Maximum speech score used to return to silence.
LOOK_BACKWARD = 0.3                     # Overlap/future score window, in seconds, used to stabilize decisions.
ORT_ACCELERATE_PROVIDERS = []           # Add installed execution providers in priority order; [] uses ORT defaults.
CPU_DISABLE_MATMUL_ADD_FUSION = True    # Keep True if MatMul+Add fusion is slower on your CPU.
CPU_DISABLE_NCHWC = True                # Keep True if NCHWc layout conversion is slower on your CPU.
CPU_EXTRA_DISABLED_OPTIMIZERS = [       # Optional extra CPU graph optimizers to disable.
    "ConvAddActivationFusion",
    "MatmulTransposeFusion",
]
MAX_THREADS = 4                  # CPU thread count; use 0 to let ORT choose automatically.
DEVICE_ID = 0                    # Accelerator device index when a non-CPU provider is selected.

REQUIRED_DFSMN_METADATA_KEYS = (
    *REQUIRED_VAD_METADATA_KEYS,
    "light_aec_model",
    "output_frame_shift_samples",
    "fbank_window_length_samples",
    "num_audio_inputs",
    "num_outputs",
)


def dfsmn_frame_count(input_samples, input_sample_rate, model_sample_rate,
                      fbank_window_length, fbank_hop_length, light_aec_model):
    model_samples = int(input_samples * model_sample_rate / input_sample_rate)
    if light_aec_model != "SDAEC":
        raise ValueError(
            f"Dynamic DFSMN inference is unsupported for {light_aec_model}."
        )
    aec_hop_length = 160
    aec_frames = (model_samples - 1) // aec_hop_length + 1
    aec_samples = (aec_frames - 1) * aec_hop_length + 1
    if aec_samples < fbank_window_length:
        return 0
    return 1 + (aec_samples - fbank_window_length) // fbank_hop_length


def numpy_dtype_from_onnx_type(type_name):
    input_dtypes = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(int16)": np.int16,
    }
    try:
        return input_dtypes[type_name]
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


def load_mono_int16(path, sample_rate):
    segment = AudioSegment.from_file(path).set_channels(1).set_frame_rate(sample_rate)
    samples = np.asarray(segment.get_array_of_samples())
    if samples.dtype == np.int16:
        return samples
    scale = float(1 << (8 * samples.dtype.itemsize - 1))
    return np.clip(
        samples.astype(np.float32) * (32768.0 / scale),
        -32768.0,
        32767.0,
    ).astype(np.int16)


def normalise_audio(audio: np.ndarray, input_dtype_np, target_rms=None) -> np.ndarray:
    if target_rms is None:
        target_rms = NORMALIZE_TARGET_RMS
    if NORMALIZE_AUDIO:
        prepared = audio.astype(np.float32)
        rms = np.sqrt(np.mean(prepared * prepared, dtype=np.float32), dtype=np.float32)
        if rms > 0.0:
            prepared *= target_rms / (rms + 1e-7)
        np.clip(prepared, -32768.0, 32767.0, out=prepared)
    elif input_dtype_np == np.int16:
        return audio
    else:
        prepared = audio.astype(np.float32)
    if input_dtype_np == np.int16:
        return prepared.astype(np.int16)
    prepared *= np.float32(1.0 / 32768.0)
    return prepared.astype(input_dtype_np, copy=False)


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


if "OpenVINOExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
    provider_options = [
        {
            'device_type': 'CPU',                         # [CPU, NPU, GPU, GPU.0, GPU.1]]
            'precision': 'ACCURACY',                      # [FP32, FP16, ACCURACY]
            'num_of_threads': MAX_THREADS,
            'num_streams': 1,
            'enable_opencl_throttling': True,
            'enable_qdq_optimizer': False,                # Enable it carefully
            'disable_dynamic_shapes': True
        }
    ]
    device_type = 'cpu'
elif "CUDAExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'gpu_mem_limit': 24 * 1024 * 1024 * 1024,     # 24 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',   # ["kNextPowerOfTwo", "kSameAsRequested"]
            'cudnn_conv_algo_search': 'EXHAUSTIVE',       # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]   The MossFormer_SE must using DEFAULT.
            'sdpa_kernel': '2',                           # ["0", "1", "2"]
            'use_tf32': '1',
            'fuse_conv_bias': '1',                        # Set to '0' to avoid potential errors when enabled.
            'cudnn_conv_use_max_workspace': '1',
            'cudnn_conv1d_pad_to_nc1d': '1',
            'tunable_op_enable': '0',
            'tunable_op_tuning_enable': '0',
            'tunable_op_max_tuning_duration_ms': 10,
            'do_copy_in_default_stream': '0',
            'enable_cuda_graph': '0',                     # Set to '0' to avoid potential errors when enabled.
            'prefer_nhwc': '0',
            'enable_skip_layer_norm_strict_mode': '0',
            'use_ep_level_unified_stream': '0',
        }
    ]
    device_type = 'cuda'
elif "DmlExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'performance_preference': 'high_performance',  # [high_performance, default, minimum_power]
            'device_filter': 'npu'                         # [any, npu, gpu]
        }
    ]
    device_type = 'dml'
else:
    # Please config by yourself for others providers.
    device_type = 'cpu'
    provider_options = None


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4                   # Fatal level, it an adjustable value.
session_opts.inter_op_num_threads = MAX_THREADS       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = MAX_THREADS       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True              # True for execute speed; False for less memory usage.
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

ort_session_A = onnxruntime.InferenceSession(
    onnx_model_A,
    sess_options=session_opts,
    providers=ORT_ACCELERATE_PROVIDERS,
    provider_options=provider_options,
    disabled_optimizers=disabled_opts,
)
model_metadata = load_runtime_metadata(
    onnx_model_A,
    lambda model_path: onnxruntime.InferenceSession(
        model_path,
        sess_options=session_opts,
        providers=ORT_ACCELERATE_PROVIDERS,
        provider_options=provider_options,
        disabled_optimizers=disabled_opts,
    ),
    required_keys=REQUIRED_DFSMN_METADATA_KEYS,
)
validate_vad_metadata(
    model_metadata,
    ort_session_A,
    expected_inputs=model_metadata.required_int("num_audio_inputs"),
    expected_outputs=model_metadata.required_int("num_outputs"),
)
IN_SAMPLE_RATE = model_metadata.required_int("input_sample_rate")
MODEL_SAMPLE_RATE = model_metadata.required_int("model_sample_rate")
LIGHT_AEC_MODEL = model_metadata.string("light_aec_model", required=True)
OUTPUT_FRAME_LENGTH = model_metadata.required_int("output_frame_shift_samples")
OUTPUT_FRAME_SHIFT_SECONDS = model_metadata.required_float(
    "output_frame_shift_seconds")
FBANK_WINDOW_LENGTH = model_metadata.required_int("fbank_window_length_samples")
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
input_dtypes = [numpy_dtype_from_onnx_type(model_input.type) for model_input in in_name_A]
if len(input_dtypes) != 2:
    raise TypeError(
        f"Expected two audio input types, got "
        f"{[model_input.type for model_input in in_name_A]}."
    )
output_dtype = numpy_dtype_from_onnx_type(out_name_A[0].type)
in_name_A0 = in_name_A[0].name
in_name_A1 = in_name_A[1].name
out_name_A0 = out_name_A[0].name


print(f"\nTest Input Near_End Audio: {test_near_end_audio}\nTest Input Far_End Audio: {test_far_end_audio}")
near_end_audio = load_mono_int16(test_near_end_audio, IN_SAMPLE_RATE)
far_end_audio = load_mono_int16(test_far_end_audio, IN_SAMPLE_RATE)
min_len = min(len(near_end_audio), len(far_end_audio))
near_end_audio = normalise_audio(near_end_audio[:min_len], input_dtypes[0])
far_end_audio = normalise_audio(far_end_audio[:min_len], input_dtypes[1])

static_input_lengths = {
    model_input.shape[-1]
    for model_input in in_name_A
    if isinstance(model_input.shape[-1], int)
}
if len(static_input_lengths) > 1:
    raise ValueError(
        f"DFSMN audio inputs have incompatible lengths: "
        f"{[model_input.shape for model_input in in_name_A]}."
    )
if static_input_lengths:
    INPUT_AUDIO_LENGTH = static_input_lengths.pop()
else:
    INPUT_AUDIO_LENGTH = min(
        IN_SAMPLE_RATE * MAX_DYNAMIC_AUDIO_SECONDS, min_len)
input_shapes = [
    resolve_tensor_shape(
        model_input,
        {-1: INPUT_AUDIO_LENGTH}
        if not isinstance(model_input.shape[-1], int)
        else {},
    )
    for model_input in in_name_A
]
near_end_audio = near_end_audio.reshape(*input_shapes[0][:-1], -1)
far_end_audio = far_end_audio.reshape(*input_shapes[1][:-1], -1)
output_extents = {}
if not isinstance(out_name_A[0].shape[0], int):
    output_extents[0] = dfsmn_frame_count(
        INPUT_AUDIO_LENGTH,
        IN_SAMPLE_RATE,
        MODEL_SAMPLE_RATE,
        FBANK_WINDOW_LENGTH,
        OUTPUT_FRAME_LENGTH,
        LIGHT_AEC_MODEL,
    )
output_shape = resolve_tensor_shape(out_name_A[0], output_extents)
OUTPUT_FRAME_COUNT = int(np.prod(output_shape))


look_backward = max(1, int(LOOK_BACKWARD / OUTPUT_FRAME_SHIFT_SECONDS))


def align_audio(audio, audio_len):
    retained_frames = OUTPUT_FRAME_COUNT - look_backward
    stride_step = round(
        (
            (retained_frames - 1) * OUTPUT_FRAME_SHIFT_SECONDS
            + FBANK_WINDOW_LENGTH / MODEL_SAMPLE_RATE
        )
        * IN_SAMPLE_RATE
    )
    if stride_step <= 0:
        raise ValueError("LOOK_BACKWARD is too large for the model input length.")
    if audio_len > INPUT_AUDIO_LENGTH:
        num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
        total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
        pad_amount = total_length_needed - audio_len
        padding = [(0, 0)] * audio.ndim
        padding[-1] = (0, pad_amount)
        audio = np.pad(audio, padding)
    elif audio_len < INPUT_AUDIO_LENGTH:
        padding = [(0, 0)] * audio.ndim
        padding[-1] = (0, INPUT_AUDIO_LENGTH - audio_len)
        audio = np.pad(audio, padding)
    aligned_len = audio.shape[-1]
    return audio, aligned_len, stride_step


near_end_audio, _, _ = align_audio(near_end_audio, min_len)
far_end_audio, aligned_len, stride_step = align_audio(far_end_audio, min_len)
inv_audio_len = float(100.0 / aligned_len)


def process_timestamps(timestamps, fusion_threshold, min_duration):
    # Filter out short durations
    filtered_timestamps = [(start, end) for start, end in timestamps if (end - start) >= min_duration]
    del timestamps
    # Fuse and filter timestamps
    fused_timestamps_1st = []
    for start, end in filtered_timestamps:
        # Merge with the previous segment if within the fusion threshold
        if fused_timestamps_1st and (start - fused_timestamps_1st[-1][1] <= fusion_threshold):
            fused_timestamps_1st[-1] = (fused_timestamps_1st[-1][0], end)
        else:
            fused_timestamps_1st.append((start, end))
    del filtered_timestamps
    fused_timestamps_2nd = []
    for start, end in fused_timestamps_1st:
        # Merge with the previous segment if within the fusion threshold
        if fused_timestamps_2nd and (start - fused_timestamps_2nd[-1][1] <= fusion_threshold):
            fused_timestamps_2nd[-1] = (fused_timestamps_2nd[-1][0], end)
        else:
            fused_timestamps_2nd.append((start, end))
    return fused_timestamps_2nd


def vad_to_timestamps(vad_output, frame_duration):
    timestamps = []
    start = None
    # Extract raw timestamps
    for i, silence in enumerate(vad_output):
        if silence:
            if start is not None:   # End of the current speaking segment
                end = i * frame_duration + frame_duration
                timestamps.append((start, end))
                start = None
        else:
            if start is None:       # Start of a new speaking segment
                start = i * frame_duration
    # Handle the case where speech continues until the end
    if start is not None:
        timestamps.append((start, len(vad_output) * frame_duration))
    return timestamps


def format_time(seconds):
    """Convert seconds to VTT time format 'hh:mm:ss.mmm'."""
    td = timedelta(seconds=seconds)
    td_sec = td.total_seconds()
    total_seconds = int(td_sec)
    milliseconds = int((td_sec - total_seconds) * 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


# Start to run DFSMN_VAD
inv_look_backward = float(1.0 / look_backward)
slice_start = 0
slice_end = INPUT_AUDIO_LENGTH
silence = True
saved = []
device_type = ortvalue_device_type(ort_session_A)
near_buffer = create_ort_buffer(
    input_shapes[0], input_dtypes[0], device_type)
far_buffer = create_ort_buffer(
    input_shapes[1], input_dtypes[1], device_type)
output_buffer = create_ort_buffer(
    output_shape, output_dtype, device_type)
binding_A = ort_session_A.io_binding()
binding_A.bind_ortvalue_input(in_name_A0, near_buffer)
binding_A.bind_ortvalue_input(in_name_A1, far_buffer)
binding_A.bind_ortvalue_output(out_name_A0, output_buffer)
print("\nRunning the DFSMN_VAD by ONNX Runtime.\n\nThe VAD result is focused on the Near-End microphone recording.\n")
start_time = time.time()
while slice_end <= aligned_len:
    update_ort_buffer(
        near_buffer, near_end_audio[..., slice_start: slice_end])
    update_ort_buffer(
        far_buffer, far_end_audio[..., slice_start: slice_end])
    ort_session_A.run_with_iobinding(binding_A, run_options=run_options)
    vad_results = output_buffer.numpy().reshape(-1)
    for i in range(len(vad_results) - look_backward):
        if silence:
            if vad_results[i] >= SPEAKING_SCORE:
                activate = 1
                for j in range(1, look_backward):
                    if vad_results[i + j] >= SPEAKING_SCORE:
                        activate += 1
                activate = activate * inv_look_backward
                if activate >= SPEAKING_SCORE:
                    silence = False
                else:
                    silence = True
            else:
                silence = True
        else:
            if vad_results[i] <= SILENCE_SCORE:
                activate = 1
                for j in range(1, look_backward):
                    if vad_results[i + j] <= SILENCE_SCORE:
                        activate += 1
                activate = activate * inv_look_backward
                if activate <= SILENCE_SCORE:
                    silence = False
                else:
                    silence = True
            else:
                silence = False
        saved.append(silence)
    slice_start += stride_step
    slice_end = slice_start + INPUT_AUDIO_LENGTH
    print(f"Complete: {slice_start * inv_audio_len:.3f}%")
for i in range(len(vad_results) - look_backward, len(vad_results)):
    if silence:
        if vad_results[i] >= SPEAKING_SCORE:
            silence = False
        else:
            silence = True
    else:
        if vad_results[i] <= SILENCE_SCORE:
            silence = True
        else:
            silence = False
    saved.append(silence)
end_time = time.time()

# Generate timestamps.
timestamps = vad_to_timestamps(saved, OUTPUT_FRAME_SHIFT_SECONDS)
timestamps = process_timestamps(timestamps, FUSION_THRESHOLD, MIN_SPEECH_DURATION)
print(f"Complete: 100.00%")

# Save the timestamps.
with open(save_timestamps_second, "w", encoding='UTF-8') as file:
    print("\nTimestamps in Second:")
    for start, end in timestamps:
        s_time = format_time(start)
        e_time = format_time(end)
        line = f"{s_time} --> {e_time}\n"
        file.write(line)
        print(line.replace("\n", ""))

with open(save_timestamps_indices, "w", encoding='UTF-8') as file:
    print("\nTimestamps in Indices:")
    for start, end in timestamps:
        line = f"{int(start * IN_SAMPLE_RATE)} --> {int(end * IN_SAMPLE_RATE)}\n"
        file.write(line)
        print(line.replace("\n", ""))

audio_duration_s = min_len / IN_SAMPLE_RATE
rtf = (end_time - start_time) / audio_duration_s if audio_duration_s > 0.0 else 0.0
print(f"\nVAD Process Complete.\n\nRTF: {rtf:.4f}")
