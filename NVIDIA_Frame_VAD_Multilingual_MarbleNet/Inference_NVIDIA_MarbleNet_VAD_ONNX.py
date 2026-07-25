import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime
from pydub import AudioSegment

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "vad_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from vad_onnx_metadata import (
    load_runtime_metadata,
    validate_vad_metadata,
)
from Example_Audio import model_audio_path


script_dir = Path(__file__).resolve().parent
argument_parser = argparse.ArgumentParser(
    description="Run NVIDIA MarbleNet VAD inference with an exported ONNX model."
)
argument_parser.add_argument(
    "model_dir",
    nargs="?",
    type=Path,
    default=script_dir / "NVIDIA_MarbleNet_Optimized",
    help="Directory containing NVIDIA_MarbleNet.onnx.",
)
arguments = argument_parser.parse_args()
onnx_model_A = arguments.model_dir.expanduser().resolve() / "NVIDIA_MarbleNet.onnx"
test_vad_audio = model_audio_path("nvidia_marblenet")
save_timestamps_second = script_dir / "timestamps_second.txt"
save_timestamps_indices = script_dir / "timestamps_indices.txt"

# User-adjustable inference settings.
NORMALIZE_AUDIO = False           # True applies RMS normalization before inference.
NORMALIZE_TARGET_RMS = 4096.0     # Target PCM16 RMS; used only when normalization is enabled.
MAX_DYNAMIC_AUDIO_SECONDS = 3600  # Maximum window size for a dynamic-length model.
SPEAKING_SCORE = 0.5              # Minimum smoothed score classified as speech.
SMOOTH_WINDOW_SIZE = 3            # Number of frames averaged before thresholding.
MIN_SPEECH_FRAME = 10             # Drop speech segments shorter than this many frames.
MAX_SPEECH_FRAME = 1000           # Split speech segments longer than this many frames.
MIN_SILENCE_FRAME = 10            # Silence frames required to close a speech segment.
MERGE_SILENCE_FRAME = 3           # Merge speech segments across shorter silence gaps.
EXTEND_SPEECH_FRAME = 0           # Pad each speech segment by this many frames per side.
ORT_ACCELERATE_PROVIDERS = []     # Add installed execution providers in priority order; [] uses ORT defaults.
CPU_DISABLE_MATMUL_ADD_FUSION = True  # ORT 1.27 wraps rank-3 MatMul+Add in costly Reshape/Gemm/Reshape chains.
CPU_DISABLE_NCHWC = True              # NCHWc reorders regress mean/tail latency on the target i7-1165G7.
CPU_EXTRA_DISABLED_OPTIMIZERS = [     # Individually benchmarked on the same CPU / ORT build.
    "ConvAddActivationFusion",
    "MatmulTransposeFusion",
]
MAX_THREADS = 4                  # OpenVINO option value; ORT session thread counts below remain automatic.
DEVICE_ID = 0                    # Device index used when allocating accelerator OrtValue buffers.


if "OpenVINOExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
    provider_options = [
        {
            'device_type': 'CPU',                         # [CPU, NPU, GPU, GPU.0, GPU.1]]
            'precision': 'ACCURACY',                      # [FP32, FP16, ACCURACY]
            'num_of_threads': MAX_THREADS,
            'num_streams': 1,
            'enable_opencl_throttling': True,
            'enable_qdq_optimizer': False,                # Enable it carefully
            'disable_dynamic_shapes': False
        }
    ]
elif "CUDAExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'gpu_mem_limit': 8 * 1024 * 1024 * 1024,     # 8 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',  # ["kNextPowerOfTwo", "kSameAsRequested"]
            'cudnn_conv_algo_search': 'EXHAUSTIVE',      # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
            'sdpa_kernel': '2',                          # ["0", "1", "2"]
            'use_tf32': '1',
            'fuse_conv_bias': '1',
            'cudnn_conv_use_max_workspace': '1',
            'cudnn_conv1d_pad_to_nc1d': '1',
            'tunable_op_enable': '0',
            'tunable_op_tuning_enable': '0',
            'tunable_op_max_tuning_duration_ms': 10,
            'do_copy_in_default_stream': '0',
            'enable_cuda_graph': '0',                    # Set to '0' to avoid potential errors when enabled.
            'prefer_nhwc': '0',
            'enable_skip_layer_norm_strict_mode': '0',
            'use_ep_level_unified_stream': '0',
        }
    ]
elif "DmlExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
     provider_options = [
         {
             'device_id': DEVICE_ID,
             'performance_preference': 'high_performance',  # [high_performance, default, minimum_power]
             'device_filter': 'any'                         # [any, npu, gpu]
         }
     ]
else:
    # Please config by yourself for others providers.
    provider_options = None


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4  # Fatal level, it an adjustable value.
session_opts.log_verbosity_level = 4  # Fatal level, it an adjustable value.
session_opts.inter_op_num_threads = 0  # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0  # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True  # True for execute speed; False for less memory usage.
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
        disabled_optimizers=disabled_opts,
    )


ort_session_A = make_session(onnx_model_A)
model_metadata = load_runtime_metadata(onnx_model_A, make_session)
validate_vad_metadata(model_metadata, ort_session_A)
runtime_config = SimpleNamespace(
    IN_SAMPLE_RATE=model_metadata.required_int("input_sample_rate"),
    MODEL_SAMPLE_RATE=model_metadata.required_int("model_sample_rate"),
    OUTPUT_FRAME_SHIFT_S=model_metadata.required_float(
        "output_frame_shift_seconds"),
    NORMALIZE_TARGET_RMS=NORMALIZE_TARGET_RMS,
    MAX_DYNAMIC_AUDIO_SECONDS=MAX_DYNAMIC_AUDIO_SECONDS,
    SPEAKING_SCORE=SPEAKING_SCORE,
    SMOOTH_WINDOW_SIZE=SMOOTH_WINDOW_SIZE,
    MIN_SPEECH_FRAME=MIN_SPEECH_FRAME,
    MAX_SPEECH_FRAME=MAX_SPEECH_FRAME,
    MIN_SILENCE_FRAME=MIN_SILENCE_FRAME,
    MERGE_SILENCE_FRAME=MERGE_SILENCE_FRAME,
    EXTEND_SPEECH_FRAME=EXTEND_SPEECH_FRAME,
)

print(f"\nUsable Providers: {ort_session_A.get_providers()}")
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A0 = out_name_A[0].name
out_name_A1 = out_name_A[1].name
out_name_A2 = out_name_A[2].name


def _numpy_dtype_from_meta(meta):
    tensor_dtypes = {
        "tensor(bool)": np.bool_,
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(int16)": np.int16,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
    }
    try:
        return tensor_dtypes[meta.type]
    except KeyError as error:
        raise TypeError(f"Unsupported ONNX tensor type: {meta.type}") from error


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


input_dtype_np = _numpy_dtype_from_meta(in_name_A[0])


def normalise_audio(audio: np.ndarray, input_dtype, target_rms: float) -> np.ndarray:
    if input_dtype == np.int16 and not NORMALIZE_AUDIO:
        return audio

    prepared = audio.astype(np.float32)
    if NORMALIZE_AUDIO:
        rms = np.sqrt(np.mean(prepared * prepared, dtype=np.float32), dtype=np.float32)
        if rms > 0.0:
            prepared *= target_rms / (rms + 1e-7)
        np.clip(prepared, -32768.0, 32767.0, out=prepared)

    if input_dtype == np.int16:
        return prepared.astype(np.int16)

    prepared *= np.float32(1.0 / 32768.0)
    return prepared.astype(input_dtype, copy=False)


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


# # Load the input audio
print(f"\nTest Input Audio: {test_vad_audio}")
audio = np.array(AudioSegment.from_file(test_vad_audio).set_channels(1).set_frame_rate(runtime_config.IN_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
audio = normalise_audio(
    audio,
    input_dtype_np,
    runtime_config.NORMALIZE_TARGET_RMS,
)
audio_len = len(audio)

shape_value_in = in_name_A[0].shape[-1]
if not isinstance(shape_value_in, int):
    INPUT_AUDIO_LENGTH = min(
        runtime_config.IN_SAMPLE_RATE * runtime_config.MAX_DYNAMIC_AUDIO_SECONDS,
        audio_len,
    )
else:
    INPUT_AUDIO_LENGTH = shape_value_in
input_shape = resolve_tensor_shape(
    in_name_A[0],
    {-1: INPUT_AUDIO_LENGTH}
    if not isinstance(in_name_A[0].shape[-1], int)
    else {},
)
audio = audio.reshape(*input_shape[:-1], -1)
stride_step = INPUT_AUDIO_LENGTH
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
    audio = np.pad(
        audio,
        padding,
    )
aligned_len = audio.shape[-1]

output_frame_shift = round(
    runtime_config.OUTPUT_FRAME_SHIFT_S * runtime_config.MODEL_SAMPLE_RATE)
model_audio_length = int(
    INPUT_AUDIO_LENGTH
    * runtime_config.MODEL_SAMPLE_RATE
    / runtime_config.IN_SAMPLE_RATE
)
score_frames = model_audio_length // output_frame_shift + 1
device_type = ortvalue_device_type(ort_session_A)
input_buffer = create_ort_buffer(
    input_shape, input_dtype_np, device_type)
score_silence_buffer = create_ort_buffer(
    resolve_tensor_shape(
        out_name_A[0],
        {1: score_frames}
        if not isinstance(out_name_A[0].shape[1], int)
        else {},
    ),
    _numpy_dtype_from_meta(out_name_A[0]),
    device_type,
)
score_active_buffer = create_ort_buffer(
    resolve_tensor_shape(
        out_name_A[1],
        {1: score_frames}
        if not isinstance(out_name_A[1].shape[1], int)
        else {},
    ),
    _numpy_dtype_from_meta(out_name_A[1]),
    device_type,
)
signal_len_buffer = create_ort_buffer(
    resolve_tensor_shape(out_name_A[2]),
    _numpy_dtype_from_meta(out_name_A[2]),
    device_type,
)
binding_A = ort_session_A.io_binding()
binding_A.bind_ortvalue_input(in_name_A0, input_buffer)
binding_A.bind_ortvalue_output(out_name_A0, score_silence_buffer)
binding_A.bind_ortvalue_output(out_name_A1, score_active_buffer)
binding_A.bind_ortvalue_output(out_name_A2, signal_len_buffer)


# ═══════════════════════════════════════════════════════════════════════════════
# VAD Post-processing (from FireRedVAD, adapted for NVIDIA MarbleNet 20ms frames)
# ═══════════════════════════════════════════════════════════════════════════════

_VAD_SILENCE = 0
_VAD_POSSIBLE_SPEECH = 1
_VAD_SPEECH = 2
_VAD_POSSIBLE_SILENCE = 3


class VadPostprocessor:
    __slots__ = ('smooth_window_size', 'prob_threshold', 'min_speech_frame',
                 'max_speech_frame', 'min_silence_frame', 'merge_silence_frame',
                 'extend_speech_frame', 'frame_shift_s', '_inv_ws', '_half_max')

    def __init__(self, smooth_window_size, prob_threshold, min_speech_frame,
                 max_speech_frame, min_silence_frame, merge_silence_frame,
                 extend_speech_frame, frame_shift_s):
        self.smooth_window_size = max(1, smooth_window_size)
        self.prob_threshold = np.float32(prob_threshold)
        self.min_speech_frame = min_speech_frame
        self.max_speech_frame = max_speech_frame
        self.min_silence_frame = min_silence_frame
        self.merge_silence_frame = merge_silence_frame
        self.extend_speech_frame = extend_speech_frame
        self.frame_shift_s = np.float32(frame_shift_s)
        self._inv_ws = np.float32(1.0 / self.smooth_window_size)
        self._half_max = max_speech_frame >> 1

    def process(self, raw_probs):
        """Process raw probabilities into binary speech decisions (numpy int8 array)."""
        if isinstance(raw_probs, np.ndarray):
            n = raw_probs.shape[0]
            if n == 0:
                return np.empty(0, dtype=np.int8)
            probs = raw_probs.astype(np.float32, copy=False)
        else:
            n = len(raw_probs)
            if n == 0:
                return np.empty(0, dtype=np.int8)
            probs = np.asarray(raw_probs, dtype=np.float32)

        decisions = self._smooth_threshold_state_machine(probs, n)
        self._fix_starts_inplace(decisions, n)
        if self.merge_silence_frame > 0:
            self._merge_silence_inplace(decisions, n)
        if self.extend_speech_frame > 0:
            self._extend_inplace(decisions, n)
        self._split_long_inplace(decisions, probs, n)
        return decisions

    def decision_to_segment(self, decisions, wav_dur=None):
        """Extract (start_sec, end_sec) segments from binary decision array."""
        if isinstance(decisions, np.ndarray):
            dec = decisions
            n = dec.shape[0]
        else:
            dec = np.asarray(decisions, dtype=np.int8)
            n = dec.shape[0]
        if n == 0:
            return []

        padded = np.empty(n + 2, dtype=np.int8)
        padded[0] = 0
        padded[n + 1] = 0
        padded[1:n + 1] = dec
        diff = np.diff(padded)
        starts = np.flatnonzero(diff == 1).astype(np.float32)
        ends = np.flatnonzero(diff == -1).astype(np.float32)

        num_segs = starts.shape[0]
        if num_segs == 0:
            return []

        segments = np.empty((num_segs, 2), dtype=np.float32)
        segments[:, 0] = starts * self.frame_shift_s
        segments[:, 1] = ends * self.frame_shift_s

        if dec[n - 1] != 0:
            end_time = n * self.frame_shift_s
            if wav_dur is not None and wav_dur < end_time:
                end_time = wav_dur
            segments[-1, 1] = end_time

        return [(round(s, 3), round(e, 3)) for s, e in segments.tolist()]

    def _smooth_threshold_state_machine(self, probs, n):
        decisions = np.zeros(n, dtype=np.int8)
        ws = self.smooth_window_size
        threshold = self.prob_threshold
        min_sp = self.min_speech_frame
        min_si = self.min_silence_frame

        if ws > 1:
            cumsum = np.empty(n + 1, dtype=np.float32)
            cumsum[0] = 0.0
            np.cumsum(probs, out=cumsum[1:])
            smoothed = np.empty(n, dtype=np.float32)
            edge_end = min(ws - 1, n)
            for i in range(edge_end):
                smoothed[i] = cumsum[i + 1] / (i + 1)
            if n >= ws:
                smoothed[ws - 1:] = (cumsum[ws:] - cumsum[:n - ws + 1]) * self._inv_ws
        else:
            smoothed = probs

        if min_sp <= 0 and min_si <= 0:
            decisions[:] = (smoothed >= threshold)
        else:
            state = _VAD_SILENCE
            speech_start = 0
            for t in range(n):
                is_speech = smoothed[t] >= threshold
                if state == _VAD_SILENCE:
                    if is_speech:
                        state = _VAD_POSSIBLE_SPEECH
                        speech_start = t
                elif state == _VAD_POSSIBLE_SPEECH:
                    if is_speech:
                        if t - speech_start >= min_sp:
                            state = _VAD_SPEECH
                            decisions[speech_start:t] = 1
                    else:
                        state = _VAD_SILENCE
                elif state == _VAD_SPEECH:
                    if not is_speech:
                        state = _VAD_POSSIBLE_SILENCE
                        silence_start = t
                else:  # _VAD_POSSIBLE_SILENCE
                    if not is_speech:
                        if t - silence_start >= min_si:
                            state = _VAD_SILENCE
                    else:
                        state = _VAD_SPEECH
                decisions[t] = 1 if state >= _VAD_SPEECH else 0

        return decisions

    def _fix_starts_inplace(self, decisions, n):
        ws = self.smooth_window_size
        if ws <= 1:
            return
        for t in range(1, n):
            if decisions[t] == 1 and decisions[t - 1] == 0:
                start = t - ws if t >= ws else 0
                decisions[start:t] = 1

    def _merge_silence_inplace(self, decisions, n):
        merge_thr = self.merge_silence_frame
        silence_start = -1
        for t in range(1, n):
            prev = decisions[t - 1]
            curr = decisions[t]
            if prev == 1 and curr == 0 and silence_start < 0:
                silence_start = t
            elif prev == 0 and curr == 1 and silence_start >= 0:
                if t - silence_start < merge_thr:
                    decisions[silence_start:t] = 1
                silence_start = -1

    def _extend_inplace(self, decisions, n):
        ext = self.extend_speech_frame
        dist = ext + 1
        for t in range(n):
            if decisions[t]:
                dist = 0
            else:
                dist += 1
                if dist <= ext:
                    decisions[t] = 1
        dist = ext + 1
        for t in range(n - 1, -1, -1):
            if decisions[t]:
                dist = 0
            else:
                dist += 1
                if dist <= ext:
                    decisions[t] = 1

    def _split_long_inplace(self, decisions, probs, n):
        max_sf = self.max_speech_frame
        half_max = self._half_max
        t = 0
        while t < n:
            if decisions[t]:
                seg_start = t
                while t < n and decisions[t]:
                    t += 1
                dur = t - seg_start
                if dur > max_sf:
                    pos = seg_start
                    seg_end = t
                    while pos + max_sf < seg_end:
                        w_start = pos + half_max
                        w_end = pos + max_sf
                        if w_end > seg_end:
                            w_end = seg_end
                        if w_start >= w_end:
                            break
                        min_idx = w_start + int(np.argmin(probs[w_start:w_end]))
                        decisions[min_idx] = 0
                        pos = min_idx + 1
            else:
                t += 1


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


# Start to run NVIDIA_VAD
slice_start = 0
slice_end = INPUT_AUDIO_LENGTH
all_vad_probs = []
print("\nRunning the NVIDIA_VAD by ONNX Runtime.")
start_time = time.time()
while slice_end <= aligned_len:
    update_ort_buffer(input_buffer, audio[..., slice_start:slice_end])
    ort_session_A.run_with_iobinding(binding_A, run_options=run_options)
    active_scores = score_active_buffer.numpy().reshape(-1)
    valid_frames = min(
        int(signal_len_buffer.numpy().reshape(-1)[0]),
        active_scores.size,
    )
    all_vad_probs.append(np.array(active_scores[:valid_frames], copy=True))
    slice_start += stride_step
    slice_end = slice_start + INPUT_AUDIO_LENGTH
end_time = time.time()

# Concatenate all speech probabilities
if all_vad_probs:
    all_vad_probs = np.concatenate(all_vad_probs, axis=0)
else:
    all_vad_probs = np.zeros((0,), dtype=np.float32)

# Post-process using VadPostprocessor (same approach as FireRedVAD)
vad_postprocessor = VadPostprocessor(
    smooth_window_size=runtime_config.SMOOTH_WINDOW_SIZE,
    prob_threshold=runtime_config.SPEAKING_SCORE,
    min_speech_frame=runtime_config.MIN_SPEECH_FRAME,
    max_speech_frame=runtime_config.MAX_SPEECH_FRAME,
    min_silence_frame=runtime_config.MIN_SILENCE_FRAME,
    merge_silence_frame=runtime_config.MERGE_SILENCE_FRAME,
    extend_speech_frame=runtime_config.EXTEND_SPEECH_FRAME,
    frame_shift_s=runtime_config.OUTPUT_FRAME_SHIFT_S,
)
vad_decisions = vad_postprocessor.process(all_vad_probs)
timestamps = vad_postprocessor.decision_to_segment(
    vad_decisions,
    audio_len / runtime_config.IN_SAMPLE_RATE,
)
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
        line = (
            f"{int(start * runtime_config.IN_SAMPLE_RATE)} --> "
            f"{int(end * runtime_config.IN_SAMPLE_RATE)}\n"
        )
        file.write(line)
        print(line.replace("\n", ""))

    audio_duration_s = audio_len / runtime_config.IN_SAMPLE_RATE
    rtf = (end_time - start_time) / audio_duration_s if audio_duration_s > 0.0 else 0.0
    print(f"\nVAD Process Complete.\n\nRTF: {rtf:.4f}")
