import argparse
import sys
import time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime
from datetime import timedelta
from pydub import AudioSegment

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "vad_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from vad_onnx_metadata import (
    MetadataReader,
    REQUIRED_VAD_METADATA_KEYS,
    validate_vad_metadata,
)
from Example_Audio import model_audio_path


# ─── Model Paths ──────────────────────────────────────────────────────────────
argument_parser = argparse.ArgumentParser(
    description='Run FireRedVAD inference with exported ONNX models.'
)
argument_parser.add_argument(
    'model_dir',
    nargs='?',
    type=Path,
    default=Path(__file__).resolve().parent / "FireRedVAD_Optimized",
    help='Directory containing FireRedVAD.onnx, FireRedAED.onnx, and FireRedStreamVAD.onnx.',
)
arguments = argument_parser.parse_args()
onnx_model_dir          = arguments.model_dir.expanduser().resolve()
onnx_model_vad          = str(onnx_model_dir / "FireRedVAD.onnx")            # The exported VAD onnx model path.
onnx_model_aed          = str(onnx_model_dir / "FireRedAED.onnx")            # The exported AED onnx model path.
onnx_model_stream_vad   = str(onnx_model_dir / "FireRedStreamVAD.onnx")      # The exported Stream-VAD onnx model path.
test_vad_audio          = model_audio_path("firered_vad")                                          # The VAD test audio path.
test_aed_audio          = model_audio_path("firered_aed")                                          # The AED test audio path.
test_stream_vad_audio   = model_audio_path("firered_stream_vad")                                   # The Stream-VAD test audio path.
save_timestamps_second  = "./timestamps_second.txt"                                                # The saved path.
save_timestamps_indices = "./timestamps_indices.txt"                                               # The saved path.


# ─── Inference Settings ───────────────────────────────────────────────────────
RUN_VAD = True                          # Run the VAD model inference.
RUN_AED = True                          # Run the AED model inference.
RUN_STREAM_VAD = True                   # Run the Stream-VAD model inference.
NORMALIZE_AUDIO = False                 # True applies RMS normalization before inference.
NORMALIZE_TARGET_RMS = 4096.0           # Target PCM16 RMS; used only when normalization is enabled.
MAX_DYNAMIC_AUDIO_SECONDS = 3600        # Maximum window size for a dynamic-length offline model.
STREAM_CHUNK_MS = 160                   # Audio processed per streaming call, in milliseconds.
STREAM_PAD_START_FRAME = 5              # History frames used to warm up streaming decisions.
EVENT_THRESHOLDS = {
    'speech': 0.4,                       # Minimum probability for a speech event.
    'singing': 0.5,                      # Minimum probability for a singing event.
    'music': 0.5,                        # Minimum probability for a music event.
}
COMMON_POSTPROCESS_CONFIG = {
    'SMOOTH_WINDOW_SIZE': 5,             # Frames averaged before thresholding.
    'MAX_SPEECH_FRAME': 2000,            # Split speech segments longer than this many frames.
    'MIN_SILENCE_FRAME': 20,             # Silence frames required to close a segment.
    'EXTEND_SPEECH_FRAME': 0,            # Segment padding in frames per side; 0 disables it.
}
POSTPROCESS_CONFIGS = {
    'vad': {
        **COMMON_POSTPROCESS_CONFIG,
        'SPEAKING_SCORE': 0.4,            # Minimum smoothed score classified as speech.
        'MIN_SPEECH_FRAME': 20,           # Drop shorter speech segments.
        'MERGE_SILENCE_FRAME': 5,         # Merge across shorter silence gaps.
    },
    'aed': {
        **COMMON_POSTPROCESS_CONFIG,
        'SPEAKING_SCORE': 0.4,            # Minimum smoothed event score.
        'MIN_SPEECH_FRAME': 20,           # Drop shorter event segments.
        'MERGE_SILENCE_FRAME': 5,         # Merge across shorter inactive gaps.
    },
    'stream_vad': {
        **COMMON_POSTPROCESS_CONFIG,
        'SPEAKING_SCORE': 0.4,            # Minimum smoothed score classified as speech.
        'MIN_SPEECH_FRAME': 8,            # Drop shorter streaming speech segments.
        'MERGE_SILENCE_FRAME': 0,         # Silence-gap merging; 0 disables it.
    },
}
ORT_ACCELERATE_PROVIDERS = []           # e.g. ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'MIGraphXExecutionProvider']
CPU_DISABLE_MATMUL_ADD_FUSION = True  # ORT 1.27 wraps rank-3 MatMul+Add in costly Reshape/Gemm/Reshape chains.
CPU_DISABLE_NCHWC = True              # NCHWc reorders regress mean/tail latency on the target i7-1165G7.
CPU_EXTRA_DISABLED_OPTIMIZERS = [     # Individually benchmarked on the same CPU / ORT build.
    "ConvAddActivationFusion",
    "MatmulTransposeFusion",
]
DEVICE_ID = 0                           # Accelerator device index for the selected provider.

if any(provider in ORT_ACCELERATE_PROVIDERS for provider in (
        'CUDAExecutionProvider', 'TensorrtExecutionProvider')):
    onnxruntime.preload_dlls(cuda=False, cudnn=True, directory="")

if any(provider in ORT_ACCELERATE_PROVIDERS for provider in (
        'CUDAExecutionProvider', 'TensorrtExecutionProvider')):
    ORT_BUFFER_DEVICE = 'cuda'
elif 'DmlExecutionProvider' in ORT_ACCELERATE_PROVIDERS:
    ORT_BUFFER_DEVICE = 'dml'
else:
    ORT_BUFFER_DEVICE = 'cpu'


# ═══════════════════════════════════════════════════════════════════════════════
# ONNX Runtime Inference Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def numpy_dtype_from_onnx_type(type_name):
    tensor_dtypes = {
        'tensor(bool)': np.bool_,
        'tensor(float16)': np.float16,
        'tensor(float)': np.float32,
        'tensor(int16)': np.int16,
        'tensor(int32)': np.int32,
        'tensor(int64)': np.int64,
    }
    try:
        return tensor_dtypes[type_name]
    except KeyError as error:
        raise TypeError(f'Unsupported ONNX tensor type: {type_name}') from error


def ort_zeros(shape, dtype):
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.zeros(shape, dtype=dtype),
        ORT_BUFFER_DEVICE,
        DEVICE_ID,
    )


def update_ortvalue(ort_value, array):
    array = np.ascontiguousarray(array)
    if hasattr(ort_value, 'update_inplace'):
        ort_value.update_inplace(array)
    else:
        np.copyto(ort_value.numpy(), array)


def run_iobinding(session, binding):
    session.run_with_iobinding(binding, run_options=run_options)


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
            f'Tensor {node_arg.name!r} has dynamic axes {sorted(dynamic_axes)} '
            f'in shape {node_arg.shape}, but extents were supplied for '
            f'{sorted(normalized_extents)}.'
        )
    return tuple(
        dimension if isinstance(dimension, int) else normalized_extents[axis]
        for axis, dimension in enumerate(node_arg.shape)
    )


def normalise_audio(audio: np.ndarray, input_dtype, target_rms) -> np.ndarray:
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


def valid_frame_count(num_samples, input_sample_rate, model_sample_rate,
                      window_length, hop_length):
    """Return the number of valid frames for snip_edges=True fbank extraction."""
    resampled_samples = int(num_samples * model_sample_rate / input_sample_rate)
    if resampled_samples < window_length:
        return 0
    return 1 + (resampled_samples - window_length) // hop_length


# ═══════════════════════════════════════════════════════════════════════════════
# VAD Post-processing (Optimized: pre-allocated numpy, fused ops, C++ portable)
# ═══════════════════════════════════════════════════════════════════════════════

_VAD_SILENCE = 0
_VAD_POSSIBLE_SPEECH = 1
_VAD_SPEECH = 2
_VAD_POSSIBLE_SILENCE = 3


class VadPostprocessor:
    __slots__ = ('smooth_window_size', 'prob_threshold', 'min_speech_frame',
                 'max_speech_frame', 'min_silence_frame', 'merge_silence_frame',
                 'extend_speech_frame', 'frame_shift', 'frame_length',
                 '_inv_ws', '_half_max')

    def __init__(self, smooth_window_size, prob_threshold, min_speech_frame,
                 max_speech_frame, min_silence_frame, merge_silence_frame,
                 extend_speech_frame, frame_shift, frame_length):
        self.smooth_window_size = max(1, smooth_window_size)
        self.prob_threshold = np.float32(prob_threshold)
        self.min_speech_frame = min_speech_frame
        self.max_speech_frame = max_speech_frame
        self.min_silence_frame = min_silence_frame
        self.merge_silence_frame = merge_silence_frame
        self.extend_speech_frame = extend_speech_frame
        self.frame_shift = np.float32(frame_shift)
        self.frame_length = np.float32(frame_length)
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
        segments[:, 0] = starts * self.frame_shift
        segments[:, 1] = ends * self.frame_shift

        if dec[n - 1] != 0:
            end_time = n * self.frame_shift + self.frame_length
            if wav_dur is not None and wav_dur < end_time:
                end_time = wav_dur
            segments[-1, 1] = end_time

        return [(round(s, 3), round(e, 3)) for s, e in segments.tolist()]

    def _smooth_threshold_state_machine(self, probs, n):
        """Fused: cumsum-based moving average -> threshold -> state machine -> int8 array."""
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
        """Expand each rising edge leftward by smooth_window_size (in-place)."""
        ws = self.smooth_window_size
        if ws <= 1:
            return
        for t in range(1, n):
            if decisions[t] == 1 and decisions[t - 1] == 0:
                start = t - ws if t >= ws else 0
                decisions[start:t] = 1

    def _merge_silence_inplace(self, decisions, n):
        """Fill silence gaps shorter than merge_silence_frame (in-place)."""
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
        """Morphological dilation: two-pass O(n), no allocation."""
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
        """Split speech segments exceeding max_speech_frame at lowest-prob points."""
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


class StreamVadPostprocessor:
    """Streaming VAD postprocessor with circular buffer smoothing."""
    __slots__ = ('smooth_window_size', 'speech_threshold', 'pad_start_frame',
                 'min_speech_frame', 'max_speech_frame', 'min_silence_frame',
                 '_window_buf', '_window_sum', '_window_pos', '_window_count',
                 'frame_cnt', 'state', 'speech_cnt', 'silence_cnt',
                 'hit_max_speech', 'last_speech_start_frame', 'last_speech_end_frame',
                 'frame_shift')

    def __init__(self, smooth_window_size, speech_threshold, pad_start_frame,
                 min_speech_frame, max_speech_frame, min_silence_frame,
                 frame_shift):
        self.smooth_window_size = max(1, smooth_window_size)
        self.speech_threshold = np.float32(speech_threshold)
        self.pad_start_frame = max(self.smooth_window_size, pad_start_frame)
        self.min_speech_frame = min_speech_frame
        self.max_speech_frame = max_speech_frame
        self.min_silence_frame = min_silence_frame
        self._window_buf = np.zeros(self.smooth_window_size, dtype=np.float32)
        self._window_sum = np.float32(0.0)
        self._window_pos = 0
        self._window_count = 0
        self.frame_cnt = 0
        self.state = _VAD_SILENCE
        self.speech_cnt = 0
        self.silence_cnt = 0
        self.hit_max_speech = False
        self.last_speech_start_frame = -1
        self.last_speech_end_frame = -1
        self.frame_shift = frame_shift

    def reset(self):
        self.frame_cnt = 0
        self._window_buf[:] = 0.0
        self._window_sum = np.float32(0.0)
        self._window_pos = 0
        self._window_count = 0
        self.state = _VAD_SILENCE
        self.speech_cnt = 0
        self.silence_cnt = 0
        self.hit_max_speech = False
        self.last_speech_start_frame = -1
        self.last_speech_end_frame = -1

    def process_batch(self, raw_probs):
        """Process all frame probabilities and return timestamps directly."""
        if isinstance(raw_probs, np.ndarray):
            probs = raw_probs
            n = probs.shape[0]
        else:
            probs = np.asarray(raw_probs, dtype=np.float32)
            n = probs.shape[0]
        if n == 0:
            return []

        frame_shift = self.frame_shift
        timestamps = []
        ws = self.smooth_window_size
        threshold = self.speech_threshold
        min_sp = self.min_speech_frame
        max_sp = self.max_speech_frame
        min_si = self.min_silence_frame
        pad_start = self.pad_start_frame

        buf = self._window_buf
        buf_sum = self._window_sum
        buf_pos = self._window_pos
        buf_count = self._window_count
        frame_cnt = self.frame_cnt
        state = self.state
        speech_cnt = self.speech_cnt
        silence_cnt = self.silence_cnt
        hit_max = self.hit_max_speech
        last_start = self.last_speech_start_frame
        last_end = self.last_speech_end_frame

        for t in range(n):
            raw_p = probs[t]
            frame_cnt += 1

            if ws <= 1:
                smoothed = raw_p
            else:
                old_val = buf[buf_pos]
                buf[buf_pos] = raw_p
                buf_sum += raw_p - old_val
                buf_pos = (buf_pos + 1) % ws
                if buf_count < ws:
                    buf_count += 1
                smoothed = buf_sum / buf_count

            is_speech = 1 if smoothed >= threshold else 0

            seg_start_out = -1
            seg_end_out = -1

            if hit_max:
                seg_start_out = frame_cnt
                last_start = frame_cnt
                hit_max = False

            if state == _VAD_SILENCE:
                if is_speech:
                    state = _VAD_POSSIBLE_SPEECH
                    speech_cnt = 1
                else:
                    silence_cnt += 1
                    speech_cnt = 0

            elif state == _VAD_POSSIBLE_SPEECH:
                if is_speech:
                    speech_cnt += 1
                    if speech_cnt >= min_sp:
                        state = _VAD_SPEECH
                        seg_start_out = max(
                            1,
                            frame_cnt - speech_cnt + 1 - pad_start,
                            last_end + 1
                        )
                        last_start = seg_start_out
                        silence_cnt = 0
                else:
                    state = _VAD_SILENCE
                    silence_cnt = 1
                    speech_cnt = 0

            elif state == _VAD_SPEECH:
                speech_cnt += 1
                if is_speech:
                    silence_cnt = 0
                    if speech_cnt >= max_sp:
                        hit_max = True
                        speech_cnt = 0
                        seg_end_out = frame_cnt
                        seg_start_out = last_start
                        last_start = -1
                        last_end = frame_cnt
                else:
                    state = _VAD_POSSIBLE_SILENCE
                    silence_cnt = 1

            else:  # _VAD_POSSIBLE_SILENCE
                speech_cnt += 1
                if is_speech:
                    state = _VAD_SPEECH
                    silence_cnt = 0
                    if speech_cnt >= max_sp:
                        hit_max = True
                        speech_cnt = 0
                        seg_end_out = frame_cnt
                        seg_start_out = last_start
                        last_start = -1
                        last_end = frame_cnt
                else:
                    silence_cnt += 1
                    if silence_cnt >= min_si:
                        state = _VAD_SILENCE
                        seg_end_out = frame_cnt
                        seg_start_out = last_start
                        last_end = frame_cnt
                        last_start = -1
                        speech_cnt = 0

            if seg_end_out > 0 and seg_start_out > 0:
                s_frame = max(0, seg_start_out - 1)
                e_frame = max(0, seg_end_out - 1)
                timestamps.append((s_frame * frame_shift, e_frame * frame_shift))

        # Handle unterminated segment at end of stream
        if last_start > 0:
            s_frame = max(0, last_start - 1)
            e_frame = frame_cnt - 1
            timestamps.append((s_frame * frame_shift, e_frame * frame_shift))

        # Write back state for continued streaming
        self._window_sum = buf_sum
        self._window_pos = buf_pos
        self._window_count = buf_count
        self.frame_cnt = frame_cnt
        self.state = state
        self.speech_cnt = speech_cnt
        self.silence_cnt = silence_cnt
        self.hit_max_speech = hit_max
        self.last_speech_start_frame = last_start
        self.last_speech_end_frame = last_end

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


# ═══════════════════════════════════════════════════════════════════════════════
# Run VAD Inference with ONNX Runtime
# ═══════════════════════════════════════════════════════════════════════════════

# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4
session_opts.log_verbosity_level = 4
session_opts.inter_op_num_threads = 0
session_opts.intra_op_num_threads = 0
session_opts.enable_cpu_mem_arena = True
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")

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
if "CUDAExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
    disabled_opts.extend(("ConvAddFusion", "ConvAddActivationFusion"))
disabled_opts = disabled_opts or None


_cuda_model_cache = {}


def cuda_compatible_model(model_path):
    if "CUDAExecutionProvider" not in ORT_ACCELERATE_PROVIDERS:
        return str(model_path)

    resolved_path = str(Path(model_path).resolve())
    cached_model = _cuda_model_cache.get(resolved_path)
    if cached_model is not None:
        return cached_model

    model = onnx.load(resolved_path, load_external_data=True)
    tensor_names = {
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
        if name
    }
    node_names = {node.name for node in model.graph.node if node.name}

    def unique_name(existing_names, base_name):
        candidate = base_name
        suffix = 2
        while candidate in existing_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        existing_names.add(candidate)
        return candidate

    rewritten_nodes = []
    for node in model.graph.node:
        if node.domain != "com.microsoft" or node.op_type != "FusedConv":
            rewritten_nodes.append(node)
            continue

        node_attributes = {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        activation = node_attributes.pop("activation", None)
        activation_params = node_attributes.pop("activation_params", None)
        if isinstance(activation, bytes):
            activation = activation.decode("utf-8")
        if activation not in (None, "Relu", "Sigmoid") or activation_params:
            raise ValueError(
                f"Unsupported CUDA FusedConv activation at {node.name!r}: "
                f"{activation!r}, params={activation_params!r}"
            )

        residual_input = node.input[3] if len(node.input) > 3 else ""
        has_post_op = bool(residual_input or activation)
        conv_output = (
            unique_name(tensor_names, f"{node.output[0]}_cuda_conv")
            if has_post_op
            else node.output[0]
        )
        conv_inputs = list(node.input[:3])
        while conv_inputs and not conv_inputs[-1]:
            conv_inputs.pop()
        rewritten_nodes.append(
            onnx.helper.make_node(
                "Conv",
                conv_inputs,
                [conv_output],
                name=unique_name(node_names, f"{node.name}_CUDAConv"),
                **node_attributes,
            )
        )

        post_output = conv_output
        if residual_input:
            add_output = (
                unique_name(tensor_names, f"{node.output[0]}_cuda_add")
                if activation
                else node.output[0]
            )
            rewritten_nodes.append(
                onnx.helper.make_node(
                    "Add",
                    [post_output, residual_input],
                    [add_output],
                    name=unique_name(node_names, f"{node.name}_CUDAAdd"),
                )
            )
            post_output = add_output
        if activation:
            rewritten_nodes.append(
                onnx.helper.make_node(
                    activation,
                    [post_output],
                    list(node.output),
                    name=unique_name(node_names, f"{node.name}_CUDA{activation}"),
                )
            )

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    onnx.checker.check_model(model, check_custom_domain=False)
    serialized_model = model.SerializeToString()
    _cuda_model_cache[resolved_path] = serialized_model
    return serialized_model


def make_session(model_path):
    return onnxruntime.InferenceSession(
        cuda_compatible_model(model_path),
        sess_options=session_opts,
        providers=ORT_ACCELERATE_PROVIDERS,
        disabled_optimizers=disabled_opts,
    )


def load_firered_runtime(
    model_path, expected_task, expected_inputs, expected_outputs,
    postprocess_config):
    session = make_session(model_path)
    metadata = MetadataReader(session.get_modelmeta().custom_metadata_map)
    for key in REQUIRED_VAD_METADATA_KEYS:
        metadata.string(key, required=True)
    validate_vad_metadata(
        metadata,
        session,
        expected_inputs=expected_inputs,
        expected_outputs=expected_outputs,
    )
    actual_task = metadata.string('task', required=True)
    if actual_task != expected_task:
        raise ValueError(
            f'Expected metadata task {expected_task!r}, got {actual_task!r} for {model_path}.'
        )
    config = {
        'IN_SAMPLE_RATE': metadata.required_int('input_sample_rate'),
        'MODEL_SAMPLE_RATE': metadata.required_int('model_sample_rate'),
        'MAX_DYNAMIC_AUDIO_SECONDS': MAX_DYNAMIC_AUDIO_SECONDS,
        'NORMALIZE_TARGET_RMS': NORMALIZE_TARGET_RMS,
        'OUTPUT_FRAME_SHIFT_S': metadata.required_float('output_frame_shift_seconds'),
        'OUTPUT_FRAME_LENGTH_S': metadata.required_float('output_frame_length_seconds'),
        'WINDOW_LENGTH': metadata.required_int('window_length_samples'),
        'HOP_LENGTH': metadata.required_int('hop_length_samples'),
        **postprocess_config,
    }
    return session, metadata, config


if RUN_VAD:
    print('\n\nStart to run FireRedVAD by ONNX Runtime.\n\nNow, loading the model...')

    ort_session_A, model_metadata_A, config_A = load_firered_runtime(
        onnx_model_vad,
        expected_task='voice_activity_detection',
        expected_inputs=1,
        expected_outputs=1,
        postprocess_config=POSTPROCESS_CONFIGS['vad'],
    )
    print(f"\nUsable Providers: {ort_session_A.get_providers()}")
    in_name_A = ort_session_A.get_inputs()
    out_name_A = ort_session_A.get_outputs()
    in_name_A0 = in_name_A[0].name
    out_name_A0 = out_name_A[0].name
    input_dtype_A = numpy_dtype_from_onnx_type(in_name_A[0].type)

    # Load the input audio
    print(f"\nTest Input Audio: {test_vad_audio}")
    audio = np.array(AudioSegment.from_file(test_vad_audio).set_channels(1).set_frame_rate(config_A['IN_SAMPLE_RATE']).get_array_of_samples(), dtype=np.int16)
    audio = normalise_audio(
        audio,
        input_dtype_A,
        config_A['NORMALIZE_TARGET_RMS'],
    )
    audio_len = len(audio)
    shape_value_in = in_name_A[0].shape[-1]
    if not isinstance(shape_value_in, int):
        INPUT_AUDIO_LENGTH_RUN = min(
            config_A['IN_SAMPLE_RATE'] * config_A['MAX_DYNAMIC_AUDIO_SECONDS'],
            audio_len,
        )
    else:
        INPUT_AUDIO_LENGTH_RUN = shape_value_in
    input_shape_A = resolve_tensor_shape(
        in_name_A[0],
        {-1: INPUT_AUDIO_LENGTH_RUN}
        if not isinstance(in_name_A[0].shape[-1], int)
        else {},
    )
    audio = audio.reshape(*input_shape_A[:-1], -1)

    stride_step = INPUT_AUDIO_LENGTH_RUN
    if audio_len > INPUT_AUDIO_LENGTH_RUN:
        num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH_RUN) / stride_step)) + 1
        total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH_RUN
        pad_amount = total_length_needed - audio_len
        padding = [(0, 0)] * audio.ndim
        padding[-1] = (0, pad_amount)
        audio = np.pad(audio, padding, mode='constant')
    elif audio_len < INPUT_AUDIO_LENGTH_RUN:
        padding = [(0, 0)] * audio.ndim
        padding[-1] = (0, INPUT_AUDIO_LENGTH_RUN - audio_len)
        audio = np.pad(
            audio,
            padding,
            mode='constant',
        )
    aligned_len = audio.shape[-1]

    output_frames_A = valid_frame_count(
        INPUT_AUDIO_LENGTH_RUN,
        config_A['IN_SAMPLE_RATE'],
        config_A['MODEL_SAMPLE_RATE'],
        config_A['WINDOW_LENGTH'],
        config_A['HOP_LENGTH'],
    )
    input_buffer_A = ort_zeros(input_shape_A, input_dtype_A)
    output_buffer_A = ort_zeros(
        resolve_tensor_shape(
            out_name_A[0],
            {-1: output_frames_A}
            if not isinstance(out_name_A[0].shape[-1], int)
            else {},
        ),
        numpy_dtype_from_onnx_type(out_name_A[0].type),
    )
    binding_A = ort_session_A.io_binding()
    binding_A.bind_ortvalue_input(in_name_A0, input_buffer_A)
    binding_A.bind_ortvalue_output(out_name_A0, output_buffer_A)

    # Start to run FireRedVAD
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH_RUN
    all_vad_probs = []
    print("\nRunning the FireRedVAD by ONNX Runtime.")
    start_time = time.time()
    while slice_end <= aligned_len:
        update_ortvalue(input_buffer_A, audio[..., slice_start:slice_end])
        run_iobinding(ort_session_A, binding_A)
        all_vad_probs.append(np.array(output_buffer_A.numpy().reshape(-1), copy=True))
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH_RUN
    end_time = time.time()

    valid_vad_frames = valid_frame_count(
        audio_len,
        config_A['IN_SAMPLE_RATE'],
        config_A['MODEL_SAMPLE_RATE'],
        config_A['WINDOW_LENGTH'],
        config_A['HOP_LENGTH'],
    )
    if all_vad_probs:
        all_vad_probs = np.concatenate(all_vad_probs, axis=0)[:valid_vad_frames]
    else:
        all_vad_probs = np.zeros((0,), dtype=np.float32)

    vad_postprocessor = VadPostprocessor(
        config_A['SMOOTH_WINDOW_SIZE'],
        config_A['SPEAKING_SCORE'],
        config_A['MIN_SPEECH_FRAME'],
        config_A['MAX_SPEECH_FRAME'],
        config_A['MIN_SILENCE_FRAME'],
        config_A['MERGE_SILENCE_FRAME'],
        config_A['EXTEND_SPEECH_FRAME'],
        config_A['OUTPUT_FRAME_SHIFT_S'],
        config_A['OUTPUT_FRAME_LENGTH_S'],
    )
    vad_decisions = vad_postprocessor.process(all_vad_probs)
    timestamps = vad_postprocessor.decision_to_segment(
        vad_decisions,
        audio_len / config_A['IN_SAMPLE_RATE'],
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
            line = f"{int(start * config_A['IN_SAMPLE_RATE'])} --> {int(end * config_A['IN_SAMPLE_RATE'])}\n"
            file.write(line)
            print(line.replace("\n", ""))

    vad_rtf = (end_time - start_time) / (audio_len / config_A['IN_SAMPLE_RATE'])
    print(f"\nVAD Process Complete.\n\nTime Cost: {end_time - start_time:.3f} Seconds")
    print(f"RTF: {vad_rtf:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Run AED (Audio Event Detection) Inference with ONNX Runtime
# ═══════════════════════════════════════════════════════════════════════════════

if RUN_AED:
    print('\n\n' + '=' * 70)
    print('Start to run FireRedAED by ONNX Runtime.\n\nNow, loading the model...')

    ort_session_B, model_metadata_B, config_B = load_firered_runtime(
        onnx_model_aed,
        expected_task='audio_event_detection',
        expected_inputs=1,
        expected_outputs=1,
        postprocess_config=POSTPROCESS_CONFIGS['aed'],
    )
    print(f"\nUsable Providers: {ort_session_B.get_providers()}")
    in_name_B = ort_session_B.get_inputs()
    out_name_B = ort_session_B.get_outputs()
    in_name_B0 = in_name_B[0].name
    out_name_B0 = out_name_B[0].name
    input_dtype_B = numpy_dtype_from_onnx_type(in_name_B[0].type)

    # AED event mapping: index -> event name
    event_labels = model_metadata_B.string('event_labels', required=True).split(',')
    if set(event_labels) != EVENT_THRESHOLDS.keys():
        raise ValueError(
            'AED metadata event_labels must match the configured EVENT_THRESHOLDS.'
        )
    output_channels = out_name_B[0].shape[1]
    if not isinstance(output_channels, int):
        raise ValueError(
            f'AED output channel axis must be static, got {out_name_B[0].shape}.'
        )
    if output_channels != len(event_labels):
        raise ValueError(
            f'AED output has {output_channels} channels, but metadata defines '
            f'{len(event_labels)} event labels.'
        )
    IDX2EVENT = dict(enumerate(event_labels))

    # Load the input audio
    print(f"\nTest Input Audio: {test_aed_audio}")
    audio_aed = np.array(AudioSegment.from_file(test_aed_audio).set_channels(1).set_frame_rate(config_B['IN_SAMPLE_RATE']).get_array_of_samples(), dtype=np.int16)
    audio_aed = normalise_audio(
        audio_aed,
        input_dtype_B,
        config_B['NORMALIZE_TARGET_RMS'],
    )
    audio_aed_len = len(audio_aed)
    audio_aed_dur = audio_aed_len / config_B['IN_SAMPLE_RATE']
    shape_value_in_B = in_name_B[0].shape[-1]
    if not isinstance(shape_value_in_B, int):
        INPUT_AUDIO_LENGTH_RUN_B = min(
            config_B['IN_SAMPLE_RATE'] * config_B['MAX_DYNAMIC_AUDIO_SECONDS'],
            audio_aed_len,
        )
    else:
        INPUT_AUDIO_LENGTH_RUN_B = shape_value_in_B
    input_shape_B = resolve_tensor_shape(
        in_name_B[0],
        {-1: INPUT_AUDIO_LENGTH_RUN_B}
        if not isinstance(in_name_B[0].shape[-1], int)
        else {},
    )
    audio_aed = audio_aed.reshape(*input_shape_B[:-1], -1)

    stride_step_B = INPUT_AUDIO_LENGTH_RUN_B
    if audio_aed_len > INPUT_AUDIO_LENGTH_RUN_B:
        num_windows_B = int(np.ceil((audio_aed_len - INPUT_AUDIO_LENGTH_RUN_B) / stride_step_B)) + 1
        total_length_needed_B = (num_windows_B - 1) * stride_step_B + INPUT_AUDIO_LENGTH_RUN_B
        pad_amount_B = total_length_needed_B - audio_aed_len
        padding_B = [(0, 0)] * audio_aed.ndim
        padding_B[-1] = (0, pad_amount_B)
        audio_aed = np.pad(
            audio_aed,
            padding_B,
            mode='constant',
        )
    elif audio_aed_len < INPUT_AUDIO_LENGTH_RUN_B:
        padding_B = [(0, 0)] * audio_aed.ndim
        padding_B[-1] = (0, INPUT_AUDIO_LENGTH_RUN_B - audio_aed_len)
        audio_aed = np.pad(
            audio_aed,
            padding_B,
            mode='constant',
        )
    aligned_len_B = audio_aed.shape[-1]

    output_frames_B = valid_frame_count(
        INPUT_AUDIO_LENGTH_RUN_B,
        config_B['IN_SAMPLE_RATE'],
        config_B['MODEL_SAMPLE_RATE'],
        config_B['WINDOW_LENGTH'],
        config_B['HOP_LENGTH'],
    )
    input_buffer_B = ort_zeros(input_shape_B, input_dtype_B)
    output_buffer_B = ort_zeros(
        resolve_tensor_shape(
            out_name_B[0],
            {-1: output_frames_B}
            if not isinstance(out_name_B[0].shape[-1], int)
            else {},
        ),
        numpy_dtype_from_onnx_type(out_name_B[0].type),
    )
    binding_B = ort_session_B.io_binding()
    binding_B.bind_ortvalue_input(in_name_B0, input_buffer_B)
    binding_B.bind_ortvalue_output(out_name_B0, output_buffer_B)

    # Run AED inference
    print("\nRunning the FireRedAED by ONNX Runtime.")
    start_time = time.time()
    all_probs = []
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH_RUN_B
    while slice_end <= aligned_len_B:
        update_ortvalue(input_buffer_B, audio_aed[..., slice_start:slice_end])
        run_iobinding(ort_session_B, binding_B)
        all_probs.append(
            np.array(output_buffer_B.numpy().reshape(output_channels, -1), copy=True)
        )
        slice_start += stride_step_B
        slice_end = slice_start + INPUT_AUDIO_LENGTH_RUN_B
    end_time = time.time()

    valid_aed_frames = valid_frame_count(
        audio_aed_len,
        config_B['IN_SAMPLE_RATE'],
        config_B['MODEL_SAMPLE_RATE'],
        config_B['WINDOW_LENGTH'],
        config_B['HOP_LENGTH'],
    )
    if all_probs:
        all_probs = np.concatenate(all_probs, axis=1)[:, :valid_aed_frames]
    else:
        all_probs = np.zeros((len(IDX2EVENT), 0), dtype=np.float32)

    # Post-process each event type
    event2timestamps = {}
    event2ratio = {}
    event2postprocessor = {
        event: VadPostprocessor(
            config_B['SMOOTH_WINDOW_SIZE'],
            threshold,
            config_B['MIN_SPEECH_FRAME'],
            config_B['MAX_SPEECH_FRAME'],
            config_B['MIN_SILENCE_FRAME'],
            config_B['MERGE_SILENCE_FRAME'],
            config_B['EXTEND_SPEECH_FRAME'],
            config_B['OUTPUT_FRAME_SHIFT_S'],
            config_B['OUTPUT_FRAME_LENGTH_S'],
        )
        for event, threshold in EVENT_THRESHOLDS.items()
    }
    for idx, event in IDX2EVENT.items():
        threshold = EVENT_THRESHOLDS[event]
        postprocessor = event2postprocessor[event]
        event_probs = all_probs[idx]
        decisions = postprocessor.process(event_probs)
        event2timestamps[event] = postprocessor.decision_to_segment(decisions, audio_aed_dur)
        raw_ratio = float(np.mean(event_probs >= threshold)) if len(event_probs) > 0 else 0.0
        event2ratio[event] = round(raw_ratio, 3)

    # Print AED results
    print(f"Complete: 100.00%")
    print(f"\nAED Results:")
    print(f"  Audio duration: {audio_aed_dur:.3f}s")
    print(f"  Event ratios: {event2ratio}")
    for event, ts_list in event2timestamps.items():
        print(f"\n  [{event}] segments ({len(ts_list)}):")
        for start_s, end_s in ts_list:
            print(f"    {format_time(start_s)} --> {format_time(end_s)}")

    aed_rtf = (end_time - start_time) / audio_aed_dur
    print(f"\nAED Process Complete.\n\nTime Cost: {end_time - start_time:.3f} Seconds")
    print(f"RTF: {aed_rtf:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Run Stream-VAD Inference with ONNX Runtime (frame-by-frame with caches)
# ═══════════════════════════════════════════════════════════════════════════════

if RUN_STREAM_VAD:
    print('\n\n' + '=' * 70)
    print('Start to run FireRedStreamVAD by ONNX Runtime.\n\nNow, loading the model...')

    ort_session_C, model_metadata_C, config_C = load_firered_runtime(
        onnx_model_stream_vad,
        expected_task='streaming_voice_activity_detection',
        expected_inputs=2,
        expected_outputs=2,
        postprocess_config=POSTPROCESS_CONFIGS['stream_vad'],
    )
    stream_chunk_ms = STREAM_CHUNK_MS
    stream_chunk_samples = int(
        round(config_C['IN_SAMPLE_RATE'] * stream_chunk_ms / 1000.0)
    )
    pad_start_frame = STREAM_PAD_START_FRAME
    expected_chunk_samples = int(
        round(config_C['IN_SAMPLE_RATE'] * stream_chunk_ms / 1000.0)
    )
    if stream_chunk_samples != expected_chunk_samples:
        raise ValueError(
            f'Stream metadata chunk size {stream_chunk_samples} does not match '
            f'{stream_chunk_ms} ms at {config_C["IN_SAMPLE_RATE"]} Hz '
            f'({expected_chunk_samples} samples).'
        )
    print(f"\nUsable Providers: {ort_session_C.get_providers()}")
    in_name_C = ort_session_C.get_inputs()
    out_name_C = ort_session_C.get_outputs()
    in_name_C0 = in_name_C[0].name  # audio
    in_name_C1 = in_name_C[1].name  # caches_in
    out_name_C0 = out_name_C[0].name  # probs
    out_name_C1 = out_name_C[1].name  # caches_out
    input_dtype_C = numpy_dtype_from_onnx_type(in_name_C[0].type)

    # Determine cache shape from the model metadata
    cache_shape = resolve_tensor_shape(in_name_C[1])
    output_cache_shape = resolve_tensor_shape(
        out_name_C[1],
        {
            axis: cache_shape[axis]
            for axis, dimension in enumerate(out_name_C[1].shape)
            if not isinstance(dimension, int)
        },
    )
    if cache_shape != output_cache_shape:
        raise ValueError(
            f'Stream-VAD cache input/output shapes must match, got '
            f'{cache_shape} and {output_cache_shape}.'
        )
    print(f"  Cache shape: {list(cache_shape)}")

    # Load the input audio
    print(f"\nTest Input Audio: {test_stream_vad_audio}")
    audio_stream = np.array(AudioSegment.from_file(test_stream_vad_audio).set_channels(1).set_frame_rate(config_C['IN_SAMPLE_RATE']).get_array_of_samples(), dtype=np.int16)
    audio_stream = normalise_audio(
        audio_stream,
        input_dtype_C,
        config_C['NORMALIZE_TARGET_RMS'],
    )
    audio_stream_len = len(audio_stream)
    audio_stream_dur = audio_stream_len / config_C['IN_SAMPLE_RATE']

    cache_dtype_C = numpy_dtype_from_onnx_type(in_name_C[1].type)
    output_cache_dtype_C = numpy_dtype_from_onnx_type(out_name_C[1].type)
    if cache_dtype_C != output_cache_dtype_C:
        raise TypeError('Stream-VAD cache input and output dtypes must match.')

    cache_buffers_C = [
        ort_zeros(cache_shape, cache_dtype_C),
        ort_zeros(cache_shape, cache_dtype_C),
    ]
    stream_io_by_length = {}

    def stream_io_for_length(input_length):
        if input_length not in stream_io_by_length:
            output_frames = valid_frame_count(
                input_length,
                config_C['IN_SAMPLE_RATE'],
                config_C['MODEL_SAMPLE_RATE'],
                config_C['WINDOW_LENGTH'],
                config_C['HOP_LENGTH'],
            )
            audio_buffer = ort_zeros(
                resolve_tensor_shape(
                    in_name_C[0],
                    {-1: input_length}
                    if not isinstance(in_name_C[0].shape[-1], int)
                    else {},
                ),
                input_dtype_C,
            )
            probs_buffer = ort_zeros(
                resolve_tensor_shape(
                    out_name_C[0],
                    {-1: output_frames}
                    if not isinstance(out_name_C[0].shape[-1], int)
                    else {},
                ),
                numpy_dtype_from_onnx_type(out_name_C[0].type),
            )
            bindings = []
            for cache_index in range(2):
                binding = ort_session_C.io_binding()
                binding.bind_ortvalue_input(in_name_C0, audio_buffer)
                binding.bind_ortvalue_input(in_name_C1, cache_buffers_C[cache_index])
                binding.bind_ortvalue_output(out_name_C0, probs_buffer)
                binding.bind_ortvalue_output(out_name_C1, cache_buffers_C[1 - cache_index])
                bindings.append(binding)
            stream_io_by_length[input_length] = (audio_buffer, probs_buffer, bindings)
        return stream_io_by_length[input_length]

    stream_io_for_length(stream_chunk_samples)
    final_chunk_samples = audio_stream_len % stream_chunk_samples
    if final_chunk_samples:
        stream_io_for_length(max(final_chunk_samples, config_C['WINDOW_LENGTH']))

    # Collect all frame probabilities
    all_stream_probs = []

    print(f"\nRunning the FireRedStreamVAD by ONNX Runtime.")
    print(f"  Audio duration: {audio_stream_dur:.3f}s")
    print(f"  Chunk size: {stream_chunk_ms}ms ({stream_chunk_samples} samples)")
    start_time = time.time()

    # Process audio in streaming chunks
    pos = 0
    cache_index = 0
    while pos < audio_stream_len:
        chunk_end = min(pos + stream_chunk_samples, audio_stream_len)
        chunk = audio_stream[pos:chunk_end]

        # Pad last chunk if needed (must be at least one analysis window).
        if len(chunk) < config_C['WINDOW_LENGTH']:
            chunk = np.pad(
                chunk,
                (0, config_C['WINDOW_LENGTH'] - len(chunk)),
                mode='constant',
            )

        audio_buffer_C, probs_buffer_C, bindings_C = stream_io_for_length(len(chunk))
        chunk_input = chunk.reshape(audio_buffer_C.shape()).astype(
            input_dtype_C, copy=False
        )

        update_ortvalue(audio_buffer_C, chunk_input)
        run_iobinding(ort_session_C, bindings_C[cache_index])
        all_stream_probs.append(
            np.array(probs_buffer_C.numpy().reshape(-1), copy=True)
        )
        cache_index = 1 - cache_index

        pos = chunk_end

    end_time = time.time()

    # Concatenate all frame probabilities
    valid_stream_frames = valid_frame_count(
        audio_stream_len,
        config_C['IN_SAMPLE_RATE'],
        config_C['MODEL_SAMPLE_RATE'],
        config_C['WINDOW_LENGTH'],
        config_C['HOP_LENGTH'],
    )
    if all_stream_probs:
        all_stream_probs = np.concatenate(all_stream_probs, axis=0)[:valid_stream_frames]
    else:
        all_stream_probs = np.zeros((0,), dtype=np.float32)

    stream_postprocessor = StreamVadPostprocessor(
        config_C['SMOOTH_WINDOW_SIZE'],
        config_C['SPEAKING_SCORE'],
        pad_start_frame,
        config_C['MIN_SPEECH_FRAME'],
        config_C['MAX_SPEECH_FRAME'],
        config_C['MIN_SILENCE_FRAME'],
        config_C['OUTPUT_FRAME_SHIFT_S'],
    )
    total_frames = len(all_stream_probs)
    stream_timestamps = stream_postprocessor.process_batch(all_stream_probs)

    # Print Stream-VAD results
    print(f"Complete: 100.00%")
    print(f"\nStream-VAD Results:")
    print(f"  Audio duration: {audio_stream_dur:.3f}s")
    print(f"  Total frames: {total_frames}")
    print(f"  Segments detected: {len(stream_timestamps)}")
    print(f"\n  Timestamps in Second:")
    for start_s, end_s in stream_timestamps:
        print(f"    {format_time(start_s)} --> {format_time(end_s)}")

    stream_vad_rtf = (end_time - start_time) / audio_stream_dur
    print(f"\nStream-VAD Process Complete.\n\nTime Cost: {end_time - start_time:.3f} Seconds")
    print(f"RTF: {stream_vad_rtf:.4f}")
