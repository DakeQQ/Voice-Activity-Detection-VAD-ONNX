"""Run the standalone FSMN-VAD ONNX model and emit speech timestamps."""

import math
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path

import numpy as np
import onnxruntime as ort
from pydub import AudioSegment

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "vad_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from vad_onnx_metadata import REQUIRED_VAD_METADATA_KEYS, load_runtime_metadata
from Example_Audio import model_audio_path


SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_PATH = SCRIPT_DIR / "FSMN_VAD_ONNX" / "FSMN.onnx"
TEST_AUDIO = Path(model_audio_path("fsmn"))
PROVIDERS = ["CPUExecutionProvider"]  # Installed execution providers in priority order.
TIMESTAMPS_SECONDS = SCRIPT_DIR / "timestamps_second.txt"
TIMESTAMPS_INDICES = SCRIPT_DIR / "timestamps_indices.txt"

# User-adjustable inference settings.
NORMALIZE_AUDIO = False                  # True applies RMS normalization before inference.
NORMALIZE_TARGET_RMS = 4096.0            # Target PCM16 RMS; used only when normalization is enabled.
MAX_DYNAMIC_AUDIO_SECONDS = 360          # Maximum window size for a dynamic-length model.
LFR_HALF_CONTEXT_FRAMES = 2              # The exported LFR-5 transform looks two frames each way.
FSMN_POSTPROCESS_CONFIG = {
    "detect_mode": 1,                    # 1 detects every segment; 0 stops after the first endpoint.
    "max_end_silence_time": 800,         # Silence needed to end a segment, in milliseconds.
    "max_start_silence_time": 3000,      # Leading-silence timeout for detect_mode 0, in milliseconds.
    "window_size_ms": 200,               # Smoothing decision window, in milliseconds.
    "sil_to_speech_time_thres": 150,     # Speech needed in the window to start, in milliseconds.
    "speech_to_sil_time_thres": 150,     # Window threshold for returning to silence, in milliseconds.
    "speech_2_noise_ratio": 1.0,         # Weight used when comparing speech and noise scores.
    "do_extend": 1,                      # 1 applies the start/end padding below; 0 disables it.
    "lookback_time_start_point": 200,    # Padding before a detected start, in milliseconds.
    "lookahead_time_end_point": 100,     # Padding after a detected end, in milliseconds.
    "max_single_segment_time": 60000,    # Split longer speech segments at this duration, in milliseconds.
    "snr_thres": -100.0,                 # Minimum SNR in dB; -100 effectively disables this filter.
    "noise_frame_num_used_for_snr": 100, # Frames used by the rolling noise-level estimate.
    "decibel_thres": -100.0,             # Force quieter frames to silence; -100 effectively disables it.
    "speech_noise_thres": 0.6,           # Required margin between speech and noise probabilities.
}
CPU_DISABLE_MATMUL_ADD_FUSION = True  # ORT 1.27 wraps rank-3 MatMul+Add in costly Reshape/Gemm/Reshape chains.
CPU_DISABLE_NCHWC = True              # NCHWc reorders regress mean/tail latency on the target i7-1165G7.
CPU_EXTRA_DISABLED_OPTIMIZERS = [     # Individually benchmarked on the same CPU / ORT build.
    "ConvAddActivationFusion",
    "MatmulTransposeFusion",
]
DEVICE_ID = 0                    # Accelerator device index when a non-CPU provider is selected.

REQUIRED_FSMN_METADATA_KEYS = (
    *REQUIRED_VAD_METADATA_KEYS,
    "output_frame_shift_samples",
    "frame_length_samples",
    "num_inputs",
    "num_outputs",
    "cache_count",
    "cache_channels",
    "cache_frames",
    "score_classes",
    "sil_pdf_ids",
    "frame_in_ms",
)


class VadState(Enum):
    START_NOT_DETECTED = 1
    IN_SPEECH = 2
    END_DETECTED = 3


class FrameState(Enum):
    INVALID = -1
    SPEECH = 1
    SILENCE = 0


class AudioChangeState(Enum):
    SPEECH_TO_SPEECH = 0
    SPEECH_TO_SILENCE = 1
    SILENCE_TO_SILENCE = 2
    SILENCE_TO_SPEECH = 3
    INVALID = 5


class WindowDetector:
    def __init__(self, window_size_ms, silence_to_speech_ms, speech_to_silence_ms, frame_ms):
        self.window_frames = int(window_size_ms / frame_ms)
        self.silence_to_speech_frames = int(silence_to_speech_ms / frame_ms)
        self.speech_to_silence_frames = int(speech_to_silence_ms / frame_ms)
        self.reset()

    def reset(self):
        self.window = [0] * self.window_frames
        self.window_sum = 0
        self.position = 0
        self.previous_state = FrameState.SILENCE

    def detect(self, frame_state):
        current = 1 if frame_state == FrameState.SPEECH else 0
        self.window_sum -= self.window[self.position]
        self.window_sum += current
        self.window[self.position] = current
        self.position = (self.position + 1) % self.window_frames

        if self.previous_state == FrameState.SILENCE and self.window_sum >= self.silence_to_speech_frames:
            self.previous_state = FrameState.SPEECH
            return AudioChangeState.SILENCE_TO_SPEECH
        if self.previous_state == FrameState.SPEECH and self.window_sum <= self.speech_to_silence_frames:
            self.previous_state = FrameState.SILENCE
            return AudioChangeState.SPEECH_TO_SILENCE
        if self.previous_state == FrameState.SPEECH:
            return AudioChangeState.SPEECH_TO_SPEECH
        return AudioChangeState.SILENCE_TO_SILENCE


@dataclass
class SpeechSegment:
    start_ms: int = 0
    end_ms: int = 0
    has_start: bool = False
    has_end: bool = False


class E2EVadPostProcessor:
    def __init__(self, config):
        self.sample_rate = config["sample_rate"]
        self.detect_mode = config["detect_mode"]
        self.max_end_silence_time = config["max_end_silence_time"]
        self.max_start_silence_time = config["max_start_silence_time"]
        self.window_size_ms = config["window_size_ms"]
        self.silence_to_speech_ms = config["sil_to_speech_time_thres"]
        self.speech_to_silence_ms = config["speech_to_sil_time_thres"]
        self.speech_to_noise_ratio = config["speech_2_noise_ratio"]
        self.do_extend = config["do_extend"]
        self.lookback_start_ms = config["lookback_time_start_point"]
        self.lookahead_end_ms = config["lookahead_time_end_point"]
        self.max_segment_ms = config["max_single_segment_time"]
        self.snr_threshold = config["snr_thres"]
        self.noise_frames = config["noise_frame_num_used_for_snr"]
        self.decibel_threshold = config["decibel_thres"]
        self.speech_noise_threshold = config["speech_noise_thres"]
        self.silence_pdf_ids = config["sil_pdf_ids"]
        self.frame_ms = config["frame_in_ms"]
        self.window_detector = WindowDetector(
            self.window_size_ms,
            self.silence_to_speech_ms,
            self.speech_to_silence_ms,
            self.frame_ms,
        )
        self.reset_all()

    def reset_all(self):
        self.data_start_frame = 0
        self.frame_count = 0
        self.last_drop_frames = 0
        self.latest_speech_frame = 0
        self.latest_silence_frame = -1
        self.continuous_silence_frames = 0
        self.state = VadState.START_NOT_DETECTED
        self.confirmed_start_frame = -1
        self.confirmed_end_frame = -1
        self.end_count = 0
        self.noise_average_decibel = -100.0
        self.next_segment = True
        self.output_segments = []
        self.output_offset = 0
        self.decibels = []
        self.scores = None
        self.window_detector.reset()

    def reset_detection(self):
        self.continuous_silence_frames = 0
        self.latest_speech_frame = 0
        self.latest_silence_frame = -1
        self.confirmed_start_frame = -1
        self.confirmed_end_frame = -1
        self.state = VadState.START_NOT_DETECTED
        self.window_detector.reset()
        if self.output_segments:
            drop_frames = int(self.output_segments[-1].end_ms / self.frame_ms)
            real_drop_frames = drop_frames - self.last_drop_frames
            self.last_drop_frames = drop_frames
            self.decibels = self.decibels[real_drop_frames:]
            self.scores = self.scores[real_drop_frames:]

    def latency_frames(self):
        latency = self.window_detector.window_frames
        if self.do_extend:
            latency += int(self.lookback_start_ms / self.frame_ms)
        return latency

    def pop_until(self, frame_index):
        self.data_start_frame = max(self.data_start_frame, frame_index)

    def pop_segment(self, start_frame, frame_count, first_is_start, last_is_end):
        self.pop_until(start_frame)
        if not self.output_segments or first_is_start:
            self.output_segments.append(SpeechSegment(start_ms=start_frame * self.frame_ms))
        segment = self.output_segments[-1]
        self.data_start_frame += frame_count
        segment.end_ms = (start_frame + frame_count) * self.frame_ms
        segment.has_start = segment.has_start or first_is_start
        segment.has_end = segment.has_end or last_is_end

    def on_silence(self, frame_index):
        self.latest_silence_frame = frame_index
        if self.state == VadState.START_NOT_DETECTED:
            self.pop_until(frame_index)

    def on_voice(self, frame_index):
        self.latest_speech_frame = frame_index
        self.pop_segment(frame_index, 1, False, False)

    def on_voice_start(self, frame_index, fake=False):
        self.confirmed_start_frame = frame_index
        if not fake and self.state == VadState.START_NOT_DETECTED:
            self.pop_segment(frame_index, 1, True, False)

    def on_voice_end(self, frame_index, fake, is_last_frame):
        for index in range(self.latest_speech_frame + 1, frame_index):
            self.on_voice(index)
        self.confirmed_end_frame = frame_index
        if not fake:
            self.pop_segment(frame_index, 1, False, True)
        self.end_count += 1

    def maybe_end_on_last_frame(self, is_final_frame, frame_index):
        if is_final_frame:
            self.on_voice_end(frame_index, False, True)
            self.state = VadState.END_DETECTED

    def get_frame_state(self, local_frame_index):
        decibel = self.decibels[local_frame_index]
        if decibel < self.decibel_threshold:
            self.detect_one_frame(FrameState.SILENCE, local_frame_index, False)
            return FrameState.SILENCE

        noise_score = float(sum(self.scores[local_frame_index, pdf] for pdf in self.silence_pdf_ids))
        speech_score = 1.0 - noise_score
        noise_probability = math.log(noise_score) * self.speech_to_noise_ratio
        speech_probability = math.log(speech_score)
        snr = decibel - self.noise_average_decibel
        if math.exp(speech_probability) >= math.exp(noise_probability) + self.speech_noise_threshold:
            if snr >= self.snr_threshold:
                return FrameState.SPEECH

        if self.noise_average_decibel < -99.9:
            self.noise_average_decibel = decibel
        else:
            self.noise_average_decibel = (
                decibel + self.noise_average_decibel * (self.noise_frames - 1)
            ) / self.noise_frames
        return FrameState.SILENCE

    def detect_one_frame(self, frame_state, frame_index, is_final_frame):
        change = self.window_detector.detect(frame_state)

        if change == AudioChangeState.SILENCE_TO_SPEECH:
            self.continuous_silence_frames = 0
            if self.state == VadState.START_NOT_DETECTED:
                start_frame = max(self.data_start_frame, frame_index - self.latency_frames())
                self.on_voice_start(start_frame)
                self.state = VadState.IN_SPEECH
                for index in range(start_frame + 1, frame_index + 1):
                    self.on_voice(index)
            elif self.state == VadState.IN_SPEECH:
                for index in range(self.latest_speech_frame + 1, frame_index):
                    self.on_voice(index)
                if frame_index - self.confirmed_start_frame + 1 > self.max_segment_ms / self.frame_ms:
                    self.on_voice_end(frame_index, False, False)
                    self.state = VadState.END_DETECTED
                elif not is_final_frame:
                    self.on_voice(frame_index)
                else:
                    self.maybe_end_on_last_frame(is_final_frame, frame_index)

        elif change == AudioChangeState.SPEECH_TO_SILENCE:
            self.continuous_silence_frames = 0
            if self.state == VadState.IN_SPEECH:
                if frame_index - self.confirmed_start_frame + 1 > self.max_segment_ms / self.frame_ms:
                    self.on_voice_end(frame_index, False, False)
                    self.state = VadState.END_DETECTED
                elif not is_final_frame:
                    self.on_voice(frame_index)
                else:
                    self.maybe_end_on_last_frame(is_final_frame, frame_index)

        elif change == AudioChangeState.SPEECH_TO_SPEECH:
            self.continuous_silence_frames = 0
            if self.state == VadState.IN_SPEECH:
                if frame_index - self.confirmed_start_frame + 1 > self.max_segment_ms / self.frame_ms:
                    self.on_voice_end(frame_index, False, False)
                    self.state = VadState.END_DETECTED
                elif not is_final_frame:
                    self.on_voice(frame_index)
                else:
                    self.maybe_end_on_last_frame(is_final_frame, frame_index)

        elif change == AudioChangeState.SILENCE_TO_SILENCE:
            self.continuous_silence_frames += 1
            if self.state == VadState.START_NOT_DETECTED:
                start_timeout = (
                    self.detect_mode == 0
                    and self.continuous_silence_frames * self.frame_ms > self.max_start_silence_time
                )
                if start_timeout or (is_final_frame and self.end_count == 0):
                    for index in range(self.latest_silence_frame + 1, frame_index):
                        self.on_silence(index)
                    self.on_voice_start(0, fake=True)
                    self.on_voice_end(0, fake=True, is_last_frame=False)
                    self.state = VadState.END_DETECTED
                elif frame_index >= self.latency_frames():
                    self.on_silence(frame_index - self.latency_frames())
            elif self.state == VadState.IN_SPEECH:
                end_silence_ms = self.max_end_silence_time - self.speech_to_silence_ms
                if self.continuous_silence_frames * self.frame_ms >= end_silence_ms:
                    lookback = int(end_silence_ms / self.frame_ms)
                    if self.do_extend:
                        lookback -= int(self.lookahead_end_ms / self.frame_ms)
                        lookback -= 1
                        lookback = max(0, lookback)
                    self.on_voice_end(frame_index - lookback, False, False)
                    self.state = VadState.END_DETECTED
                elif frame_index - self.confirmed_start_frame + 1 > self.max_segment_ms / self.frame_ms:
                    self.on_voice_end(frame_index, False, False)
                    self.state = VadState.END_DETECTED
                elif self.do_extend and not is_final_frame:
                    if self.continuous_silence_frames <= int(self.lookahead_end_ms / self.frame_ms):
                        self.on_voice(frame_index)
                else:
                    self.maybe_end_on_last_frame(is_final_frame, frame_index)

        if self.state == VadState.END_DETECTED and self.detect_mode == 1:
            self.reset_detection()

    def __call__(self, scores, decibels, is_final=True, online=False):
        scores = np.asarray(scores, dtype=np.float32)
        decibels = np.asarray(decibels, dtype=np.float32)
        if scores.ndim != 2 or decibels.ndim != 1 or scores.shape[0] != decibels.shape[0]:
            raise ValueError("Expected scores [frames, classes] and decibels [frames]")

        first_frame = self.frame_count
        self.frame_count += scores.shape[0]
        self.scores = scores if self.scores is None else np.concatenate((self.scores, scores), axis=0)
        self.decibels.extend(float(value) for value in decibels)
        for local_index in range(scores.shape[0]):
            frame_index = first_frame + local_index
            history_index = frame_index - self.last_drop_frames
            self.detect_one_frame(
                self.get_frame_state(history_index),
                frame_index,
                is_final and local_index == scores.shape[0] - 1,
            )

        segments = []
        for index in range(self.output_offset, len(self.output_segments)):
            segment = self.output_segments[index]
            if online:
                if not segment.has_start:
                    continue
                if not self.next_segment and not segment.has_end:
                    continue
                start_ms = segment.start_ms if self.next_segment else -1
                if segment.has_end:
                    end_ms = segment.end_ms
                    self.next_segment = True
                    self.output_offset += 1
                else:
                    end_ms = -1
                    self.next_segment = False
                segments.append([start_ms, end_ms])
            else:
                if not is_final and (not segment.has_start or not segment.has_end):
                    continue
                segments.append([segment.start_ms, segment.end_ms])
                self.output_offset += 1

        if is_final:
            self.reset_all()
        return segments


def config_from_metadata(metadata):
    return {
        "encoder_conf": {
            "fsmn_layers": metadata.required_int("cache_count"),
            "proj_dim": metadata.required_int("cache_channels"),
            "lorder": metadata.required_int("cache_frames") + 1,
            "lstride": 1,
            "output_dim": metadata.required_int("score_classes"),
        },
        "model_conf": {
            **FSMN_POSTPROCESS_CONFIG,
            "sample_rate": metadata.required_int("model_sample_rate"),
            "sil_pdf_ids": [
                int(value)
                for value in metadata.string("sil_pdf_ids", required=True).split(",")
            ],
            "frame_in_ms": metadata.required_int("frame_in_ms"),
        },
    }


def validate_fsmn_metadata(metadata, session):
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    expected_inputs = metadata.required_int("num_inputs")
    expected_outputs = metadata.required_int("num_outputs")
    if len(inputs) != expected_inputs or len(outputs) != expected_outputs:
        raise ValueError(
            f"FSMN model has {len(inputs)} inputs and {len(outputs)} outputs; "
            f"metadata requires {expected_inputs} and {expected_outputs}."
        )

    audio_input = inputs[0]
    if len(audio_input.shape) != 2:
        raise ValueError(f"FSMN audio input must have rank 2, got {audio_input.shape}.")
    metadata_dtype = metadata.string("input_audio_dtype", required=True).lower()
    expected_dtype = metadata_dtype if metadata_dtype.startswith("tensor(") else f"tensor({metadata_dtype})"
    if audio_input.type.lower() != expected_dtype:
        raise ValueError(
            f"ONNX input dtype {audio_input.type} does not match metadata "
            f"input_audio_dtype={metadata_dtype}."
        )

    input_length = audio_input.shape[1]
    model_dynamic = not isinstance(input_length, int)
    if model_dynamic != metadata.required_bool("dynamic_axes"):
        raise ValueError("FSMN audio dynamic-axis state does not match metadata.")
    if (
        isinstance(input_length, int)
        and input_length != metadata.required_int("input_audio_length")
    ):
        raise ValueError("FSMN static audio length does not match metadata.")

    cache_shapes = [tuple(model_input.shape) for model_input in inputs[1:]]
    if cache_shapes and any(
        shape != cache_shapes[0] for shape in cache_shapes[1:]
    ):
        raise ValueError(f"FSMN cache input shapes disagree: {cache_shapes}.")
    for model_input in inputs[1:]:
        if (
            len(model_input.shape) != 4
            or model_input.shape[1] != metadata.required_int("cache_channels")
            or model_input.shape[2] != metadata.required_int("cache_frames")
        ):
            raise ValueError(f"Unexpected FSMN cache input shape: {model_input.shape}.")
    if outputs[0].shape[-1] != metadata.required_int("score_classes"):
        raise ValueError("FSMN score class count does not match metadata.")


def create_session_options():
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "1")
    options.add_session_config_entry("session.inter_op.allow_spinning", "1")
    options.add_session_config_entry("session.set_denormal_as_zero", "1")
    return options


def cpu_disabled_optimizers(providers):
    cpu_ep_only = not providers or set(providers) == {"CPUExecutionProvider"}
    disabled_opts = []
    if cpu_ep_only and CPU_DISABLE_MATMUL_ADD_FUSION:
        disabled_opts.append("MatMulAddFusion")
    if cpu_ep_only and CPU_DISABLE_NCHWC:
        disabled_opts.append("NchwcTransformer")
    if cpu_ep_only:
        disabled_opts.extend(CPU_EXTRA_DISABLED_OPTIMIZERS)
    return disabled_opts or None


def numpy_dtype_from_onnx_type(type_name):
    tensor_dtypes = {
        "tensor(bool)": np.bool_,
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(int16)": np.int16,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
    }
    try:
        return tensor_dtypes[type_name]
    except KeyError as error:
        raise TypeError(f"Unsupported ONNX tensor type: {type_name}") from error


def ortvalue_device_type(session):
    primary_provider = session.get_providers()[0]
    if primary_provider in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}:
        return "cuda"
    if primary_provider == "DmlExecutionProvider":
        return "dml"
    return "cpu"


def create_ort_buffer(shape, dtype, device_type):
    return ort.OrtValue.ortvalue_from_numpy(
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


def fsmn_frame_count(input_samples, input_sample_rate, model_sample_rate,
                     frame_length, frame_shift):
    model_samples = int(input_samples * model_sample_rate / input_sample_rate)
    if model_samples < frame_length:
        return 0
    return 1 + (model_samples - frame_length) // frame_shift


def static_window_plan(total_frames, window_frames, left_context_frames,
                       right_context_frames):
    if total_frames <= 0:
        return []
    if total_frames <= window_frames:
        return [(0, 0, total_frames)]

    retained_frames = window_frames - left_context_frames - right_context_frames
    if retained_frames <= 0:
        raise ValueError(
            "Static FSMN input is too short for sliding inference: "
            f"{window_frames} output frames must exceed "
            f"{left_context_frames + right_context_frames} context frames."
        )

    plan = [(0, 0, window_frames - right_context_frames)]
    next_frame = window_frames - right_context_frames
    final_window_capacity = window_frames - left_context_frames
    while total_frames - next_frame > final_window_capacity:
        window_start = next_frame - left_context_frames
        plan.append((
            window_start,
            left_context_frames,
            window_frames - right_context_frames,
        ))
        next_frame += retained_frames

    window_start = total_frames - window_frames
    output_start = next_frame - window_start
    if output_start < left_context_frames:
        raise ValueError("Static FSMN final window does not contain enough left context.")
    plan.append((window_start, output_start, window_frames))
    return plan


def make_static_audio_window(audio, start_frame, window_samples, frame_shift,
                             input_sample_rate, model_sample_rate):
    start_sample = round(
        start_frame * frame_shift * input_sample_rate / model_sample_rate)
    available = audio[:, start_sample:start_sample + window_samples]
    if available.shape[1] == window_samples:
        return available
    window = np.zeros((audio.shape[0], window_samples), dtype=audio.dtype)
    window[:, :available.shape[1]] = available
    return window


class FSMNIOBinding:
    """Own reusable OrtValues and a single binding for one execution shape."""

    def __init__(self, session, config, audio_shape, frame_length, frame_shift,
                 input_sample_rate, model_sample_rate):
        self.session = session
        self.run_options = ort.RunOptions()
        self.run_options.log_severity_level = 3
        self.run_options.log_verbosity_level = 3
        self.run_options.add_run_config_entry(
            "disable_synchronize_execution_providers", "1"
        )
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if not inputs or len(outputs) < 2 or len(inputs) - 1 != len(outputs) - 2:
            raise ValueError(
                f"FSMN cache I/O is unbalanced: {len(inputs)} inputs and "
                f"{len(outputs)} outputs."
            )

        audio_samples = audio_shape[1]
        frame_count = fsmn_frame_count(
            audio_samples,
            input_sample_rate,
            model_sample_rate,
            frame_length,
            frame_shift,
        )
        device_type = ortvalue_device_type(session)

        self.audio_buffer = create_ort_buffer(
            resolve_tensor_shape(
                inputs[0],
                {1: audio_samples}
                if not isinstance(inputs[0].shape[1], int)
                else {},
            ),
            numpy_dtype_from_onnx_type(inputs[0].type),
            device_type,
        )
        self.cache_input_buffers = [
            create_ort_buffer(
                resolve_tensor_shape(metadata),
                numpy_dtype_from_onnx_type(metadata.type),
                device_type,
            )
            for metadata in inputs[1:]
        ]
        self.score_buffer = create_ort_buffer(
            resolve_tensor_shape(
                outputs[0],
                {1: frame_count}
                if not isinstance(outputs[0].shape[1], int)
                else {},
            ),
            numpy_dtype_from_onnx_type(outputs[0].type),
            device_type,
        )
        self.decibel_buffer = create_ort_buffer(
            resolve_tensor_shape(
                outputs[1],
                {1: frame_count}
                if not isinstance(outputs[1].shape[1], int)
                else {},
            ),
            numpy_dtype_from_onnx_type(outputs[1].type),
            device_type,
        )
        self.cache_output_buffers = [
            create_ort_buffer(
                resolve_tensor_shape(metadata),
                numpy_dtype_from_onnx_type(metadata.type),
                device_type,
            )
            for metadata in outputs[2:]
        ]

        self.binding = session.io_binding()
        self.binding.bind_ortvalue_input(inputs[0].name, self.audio_buffer)
        for metadata, buffer in zip(inputs[1:], self.cache_input_buffers):
            self.binding.bind_ortvalue_input(metadata.name, buffer)
        self.binding.bind_ortvalue_output(outputs[0].name, self.score_buffer)
        self.binding.bind_ortvalue_output(outputs[1].name, self.decibel_buffer)
        for metadata, buffer in zip(outputs[2:], self.cache_output_buffers):
            self.binding.bind_ortvalue_output(metadata.name, buffer)

    def run(self, audio):
        if tuple(audio.shape) != tuple(self.audio_buffer.shape()):
            raise ValueError(
                f"Audio shape changed from bound shape {self.audio_buffer.shape()} to {audio.shape}"
            )
        update_ort_buffer(self.audio_buffer, audio)
        self.session.run_with_iobinding(
            self.binding,
            run_options=self.run_options,
        )
        return self.score_buffer.numpy(), self.decibel_buffer.numpy()


def load_audio(audio_path, sample_rate, frame_length):
    segment = AudioSegment.from_file(audio_path)
    samples = np.asarray(segment.get_array_of_samples())
    if segment.channels > 1:
        samples = samples.reshape(-1, segment.channels)[:, 0]
    scale = float(1 << (8 * segment.sample_width - 1))
    audio = samples.astype(np.float32) / scale
    if segment.frame_rate != sample_rate:
        import torch
        from torchaudio.functional import resample

        audio = resample(torch.from_numpy(audio), segment.frame_rate, sample_rate).numpy()
    if audio.size < frame_length:
        raise ValueError(f"Audio must contain at least {frame_length} samples")
    return audio


def normalise_audio(audio: np.ndarray, input_dtype_np, target_rms) -> np.ndarray:
    prepared = audio.astype(np.float32, copy=False)
    if NORMALIZE_AUDIO:
        rms = np.sqrt(np.mean(prepared * prepared, dtype=np.float32), dtype=np.float32)
        if rms > 0.0:
            prepared = prepared * np.float32(
                target_rms / (32768.0 * (rms + 1e-7)))
        np.clip(prepared, -1.0, 32767.0 / 32768.0, out=prepared)
    if input_dtype_np == np.int16:
        return np.clip(
            prepared * 32768.0, -32768.0, 32767.0
        ).astype(np.int16)
    return prepared.astype(input_dtype_np, copy=False)


def run_inference(onnx_path, audio_path, providers):
    disabled_opts = cpu_disabled_optimizers(providers)
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=create_session_options(),
        providers=providers,
        disabled_optimizers=disabled_opts,
    )
    metadata = load_runtime_metadata(
        onnx_path,
        lambda model_path: ort.InferenceSession(
            model_path,
            sess_options=create_session_options(),
            providers=providers,
            disabled_optimizers=disabled_opts,
        ),
        required_keys=REQUIRED_FSMN_METADATA_KEYS,
    )
    validate_fsmn_metadata(metadata, session)
    config = config_from_metadata(metadata)
    sample_rate = metadata.required_int("input_sample_rate")
    model_sample_rate = metadata.required_int("model_sample_rate")
    frame_length = metadata.required_int("frame_length_samples")
    frame_shift = metadata.required_int("output_frame_shift_samples")
    minimum_input_samples = math.ceil(
        frame_length * sample_rate / model_sample_rate)
    audio = load_audio(audio_path, sample_rate, minimum_input_samples)
    audio_metadata = session.get_inputs()[0]
    expected_length = audio_metadata.shape[1]
    if not isinstance(expected_length, int):
        max_samples = sample_rate * MAX_DYNAMIC_AUDIO_SECONDS
        if audio.size > max_samples:
            raise ValueError(
                f"Audio has {audio.size} samples; MAX_DYNAMIC_AUDIO_SECONDS "
                f"limits dynamic input to {max_samples}."
            )

    audio_descriptor_shape = resolve_tensor_shape(
        audio_metadata,
        {1: audio.size} if not isinstance(expected_length, int) else {},
    )
    prepared_audio = normalise_audio(
        audio.reshape(*audio_descriptor_shape[:-1], -1),
        numpy_dtype_from_onnx_type(audio_metadata.type),
        NORMALIZE_TARGET_RMS,
    )
    postprocessor = E2EVadPostProcessor(config["model_conf"])
    start_time = time.perf_counter()
    if isinstance(expected_length, int):
        output_frames = session.get_outputs()[0].shape[1]
        if not isinstance(output_frames, int):
            raise ValueError("Static FSMN audio input requires a static score-frame output.")
        total_frames = fsmn_frame_count(
            audio.size,
            sample_rate,
            model_sample_rate,
            frame_length,
            frame_shift,
        )
        encoder_config = config["encoder_conf"]
        left_context_frames = (
            encoder_config["fsmn_layers"]
            * (encoder_config["lorder"] - 1)
            * encoder_config["lstride"]
            + LFR_HALF_CONTEXT_FRAMES
        )
        plan = static_window_plan(
            total_frames,
            output_frames,
            left_context_frames,
            LFR_HALF_CONTEXT_FRAMES,
        )
        runner = FSMNIOBinding(
            session,
            config,
            resolve_tensor_shape(audio_metadata),
            frame_length,
            frame_shift,
            sample_rate,
            model_sample_rate,
        )
        print(
            f"Running FSMN-VAD on {audio_path} in {len(plan)} fixed sliding windows "
            f"with I/O binding and {session.get_providers()} ..."
        )
        segments = []
        # The discarded left overlap flushes all four finite FSMN memories, so
        # each window intentionally starts from the binding's zero caches.
        for window_index, (window_start, output_start, output_stop) in enumerate(plan):
            audio_window = make_static_audio_window(
                prepared_audio,
                window_start,
                expected_length,
                frame_shift,
                sample_rate,
                model_sample_rate,
            )
            scores, decibels = runner.run(audio_window)
            is_final = window_index == len(plan) - 1
            segments.extend(postprocessor(
                scores[0, output_start:output_stop].copy(),
                decibels[0, output_start:output_stop].copy(),
                is_final=is_final,
            ))
    else:
        runner = FSMNIOBinding(
            session,
            config,
            prepared_audio.shape,
            frame_length,
            frame_shift,
            sample_rate,
            model_sample_rate,
        )
        print(
            f"Running FSMN-VAD on {audio_path} with dynamic I/O binding "
            f"and {session.get_providers()} ..."
        )
        scores, decibels = runner.run(prepared_audio)
        segments = postprocessor(scores[0], decibels[0], is_final=True)
    elapsed = time.perf_counter() - start_time
    print(f"Inference complete in {elapsed:.3f} seconds.")
    return segments, audio.size, sample_rate


def format_time(milliseconds):
    duration = timedelta(milliseconds=milliseconds)
    total_milliseconds = int(duration.total_seconds() * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def save_timestamps(segments, seconds_path, indices_path, sample_rate):
    seconds_path = Path(seconds_path)
    indices_path = Path(indices_path)
    seconds_path.parent.mkdir(parents=True, exist_ok=True)
    indices_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nTimestamps in seconds:")
    with open(seconds_path, "w", encoding="utf-8") as output:
        for start_ms, end_ms in segments:
            line = f"{format_time(start_ms)} --> {format_time(end_ms)}"
            output.write(line + "\n")
            print(line)

    print("\nTimestamps in sample indices:")
    with open(indices_path, "w", encoding="utf-8") as output:
        for start_ms, end_ms in segments:
            start_sample = int(start_ms * sample_rate / 1000)
            end_sample = int(end_ms * sample_rate / 1000)
            line = f"{start_sample} --> {end_sample}"
            output.write(line + "\n")
            print(line)


def main():
    segments, _, sample_rate = run_inference(ONNX_PATH, TEST_AUDIO, PROVIDERS)
    save_timestamps(
        segments,
        TIMESTAMPS_SECONDS,
        TIMESTAMPS_INDICES,
        sample_rate,
    )


if __name__ == "__main__":
    main()
