import time
from datetime import timedelta

import numpy as np
import onnxruntime
from pydub import AudioSegment

onnx_model_A = "/home/DakeQQ/Downloads/NVIDIA_MarbleNet_Optimized/NVIDIA_MarbleNet.onnx"  # The exported onnx model path.
test_vad_audio = "./vad_sample.wav"                                                       # The test audio path.
save_timestamps_second = "./timestamps_second.txt"                                        # The saved path.
save_timestamps_indices = "./timestamps_indices.txt"                                      # The saved path.

ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
SAMPLE_RATE = 16000                     # The model parameter, do not edit the value.
OUTPUT_FRAME_LENGTH = 320               # The model parameter, do not edit it.
MAX_THREADS = 4

# ─── VAD Postprocessing Parameters ───────────────────────────────────────────
OUTPUT_FRAME_SHIFT_S = OUTPUT_FRAME_LENGTH / SAMPLE_RATE    # 0.02 seconds (20ms frame shift for NVIDIA MarbleNet output)
SPEAKING_SCORE = 0.5                                        # Speech probability threshold.
SMOOTH_WINDOW_SIZE = 3                                      # Probability smoothing window size (in output frames).
MIN_SPEECH_FRAME = 10                                       # Min speech frames to confirm a segment.
MAX_SPEECH_FRAME = 1000                                     # Max speech frames before forced split.
MIN_SILENCE_FRAME = 10                                      # Min silence frames to confirm end of speech.
MERGE_SILENCE_FRAME = 3                                     # Merge silence gaps shorter than this (frames).
EXTEND_SPEECH_FRAME = 0                                     # Extend speech regions by this many frames.
DEVICE_ID = 0
NORMALIZE_AUDIO = False                 # Normalize the input audio to a target RMS level (e.g., 8192) before processing. It can help improve the performance of the model, especially for low-volume audio. Set it to True if you want to enable it.


if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
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
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
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
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
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


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A0 = out_name_A[0].name
out_name_A1 = out_name_A[1].name
out_name_A2 = out_name_A[2].name


def normalise_audio(audio: np.ndarray, target_rms: float = 8192.0) -> np.ndarray:
    _audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(_audio * _audio, dtype=np.float32), dtype=np.float32)
    if rms > 0:
        _audio *= (target_rms / (rms + 1e-7))
        np.clip(_audio, -32768.0, 32767.0, out=_audio)
        return _audio.astype(np.int16)
    else:
        return audio


# # Load the input audio
print(f"\nTest Input Audio: {test_vad_audio}")
audio = np.array(AudioSegment.from_file(test_vad_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
if NORMALIZE_AUDIO:
    audio = normalise_audio(audio)
audio_len = len(audio)
inv_audio_len = float(100.0 / audio_len)
audio = audio.reshape(1, 1, -1)

shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = min(SAMPLE_RATE * 3600, audio_len)  # You can adjust it.
else:
    INPUT_AUDIO_LENGTH = shape_value_in
stride_step = INPUT_AUDIO_LENGTH
if audio_len > INPUT_AUDIO_LENGTH:
    num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
    total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
    pad_amount = total_length_needed - audio_len
    final_slice = audio[:, :, -pad_amount:].astype(np.float32)
    white_noise = (np.sqrt(np.mean(final_slice * final_slice)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
elif audio_len < INPUT_AUDIO_LENGTH:
    audio_float = audio.astype(np.float32)
    white_noise = (np.sqrt(np.mean(audio_float * audio_float)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
aligned_len = audio.shape[-1]


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
    score_silence, score_active, signal_len = ort_session_A.run(
        [out_name_A0, out_name_A1, out_name_A2], {in_name_A0: audio[:, :, slice_start: slice_end]})
    # score_active shape: [1, signal_len, 1] -> extract speech probs as 1D
    valid_frames = min(int(signal_len[0]), score_active.shape[1])
    all_vad_probs.append(score_active[0, :valid_frames, 0])
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
    smooth_window_size=SMOOTH_WINDOW_SIZE,
    prob_threshold=SPEAKING_SCORE,
    min_speech_frame=MIN_SPEECH_FRAME,
    max_speech_frame=MAX_SPEECH_FRAME,
    min_silence_frame=MIN_SILENCE_FRAME,
    merge_silence_frame=MERGE_SILENCE_FRAME,
    extend_speech_frame=EXTEND_SPEECH_FRAME,
    frame_shift_s=OUTPUT_FRAME_SHIFT_S,
)
vad_decisions = vad_postprocessor.process(all_vad_probs)
timestamps = vad_postprocessor.decision_to_segment(vad_decisions, audio_len / SAMPLE_RATE)
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
        line = f"{int(start * SAMPLE_RATE)} --> {int(end * SAMPLE_RATE)}\n"
        file.write(line)
        print(line.replace("\n", ""))

print(f"\nVAD Process Complete.\n\nTime Cost: {end_time - start_time:.3f} Seconds")
