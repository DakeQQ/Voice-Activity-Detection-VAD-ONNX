import time
import numpy as np
import onnxruntime
from datetime import timedelta
from pydub import AudioSegment


# ─── Model Paths ──────────────────────────────────────────────────────────────
onnx_model_vad          = "/home/DakeQQ/Downloads/FireRedVAD_Optimized/FireRedVAD.onnx"            # The exported VAD onnx model path.
onnx_model_aed          = "/home/DakeQQ/Downloads/FireRedVAD_Optimized/FireRedAED.onnx"            # The exported AED onnx model path.
onnx_model_stream_vad   = "/home/DakeQQ/Downloads/FireRedVAD_Optimized/FireRedStreamVAD.onnx"      # The exported Stream-VAD onnx model path.
test_vad_audio          = "./vad_sample.wav"                                                       # The VAD test audio path.
test_aed_audio          = "./vad_sample.wav"                                                       # The AED test audio path.
save_timestamps_second  = "./timestamps_second.txt"                                                # The saved path.
save_timestamps_indices = "./timestamps_indices.txt"                                               # The saved path.


# ─── Inference Settings ───────────────────────────────────────────────────────
RUN_VAD = True                          # Run the VAD model inference.
RUN_AED = True                          # Run the AED model inference.
RUN_STREAM_VAD = True                   # Run the Stream-VAD model inference.
NORMALIZE_AUDIO = False                 # Normalize input audio to target RMS level.
ORT_Accelerate_Providers = []           # e.g. ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'MIGraphXExecutionProvider']


# ─── Audio & STFT Parameters (do not edit) ────────────────────────────────────
SAMPLE_RATE = 16000                     # Hz
FRAME_SHIFT_MS = 10                     # Frame shift (ms).
FRAME_LENGTH_MS = 25                    # Frame length (ms).
HOP_LENGTH = 160                        # Samples between frames (frame_shift_ms * sample_rate / 1000).
WINDOW_LENGTH = 400                     # Window length in samples (frame_length_ms * sample_rate / 1000).

# ─── Stream-VAD Fixed Chunk ──────────────────────────────────────────────────
STREAM_CHUNK_MS = 160                   # [80, 160, 200] Fixed streaming chunk size (ms).
STREAM_CHUNK_SAMPLES = int(SAMPLE_RATE * STREAM_CHUNK_MS / 1000)  # 2560 samples.

# ─── Derived Constants ────────────────────────────────────────────────────────
FRAME_SHIFT_S = FRAME_SHIFT_MS / 1000.0
FRAME_LENGTH_S = FRAME_LENGTH_MS / 1000.0
FRAME_PER_SECONDS = int(1000 / FRAME_SHIFT_MS)
_FRAME_SHIFT_F32 = np.float32(FRAME_SHIFT_S)
_FRAME_LENGTH_F32 = np.float32(FRAME_LENGTH_S)

# ─── VAD Postprocessing ──────────────────────────────────────────────────────
SPEAKING_SCORE = 0.4                    # Speech probability threshold.
MIN_SPEECH_FRAME = 20                   # Min speech frames to confirm a segment.
MAX_SPEECH_FRAME = 2000                 # Max speech frames before forced split.
MIN_SILENCE_FRAME = 20                  # Min silence frames to confirm end of speech.
MERGE_SILENCE_FRAME = 5                 # Merge silence gaps shorter than this (frames).
EXTEND_SPEECH_FRAME = 0                 # Extend speech regions by this many frames.
SMOOTH_WINDOW_SIZE = 5                  # Probability smoothing window size.

# ─── AED (Audio Event Detection) Postprocessing ──────────────────────────────
MIN_EVENT_FRAME = 20                    # Min event frames to confirm a segment.
MAX_EVENT_FRAME = 2000                  # Max event frames before forced split.
SINGING_THRESHOLD = 0.5                 # Singing detection threshold.
MUSIC_THRESHOLD = 0.5                   # Music detection threshold.

# ─── Stream-VAD Settings ─────────────────────────────────────────────────────
STREAM_VAD_THRESHOLD = 0.4              # Speech probability threshold.
PAD_START_FRAME = 5                     # Frames to pad at segment start.
MIN_SPEECH_FRAME_STREAM = 8             # Min speech frames to confirm a segment.
MAX_SPEECH_FRAME_STREAM = 2000          # Max speech frames before forced split.
MIN_SILENCE_FRAME_STREAM = 20           # Min silence frames to confirm end of speech.


# ═══════════════════════════════════════════════════════════════════════════════
# ONNX Runtime Inference Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def normalise_audio(audio: np.ndarray, target_rms: float = 8192.0) -> np.ndarray:
    _audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(_audio * _audio, dtype=np.float32), dtype=np.float32)
    if rms > 0:
        _audio *= (target_rms / (rms + 1e-7))
        np.clip(_audio, -32768.0, 32767.0, out=_audio)
        return _audio.astype(np.int16)
    else:
        return audio


def valid_frame_count(num_samples: int) -> int:
    """Return the number of valid frames for snip_edges=True fbank extraction."""
    if num_samples < WINDOW_LENGTH:
        return 0
    return 1 + (num_samples - WINDOW_LENGTH) // HOP_LENGTH


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
                 'extend_speech_frame', '_inv_ws', '_half_max')

    def __init__(self, smooth_window_size, prob_threshold, min_speech_frame,
                 max_speech_frame, min_silence_frame, merge_silence_frame,
                 extend_speech_frame):
        self.smooth_window_size = max(1, smooth_window_size)
        self.prob_threshold = np.float32(prob_threshold)
        self.min_speech_frame = min_speech_frame
        self.max_speech_frame = max_speech_frame
        self.min_silence_frame = min_silence_frame
        self.merge_silence_frame = merge_silence_frame
        self.extend_speech_frame = extend_speech_frame
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
        segments[:, 0] = starts * _FRAME_SHIFT_F32
        segments[:, 1] = ends * _FRAME_SHIFT_F32

        if dec[n - 1] != 0:
            end_time = n * _FRAME_SHIFT_F32 + _FRAME_LENGTH_F32
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
                 'hit_max_speech', 'last_speech_start_frame', 'last_speech_end_frame')

    def __init__(self, smooth_window_size, speech_threshold, pad_start_frame,
                 min_speech_frame, max_speech_frame, min_silence_frame):
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

        inv_fps = 1.0 / FRAME_PER_SECONDS
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
                timestamps.append((s_frame * inv_fps, e_frame * inv_fps))

        # Handle unterminated segment at end of stream
        if last_start > 0:
            s_frame = max(0, last_start - 1)
            e_frame = frame_cnt - 1
            timestamps.append((s_frame * inv_fps, e_frame * inv_fps))

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


if RUN_VAD:
    print('\n\nStart to run FireRedVAD by ONNX Runtime.\n\nNow, loading the model...')

    ort_session_A = onnxruntime.InferenceSession(onnx_model_vad, sess_options=session_opts, providers=ORT_Accelerate_Providers)
    print(f"\nUsable Providers: {ort_session_A.get_providers()}")
    in_name_A = ort_session_A.get_inputs()
    out_name_A = ort_session_A.get_outputs()
    in_name_A0 = in_name_A[0].name
    out_name_A0 = out_name_A[0].name

    # Load the input audio
    print(f"\nTest Input Audio: {test_vad_audio}")
    audio = np.array(AudioSegment.from_file(test_vad_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
    if NORMALIZE_AUDIO:
        audio = normalise_audio(audio)
    audio_len = len(audio)
    audio = audio.reshape(1, 1, -1)

    shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
    if isinstance(shape_value_in, str):
        INPUT_AUDIO_LENGTH_RUN = min(SAMPLE_RATE * 3600, audio_len)  # You can adjust it.
    else:
        INPUT_AUDIO_LENGTH_RUN = shape_value_in

    stride_step = INPUT_AUDIO_LENGTH_RUN
    if audio_len > INPUT_AUDIO_LENGTH_RUN:
        num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH_RUN) / stride_step)) + 1
        total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH_RUN
        pad_amount = total_length_needed - audio_len
        final_slice = audio[:, :, -pad_amount:].astype(np.float32)
        white_noise = (np.sqrt(np.mean(final_slice * final_slice)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio.dtype)
        audio = np.concatenate((audio, white_noise), axis=-1)
    elif audio_len < INPUT_AUDIO_LENGTH_RUN:
        audio_float = audio.astype(np.float32)
        white_noise = (np.sqrt(np.mean(audio_float * audio_float)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH_RUN - audio_len))).astype(audio.dtype)
        audio = np.concatenate((audio, white_noise), axis=-1)
    aligned_len = audio.shape[-1]

    # Start to run FireRedVAD
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH_RUN
    all_vad_probs = []
    print("\nRunning the FireRedVAD by ONNX Runtime.")
    start_time = time.time()
    while slice_end <= aligned_len:
        probs = ort_session_A.run(
            [out_name_A0], {in_name_A0: audio[:, :, slice_start: slice_end]})[0]
        all_vad_probs.append(probs[0, 0])
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH_RUN
    end_time = time.time()

    valid_vad_frames = valid_frame_count(audio_len)
    if all_vad_probs:
        all_vad_probs = np.concatenate(all_vad_probs, axis=0)[:valid_vad_frames]
    else:
        all_vad_probs = np.zeros((0,), dtype=np.float32)

    vad_postprocessor = VadPostprocessor(
        SMOOTH_WINDOW_SIZE,
        SPEAKING_SCORE,
        MIN_SPEECH_FRAME,
        MAX_SPEECH_FRAME,
        MIN_SILENCE_FRAME,
        MERGE_SILENCE_FRAME,
        EXTEND_SPEECH_FRAME,
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

    vad_rtf = (end_time - start_time) / (audio_len / SAMPLE_RATE)
    print(f"\nVAD Process Complete.\n\nTime Cost: {end_time - start_time:.3f} Seconds")
    print(f"RTF: {vad_rtf:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Run AED (Audio Event Detection) Inference with ONNX Runtime
# ═══════════════════════════════════════════════════════════════════════════════

if RUN_AED:
    print('\n\n' + '=' * 70)
    print('Start to run FireRedAED by ONNX Runtime.\n\nNow, loading the model...')

    ort_session_B = onnxruntime.InferenceSession(onnx_model_aed, sess_options=session_opts, providers=ORT_Accelerate_Providers)
    print(f"\nUsable Providers: {ort_session_B.get_providers()}")
    in_name_B = ort_session_B.get_inputs()
    out_name_B = ort_session_B.get_outputs()
    in_name_B0 = in_name_B[0].name
    out_name_B0 = out_name_B[0].name

    # AED event mapping: index -> event name
    IDX2EVENT = {0: "speech", 1: "singing", 2: "music"}
    EVENT_THRESHOLDS = {"speech": SPEAKING_SCORE, "singing": SINGING_THRESHOLD, "music": MUSIC_THRESHOLD}

    # Load the input audio
    print(f"\nTest Input Audio: {test_aed_audio}")
    audio_aed = np.array(AudioSegment.from_file(test_aed_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
    if NORMALIZE_AUDIO:
        audio_aed = normalise_audio(audio_aed)
    audio_aed_len = len(audio_aed)
    audio_aed_dur = audio_aed_len / SAMPLE_RATE
    audio_aed = audio_aed.reshape(1, 1, -1)

    shape_value_in_B = ort_session_B._inputs_meta[0].shape[-1]
    if isinstance(shape_value_in_B, str):
        INPUT_AUDIO_LENGTH_RUN_B = min(SAMPLE_RATE * 3600, audio_aed_len)
    else:
        INPUT_AUDIO_LENGTH_RUN_B = shape_value_in_B

    stride_step_B = INPUT_AUDIO_LENGTH_RUN_B
    if audio_aed_len > INPUT_AUDIO_LENGTH_RUN_B:
        num_windows_B = int(np.ceil((audio_aed_len - INPUT_AUDIO_LENGTH_RUN_B) / stride_step_B)) + 1
        total_length_needed_B = (num_windows_B - 1) * stride_step_B + INPUT_AUDIO_LENGTH_RUN_B
        pad_amount_B = total_length_needed_B - audio_aed_len
        final_slice_B = audio_aed[:, :, -pad_amount_B:].astype(np.float32)
        white_noise_B = (np.sqrt(np.mean(final_slice_B * final_slice_B)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount_B))).astype(audio_aed.dtype)
        audio_aed = np.concatenate((audio_aed, white_noise_B), axis=-1)
    elif audio_aed_len < INPUT_AUDIO_LENGTH_RUN_B:
        audio_aed_float = audio_aed.astype(np.float32)
        white_noise_B = (np.sqrt(np.mean(audio_aed_float * audio_aed_float)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH_RUN_B - audio_aed_len))).astype(audio_aed.dtype)
        audio_aed = np.concatenate((audio_aed, white_noise_B), axis=-1)
    aligned_len_B = audio_aed.shape[-1]

    # Run AED inference
    print("\nRunning the FireRedAED by ONNX Runtime.")
    start_time = time.time()
    all_probs = []
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH_RUN_B
    while slice_end <= aligned_len_B:
        probs_aed = ort_session_B.run(
            [out_name_B0], {in_name_B0: audio_aed[:, :, slice_start: slice_end]})[0]
        # probs_aed shape: [1, 3, T]
        all_probs.append(probs_aed[0])  # [3, T]
        slice_start += stride_step_B
        slice_end = slice_start + INPUT_AUDIO_LENGTH_RUN_B
    end_time = time.time()

    valid_aed_frames = valid_frame_count(audio_aed_len)
    if all_probs:
        all_probs = np.concatenate(all_probs, axis=1)[:, :valid_aed_frames]
    else:
        all_probs = np.zeros((len(IDX2EVENT), 0), dtype=np.float32)

    # Post-process each event type
    event2timestamps = {}
    event2ratio = {}
    event2postprocessor = {
        "speech": VadPostprocessor(
            SMOOTH_WINDOW_SIZE,
            SPEAKING_SCORE,
            MIN_EVENT_FRAME,
            MAX_EVENT_FRAME,
            MIN_SILENCE_FRAME,
            MERGE_SILENCE_FRAME,
            EXTEND_SPEECH_FRAME,
        ),
        "singing": VadPostprocessor(
            SMOOTH_WINDOW_SIZE,
            SINGING_THRESHOLD,
            MIN_EVENT_FRAME,
            MAX_EVENT_FRAME,
            MIN_SILENCE_FRAME,
            MERGE_SILENCE_FRAME,
            EXTEND_SPEECH_FRAME,
        ),
        "music": VadPostprocessor(
            SMOOTH_WINDOW_SIZE,
            MUSIC_THRESHOLD,
            MIN_EVENT_FRAME,
            MAX_EVENT_FRAME,
            MIN_SILENCE_FRAME,
            MERGE_SILENCE_FRAME,
            EXTEND_SPEECH_FRAME,
        ),
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

    ort_session_C = onnxruntime.InferenceSession(onnx_model_stream_vad, sess_options=session_opts, providers=ORT_Accelerate_Providers)
    print(f"\nUsable Providers: {ort_session_C.get_providers()}")
    in_name_C = ort_session_C.get_inputs()
    out_name_C = ort_session_C.get_outputs()
    in_name_C0 = in_name_C[0].name  # audio
    in_name_C1 = in_name_C[1].name  # caches_in
    out_name_C0 = out_name_C[0].name  # probs
    out_name_C1 = out_name_C[1].name  # caches_out

    # Determine cache shape from the model metadata
    cache_shape = ort_session_C._inputs_meta[1].shape
    # cache_shape should be [R, 1, P, lookback_padding]
    R_val = cache_shape[0]
    P_val = cache_shape[2]
    L_val = cache_shape[3]
    print(f"  Cache shape: [{R_val}, 1, {P_val}, {L_val}]")

    # Load the input audio
    print(f"\nTest Input Audio: {test_vad_audio}")
    audio_stream = np.array(AudioSegment.from_file(test_vad_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
    if NORMALIZE_AUDIO:
        audio_stream = normalise_audio(audio_stream)
    audio_stream_len = len(audio_stream)
    audio_stream_dur = audio_stream_len / SAMPLE_RATE

    # Initialize caches to zero
    caches = np.zeros((R_val, 1, P_val, L_val), dtype=np.float32)

    # Collect all frame probabilities
    all_stream_probs = []

    print(f"\nRunning the FireRedStreamVAD by ONNX Runtime.")
    print(f"  Audio duration: {audio_stream_dur:.3f}s")
    print(f"  Chunk size: {STREAM_CHUNK_MS}ms ({STREAM_CHUNK_SAMPLES} samples)")
    start_time = time.time()

    # Process audio in streaming chunks
    pos = 0
    while pos < audio_stream_len:
        chunk_end = min(pos + STREAM_CHUNK_SAMPLES, audio_stream_len)
        chunk = audio_stream[pos:chunk_end]

        # Pad last chunk if needed (must be at least WINDOW_LENGTH samples for 1 frame)
        if len(chunk) < WINDOW_LENGTH:
            chunk = np.pad(chunk, (0, WINDOW_LENGTH - len(chunk)), mode='constant')

        chunk_input = chunk.reshape(1, 1, -1).astype(np.int16)

        # Run inference with caches
        probs_out, caches = ort_session_C.run(
            [out_name_C0, out_name_C1],
            {in_name_C0: chunk_input, in_name_C1: caches}
        )
        # probs_out shape: [1, 1, T]
        all_stream_probs.append(probs_out[0, 0])  # [T]

        pos = chunk_end

    end_time = time.time()

    # Concatenate all frame probabilities
    valid_stream_frames = valid_frame_count(audio_stream_len)
    if all_stream_probs:
        all_stream_probs = np.concatenate(all_stream_probs, axis=0)[:valid_stream_frames]
    else:
        all_stream_probs = np.zeros((0,), dtype=np.float32)

    stream_postprocessor = StreamVadPostprocessor(
        SMOOTH_WINDOW_SIZE,
        STREAM_VAD_THRESHOLD,
        PAD_START_FRAME,
        MIN_SPEECH_FRAME_STREAM,
        MAX_SPEECH_FRAME_STREAM,
        MIN_SILENCE_FRAME_STREAM,
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
