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
from vad_onnx_metadata import REQUIRED_VAD_METADATA_KEYS, load_runtime_metadata
from Example_Audio import model_audio_path


script_dir = Path(__file__).resolve().parent
output_onnx_model = script_dir / "silero_vad_optimized.onnx"
test_vad_audio = model_audio_path("silero")                     # The test audio path.
save_timestamps_second = script_dir / "timestamps_second.txt"   # The saved path.
save_timestamps_indices = script_dir / "timestamps_indices.txt" # The saved path.

# User-adjustable inference settings.
NORMALIZE_AUDIO = False                 # True applies RMS normalization before inference.
NORMALIZE_TARGET_RMS = 4096.0           # Target PCM16 RMS; used only when normalization is enabled.
MAX_DYNAMIC_AUDIO_SECONDS = 3600        # Maximum accepted duration for a dynamic-length input.
ACTIVATE_THRESHOLD = 0.5                # Speech score required to start a segment.
NEGATIVE_THRESHOLD = 0.35               # Speech score below which an active segment may end.
FUSION_THRESHOLD = 0.3                  # Merge speech segments separated by at most this many seconds.
MIN_SPEECH_DURATION_MS = 250            # Drop speech segments shorter than this many milliseconds.
MIN_SPEECH_DURATION = MIN_SPEECH_DURATION_MS * 0.001  # Seconds used by final timestamp filtering.
MAX_SPEECH_DURATION = 20.0              # Split speech segments longer than this many seconds.
MIN_SILENCE_DURATION = 250              # Silence required to close a segment, in milliseconds.
SPEECH_PAD_MS = 30                      # Pad each speech segment by this many milliseconds per side.
MIN_SILENCE_AT_MAX_SPEECH_MS = 98       # Candidate silence used when splitting long speech.
USE_MAX_POSS_SIL_AT_MAX_SPEECH = True   # Prefer the longest candidate silence for long-speech splits.
USE_CPU = True                          # True forces CPU; False tries available providers in ORT order.
MAX_THREADS = 0                         # 0 uses all available threads; >0 limits intra-op threads.

CPU_DISABLE_MATMUL_ADD_FUSION = True    # ORT 1.27 wraps rank-3 MatMul+Add in costly Reshape/Gemm/Reshape chains.
CPU_DISABLE_NCHWC = True                # NCHWc reorders regress mean/tail latency on the target i7-1165G7.
CPU_EXTRA_DISABLED_OPTIMIZERS = [       # Individually benchmarked on the same CPU / ORT build.
    "ConvAddActivationFusion",
    "MatmulTransposeFusion",
]

CPU_DISABLED_OPTIMIZERS = []
if CPU_DISABLE_MATMUL_ADD_FUSION:
    CPU_DISABLED_OPTIMIZERS.append("MatMulAddFusion")
if CPU_DISABLE_NCHWC:
    CPU_DISABLED_OPTIMIZERS.append("NchwcTransformer")
CPU_DISABLED_OPTIMIZERS.extend(CPU_EXTRA_DISABLED_OPTIMIZERS)


class OnnxWrapper:
    def __init__(self, path, force_onnx_cpu, sample_rate,
                 window_size_samples, context_size_samples,
                 cpu_disabled_optimizers=None):
        session_options = onnxruntime.SessionOptions()
        run_options = onnxruntime.RunOptions()
        session_options.log_severity_level = 4
        session_options.log_verbosity_level = 4
        run_options.log_severity_level = 4
        run_options.log_verbosity_level = 4
        session_options.inter_op_num_threads = MAX_THREADS
        session_options.intra_op_num_threads = MAX_THREADS
        session_options.enable_cpu_mem_arena = True
        session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.add_session_config_entry("session.set_denormal_as_zero", "1")
        session_options.add_session_config_entry("session.intra_op.allow_spinning", "1")
        session_options.add_session_config_entry("session.inter_op.allow_spinning", "1")
        session_options.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
        session_options.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
        session_options.add_session_config_entry("optimization.enable_gelu_approximation", "1")
        session_options.add_session_config_entry("optimization.minimal_build_optimizations", "")
        session_options.add_session_config_entry(
            "session.use_device_allocator_for_initializers", "1")
        session_options.add_session_config_entry(
            "optimization.enable_cast_chain_elimination", "1")
        session_options.add_session_config_entry(
            "session.graph_optimizations_loop_level", "2")
        run_options.add_run_config_entry(
            "disable_synchronize_execution_providers", "1")

        self._run_options = run_options
        self.session = self._create_session(
            path,
            force_onnx_cpu,
            session_options,
            cpu_disabled_optimizers,
        )
        self.sample_rate = sample_rate
        self.window_size_samples = window_size_samples
        self.context_size_samples = context_size_samples
        self.input_size_samples = context_size_samples + window_size_samples
        self._device = self._device_type()
        self._io_key = None
        self._bindings = []
        self._state_buffers = []
        self._state_hosts = []
        self._state_index = 0
        self._last_batch_size = 0
        self._build_bindings(1, self.input_size_samples)

    @staticmethod
    def _create_session(path, force_onnx_cpu, session_options,
                        cpu_disabled_optimizers):
        available_providers = onnxruntime.get_available_providers()
        if force_onnx_cpu and "CPUExecutionProvider" in available_providers:
            return onnxruntime.InferenceSession(
                path,
                providers=["CPUExecutionProvider"],
                sess_options=session_options,
                disabled_optimizers=cpu_disabled_optimizers,
            )

        failures = []
        for provider in available_providers:
            try:
                print(f"Attempting to create InferenceSession with provider: {provider}")
                session = onnxruntime.InferenceSession(
                    path,
                    providers=[provider],
                    sess_options=session_options,
                    disabled_optimizers=(
                        cpu_disabled_optimizers
                        if provider == "CPUExecutionProvider" else None
                    ),
                )
                print(f"Successfully created InferenceSession with {provider}.")
                return session
            except Exception as error:
                failures.append(f"{provider}: {error}")
        raise RuntimeError(
            "Unable to create a Silero ONNX Runtime session: " + "; ".join(failures)
        )

    @staticmethod
    def _numpy_dtype(type_name):
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

    @staticmethod
    def _runtime_shape(node_arg, runtime_shape):
        if len(node_arg.shape) != len(runtime_shape):
            raise ValueError(
                f"Tensor {node_arg.name!r} expects rank {len(node_arg.shape)}, "
                f"got runtime shape {runtime_shape}."
            )
        resolved = []
        for dimension, runtime_dimension in zip(node_arg.shape, runtime_shape):
            if isinstance(dimension, int) and dimension != runtime_dimension:
                raise ValueError(
                    f"Tensor {node_arg.name!r} expects shape {node_arg.shape}, "
                    f"got runtime shape {runtime_shape}."
                )
            resolved.append(
                dimension if isinstance(dimension, int) else runtime_dimension
            )
        return tuple(resolved)

    @staticmethod
    def _single_dynamic_shape(node_arg, dynamic_extent=None):
        dynamic_axes = [
            axis
            for axis, dimension in enumerate(node_arg.shape)
            if not isinstance(dimension, int)
        ]
        if dynamic_axes and dynamic_extent is None:
            raise ValueError(
                f"Tensor {node_arg.name!r} has unresolved shape {node_arg.shape}."
            )
        if len(dynamic_axes) > 1:
            raise ValueError(
                f"Tensor {node_arg.name!r} has multiple dynamic axes "
                f"{node_arg.shape}."
            )
        return tuple(
            dimension if isinstance(dimension, int) else dynamic_extent
            for dimension in node_arg.shape
        )

    def _device_type(self):
        provider = self.session.get_providers()[0]
        if provider in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}:
            return "cuda"
        if provider == "DmlExecutionProvider":
            return "dml"
        return "cpu"

    def _create_buffer(self, shape, dtype):
        host_array = np.zeros(shape, dtype=dtype)
        ort_value = onnxruntime.OrtValue.ortvalue_from_numpy(
            host_array,
            self._device,
            0,
        )
        return host_array, ort_value

    def _sync_to_device(self, buffer, host_array):
        if self._device != "cpu":
            buffer.update_inplace(host_array)

    def _build_bindings(self, batch_size, input_length):
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if [item.name for item in inputs] != ["input", "state", "sr"]:
            raise ValueError(f"Unexpected Silero inputs: {[item.name for item in inputs]}")
        if [item.name for item in outputs] != ["output", "stateN"]:
            raise ValueError(f"Unexpected Silero outputs: {[item.name for item in outputs]}")

        audio_shape = self._runtime_shape(
            inputs[0], (batch_size, input_length)
        )
        state_shape = self._single_dynamic_shape(inputs[1], batch_size)
        output_state_shape = self._runtime_shape(outputs[1], state_shape)
        if state_shape != output_state_shape:
            raise ValueError(
                f"Silero state input/output shapes must match, got "
                f"{state_shape} and {output_state_shape}."
            )
        state_dtype = self._numpy_dtype(inputs[1].type)
        if state_dtype != self._numpy_dtype(outputs[1].type):
            raise TypeError("Silero state input and output dtypes must match.")

        self._audio_dtype = self._numpy_dtype(inputs[0].type)
        self._sample_rate_dtype = self._numpy_dtype(inputs[2].type)
        self._state_dtype = state_dtype
        self._audio_host, self._audio_buffer = self._create_buffer(
            audio_shape, self._audio_dtype)
        self._audio_window_host = self._audio_host[:, self.context_size_samples:]
        if self.context_size_samples:
            self._context_host = self._audio_host[:, :self.context_size_samples]
            self._next_context_host = self._audio_host[:, -self.context_size_samples:]
        else:
            self._context_host = None
            self._next_context_host = None
        self._sample_rate_host, self._sample_rate_buffer = self._create_buffer(
            self._single_dynamic_shape(inputs[2]), self._sample_rate_dtype)
        self._output_host, self._output_buffer = self._create_buffer(
            self._single_dynamic_shape(outputs[0], batch_size),
            self._numpy_dtype(outputs[0].type),
        )
        state_buffer_pairs = [
            self._create_buffer(state_shape, state_dtype),
            self._create_buffer(state_shape, state_dtype),
        ]
        self._state_hosts = [pair[0] for pair in state_buffer_pairs]
        self._state_buffers = [pair[1] for pair in state_buffer_pairs]
        self._sample_rate_host[...] = self.sample_rate
        self._sync_to_device(self._sample_rate_buffer, self._sample_rate_host)
        self._bindings = []
        for state_index in range(2):
            binding = self.session.io_binding()
            binding.bind_ortvalue_input(inputs[0].name, self._audio_buffer)
            binding.bind_ortvalue_input(
                inputs[1].name, self._state_buffers[state_index])
            binding.bind_ortvalue_input(inputs[2].name, self._sample_rate_buffer)
            binding.bind_ortvalue_output(outputs[0].name, self._output_buffer)
            binding.bind_ortvalue_output(
                outputs[1].name, self._state_buffers[1 - state_index])
            self._bindings.append(binding)
        self._io_key = (batch_size, input_length)
        self._state_index = 0

    def reset_states(self):
        self._state_hosts[0].fill(0)
        self._sync_to_device(self._state_buffers[0], self._state_hosts[0])
        if self._io_key is not None and self._context_host is not None:
            self._context_host.fill(0)
        self._state_index = 0
        self._last_batch_size = 0

    def __call__(self, audio_chunk: np.ndarray, sample_rate):
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Expected {self.sample_rate} Hz audio, got {sample_rate} Hz."
            )
        if audio_chunk.ndim == 1:
            audio_chunk = audio_chunk[np.newaxis, :]
        if audio_chunk.ndim != 2:
            raise ValueError(
                f"Silero audio chunks must have one or two dimensions, got "
                f"{audio_chunk.ndim}."
            )
        chunk_length = audio_chunk.shape[1]
        if not 0 < chunk_length <= self.window_size_samples:
            raise ValueError(
                f"Provided number of samples is {chunk_length}; expected between "
                f"1 and {self.window_size_samples} for {sample_rate} Hz."
            )

        batch_size = audio_chunk.shape[0]
        if self._io_key is None or self._io_key[0] != batch_size:
            self._build_bindings(batch_size, self.input_size_samples)
        elif self._context_host is not None and self._last_batch_size:
            self._context_host[...] = self._next_context_host
        if chunk_length == self.window_size_samples:
            self._audio_window_host[...] = audio_chunk
        else:
            self._audio_window_host[:, :chunk_length] = audio_chunk
            self._audio_window_host[:, chunk_length:].fill(0)
        if self._device != "cpu":
            self._audio_buffer.update_inplace(self._audio_host)
        self.session.run_with_iobinding(
            self._bindings[self._state_index],
            run_options=self._run_options,
        )
        if self._device == "cpu":
            speech_probability = float(self._output_host.reshape(-1)[0])
        else:
            speech_probability = float(self._output_buffer.numpy().reshape(-1)[0])
        self._state_index = 1 - self._state_index
        self._last_batch_size = batch_size
        return speech_probability


def get_speech_timestamps(audio: np.ndarray, model, *, threshold,
                          neg_threshold, sampling_rate,
                          min_speech_samples, max_speech_samples,
                          min_silence_samples, speech_pad_samples,
                          min_silence_samples_at_max_speech,
                          window_size_samples,
                          use_max_poss_sil_at_max_speech,
                          return_seconds=False, time_resolution=1):
    if audio.ndim != 1:
        raise ValueError("Silero VAD expects one-dimensional mono audio.")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")

    model.reset_states()
    audio_length_samples = len(audio)

    speeches = []
    current_speech_start = None
    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)
    temp_end = 0
    prev_end = 0
    next_start = 0
    best_silence_end = 0
    best_silence_duration = 0

    for current_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_sample:current_sample + window_size_samples]
        speech_prob = model(chunk, sampling_rate)

        if speech_prob >= threshold and temp_end:
            silence_duration = current_sample - temp_end
            if (
                silence_duration > min_silence_samples_at_max_speech
                and silence_duration > best_silence_duration
            ):
                best_silence_end = temp_end
                best_silence_duration = silence_duration
            temp_end = 0
            if next_start < prev_end:
                next_start = current_sample

        if speech_prob >= threshold and current_speech_start is None:
            current_speech_start = current_sample
            continue

        if (
            current_speech_start is not None
            and current_sample - current_speech_start > max_speech_samples
        ):
            if use_max_poss_sil_at_max_speech and best_silence_end:
                prev_end = best_silence_end
                silence_duration = best_silence_duration
                speeches.append({
                    "start": current_speech_start,
                    "end": prev_end,
                })
                next_start = prev_end + silence_duration
                if next_start < prev_end + current_sample:
                    current_speech_start = next_start
                else:
                    current_speech_start = None
                prev_end = next_start = temp_end = 0
                best_silence_end = best_silence_duration = 0
            elif prev_end:
                speeches.append({
                    "start": current_speech_start,
                    "end": prev_end,
                })
                if next_start < prev_end:
                    current_speech_start = None
                else:
                    current_speech_start = next_start
                prev_end = next_start = temp_end = 0
                best_silence_end = best_silence_duration = 0
            else:
                speeches.append({
                    "start": current_speech_start,
                    "end": current_sample,
                })
                current_speech_start = None
                prev_end = next_start = temp_end = 0
                best_silence_end = best_silence_duration = 0
                continue

        if speech_prob < neg_threshold and current_speech_start is not None:
            if not temp_end:
                temp_end = current_sample
            current_silence_duration = current_sample - temp_end
            if (
                not use_max_poss_sil_at_max_speech
                and current_silence_duration > min_silence_samples_at_max_speech
            ):
                prev_end = temp_end
            if current_silence_duration < min_silence_samples:
                continue

            if temp_end - current_speech_start > min_speech_samples:
                speeches.append({
                    "start": current_speech_start,
                    "end": temp_end,
                })
            current_speech_start = None
            prev_end = next_start = temp_end = 0
            best_silence_end = best_silence_duration = 0

    if (
        current_speech_start is not None
        and audio_length_samples - current_speech_start > min_speech_samples
    ):
        speeches.append({
            "start": current_speech_start,
            "end": audio_length_samples,
        })

    for index, speech in enumerate(speeches):
        if index == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if index != len(speeches) - 1:
            silence_duration = speeches[index + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[index + 1]["start"] = int(max(
                    0,
                    speeches[index + 1]["start"] - silence_duration // 2,
                ))
            else:
                speech["end"] = int(min(
                    audio_length_samples,
                    speech["end"] + speech_pad_samples,
                ))
                speeches[index + 1]["start"] = int(max(
                    0,
                    speeches[index + 1]["start"] - speech_pad_samples,
                ))
        else:
            speech["end"] = int(min(
                audio_length_samples,
                speech["end"] + speech_pad_samples,
            ))

    if return_seconds:
        audio_length_seconds = audio_length_samples / sampling_rate
        for speech in speeches:
            speech["start"] = max(
                round(speech["start"] / sampling_rate, time_resolution), 0)
            speech["end"] = min(
                round(speech["end"] / sampling_rate, time_resolution),
                audio_length_seconds,
            )
    return speeches

REQUIRED_SILERO_METADATA_KEYS = (
    *REQUIRED_VAD_METADATA_KEYS,
    "window_size_samples",
    "context_size_samples",
    "state_layers",
    "state_size",
    "supported_sample_rates",
    "num_inputs",
    "num_outputs",
)


def make_metadata_session(model_path):
    return onnxruntime.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
        disabled_optimizers=CPU_DISABLED_OPTIMIZERS or None,
    )


def validate_silero_metadata(metadata, session):
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != metadata.required_int("num_inputs"):
        raise ValueError("Silero input count does not match metadata.")
    if len(outputs) != metadata.required_int("num_outputs"):
        raise ValueError("Silero output count does not match metadata.")
    if [item.name for item in inputs] != ["input", "state", "sr"]:
        raise ValueError(f"Unexpected Silero inputs: {[item.name for item in inputs]}.")
    if [item.name for item in outputs] != ["output", "stateN"]:
        raise ValueError(f"Unexpected Silero outputs: {[item.name for item in outputs]}.")

    metadata_dtype = metadata.string("input_audio_dtype", required=True).lower()
    expected_dtype = metadata_dtype if metadata_dtype.startswith("tensor(") else f"tensor({metadata_dtype})"
    if inputs[0].type.lower() != expected_dtype:
        raise ValueError(
            f"Silero audio dtype {inputs[0].type} does not match metadata {metadata_dtype}."
        )
    state_shape = inputs[1].shape
    if state_shape[0] != metadata.required_int("state_layers"):
        raise ValueError("Silero state layer count does not match metadata.")
    if state_shape[-1] != metadata.required_int("state_size"):
        raise ValueError("Silero state width does not match metadata.")
metadata_session = make_metadata_session(output_onnx_model)
model_metadata = load_runtime_metadata(
    output_onnx_model,
    make_metadata_session,
    required_keys=REQUIRED_SILERO_METADATA_KEYS,
)
validate_silero_metadata(model_metadata, metadata_session)

SAMPLE_RATE = model_metadata.required_int("input_sample_rate")
MODEL_SAMPLE_RATE = model_metadata.required_int("model_sample_rate")
supported_sample_rates = {
    int(value)
    for value in model_metadata.string("supported_sample_rates", required=True).split(",")
}
if SAMPLE_RATE not in supported_sample_rates:
    raise ValueError(
        f"Metadata sample rate {SAMPLE_RATE} is not in {sorted(supported_sample_rates)}."
    )
WINDOW_SIZE_SAMPLES = model_metadata.required_int("window_size_samples")
CONTEXT_SIZE_SAMPLES = model_metadata.required_int("context_size_samples")
expected_window_size = round(512 * SAMPLE_RATE / MODEL_SAMPLE_RATE)
expected_context_size = round(64 * SAMPLE_RATE / MODEL_SAMPLE_RATE)
if WINDOW_SIZE_SAMPLES != expected_window_size:
    raise ValueError(
        f"Silero window size {WINDOW_SIZE_SAMPLES} does not match {SAMPLE_RATE} Hz."
    )
if CONTEXT_SIZE_SAMPLES != expected_context_size:
    raise ValueError(
        f"Silero context size {CONTEXT_SIZE_SAMPLES} does not match {SAMPLE_RATE} Hz."
    )
MIN_SPEECH_SAMPLES = SAMPLE_RATE * MIN_SPEECH_DURATION_MS / 1000
SPEECH_PAD_SAMPLES = SAMPLE_RATE * SPEECH_PAD_MS / 1000
MAX_SPEECH_SAMPLES = (
    SAMPLE_RATE * MAX_SPEECH_DURATION
    - WINDOW_SIZE_SAMPLES
    - 2 * SPEECH_PAD_SAMPLES
)
MIN_SILENCE_SAMPLES = SAMPLE_RATE * MIN_SILENCE_DURATION / 1000
MIN_SILENCE_SAMPLES_AT_MAX_SPEECH = (
    SAMPLE_RATE * MIN_SILENCE_AT_MAX_SPEECH_MS / 1000
)
input_dtype = OnnxWrapper._numpy_dtype(metadata_session.get_inputs()[0].type)
del metadata_session


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


def process_timestamps(timestamps, fusion_threshold, min_duration):
    fused_timestamps = []
    for start, end in timestamps:
        if end - start < min_duration:
            continue
        if fused_timestamps and start - fused_timestamps[-1][1] <= fusion_threshold:
            fused_timestamps[-1] = (fused_timestamps[-1][0], end)
        else:
            fused_timestamps.append((start, end))
    return fused_timestamps


def vad_to_timestamps(vad_output, frame_duration):
    timestamps = []
    start = None
    # Extract raw timestamps
    for i, silence in enumerate(vad_output):
        if silence:
            if start is not None:  # End of the current speaking segment
                end = i * frame_duration + frame_duration
                timestamps.append((start, end))
                start = None
        else:
            if start is None:  # Start of a new speaking segment
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


# Load the Silero
silero_vad = OnnxWrapper(
    str(output_onnx_model),
    force_onnx_cpu=USE_CPU,
    sample_rate=SAMPLE_RATE,
    window_size_samples=WINDOW_SIZE_SAMPLES,
    context_size_samples=CONTEXT_SIZE_SAMPLES,
    cpu_disabled_optimizers=CPU_DISABLED_OPTIMIZERS,
)

# Load the audio as a zero-copy view over pydub's decoded PCM16 bytes.
decoded_audio = (
    AudioSegment.from_file(test_vad_audio)
    .set_channels(1)
    .set_frame_rate(SAMPLE_RATE)
    .set_sample_width(2)
)
audio_pcm = np.frombuffer(decoded_audio.raw_data, dtype=np.int16)
del decoded_audio
max_samples = SAMPLE_RATE * MAX_DYNAMIC_AUDIO_SECONDS
if audio_pcm.size > max_samples:
    raise ValueError(
        f"Audio has {audio_pcm.size} samples; MAX_DYNAMIC_AUDIO_SECONDS "
        f"limits input to {max_samples}."
    )
audio = normalise_audio(
    audio_pcm,
    input_dtype,
    NORMALIZE_TARGET_RMS,
)
del audio_pcm

# Start VAD
print("\nStart to run the VAD process.")
start_time = time.time()
timestamps = get_speech_timestamps(
    audio,
    model=silero_vad,
    threshold=ACTIVATE_THRESHOLD,
    neg_threshold=NEGATIVE_THRESHOLD,
    sampling_rate=SAMPLE_RATE,
    min_speech_samples=MIN_SPEECH_SAMPLES,
    max_speech_samples=MAX_SPEECH_SAMPLES,
    min_silence_samples=MIN_SILENCE_SAMPLES,
    speech_pad_samples=SPEECH_PAD_SAMPLES,
    min_silence_samples_at_max_speech=MIN_SILENCE_SAMPLES_AT_MAX_SPEECH,
    window_size_samples=WINDOW_SIZE_SAMPLES,
    use_max_poss_sil_at_max_speech=USE_MAX_POSS_SIL_AT_MAX_SPEECH,
    return_seconds=True
)
elapsed = time.time() - start_time
audio_duration = audio.size / SAMPLE_RATE
rtf = elapsed / audio_duration if audio_duration > 0.0 else 0.0
print(f"\nVAD Complete. RTF: {rtf:.4f}.")
del audio

# Generate the timestamps.
timestamps = process_timestamps(
    ((item["start"], item["end"]) for item in timestamps),
    FUSION_THRESHOLD,
    MIN_SPEECH_DURATION,
)

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
