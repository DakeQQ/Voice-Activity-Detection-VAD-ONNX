import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime
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
onnx_model = parent_path / "DFSMN_VAD_ONNX" / "DFSMN_VAD.onnx"
test_near_end_audio = model_audio_path("dfsmn", "near_end")
save_timestamps_second = parent_path / "timestamps_second.txt"
save_timestamps_indices = parent_path / "timestamps_indices.txt"


# User-adjustable inference settings.
ORT_ACCELERATE_PROVIDERS = []         # Add installed providers in priority order; [] forces CPU inference.
CPU_DISABLE_MATMUL_ADD_FUSION = True  # ORT 1.27 wraps rank-3 MatMul+Add in costly Reshape/Gemm/Reshape chains.
CPU_DISABLE_NCHWC = True              # NCHWc reorders regress mean/tail latency on the target i7-1165G7.
CPU_EXTRA_DISABLED_OPTIMIZERS = [     # Individually benchmarked on the same CPU / ORT build.
    "ConvAddActivationFusion",
    "MatmulTransposeFusion",
]
MAX_THREADS = 4                  # CPU thread count; use 0 to let ORT choose automatically.
DEVICE_ID = 0                    # Accelerator device index when a non-CPU provider is selected.
NORMALIZE_AUDIO = False          # True applies RMS normalization before inference.
NORMALIZE_TARGET_RMS = 8192.0    # Target PCM16 RMS; used only when normalization is enabled.
MAX_DYNAMIC_AUDIO_SECONDS = 360  # Maximum window size for a dynamic-length model.
FUSION_THRESHOLD = 0.3           # Merge speech segments separated by at most this many seconds.
MIN_SPEECH_DURATION = 0.2        # Drop speech segments shorter than this many seconds.
SPEAKING_SCORE = 0.5             # Minimum speech score required to leave silence.
SILENCE_SCORE = 0.5              # Maximum speech score used to return to silence.
LOOK_AHEAD = 0.3                 # Future score window, in seconds, used to stabilize decisions.

REQUIRED_DFSMN_METADATA_KEYS = (
    *REQUIRED_VAD_METADATA_KEYS,
    "light_aec_model",
    "output_frame_shift_samples",
    "fbank_window_length_samples",
    "num_audio_inputs",
    "num_outputs",
)


@dataclass(frozen=True)
class RuntimeConfig:
    sample_rate: int
    model_sample_rate: int
    light_aec_model: str
    output_frame_length: int
    output_frame_shift_seconds: float
    fbank_window_length: int
    max_dynamic_audio_seconds: int
    fusion_threshold: float
    min_speech_duration: float
    speaking_score: float
    silence_score: float
    look_ahead: float
    normalize_target_rms: float

    @classmethod
    def from_metadata(cls, metadata):
        sample_rate = metadata.required_int("input_sample_rate")
        model_sample_rate = metadata.required_int("model_sample_rate")
        output_frame_length = metadata.required_int("output_frame_shift_samples")
        output_frame_shift_seconds = metadata.required_float(
            "output_frame_shift_seconds")
        expected_shift = output_frame_length / model_sample_rate
        if not np.isclose(output_frame_shift_seconds, expected_shift):
            raise ValueError(
                f"Metadata frame shift {output_frame_shift_seconds} seconds does not "
                f"match {output_frame_length} samples at {sample_rate} Hz."
            )
        return cls(
            sample_rate=sample_rate,
            model_sample_rate=model_sample_rate,
            light_aec_model=metadata.string(
                "light_aec_model", required=True),
            output_frame_length=output_frame_length,
            output_frame_shift_seconds=output_frame_shift_seconds,
            fbank_window_length=metadata.required_int(
                "fbank_window_length_samples"),
            max_dynamic_audio_seconds=MAX_DYNAMIC_AUDIO_SECONDS,
            fusion_threshold=FUSION_THRESHOLD,
            min_speech_duration=MIN_SPEECH_DURATION,
            speaking_score=SPEAKING_SCORE,
            silence_score=SILENCE_SCORE,
            look_ahead=LOOK_AHEAD,
            normalize_target_rms=NORMALIZE_TARGET_RMS,
        )


def load_mono_int16(path, sample_rate):
    segment = AudioSegment.from_file(path).set_channels(1).set_frame_rate(sample_rate)
    samples = np.asarray(segment.get_array_of_samples())
    if samples.dtype == np.int16:
        return samples
    scale = float(1 << (8 * samples.dtype.itemsize - 1))
    return np.clip(samples.astype(np.float32) * (32768.0 / scale), -32768, 32767).astype(np.int16)


def numpy_dtype_from_onnx_type(type_name):
    input_dtypes = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(int16)": np.int16,
    }
    try:
        return input_dtypes[type_name]
    except KeyError as error:
        raise TypeError(f"Unsupported ONNX audio input type: {type_name}") from error


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


def normalise_audio(audio: np.ndarray, input_dtype_np, target_rms) -> np.ndarray:
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


def create_session(model_path=onnx_model):
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 4
    options.inter_op_num_threads = MAX_THREADS
    options.intra_op_num_threads = MAX_THREADS
    options.enable_cpu_mem_arena = True
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.set_denormal_as_zero", "1")
    options.add_session_config_entry("session.intra_op.allow_spinning", "1")
    options.add_session_config_entry("session.inter_op.allow_spinning", "1")

    providers = ORT_ACCELERATE_PROVIDERS or ["CPUExecutionProvider"]
    cpu_ep_only = set(providers) == {"CPUExecutionProvider"}
    disabled_opts = []
    if cpu_ep_only and CPU_DISABLE_MATMUL_ADD_FUSION:
        disabled_opts.append("MatMulAddFusion")
    if cpu_ep_only and CPU_DISABLE_NCHWC:
        disabled_opts.append("NchwcTransformer")
    if cpu_ep_only:
        disabled_opts.extend(CPU_EXTRA_DISABLED_OPTIMIZERS)
    return onnxruntime.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=providers,
        disabled_optimizers=disabled_opts or None,
    )


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


def pad_to_length(audio, length):
    if audio.shape[-1] >= length:
        return audio[..., :length]
    padding = [(0, 0)] * audio.ndim
    padding[-1] = (0, length - audio.shape[-1])
    return np.pad(audio, padding)


def dfsmn_frame_count(input_samples, config):
    model_samples = int(
        input_samples * config.model_sample_rate / config.sample_rate)
    if config.light_aec_model != "SDAEC":
        raise ValueError(
            f"Dynamic DFSMN inference is unsupported for {config.light_aec_model}."
        )
    aec_hop_length = 160
    aec_frames = (model_samples - 1) // aec_hop_length + 1
    aec_samples = (aec_frames - 1) * aec_hop_length + 1
    if aec_samples < config.fbank_window_length:
        return 0
    return 1 + (
        aec_samples - config.fbank_window_length
    ) // config.output_frame_length


def run_chunked(session, near_audio, config):
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    chunk_length = inputs[0].shape[-1]
    if not isinstance(chunk_length, int):
        chunk_length = min(
            len(near_audio),
            config.sample_rate * config.max_dynamic_audio_seconds,
        )

    look_ahead_frames = max(
        1,
        int(config.look_ahead / config.output_frame_shift_seconds),
    )
    output_extents = {}
    if not isinstance(outputs[0].shape[0], int):
        output_extents[0] = dfsmn_frame_count(chunk_length, config)
    output_shape = resolve_tensor_shape(outputs[0], output_extents)
    output_frames = int(np.prod(output_shape))
    retained_frames = output_frames - look_ahead_frames
    stride = round(
        (
            (retained_frames - 1) * config.output_frame_shift_seconds
            + config.fbank_window_length / config.model_sample_rate
        )
        * config.sample_rate
    )
    if stride <= 0:
        raise ValueError("LOOK_AHEAD is too large for the model input length.")

    audio_length = len(near_audio)
    input_extents = (
        {-1: chunk_length}
        if not isinstance(inputs[0].shape[-1], int)
        else {}
    )
    input_shape = resolve_tensor_shape(inputs[0], input_extents)
    near_audio = near_audio.reshape(*input_shape[:-1], -1)
    probabilities = []

    device_type = ortvalue_device_type(session)
    input_dtype = numpy_dtype_from_onnx_type(inputs[0].type)
    output_dtype = numpy_dtype_from_onnx_type(outputs[0].type)
    near_buffer = create_ort_buffer(input_shape, input_dtype, device_type)
    output_buffer = create_ort_buffer(output_shape, output_dtype, device_type)
    binding = session.io_binding()
    binding.bind_ortvalue_input(inputs[0].name, near_buffer)
    binding.bind_ortvalue_output(outputs[0].name, output_buffer)
    run_options = onnxruntime.RunOptions()
    run_options.log_severity_level = 4
    run_options.log_verbosity_level = 4
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")

    start = 0
    while start < audio_length:
        end = min(start + chunk_length, audio_length)
        near_chunk = pad_to_length(near_audio[..., start:end], chunk_length)
        update_ort_buffer(near_buffer, near_chunk)
        session.run_with_iobinding(binding, run_options=run_options)
        output = output_buffer.numpy().reshape(-1)

        if end < audio_length:
            probabilities.extend(output[:retained_frames])
        else:
            valid_frames = dfsmn_frame_count(end - start, config)
            probabilities.extend(output[:valid_frames])
            break
        start += stride

    return np.asarray(probabilities, dtype=np.float32), audio_length


def probabilities_to_silence(probabilities, config):
    look_ahead_frames = max(
        1,
        int(config.look_ahead / config.output_frame_shift_seconds),
    )
    silence = True
    states = []
    full_look_ahead_end = max(0, len(probabilities) - look_ahead_frames)
    for index in range(full_look_ahead_end):
        probability = probabilities[index]
        future = probabilities[index:index + look_ahead_frames]
        if silence:
            if probability >= config.speaking_score:
                silence = (
                    np.mean(future >= config.speaking_score)
                    < config.speaking_score
                )
            else:
                silence = True
        else:
            if probability <= config.silence_score:
                silence = (
                    np.mean(future <= config.silence_score)
                    > config.silence_score
                )
            else:
                silence = False
        states.append(silence)

    for probability in probabilities[full_look_ahead_end:]:
        if silence:
            silence = probability < config.speaking_score
        else:
            silence = probability <= config.silence_score
        states.append(silence)
    return states


def vad_to_timestamps(silence_states, frame_duration):
    timestamps = []
    start = None
    for index, silence in enumerate(silence_states):
        if silence and start is not None:
            timestamps.append((start, (index + 1) * frame_duration))
            start = None
        elif not silence and start is None:
            start = index * frame_duration
    if start is not None:
        timestamps.append((start, len(silence_states) * frame_duration))
    return timestamps


def process_timestamps(timestamps, config):
    filtered = [
        (start, end)
        for start, end in timestamps
        if end - start >= config.min_speech_duration
    ]
    fused = []
    for start, end in filtered:
        if fused and start - fused[-1][1] <= config.fusion_threshold:
            fused[-1] = (fused[-1][0], end)
        else:
            fused.append((start, end))
    return fused


def format_time(seconds):
    total_milliseconds = round(float(seconds) * 1000)
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def save_timestamps(timestamps, sample_rate):
    print("\nTimestamps in seconds:")
    with save_timestamps_second.open("w", encoding="utf-8") as stream:
        for start, end in timestamps:
            line = f"{format_time(start)} --> {format_time(end)}"
            stream.write(line + "\n")
            print(line)

    print("\nTimestamps in indices:")
    with save_timestamps_indices.open("w", encoding="utf-8") as stream:
        for start, end in timestamps:
            line = f"{int(start * sample_rate)} --> {int(end * sample_rate)}"
            stream.write(line + "\n")
            print(line)


def main():
    session = create_session()
    model_metadata = load_runtime_metadata(
        onnx_model,
        create_session,
        required_keys=REQUIRED_DFSMN_METADATA_KEYS,
    )
    validate_vad_metadata(
        model_metadata,
        session,
        expected_inputs=model_metadata.required_int("num_audio_inputs"),
        expected_outputs=model_metadata.required_int("num_outputs"),
    )
    config = RuntimeConfig.from_metadata(model_metadata)
    print(f"Usable providers: {session.get_providers()}")
    print(f"Near-end audio: {test_near_end_audio}")
    inputs = session.get_inputs()
    input_dtypes = [numpy_dtype_from_onnx_type(model_input.type) for model_input in inputs]
    if len(input_dtypes) != 1:
        raise TypeError(
            f"Expected one audio input, got {[model_input.type for model_input in inputs]}."
        )
    near_audio = normalise_audio(
        load_mono_int16(test_near_end_audio, config.sample_rate),
        input_dtypes[0],
        config.normalize_target_rms,
    )

    start_time = time.time()
    probabilities, audio_length = run_chunked(
        session, near_audio, config)
    silence_states = probabilities_to_silence(probabilities, config)
    timestamps = process_timestamps(
        vad_to_timestamps(
            silence_states,
            config.output_frame_shift_seconds,
        ),
        config,
    )
    elapsed = time.time() - start_time
    audio_duration = audio_length / config.sample_rate
    rtf = elapsed / audio_duration if audio_duration > 0.0 else 0.0

    save_timestamps(timestamps, config.sample_rate)
    print(
        f"\nVAD complete: {len(probabilities)} frames, "
        f"{audio_duration:.3f} seconds of audio, "
        f"RTF: {rtf:.4f}.")


if __name__ == "__main__":
    main()