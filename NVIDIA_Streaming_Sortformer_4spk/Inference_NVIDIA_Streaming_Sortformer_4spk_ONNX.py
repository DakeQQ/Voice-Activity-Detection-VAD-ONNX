#!/usr/bin/env python
"""Run standalone NVIDIA Streaming Sortformer v2.1 ONNX inference."""

import argparse
import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import onnxruntime


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_FOLDER: Final = _SCRIPT_DIR / "NVIDIA_Streaming_Sortformer_4spk_Optimized"
MODEL_NAME: Final = "NVIDIA_Streaming_Sortformer_4spk.onnx"

DEFAULT_ACTIVITY_THRESHOLD: Final = 0.5

DEFAULT_TEST_AUDIO_PATH: Final = (
    _SCRIPT_DIR.parent
    / "Test_Examples"
    / "en"
    / "test_sample_multitalker_overlap.wav"
)
TEST_AUDIO_PATH: Final = Path(
    os.environ.get(
        "SORTFORMER_AUDIO_PATH",
        DEFAULT_TEST_AUDIO_PATH,
    )
)
DEFAULT_ACTIVITY_PATH: Final = (
    Path(os.environ["SORTFORMER_ACTIVITY_PATH"])
    if "SORTFORMER_ACTIVITY_PATH" in os.environ
    else None
)

ORT_ACCELERATE_PROVIDERS = []
CPU_DISABLE_MATMUL_ADD_FUSION = True
CPU_DISABLE_NCHWC = True
CPU_EXTRA_DISABLED_OPTIMIZERS = [
    "ConvAddActivationFusion",
    "MatmulTransposeFusion",
]
MAX_THREADS = 0
DEVICE_ID = 0


if "OpenVINOExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
    provider_options = [
        {
            "device_type": "CPU",
            "precision": "ACCURACY",
            "num_of_threads": MAX_THREADS if MAX_THREADS > 0 else os.cpu_count(),
            "num_streams": 1,
            "enable_opencl_throttling": True,
            "enable_qdq_optimizer": False,
            "disable_dynamic_shapes": False,
        }
    ]
elif "CUDAExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
    provider_options = [
        {
            "device_id": DEVICE_ID,
            "gpu_mem_limit": 8 * 1024 * 1024 * 1024,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "sdpa_kernel": "2",
            "use_tf32": "1",
            "fuse_conv_bias": "1",
            "cudnn_conv_use_max_workspace": "1",
            "cudnn_conv1d_pad_to_nc1d": "1",
            "tunable_op_enable": "0",
            "tunable_op_tuning_enable": "0",
            "tunable_op_max_tuning_duration_ms": 10,
            "do_copy_in_default_stream": "0",
            "enable_cuda_graph": "0",
            "prefer_nhwc": "0",
            "enable_skip_layer_norm_strict_mode": "0",
            "use_ep_level_unified_stream": "0",
        }
    ]
elif "DmlExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
    provider_options = [
        {
            "device_id": DEVICE_ID,
            "performance_preference": "high_performance",
            "device_filter": "any",
        }
    ]
else:
    provider_options = None


session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4
session_opts.log_verbosity_level = 4
session_opts.inter_op_num_threads = MAX_THREADS
session_opts.intra_op_num_threads = MAX_THREADS
session_opts.enable_cpu_mem_arena = True
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "0")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")

run_options = onnxruntime.RunOptions()
run_options.log_severity_level = 4
run_options.log_verbosity_level = 4
run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")

_CPU_EP_ONLY = not ORT_ACCELERATE_PROVIDERS or set(ORT_ACCELERATE_PROVIDERS) == {
    "CPUExecutionProvider"
}
disabled_opts = []
if _CPU_EP_ONLY and CPU_DISABLE_MATMUL_ADD_FUSION:
    disabled_opts.append("MatMulAddFusion")
if _CPU_EP_ONLY and CPU_DISABLE_NCHWC:
    disabled_opts.append("NchwcTransformer")
if _CPU_EP_ONLY:
    disabled_opts.extend(CPU_EXTRA_DISABLED_OPTIMIZERS)
disabled_opts = disabled_opts or None


_ONNX_DTYPE_MAP = {
    "tensor(bool)": np.dtype(np.bool_),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(float)": np.dtype(np.float32),
    "tensor(double)": np.dtype(np.float64),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(uint16)": np.dtype(np.uint16),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(uint32)": np.dtype(np.uint32),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(uint64)": np.dtype(np.uint64),
}


@dataclass(frozen=True)
class _TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype


@dataclass(frozen=True)
class _StateTensorSpec:
    input: _TensorSpec
    output: _TensorSpec


@dataclass(frozen=True)
class _ModelIO:
    audio_input: _TensorSpec
    audio_lengths_input: _TensorSpec
    activity_threshold_input: _TensorSpec
    activity_output: _TensorSpec
    active_frame_counts_state: _StateTensorSpec
    overlap_frame_count_state: _StateTensorSpec
    state_tensors: tuple[_StateTensorSpec, ...]


def _activity_threshold(value: str) -> float:
    try:
        threshold = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "activity threshold must be a number from 0 to 1"
        ) from error
    if not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError(
            "activity threshold must be between 0 and 1"
        )
    return threshold


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the standalone Streaming Sortformer ONNX multitalker demo."
    )
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=DEFAULT_MODEL_FOLDER,
        help="Folder containing NVIDIA_Streaming_Sortformer_4spk.onnx.",
    )
    parser.add_argument(
        "--audio-path",
        type=Path,
        default=TEST_AUDIO_PATH,
        help="16 kHz mono PCM WAV input.",
    )
    parser.add_argument(
        "--activity-path",
        type=Path,
        default=DEFAULT_ACTIVITY_PATH,
        help="Optional output .npy path for float32 [chunks, frames, speakers] activity.",
    )
    parser.add_argument(
        "--activity-threshold",
        type=_activity_threshold,
        default=DEFAULT_ACTIVITY_THRESHOLD,
        metavar="PROBABILITY",
        help=(
            "Count an activity score above this probability as active "
            f"(default: {DEFAULT_ACTIVITY_THRESHOLD})."
        ),
    )
    return parser.parse_args()


def _make_session(model_path: Path) -> onnxruntime.InferenceSession:
    kwargs = {
        "sess_options": session_opts,
        "providers": ORT_ACCELERATE_PROVIDERS,
        "disabled_optimizers": disabled_opts,
    }
    if provider_options is not None:
        kwargs["provider_options"] = provider_options
    return onnxruntime.InferenceSession(str(model_path), **kwargs)


def _ortvalue_device_type(session: onnxruntime.InferenceSession) -> str:
    primary_provider = session.get_providers()[0]
    if primary_provider in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}:
        return "cuda"
    if primary_provider == "DmlExecutionProvider":
        return "dml"
    return "cpu"


def _create_ort_buffer(
    shape: tuple[int, ...],
    dtype: np.dtype,
    device_type: str,
) -> onnxruntime.OrtValue:
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.zeros(shape, dtype=dtype),
        device_type,
        DEVICE_ID,
    )


def _update_ort_buffer(buffer: onnxruntime.OrtValue, value: np.ndarray) -> None:
    if hasattr(buffer, "update_inplace"):
        buffer.update_inplace(np.ascontiguousarray(value))
    else:
        np.copyto(buffer.numpy(), value)


def _tensor_spec(node: object, direction: str) -> _TensorSpec:
    name = getattr(node, "name")
    node_type = getattr(node, "type")
    node_shape = getattr(node, "shape")
    try:
        dtype = _ONNX_DTYPE_MAP[node_type]
    except KeyError as error:
        raise TypeError(
            f"Unsupported ONNX {direction} tensor type {node_type!r} for {name!r}."
        ) from error
    if not node_shape:
        raise ValueError(
            f"ONNX {direction} tensor {name!r} must have a static, non-empty shape."
        )
    try:
        shape = tuple(int(dimension) for dimension in node_shape)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"ONNX {direction} tensor {name!r} must have a static shape; "
            f"got {node_shape!r}."
        ) from error
    if any(dimension <= 0 for dimension in shape):
        raise ValueError(
            f"ONNX {direction} tensor {name!r} must have only positive dimensions; "
            f"got {node_shape!r}."
        )
    return _TensorSpec(name=name, shape=shape, dtype=dtype)


def _model_io(session: onnxruntime.InferenceSession) -> _ModelIO:
    inputs = tuple(_tensor_spec(node, "input") for node in session.get_inputs())
    outputs = tuple(_tensor_spec(node, "output") for node in session.get_outputs())
    state_tensors = []
    for input in inputs:
        matching_outputs = tuple(
            output
            for output in outputs
            if output.name != input.name
            and output.name.endswith(input.name)
            and output.shape == input.shape
            and output.dtype == input.dtype
        )
        if len(matching_outputs) > 1:
            raise ValueError(
                f"Could not uniquely pair ONNX state input {input.name!r} with an output."
            )
        if matching_outputs:
            state_tensors.append(_StateTensorSpec(input, matching_outputs[0]))
    state_tensors = tuple(state_tensors)
    state_input_names = {state.input.name for state in state_tensors}
    state_output_names = {state.output.name for state in state_tensors}
    non_state_inputs = tuple(
        input for input in inputs if input.name not in state_input_names
    )
    non_state_outputs = tuple(
        output for output in outputs if output.name not in state_output_names
    )

    if not state_tensors:
        raise ValueError("The ONNX model has no input/output streaming state pairs.")
    for state in state_tensors:
        if state.input.shape != state.output.shape or state.input.dtype != state.output.dtype:
            raise ValueError(
                "ONNX state input/output pair has incompatible type or shape: "
                f"{state.input.name!r} -> {state.output.name!r}."
            )
    if len(non_state_inputs) != 3 or len(non_state_outputs) != 1:
        raise ValueError(
            "Expected audio, audio-length, and activity-threshold inputs plus one "
            "non-state activity output in the ONNX model. Re-export Sortformer "
            "with the current exporter."
        )

    audio_candidates = tuple(
        input
        for input in non_state_inputs
        if len(input.shape) >= 2 and np.issubdtype(input.dtype, np.number)
    )
    if len(audio_candidates) != 1:
        raise ValueError("Could not identify exactly one waveform input from ONNX I/O.")
    audio_input = audio_candidates[0]
    lengths_candidates = tuple(
        input
        for input in non_state_inputs
        if input.name != audio_input.name
        and input.shape == (audio_input.shape[0],)
        and np.issubdtype(input.dtype, np.integer)
    )
    if len(lengths_candidates) != 1:
        raise ValueError("Could not identify exactly one waveform-length input from ONNX I/O.")

    threshold_candidates = tuple(
        input
        for input in non_state_inputs
        if input.name == "activity_threshold"
        and input.shape == (1,)
        and np.issubdtype(input.dtype, np.floating)
    )
    if len(threshold_candidates) != 1:
        raise ValueError(
            "Expected one float activity_threshold input with shape (1,) in the ONNX model."
        )

    activity_candidates = tuple(
        output
        for output in non_state_outputs
        if output.name == "activity"
        and len(output.shape) == 3
        and output.shape[0] == 1
        and np.issubdtype(output.dtype, np.floating)
    )
    if len(activity_candidates) != 1:
        raise ValueError(
            "Expected one float activity output with shape [1, frames, speakers] "
            "in the ONNX model."
        )

    states_by_input_name = {state.input.name: state for state in state_tensors}
    try:
        active_frame_counts_state = states_by_input_name["active_frame_counts"]
        overlap_frame_count_state = states_by_input_name["overlap_frame_count"]
    except KeyError as error:
        raise ValueError(
            "The ONNX model has no cumulative activity-count state tensors."
        ) from error
    if (
        len(active_frame_counts_state.input.shape) != 2
        or active_frame_counts_state.input.shape[0] != 1
        or not np.issubdtype(active_frame_counts_state.input.dtype, np.integer)
    ):
        raise ValueError("Invalid active_frame_counts state tensor in the ONNX model.")
    if (
        overlap_frame_count_state.input.shape != (1,)
        or not np.issubdtype(overlap_frame_count_state.input.dtype, np.integer)
    ):
        raise ValueError("Invalid overlap_frame_count state tensor in the ONNX model.")

    return _ModelIO(
        audio_input=audio_input,
        audio_lengths_input=lengths_candidates[0],
        activity_threshold_input=threshold_candidates[0],
        activity_output=activity_candidates[0],
        active_frame_counts_state=active_frame_counts_state,
        overlap_frame_count_state=overlap_frame_count_state,
        state_tensors=state_tensors,
    )


def _load_test_audio(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        sample_rate = source.getframerate()
        channels = source.getnchannels()
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    samples = samples.reshape(-1, channels)[:, 0]
    return np.ascontiguousarray(samples.astype(np.float32) / 32768.0), sample_rate


def _convert_waveform_dtype(
    waveform: np.ndarray,
    audio_dtype: np.dtype,
) -> np.ndarray:
    if audio_dtype == np.dtype(np.int16):
        return np.clip(
            waveform * 32768.0,
            -32768.0,
            32767.0,
        ).astype(np.int16)
    if np.issubdtype(audio_dtype, np.floating):
        return np.ascontiguousarray(waveform.astype(audio_dtype))
    raise TypeError(
        f"Unsupported ONNX waveform input dtype {audio_dtype.name!r}; "
        "expected a floating-point type or int16."
    )


def _state_buffers(
    state_tensors: tuple[_StateTensorSpec, ...],
    device_type: str,
) -> dict[str, onnxruntime.OrtValue]:
    return {
        state.input.name: _create_ort_buffer(
            state.input.shape,
            state.input.dtype,
            device_type,
        )
        for state in state_tensors
    }


def _bind_step(
    session: onnxruntime.InferenceSession,
    model_io: _ModelIO,
    audio: onnxruntime.OrtValue,
    audio_lengths: onnxruntime.OrtValue,
    activity_threshold: onnxruntime.OrtValue,
    activity_output: onnxruntime.OrtValue | None,
    current_state: dict[str, onnxruntime.OrtValue],
    next_state: dict[str, onnxruntime.OrtValue],
) -> onnxruntime.IOBinding:
    binding = session.io_binding()
    binding.bind_ortvalue_input(model_io.audio_input.name, audio)
    binding.bind_ortvalue_input(model_io.audio_lengths_input.name, audio_lengths)
    binding.bind_ortvalue_input(
        model_io.activity_threshold_input.name,
        activity_threshold,
    )
    if activity_output is not None:
        binding.bind_ortvalue_output(model_io.activity_output.name, activity_output)
    for state in model_io.state_tensors:
        binding.bind_ortvalue_input(state.input.name, current_state[state.input.name])
        binding.bind_ortvalue_output(state.output.name, next_state[state.input.name])
    return binding


def main() -> None:
    args = _parse_args()
    model_folder = args.model_folder.expanduser().resolve()
    model_path = model_folder / MODEL_NAME
    metadata_path = model_path.with_name(model_path.stem + "_Metadata.onnx")
    audio_path = args.audio_path.expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    activity_path = (
        args.activity_path.expanduser().resolve()
        if args.activity_path is not None
        else None
    )
    session = _make_session(model_path)
    metadata_session = _make_session(metadata_path)
    metadata = metadata_session.get_modelmeta().custom_metadata_map

    model_io = _model_io(session)
    sample_rate = int(metadata["sample_rate"])
    audio_cache_samples = int(metadata["audio_cache_samples"])
    new_audio_samples = int(metadata["new_audio_samples"])
    waveform, input_sample_rate = _load_test_audio(audio_path)
    if input_sample_rate != sample_rate:
        raise ValueError(
            f"Audio sample rate must be {sample_rate} Hz; got {input_sample_rate} Hz."
        )
    if waveform.size == 0:
        raise ValueError("Audio input contains no samples.")
    audio_chunk_samples = audio_cache_samples + new_audio_samples
    audio_input_samples = model_io.audio_input.shape[-1]
    if audio_input_samples != audio_chunk_samples + 1:
        raise ValueError(
            "The ONNX waveform input length does not match the streaming metadata: "
            f"{audio_input_samples} != {audio_chunk_samples} + 1."
        )
    if np.prod(model_io.audio_input.shape[:-1], dtype=np.int64) != 1:
        raise ValueError(
            "The ONNX waveform input must describe exactly one audio stream per call; "
            f"got shape {model_io.audio_input.shape}."
        )
    if np.prod(model_io.audio_lengths_input.shape, dtype=np.int64) != 1:
        raise ValueError(
            "The ONNX waveform-length input must describe exactly one audio stream per call; "
            f"got shape {model_io.audio_lengths_input.shape}."
        )
    speaker_count = model_io.active_frame_counts_state.input.shape[-1]
    if speaker_count != int(metadata["num_speakers"]):
        raise ValueError(
            "The ONNX activity-count state does not match the speaker metadata: "
            f"{speaker_count} != {metadata['num_speakers']}."
        )
    output_activity_frames = int(metadata["output_activity_frames"])
    if output_activity_frames <= 0:
        raise ValueError("The ONNX output_activity_frames metadata must be positive.")
    expected_activity_shape = (1, output_activity_frames, speaker_count)
    if model_io.activity_output.shape != expected_activity_shape:
        raise ValueError(
            "The ONNX activity output does not match the streaming metadata: "
            f"{model_io.activity_output.shape} != {expected_activity_shape}."
        )
    num_chunks = (waveform.size + new_audio_samples - 1) // new_audio_samples

    device_type = _ortvalue_device_type(session)
    model_waveform = _convert_waveform_dtype(waveform, model_io.audio_input.dtype)
    audio_window = np.zeros(
        model_io.audio_input.shape,
        dtype=model_io.audio_input.dtype,
    )
    audio_samples = audio_window.reshape(-1, audio_input_samples)[0]
    audio_lengths = np.zeros(
        model_io.audio_lengths_input.shape,
        dtype=model_io.audio_lengths_input.dtype,
    )
    audio_buffer = _create_ort_buffer(
        model_io.audio_input.shape,
        model_io.audio_input.dtype,
        device_type,
    )
    audio_lengths_buffer = _create_ort_buffer(
        model_io.audio_lengths_input.shape,
        model_io.audio_lengths_input.dtype,
        device_type,
    )
    activity_threshold = np.full(
        model_io.activity_threshold_input.shape,
        args.activity_threshold,
        dtype=model_io.activity_threshold_input.dtype,
    )
    activity_threshold_buffer = _create_ort_buffer(
        model_io.activity_threshold_input.shape,
        model_io.activity_threshold_input.dtype,
        device_type,
    )
    _update_ort_buffer(activity_threshold_buffer, activity_threshold)
    activity_buffers = (
        _create_ort_buffer(
            model_io.activity_output.shape,
            model_io.activity_output.dtype,
            device_type,
        ),
        _create_ort_buffer(
            model_io.activity_output.shape,
            model_io.activity_output.dtype,
            device_type,
        ),
    )
    activity_history = (
        np.empty(
            (num_chunks, *model_io.activity_output.shape[1:]),
            dtype=np.float32,
        )
        if activity_path is not None
        else None
    )
    state_a = _state_buffers(
        model_io.state_tensors,
        device_type,
    )
    state_b = _state_buffers(
        model_io.state_tensors,
        device_type,
    )
    bindings = (
        _bind_step(
            session,
            model_io,
            audio_buffer,
            audio_lengths_buffer,
            activity_threshold_buffer,
            activity_buffers[0],
            state_a,
            state_b,
        ),
        _bind_step(
            session,
            model_io,
            audio_buffer,
            audio_lengths_buffer,
            activity_threshold_buffer,
            activity_buffers[1],
            state_b,
            state_a,
        ),
    )

    start_time = time.perf_counter()
    for chunk_index in range(num_chunks):
        new_start = chunk_index * new_audio_samples
        window_start = new_start - audio_cache_samples
        source_start = max(window_start, 0)
        source_end = min(window_start + audio_chunk_samples, waveform.size)
        destination_start = source_start - window_start
        audio_window.fill(0)
        if window_start > 0:
            audio_samples[0] = model_waveform[window_start - 1]
        audio_samples[
            1 + destination_start : 1 + destination_start + source_end - source_start
        ] = model_waveform[source_start:source_end]
        audio_lengths.fill(audio_cache_samples + min(
            new_audio_samples,
            waveform.size - new_start,
        ))
        _update_ort_buffer(audio_buffer, audio_window)
        _update_ort_buffer(audio_lengths_buffer, audio_lengths)
        session.run_with_iobinding(
            bindings[chunk_index & 1],
            run_options=run_options,
        )
        if activity_history is not None:
            activity_history[chunk_index] = activity_buffers[chunk_index & 1].numpy()[0]

    elapsed = time.perf_counter() - start_time
    final_state = state_b if num_chunks % 2 else state_a
    active_frame_counts = final_state[
        model_io.active_frame_counts_state.input.name
    ].numpy().reshape(-1)
    overlap_frame_count = int(
        final_state[model_io.overlap_frame_count_state.input.name].numpy().reshape(-1)[0]
    )
    if activity_path is not None:
        activity_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_activity_path = activity_path.with_name(activity_path.name + ".tmp")
        with temporary_activity_path.open("wb") as destination:
            np.save(destination, activity_history)
        temporary_activity_path.replace(activity_path)
    audio_seconds = waveform.size / sample_rate
    print(f"Usable Providers: {session.get_providers()}")
    print(f"ONNX model: {model_path}")
    print(f"Audio input dtype: {model_io.audio_input.dtype.name}")
    print(f"Audio: {audio_path.name} ({audio_seconds:.3f}s)")
    print(f"Activity threshold: {args.activity_threshold:.3f}")
    print(f"Activity frames: {num_chunks * output_activity_frames}")
    if activity_path is not None:
        print(f"Activity sidecar: {activity_path}")
    print(f"Active frames: {active_frame_counts.tolist()}  overlap_frames={overlap_frame_count}")
    print(f"Elapsed: {elapsed:.3f}s  RTF: {elapsed / audio_seconds:.3f}")


if __name__ == "__main__":
    main()