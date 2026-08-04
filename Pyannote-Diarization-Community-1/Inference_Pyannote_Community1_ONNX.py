import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "Example_Audio.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from Example_Audio import model_audio_path
from vad_onnx_metadata import (
    community1_metadata_path,
    parse_community1_metadata,
)


script_dir = Path(__file__).resolve().parent
ONNX_MODEL_DIR = script_dir / "Pyannote_Community1_Optimized"
TEST_AUDIO = Path(model_audio_path("pyannote_community1"))

# ONNX Runtime settings. An empty provider list explicitly selects CPU.
ORT_ACCELERATE_PROVIDERS = []
ORT_LOG = False
MAX_THREADS = 0
DEVICE_ID = 0
DISABLE_CPU_EP_FALLBACK = False  # Dynamic shape/control nodes require CPU EP.
CPU_DISABLE_MATMUL_ADD_FUSION = True
CPU_DISABLE_NCHWC = True
CPU_EXTRA_DISABLED_OPTIMIZERS = [
    "ConvAddActivationFusion",
    "MatmulTransposeFusion",
]

# Diarization settings.
NUM_SPEAKERS = None
MIN_SPEAKERS = None
MAX_SPEAKERS = None
USE_AUDIO_NORMALIZER = False
NORMALIZE_TARGET_RMS = 4096.0
FALLBACK_FINITE_BLOCK_ROWS = 4096
JSON_PATH = None
RTTM_PATH = None
EXCLUSIVE_RTTM_PATH = None

METADATA_KEY = "pyannote.community1"
MODEL_ID = "pyannote/speaker-diarization-community-1"
MODEL_NAMES = {
    "frontend": "segmentation_embedding.onnx",
    "backend": "vbx_clustering_diarization.onnx",
}
PCM16_SCALE = np.float32(1.0 / 32768.0)
_ORT_AUDIO_DTYPES = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(int16)": np.int16,
}
_ORT_TENSOR_DTYPES = {
    **_ORT_AUDIO_DTYPES,
    "tensor(bool)": np.bool_,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
}
_METADATA_AUDIO_DTYPES = {
    "float16": np.float16,
    "float32": np.float32,
    "int16": np.int16,
}


@dataclass(frozen=True)
class FrameGrid:
    start: float
    duration: float
    step: float

    def middle(self, index: int) -> float:
        return self.start + index * self.step + 0.5 * self.duration


@dataclass(frozen=True)
class SpeechTurn:
    start: float
    end: float
    speaker: str


@dataclass
class LoadedAudio:
    waveform: np.ndarray
    sample_rate: int
    source_sample_rate: int
    source_channels: int
    duration: float


@dataclass
class DiarizationResult:
    diarization: list[SpeechTurn]
    exclusive_diarization: list[SpeechTurn]
    speaker_embeddings: np.ndarray

    def serialize(self, elapsed: float) -> dict:
        def serialize_turns(turns: list[SpeechTurn]) -> list[dict]:
            return [
                {
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                    "speaker": turn.speaker,
                }
                for turn in turns
            ]

        return {
            "diarization": serialize_turns(self.diarization),
            "exclusive_diarization": serialize_turns(
                self.exclusive_diarization
            ),
            "elapsed_seconds": round(elapsed, 6),
            "speaker_embeddings": self.speaker_embeddings.tolist(),
        }


def read_metadata(session: ort.InferenceSession, component: str) -> dict:
    raw = session.get_modelmeta().custom_metadata_map.get(METADATA_KEY)
    if raw is None:
        raise ValueError(
            f"{component} model has no {METADATA_KEY!r} metadata. "
            "Re-export it with Export_Pyannote_Community1.py."
        )
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {component} model metadata: {error}") from error
    if metadata.get("component") != component:
        raise ValueError(
            f"Expected {component!r} model, got {metadata.get('component')!r}."
        )
    if metadata.get("model_id") != MODEL_ID:
        raise ValueError(
            f"Unexpected model_id in {component} metadata: "
            f"{metadata.get('model_id')!r}."
        )
    if int(metadata.get("format_version", 0)) < 3:
        raise ValueError(
            f"{component} is a legacy host-pipeline model; re-export format v3."
        )
    return metadata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_package_metadata(model_dir: Path) -> tuple[Path, dict]:
    metadata_path = community1_metadata_path(model_dir)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Required Community-1 metadata model is missing: {metadata_path}. "
            "Re-export with Export_Pyannote_Community1.py."
        )
    session = ort.InferenceSession(
        str(metadata_path), providers=["CPUExecutionProvider"]
    )
    try:
        manifest = parse_community1_metadata(
            session.get_modelmeta().custom_metadata_map
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid Community-1 package metadata in {metadata_path}: {error}"
        ) from error
    return metadata_path, manifest


def validate_package_metadata(
    manifest: dict,
    model_paths: dict[str, Path],
    component_metadata: dict[str, dict],
) -> None:
    models = manifest["models"]
    if set(models) != set(MODEL_NAMES):
        raise ValueError(
            f"Community-1 metadata describes {sorted(models)}, expected "
            f"{sorted(MODEL_NAMES)}."
        )
    for component, path in model_paths.items():
        details = models[component]
        if details.get("file") != path.name:
            raise ValueError(
                f"Community-1 {component} filename mismatch: "
                f"{details.get('file')!r} != {path.name!r}."
            )
        expected_hash = details.get("sha256")
        actual_hash = sha256(path)
        if expected_hash != actual_hash:
            raise ValueError(
                f"Community-1 {component} SHA-256 mismatch for {path}."
            )
        embedded = component_metadata[component]
        for key in ("component", "format_version", "model_id"):
            if details.get(key) != embedded.get(key):
                raise ValueError(
                    f"Community-1 {component} {key} metadata does not match "
                    "the package carrier."
                )


def resolve_model_path(model_dir: Path, component: str) -> Path:
    path = model_dir / MODEL_NAMES[component]
    if path.is_file():
        return path
    raise FileNotFoundError(
        f"Could not find the {component} ONNX model under {model_dir}."
    )


def resolve_providers(
    requested: list[str],
) -> tuple[list[str], list[dict], str]:
    available = ort.get_available_providers()
    providers = list(dict.fromkeys(requested)) or ["CPUExecutionProvider"]
    unavailable = [provider for provider in providers if provider not in available]
    if unavailable:
        raise ValueError(
            f"Unavailable ONNX Runtime providers: {unavailable}. "
            f"Available providers: {available}"
        )

    provider_options = []
    for provider in providers:
        if provider == "OpenVINOExecutionProvider":
            provider_options.append(
                {
                    "device_type": "CPU",
                    "precision": "ACCURACY",
                    "num_of_threads": MAX_THREADS or 8,
                    "num_streams": 1,
                    "enable_opencl_throttling": False,
                    "enable_qdq_optimizer": False,
                    "disable_dynamic_shapes": False,
                }
            )
        elif provider == "CUDAExecutionProvider":
            provider_options.append(
                {
                    "device_id": DEVICE_ID,
                    "gpu_mem_limit": 24 * 1024 * 1024 * 1024,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "sdpa_kernel": "2",
                    "use_tf32": "1",
                    "fuse_conv_bias": "0",
                    "cudnn_conv_use_max_workspace": "1",
                    "cudnn_conv1d_pad_to_nc1d": "0",
                    "tunable_op_enable": "0",
                    "tunable_op_tuning_enable": "0",
                    "do_copy_in_default_stream": "0",
                    "enable_cuda_graph": "0",
                    "prefer_nhwc": "0",
                }
            )
        elif provider == "DmlExecutionProvider":
            provider_options.append(
                {
                    "device_id": DEVICE_ID,
                    "performance_preference": "high_performance",
                    "device_filter": "gpu",
                    "disable_metacommands": "false",
                    "enable_graph_capture": "false",
                    "enable_graph_serialization": "false",
                }
            )
        else:
            provider_options.append({})

    primary = providers[0]
    if primary in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}:
        device_type = "cuda"
    elif primary == "DmlExecutionProvider":
        device_type = "dml"
    else:
        device_type = "cpu"
    return providers, provider_options, device_type


def create_session_options(providers: list[str]) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.log_severity_level = 0 if ORT_LOG else 4
    options.log_verbosity_level = 4
    options.inter_op_num_threads = MAX_THREADS
    options.intra_op_num_threads = MAX_THREADS
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    entries = {
        "session.set_denormal_as_zero": "1",
        "session.intra_op.allow_spinning": "1",
        "session.inter_op.allow_spinning": "1",
        "session.enable_quant_qdq_cleanup": "1",
        "session.use_device_allocator_for_initializers": "1",
        "session.graph_optimizations_loop_level": "2",
        "optimization.enable_gelu_approximation": "1",
        "optimization.enable_cast_chain_elimination": "1",
    }
    if DISABLE_CPU_EP_FALLBACK and "CPUExecutionProvider" not in providers:
        entries["session.disable_cpu_ep_fallback"] = "1"
    for key, value in entries.items():
        options.add_session_config_entry(key, value)
    return options


def create_run_options() -> ort.RunOptions:
    options = ort.RunOptions()
    options.log_severity_level = 0 if ORT_LOG else 4
    options.log_verbosity_level = 4
    options.add_run_config_entry("disable_synchronize_execution_providers", "0")
    return options


def disabled_optimizers(providers: list[str]) -> list[str] | None:
    if set(providers) != {"CPUExecutionProvider"}:
        return None
    disabled = []
    if CPU_DISABLE_MATMUL_ADD_FUSION:
        disabled.append("MatMulAddFusion")
    if CPU_DISABLE_NCHWC:
        disabled.append("NchwcTransformer")
    disabled.extend(CPU_EXTRA_DISABLED_OPTIMIZERS)
    return disabled or None


def create_session(
    path: Path,
    providers: list[str],
    provider_options: list[dict],
) -> ort.InferenceSession:
    return ort.InferenceSession(
        str(path),
        sess_options=create_session_options(providers),
        providers=providers,
        provider_options=provider_options,
        disabled_optimizers=disabled_optimizers(providers),
    )


def audio_normalizer(
    audio: np.ndarray, target_value: float = NORMALIZE_TARGET_RMS
) -> np.ndarray:
    normalized = audio.astype(np.float32)
    rms = np.sqrt(np.mean(normalized * normalized, dtype=np.float32))
    normalized *= np.float32(target_value) / (rms + np.float32(1e-7))
    np.clip(normalized, -32768.0, 32767.0, out=normalized)
    return normalized.astype(np.int16)


def load_audio(path: Path, target_sample_rate: int) -> LoadedAudio:
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        source = AudioSegment.from_file(path).set_sample_width(2)
    except (CouldntDecodeError, OSError) as error:
        raise RuntimeError(f"Could not decode {path}: {error}") from error

    source_sample_rate = source.frame_rate
    source_channels = source.channels
    duration = source.frame_count() / source_sample_rate
    converted = source.set_channels(1).set_frame_rate(target_sample_rate)
    audio = np.frombuffer(
        converted.raw_data,
        dtype="<i2",
    )
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if USE_AUDIO_NORMALIZER:
        audio = audio_normalizer(audio)
    return LoadedAudio(
        waveform=np.ascontiguousarray(audio, dtype=np.int16),
        sample_rate=target_sample_rate,
        source_sample_rate=source_sample_rate,
        source_channels=source_channels,
        duration=duration,
    )


def resolve_audio_dtype(input_type: str, metadata: dict) -> type[np.generic]:
    if input_type not in _ORT_AUDIO_DTYPES:
        raise ValueError(f"Unsupported ONNX waveform input type: {input_type!r}.")
    metadata_name = str(metadata.get("input_dtype", "")).lower()
    if metadata_name not in _METADATA_AUDIO_DTYPES:
        raise ValueError(
            f"Unsupported input_dtype in model metadata: {metadata_name!r}."
        )
    input_dtype = _ORT_AUDIO_DTYPES[input_type]
    if input_dtype != _METADATA_AUDIO_DTYPES[metadata_name]:
        raise ValueError(
            "ONNX waveform input type disagrees with embedded input_dtype metadata."
        )
    return input_dtype


def numpy_dtype_from_onnx_type(type_name: str) -> type[np.generic]:
    try:
        return _ORT_TENSOR_DTYPES[type_name]
    except KeyError as error:
        raise TypeError(f"Unsupported ONNX tensor type: {type_name}") from error


def resolve_tensor_shape(
    node_arg, *dynamic_extents: int
) -> tuple[int, ...]:
    dynamic_axes = [
        axis
        for axis, dimension in enumerate(node_arg.shape)
        if not isinstance(dimension, int)
    ]
    if len(dynamic_axes) != len(dynamic_extents):
        raise ValueError(
            f"Tensor {node_arg.name!r} shape {node_arg.shape} requires "
            f"{len(dynamic_axes)} dynamic extents, got {len(dynamic_extents)}."
        )
    extent_by_axis = dict(zip(dynamic_axes, dynamic_extents))
    return tuple(
        dimension if isinstance(dimension, int) else extent_by_axis[axis]
        for axis, dimension in enumerate(node_arg.shape)
    )


def validate_runtime_shape(node_arg, runtime_shape: tuple[int, ...]) -> None:
    if len(node_arg.shape) != len(runtime_shape):
        raise ValueError(
            f"Tensor {node_arg.name!r} expects rank {len(node_arg.shape)}, "
            f"got shape {runtime_shape}."
        )
    for dimension, runtime_dimension in zip(node_arg.shape, runtime_shape):
        if isinstance(dimension, int) and dimension != runtime_dimension:
            raise ValueError(
                f"Tensor {node_arg.name!r} expects shape {node_arg.shape}, "
                f"got shape {runtime_shape}."
            )


def prepare_waveform_input(
    waveform: np.ndarray,
    input_metadata,
    output: np.ndarray | None = None,
) -> np.ndarray:
    input_dtype = numpy_dtype_from_onnx_type(input_metadata.type)
    dynamic_axes = sum(
        not isinstance(dimension, int) for dimension in input_metadata.shape
    )
    if dynamic_axes > 1:
        raise ValueError(
            f"Waveform input has ambiguous dynamic shape {input_metadata.shape}."
        )
    shape = resolve_tensor_shape(
        input_metadata, *((waveform.size,) if dynamic_axes else ())
    )
    if input_dtype == np.int16:
        return np.ascontiguousarray(
            waveform.reshape(shape), dtype=input_dtype
        )

    if output is None or output.shape != shape or output.dtype != input_dtype:
        output = np.empty(shape, dtype=input_dtype)
    np.multiply(
        waveform,
        PCM16_SCALE,
        out=output.reshape(-1),
        casting="unsafe",
    )
    return output


def create_device_value(array: np.ndarray, device_type: str) -> ort.OrtValue:
    return ort.OrtValue.ortvalue_from_numpy(
        np.ascontiguousarray(array), device_type, DEVICE_ID
    )


def run_iobinding(
    session: ort.InferenceSession,
    inputs: dict[str, ort.OrtValue],
    device_type: str,
    run_options: ort.RunOptions,
    host_outputs: frozenset[str] = frozenset(),
    binding: ort.IOBinding | None = None,
    output_names: list[str] | None = None,
) -> list[ort.OrtValue]:
    if binding is None:
        binding = session.io_binding()
    else:
        binding.clear_binding_inputs()
        binding.clear_binding_outputs()
    for name, value in inputs.items():
        binding.bind_ortvalue_input(name, value)
    if output_names is None:
        output_names = [output.name for output in session.get_outputs()]
    for output_name in output_names:
        if output_name in host_outputs:
            binding.bind_output(output_name, "cpu", 0)
        else:
            binding.bind_output(output_name, device_type, DEVICE_ID)
    session.run_with_iobinding(binding, run_options=run_options)
    return binding.get_outputs()


def initialize_clusters(normalized: np.ndarray) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage

    dendrogram = linkage(normalized, method="centroid", metric="euclidean")
    initial = fcluster(dendrogram, 0.6, criterion="distance")
    initial -= 1
    return np.ascontiguousarray(initial, dtype=np.int32)


def resolve_cluster_target(
    num_clusters: int | None,
    min_clusters: int,
    max_clusters: float,
    automatic_clusters: int,
) -> int | None:
    if automatic_clusters < min_clusters:
        return min_clusters
    if automatic_clusters > max_clusters:
        return int(max_clusters)
    return num_clusters


def binarize_turns(
    diarization: np.ndarray,
    frames: FrameGrid,
    state_buffer: np.ndarray | None = None,
    change_buffer: np.ndarray | None = None,
) -> list[tuple[float, float, int]]:
    if diarization.shape[0] == 0:
        return []

    last_frame = diarization.shape[0] - 1
    if state_buffer is None or state_buffer.size < diarization.shape[0]:
        state_buffer = np.empty(diarization.shape[0], dtype=np.bool_)
    else:
        state_buffer = state_buffer[: diarization.shape[0]]
    if change_buffer is None or change_buffer.size < last_frame:
        change_buffer = np.empty(last_frame, dtype=np.bool_)
    else:
        change_buffer = change_buffer[:last_frame]
    turns = []
    for speaker, scores in enumerate(diarization.T):
        np.not_equal(scores, 0.5, out=state_buffer)
        states = state_buffer
        decisive_count = np.count_nonzero(states)
        if decisive_count == 0:
            continue
        if decisive_count == scores.size:
            np.greater(scores, 0.5, out=states)
            decisive = None
            first_frame = 0
        else:
            decisive = np.flatnonzero(states)
            states = scores[decisive] > 0.5
            first_frame = int(decisive[0])

        num_comparisons = states.size - 1
        np.not_equal(
            states[1:],
            states[:-1],
            out=change_buffer[:num_comparisons],
        )
        change_positions = np.flatnonzero(
            change_buffer[:num_comparisons]
        )
        change_positions += 1
        start_frame = first_frame if states[0] else None
        for position in change_positions:
            frame = int(position if decisive is None else decisive[position])
            if states[position]:
                start_frame = frame
                continue
            if start_frame is None:
                continue
            turns.append(
                (
                    frames.middle(int(start_frame)),
                    frames.middle(frame),
                    speaker,
                )
            )
            start_frame = None
        if start_frame is not None:
            turns.append(
                (
                    frames.middle(int(start_frame)),
                    frames.middle(last_frame),
                    speaker,
                )
            )
    return sorted(turns, key=lambda turn: (turn[0], turn[1], turn[2]))


def label_turns(
    regular: list[tuple[float, float, int]],
    exclusive: list[tuple[float, float, int]],
    centroids: np.ndarray,
) -> tuple[list[SpeechTurn], list[SpeechTurn], np.ndarray]:
    labels = sorted({speaker for _, _, speaker in regular})
    mapping = {
        speaker: f"SPEAKER_{index:02d}"
        for index, speaker in enumerate(labels)
    }
    for _, _, speaker in exclusive:
        if speaker not in mapping:
            mapping[speaker] = f"SPEAKER_{len(mapping):02d}"
            labels.append(speaker)

    maximum_label = max(labels, default=-1)
    if maximum_label >= centroids.shape[0]:
        centroids = np.pad(
            centroids,
            ((0, maximum_label + 1 - centroids.shape[0]), (0, 0)),
        )
    if not labels:
        ordered_centroids = np.zeros(
            (0, centroids.shape[1]), dtype=centroids.dtype
        )
    elif all(label == index for index, label in enumerate(labels)):
        ordered_centroids = centroids[: len(labels)]
    else:
        ordered_centroids = centroids[labels]
    regular_turns = [
        SpeechTurn(start, end, mapping[speaker])
        for start, end, speaker in regular
    ]
    exclusive_turns = [
        SpeechTurn(start, end, mapping[speaker])
        for start, end, speaker in exclusive
    ]
    return regular_turns, exclusive_turns, ordered_centroids


class Community1DevicePipeline:
    FRONTEND_OUTPUTS = [
        "segmentations",
        "inactive_speakers",
        "speaker_counts",
        "embeddings",
        "train_embeddings",
        "train_features",
        "normalized_train_embeddings",
    ]
    BACKEND_INPUTS = [
        "embeddings",
        "train_embeddings",
        "train_features",
        "inactive_speakers",
        "segmentations",
        "speaker_counts",
        "initial_clusters",
        "initial_cluster_ids",
        "override_centroid_weights",
        "use_override",
        "constrained",
    ]
    BACKEND_OUTPUTS = [
        "centroids",
        "regular_diarization",
        "exclusive_diarization",
    ]
    FRONTEND_HOST_OUTPUTS = frozenset()
    BACKEND_HOST_OUTPUTS = frozenset(BACKEND_OUTPUTS)

    def __init__(
        self,
        frontend_session: ort.InferenceSession,
        backend_session: ort.InferenceSession,
        frontend_metadata: dict,
        backend_metadata: dict,
        device_type: str,
    ):
        self.frontend_session = frontend_session
        self.backend_session = backend_session
        self.frontend_binding = frontend_session.io_binding()
        self.backend_binding = backend_session.io_binding()
        self.device_type = device_type
        self.run_options = create_run_options()
        self.sample_rate = int(frontend_metadata["sample_rate"])
        self.embedding_dimension = int(frontend_metadata["embedding_dimension"])
        self.plda_dimension = int(frontend_metadata["plda_dimension"])
        receptive_field = backend_metadata["receptive_field"]
        self.frame_grid = FrameGrid(
            start=float(receptive_field["start"]),
            duration=float(receptive_field["duration"]),
            step=float(receptive_field["step"]),
        )

        self._validate_session(
            frontend_session, ["waveform"], self.FRONTEND_OUTPUTS
        )
        self._validate_session(
            backend_session, self.BACKEND_INPUTS, self.BACKEND_OUTPUTS
        )
        frontend_input = frontend_session.get_inputs()[0]
        self.frontend_input = frontend_input
        self.backend_inputs = {
            value.name: value for value in backend_session.get_inputs()
        }
        self.model_inputs = {
            frontend_input.name: frontend_input,
            **self.backend_inputs,
        }
        self.input_dtype = resolve_audio_dtype(
            frontend_input.type, frontend_metadata
        )
        if (
            int(frontend_metadata["num_frames"])
            != int(frontend_metadata["mask_num_frames"])
        ):
            raise ValueError("Segmentation and embedding frame grids disagree.")
        if (
            self.plda_dimension != int(backend_metadata["plda_dimension"])
        ):
            raise ValueError("Embedding and VBx PLDA dimensions disagree.")
        if (
            self.embedding_dimension
            != int(backend_metadata["embedding_dimension"])
        ):
            raise ValueError("Embedding and clustering dimensions disagree.")

        self._device_input_cache: dict[
            str, tuple[tuple[int, ...], np.dtype, ort.OrtValue]
        ] = {}
        self._waveform_buffer: np.ndarray | None = None
        self._override_weights_buffer: np.ndarray | None = None
        self._cluster_indices: np.ndarray | None = None
        self._turn_state_buffer = np.empty(0, dtype=np.bool_)
        self._turn_change_buffer = np.empty(0, dtype=np.bool_)
        self._fallback_centroid_buffer = np.empty(
            resolve_tensor_shape(self.backend_inputs["train_embeddings"], 1),
            dtype=numpy_dtype_from_onnx_type(
                self.backend_inputs["train_embeddings"].type
            ),
        )
        self._fallback_finite_mask = np.empty(0, dtype=np.bool_)
        self._fallback_finite_scratch = np.empty(
            (0, self.embedding_dimension), dtype=np.bool_
        )
        self._empty_override: tuple[int, ort.OrtValue] | None = None
        self._use_override_values = tuple(
            self._constant_input_value("use_override", (), value)
            for value in (False, True)
        )
        self._constrained_values = tuple(
            self._constant_input_value("constrained", (), value)
            for value in (False, True)
        )
        self._fallback_initial_value = self._constant_input_value(
            "initial_clusters", (1,), 0
        )
        self._fallback_cluster_ids_value = self._constant_input_value(
            "initial_cluster_ids", (1,), 0
        )
        self._fallback_override_value = self._constant_input_value(
            "override_centroid_weights", (1, 1), 1
        )
        self._fallback_features_value = self._constant_input_value(
            "train_features", (1,), 0
        )

    @staticmethod
    def _validate_session(
        session: ort.InferenceSession,
        expected_inputs: list[str],
        expected_outputs: list[str],
    ) -> None:
        inputs = [value.name for value in session.get_inputs()]
        outputs = [value.name for value in session.get_outputs()]
        if inputs != expected_inputs or outputs != expected_outputs:
            raise ValueError(
                f"Unexpected ONNX interface: inputs={inputs}, outputs={outputs}."
            )

    def _constant_input_value(
        self,
        name: str,
        dynamic_extents: tuple[int, ...],
        fill_value,
    ) -> ort.OrtValue:
        metadata = self.backend_inputs[name]
        array = np.full(
            resolve_tensor_shape(metadata, *dynamic_extents),
            fill_value,
            dtype=numpy_dtype_from_onnx_type(metadata.type),
        )
        return create_device_value(array, self.device_type)

    def _device_input(
        self,
        name: str,
        array: np.ndarray,
        cache_key: str | None = None,
    ) -> ort.OrtValue:
        metadata = self.model_inputs[name]
        array = np.ascontiguousarray(
            array, dtype=numpy_dtype_from_onnx_type(metadata.type)
        )
        validate_runtime_shape(metadata, array.shape)
        if self.device_type == "cpu":
            return create_device_value(array, self.device_type)

        cache_key = cache_key or name
        signature = (array.shape, array.dtype)
        cached = self._device_input_cache.get(cache_key)
        if cached is not None and cached[:2] == signature:
            cached[2].update_inplace(array)
            return cached[2]

        value = create_device_value(array, self.device_type)
        self._device_input_cache[cache_key] = (*signature, value)
        return value

    def _empty_override_value(self, num_rows: int) -> ort.OrtValue:
        if self._empty_override is None or self._empty_override[0] != num_rows:
            metadata = self.backend_inputs["override_centroid_weights"]
            value = create_device_value(
                np.empty(
                    resolve_tensor_shape(metadata, num_rows, 0),
                    dtype=numpy_dtype_from_onnx_type(metadata.type),
                ),
                self.device_type,
            )
            self._empty_override = (num_rows, value)
        return self._empty_override[1]

    def _fallback_centroid(self, embeddings: np.ndarray) -> np.ndarray:
        rows = embeddings.reshape(-1, self.embedding_dimension)
        num_rows = rows.shape[0]
        if self._fallback_finite_mask.size < num_rows:
            self._fallback_finite_mask = np.empty(num_rows, dtype=np.bool_)
        finite_mask = self._fallback_finite_mask[:num_rows]

        block_rows = min(num_rows, FALLBACK_FINITE_BLOCK_ROWS)
        if self._fallback_finite_scratch.shape[0] < block_rows:
            self._fallback_finite_scratch = np.empty(
                (block_rows, self.embedding_dimension), dtype=np.bool_
            )
        for start in range(0, num_rows, FALLBACK_FINITE_BLOCK_ROWS):
            stop = min(start + FALLBACK_FINITE_BLOCK_ROWS, num_rows)
            finite_block = self._fallback_finite_scratch[: stop - start]
            np.isfinite(rows[start:stop], out=finite_block)
            np.all(finite_block, axis=1, out=finite_mask[start:stop])

        finite_count = np.count_nonzero(finite_mask)
        centroid = self._fallback_centroid_buffer
        if finite_count:
            np.sum(
                rows,
                axis=0,
                dtype=np.float32,
                out=centroid[0],
                where=finite_mask[:, None],
            )
            centroid /= np.float32(finite_count)
        else:
            centroid.fill(0.0)
        return centroid

    def _binarize_turns(
        self, diarization: np.ndarray
    ) -> list[tuple[float, float, int]]:
        num_frames = diarization.shape[0]
        if self._turn_state_buffer.size < num_frames:
            self._turn_state_buffer = np.empty(num_frames, dtype=np.bool_)
        if self._turn_change_buffer.size < max(num_frames - 1, 0):
            self._turn_change_buffer = np.empty(
                num_frames - 1, dtype=np.bool_
            )
        return binarize_turns(
            diarization,
            self.frame_grid,
            self._turn_state_buffer,
            self._turn_change_buffer,
        )

    def _run_backend(
        self,
        frontend_outputs: list[ort.OrtValue],
        train_embeddings: ort.OrtValue,
        train_features: ort.OrtValue,
        initial_clusters: ort.OrtValue,
        initial_cluster_ids: ort.OrtValue,
        override_centroid_weights: ort.OrtValue,
        use_override: bool,
        constrained: bool,
        host_outputs: bool,
    ) -> list[ort.OrtValue]:
        return run_iobinding(
            self.backend_session,
            {
                "embeddings": frontend_outputs[3],
                "train_embeddings": train_embeddings,
                "train_features": train_features,
                "inactive_speakers": frontend_outputs[1],
                "segmentations": frontend_outputs[0],
                "speaker_counts": frontend_outputs[2],
                "initial_clusters": initial_clusters,
                "initial_cluster_ids": initial_cluster_ids,
                "override_centroid_weights": override_centroid_weights,
                "use_override": self._use_override_values[use_override],
                "constrained": self._constrained_values[constrained],
            },
            self.device_type,
            self.run_options,
            self.BACKEND_HOST_OUTPUTS if host_outputs else frozenset(),
            self.backend_binding,
            self.BACKEND_OUTPUTS,
        )

    def __call__(
        self,
        audio: LoadedAudio,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> DiarizationResult:
        bounds_requested = any(
            value is not None
            for value in (num_speakers, min_speakers, max_speakers)
        )
        num_speakers, minimum, maximum = set_speaker_bounds(
            num_speakers, min_speakers, max_speakers
        )
        waveform = prepare_waveform_input(
            audio.waveform, self.frontend_input, self._waveform_buffer
        )
        if self.input_dtype != np.int16:
            self._waveform_buffer = waveform
        waveform_value = self._device_input("waveform", waveform)

        frontend_outputs = run_iobinding(
            self.frontend_session,
            {"waveform": waveform_value},
            self.device_type,
            self.run_options,
            self.FRONTEND_HOST_OUTPUTS,
            self.frontend_binding,
            self.FRONTEND_OUTPUTS,
        )

        num_training_embeddings = frontend_outputs[4].shape()[0]
        if num_training_embeddings < 2:
            if num_training_embeddings == 0:
                train_embeddings = self._device_input(
                    "train_embeddings",
                    self._fallback_centroid(frontend_outputs[3].numpy()),
                    cache_key="fallback_centroid",
                )
                train_features = self._fallback_features_value
            else:
                train_embeddings = frontend_outputs[4]
                train_features = frontend_outputs[5]
            backend_outputs = self._run_backend(
                frontend_outputs,
                train_embeddings,
                train_features,
                self._fallback_initial_value,
                self._fallback_cluster_ids_value,
                self._fallback_override_value,
                use_override=True,
                constrained=False,
                host_outputs=True,
            )
        else:
            normalized = frontend_outputs[6].numpy()
            initial_clusters = initialize_clusters(normalized)
            initial_value = self._device_input(
                "initial_clusters", initial_clusters
            )
            initial_cluster_ids = np.arange(
                int(initial_clusters.max()) + 1, dtype=np.int32
            )
            initial_cluster_ids_value = self._device_input(
                "initial_cluster_ids", initial_cluster_ids
            )
            backend_outputs = self._run_backend(
                frontend_outputs,
                frontend_outputs[4],
                frontend_outputs[5],
                initial_value,
                initial_cluster_ids_value,
                self._empty_override_value(num_training_embeddings),
                use_override=False,
                constrained=True,
                host_outputs=not bounds_requested,
            )
            automatic_clusters = backend_outputs[0].shape()[0]
            target_clusters = resolve_cluster_target(
                num_speakers, minimum, maximum, automatic_clusters
            )
            if (
                target_clusters is not None
                and target_clusters != automatic_clusters
            ):
                from sklearn.cluster import KMeans

                labels = KMeans(
                    n_clusters=target_clusters,
                    n_init=3,
                    random_state=42,
                    copy_x=False,
                ).fit_predict(normalized)
                weights_shape = (num_training_embeddings, target_clusters)
                weights_shape = resolve_tensor_shape(
                    self.backend_inputs["override_centroid_weights"],
                    *weights_shape,
                )
                if (
                    self._override_weights_buffer is None
                    or self._override_weights_buffer.shape != weights_shape
                ):
                    self._override_weights_buffer = np.empty(
                        weights_shape,
                        dtype=numpy_dtype_from_onnx_type(
                            self.backend_inputs["override_centroid_weights"].type
                        ),
                    )
                if (
                    self._cluster_indices is None
                    or self._cluster_indices.size != target_clusters
                    or self._cluster_indices.dtype != labels.dtype
                ):
                    self._cluster_indices = np.arange(
                        target_clusters, dtype=labels.dtype
                    )
                weights = self._override_weights_buffer
                np.equal(
                    labels[:, None],
                    self._cluster_indices,
                    out=weights,
                    casting="unsafe",
                )
                backend_outputs = self._run_backend(
                    frontend_outputs,
                    frontend_outputs[4],
                    frontend_outputs[5],
                    initial_value,
                    initial_cluster_ids_value,
                    self._device_input("override_centroid_weights", weights),
                    use_override=True,
                    constrained=False,
                    host_outputs=True,
                )
        regular = self._binarize_turns(backend_outputs[1].numpy())
        exclusive = self._binarize_turns(backend_outputs[2].numpy())
        regular_turns, exclusive_turns, centroids = label_turns(
            regular, exclusive, backend_outputs[0].numpy()
        )
        return DiarizationResult(
            diarization=regular_turns,
            exclusive_diarization=exclusive_turns,
            speaker_embeddings=centroids,
        )


def set_speaker_bounds(
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> tuple[int | None, int, float]:
    minimum = num_speakers or min_speakers or 1
    maximum = float(num_speakers or max_speakers or np.inf)
    if minimum > maximum:
        raise ValueError(
            "min_speakers must be smaller than or equal to max_speakers, "
            f"got {minimum} and {maximum}."
        )
    if minimum == maximum:
        num_speakers = minimum
    return num_speakers, minimum, maximum


def write_rttm(path: Path, uri: str, turns: list[SpeechTurn]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for turn in turns:
            file.write(
                f"SPEAKER {uri} 1 {turn.start:.3f} "
                f"{turn.end - turn.start:.3f} <NA> <NA> "
                f"{turn.speaker} <NA> <NA>\n"
            )


def positive_integer(value: int | None, name: str) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir",
        nargs="?",
        type=Path,
        default=ONNX_MODEL_DIR,
        help="Directory containing the merged Community-1 ONNX models.",
    )
    return parser.parse_args()


def main(model_dir: Path) -> None:
    positive_integer(NUM_SPEAKERS, "num speakers")
    positive_integer(MIN_SPEAKERS, "min speakers")
    positive_integer(MAX_SPEAKERS, "max speakers")

    model_dir = model_dir.expanduser().resolve()
    metadata_path, package_manifest = read_package_metadata(model_dir)
    model_paths = {
        component: resolve_model_path(model_dir, component)
        for component in MODEL_NAMES
    }
    providers, provider_options, device_type = resolve_providers(
        ORT_ACCELERATE_PROVIDERS
    )
    sessions = {
        component: create_session(path, providers, provider_options)
        for component, path in model_paths.items()
    }
    metadata = {
        component: read_metadata(sessions[component], component)
        for component in sessions
    }
    validate_package_metadata(package_manifest, model_paths, metadata)
    print(f"\nUsable Providers: {sessions['frontend'].get_providers()}")
    print(f"OrtValue Device: {device_type}:{DEVICE_ID}")
    print(f"Metadata: {metadata_path}")
    for component, path in model_paths.items():
        print(f"{component.title()}: {path}")

    pipeline = Community1DevicePipeline(
        sessions["frontend"],
        sessions["backend"],
        metadata["frontend"],
        metadata["backend"],
        device_type,
    )
    audio_path = TEST_AUDIO.expanduser().resolve()
    audio = load_audio(audio_path, pipeline.sample_rate)
    print(
        f"Audio: {audio_path} ({audio.duration:.3f}s, "
        f"{audio.source_sample_rate} Hz, {audio.source_channels} channel(s))"
    )

    start = time.perf_counter()
    result = pipeline(
        audio,
        num_speakers=NUM_SPEAKERS,
        min_speakers=MIN_SPEAKERS,
        max_speakers=MAX_SPEAKERS,
    )
    elapsed = time.perf_counter() - start
    for turn in result.diarization:
        print(
            f"{turn.speaker} speaks between "
            f"t={turn.start:.3f}s and t={turn.end:.3f}s"
        )
    speakers = sorted({turn.speaker for turn in result.diarization})
    rtf = elapsed / audio.duration
    print(
        f"Detected {len(speakers)} speaker(s) in {elapsed:.3f}s "
        f"(RTF: {rtf:.4f})."
    )

    if JSON_PATH is not None:
        JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        JSON_PATH.write_text(
            json.dumps(result.serialize(elapsed), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"JSON: {JSON_PATH}")
    if RTTM_PATH is not None:
        write_rttm(RTTM_PATH, audio_path.stem, result.diarization)
        print(f"RTTM: {RTTM_PATH}")
    if EXCLUSIVE_RTTM_PATH is not None:
        write_rttm(
            EXCLUSIVE_RTTM_PATH,
            audio_path.stem,
            result.exclusive_diarization,
        )
        print(f"Exclusive RTTM: {EXCLUSIVE_RTTM_PATH}")


if __name__ == "__main__":
    main(parse_args().model_dir)
