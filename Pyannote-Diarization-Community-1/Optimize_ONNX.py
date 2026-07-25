import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Merge_Pyannote_Community1 import remove_split_models
from Optimize_ONNX_Common import (
    atomic_model_path,
    convert_graph_to_fp16,
    preserve_model_properties,
    simplify_onnx,
)
from vad_onnx_metadata import (
    community1_metadata_path,
    export_community1_metadata_carrier,
    parse_community1_metadata,
    read_onnx_metadata,
)


DEFAULT_SOURCE_DIR = SCRIPT_DIR / "Pyannote_Community1_ONNX"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "Pyannote_Community1_Optimized"
MODEL_NAMES = (
    "segmentation_embedding.onnx",
    "vbx_clustering_diarization.onnx",
)
SOURCE_DIR = DEFAULT_SOURCE_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
MODELS = MODEL_NAMES
NO_SHAPE_INFER = False
ENABLE_FP16 = False  # Convert internal float compute to FP16; preserve public I/O types.
RTOL = 1e-5
ATOL = 1e-5
FP16_RTOL = 1e-2
FP16_ATOL = 1e-2
FP16_PLDA_ATOL = 7e-2
FP16_OP_BLOCK_LIST = [
    "LSTM",
    "InstanceNormalization",
    "Range",
    "ScatterElements",
]  # Required FP32 CPU islands and unsupported FP16 shape generation.
CPU_DISABLED_OPTIMIZERS = [  # Match inference; mixed LSTM graphs crash without these.
    "MatMulAddFusion",
    "NchwcTransformer",
    "ConvAddActivationFusion",
    "MatmulTransposeFusion",
]
METADATA_KEY = "pyannote.community1"
_ORT_AUDIO_DTYPES = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(int16)": np.int16,
}


def read_component_metadata(path: Path) -> dict:
    model = onnx.load(path, load_external_data=False)
    metadata = {item.key: item.value for item in model.metadata_props}
    raw = metadata.get(METADATA_KEY)
    if raw is None:
        raise ValueError(
            f"{path} has no {METADATA_KEY!r} metadata. Re-export the model first."
        )
    return json.loads(raw)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def graph_statistics(path: Path) -> dict:
    model = onnx.load(path, load_external_data=False)
    operators = Counter(node.op_type for node in model.graph.node)
    return {
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "operator_histogram": dict(sorted(operators.items())),
    }


def publish_metadata_carrier(source_dir: Path, output_dir: Path) -> Path:
    source_metadata_path = community1_metadata_path(source_dir)
    if not source_metadata_path.is_file():
        raise FileNotFoundError(
            f"Required Community-1 metadata model not found: {source_metadata_path}"
        )
    manifest = parse_community1_metadata(
        read_onnx_metadata(source_metadata_path)
    )
    for details in manifest["models"].values():
        model_path = output_dir / details["file"]
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Optimized Community-1 model not found: {model_path}"
            )
        details["sha256"] = sha256(model_path)
        details["graph"] = graph_statistics(model_path)

    metadata_path = community1_metadata_path(output_dir)
    export_community1_metadata_carrier(
        metadata_path,
        manifest,
        producer="Optimize_ONNX.py",
    )
    (output_dir / "community1_onnx.json").unlink(missing_ok=True)
    return metadata_path


def convert_model_to_fp16(path: Path) -> None:
    model = onnx.load(path, load_external_data=True)
    node_block_list = [
        node.name for node in model.graph.node if "/plda/" in node.name
    ]
    model, _ = convert_graph_to_fp16(
        model,
        force_fp16_initializers=True,
        op_block_list=FP16_OP_BLOCK_LIST,
        node_block_list=node_block_list,
    )
    onnx.save(model, path, save_as_external_data=False)


def create_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.set_denormal_as_zero", "1")
    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
        disabled_optimizers=CPU_DISABLED_OPTIMIZERS,
    )


def random_inputs(
    component: str,
    metadata: dict,
    case_size: int,
    input_types: dict[str, str],
) -> dict:
    generator = np.random.default_rng(20250721 + case_size)
    if component in {"segmentation", "frontend"}:
        input_type = input_types["waveform"]
        if input_type not in _ORT_AUDIO_DTYPES:
            raise ValueError(
                f"Unsupported ONNX waveform input type: {input_type!r}."
            )
        input_dtype = _ORT_AUDIO_DTYPES[input_type]
        num_samples = int(metadata["num_samples"])
        step_samples = round(
            float(metadata["chunk_step"]) * int(metadata["sample_rate"])
        )
        audio_len = num_samples + max(case_size - 2, 0) * step_samples
        if case_size > 1:
            audio_len += 1
        shape = (1, 1, audio_len)
        if input_dtype == np.int16:
            waveform = generator.integers(
                -32768, 32768, size=shape, dtype=np.int16
            )
        else:
            waveform = (
                0.1 * generator.standard_normal(shape, dtype=np.float32)
            ).astype(input_dtype, copy=False)
        return {"waveform": waveform}
    if component == "embedding":
        input_type = input_types["chunks"]
        if input_type not in _ORT_AUDIO_DTYPES:
            raise ValueError(
                f"Unsupported ONNX chunk input type: {input_type!r}."
            )
        input_dtype = _ORT_AUDIO_DTYPES[input_type]
        num_samples = int(metadata["num_samples"])
        chunk_shape = (case_size, 1, num_samples)
        if input_dtype == np.int16:
            chunks = generator.integers(
                -32768, 32768, size=chunk_shape, dtype=np.int16
            )
        else:
            chunks = (
                0.1
                * generator.standard_normal(chunk_shape, dtype=np.float32)
            ).astype(input_dtype, copy=False)
        mask_frames = int(metadata["mask_num_frames"])
        num_speakers = 3
        weights = (
            generator.random(
                (case_size, num_speakers, mask_frames), dtype=np.float32
            )
            > 0.35
        ).astype(np.float32)
        active = generator.integers(
            0, 2, size=(case_size, num_speakers), dtype=np.int8
        ).astype(bool)
        active.reshape(-1)[0] = True
        return {
            "chunks": chunks,
            "embedding_masks": weights,
            "active_speakers": active,
        }
    if component == "vbx":
        observations = 4 * case_size + 3
        num_clusters = case_size + 1
        features = generator.standard_normal(
            (observations, int(metadata["plda_dimension"])), dtype=np.float32
        )
        initial_clusters = (
            np.arange(observations, dtype=np.int32) % num_clusters
        )
        initial_cluster_ids = np.arange(num_clusters, dtype=np.int32)
        return {
            "train_features": features,
            "initial_clusters": initial_clusters,
            "initial_cluster_ids": initial_cluster_ids,
        }
    if component == "clustering":
        observations = 4 * case_size + 3
        num_clusters = case_size + 1
        dimension = int(metadata["embedding_dimension"])
        num_speakers = int(metadata["num_local_speakers"])
        embeddings = generator.standard_normal(
            (case_size, num_speakers, dimension), dtype=np.float32
        )
        train_embeddings = generator.standard_normal(
            (observations, dimension), dtype=np.float32
        )
        inactive = generator.integers(
            0, 2, size=(case_size, num_speakers), dtype=np.int8
        ).astype(bool)
        centroid_weights = (
            generator.random(
                (observations, num_clusters), dtype=np.float32
            )
            + 0.01
        )
        return {
            "embeddings": embeddings,
            "train_embeddings": train_embeddings,
            "inactive_speakers": inactive,
            "centroid_weights": centroid_weights,
            "constrained": np.asarray([case_size == 1], dtype=np.bool_),
        }
    if component == "reconstruction":
        num_frames = int(metadata["num_frames"])
        num_speakers = int(metadata["num_local_speakers"])
        segmentations = generator.integers(
            0,
            2,
            size=(case_size, num_frames, num_speakers),
            dtype=np.int32,
        ).astype(np.float32)
        frame_step = float(metadata["receptive_field"]["step"])
        output_frames = round(
            (
                float(metadata["chunk_duration"])
                + (case_size - 1) * float(metadata["chunk_step"])
            )
            / frame_step
        ) + 1
        speaker_counts = generator.integers(
            0,
            num_speakers + 1,
            size=(output_frames, 1),
            dtype=np.int32,
        )
        assignments = np.tile(
            np.arange(num_speakers, dtype=np.int32), (case_size, 1)
        )
        assignments[::2, -1] = -2
        return {
            "segmentations": segmentations,
            "speaker_counts": speaker_counts,
            "assignments": assignments,
        }
    if component == "backend":
        observations = 4 * case_size + 3
        num_clusters = case_size + 1
        embedding_dimension = int(metadata["embedding_dimension"])
        plda_dimension = int(metadata["plda_dimension"])
        num_speakers = int(metadata["num_local_speakers"])
        num_frames = int(metadata["num_frames"])
        embeddings = generator.standard_normal(
            (case_size, num_speakers, embedding_dimension),
            dtype=np.float32,
        )
        train_embeddings = generator.standard_normal(
            (observations, embedding_dimension), dtype=np.float32
        )
        train_features = generator.standard_normal(
            (observations, plda_dimension), dtype=np.float32
        )
        inactive = generator.integers(
            0, 2, size=(case_size, num_speakers), dtype=np.int8
        ).astype(bool)
        initial_clusters = (
            np.arange(observations, dtype=np.int32) % num_clusters
        )
        initial_cluster_ids = np.arange(num_clusters, dtype=np.int32)
        override_weights = (
            generator.random(
                (observations, num_clusters), dtype=np.float32
            )
            + 0.01
        )
        segmentations = generator.integers(
            0,
            2,
            size=(case_size, num_frames, num_speakers),
            dtype=np.int32,
        ).astype(np.float32)
        output_frames = round(
            (
                float(metadata["chunk_duration"])
                + (case_size - 1) * float(metadata["chunk_step"])
            )
            / float(metadata["receptive_field"]["step"])
        ) + 1
        speaker_counts = generator.integers(
            0,
            num_speakers + 1,
            size=(output_frames, 1),
            dtype=np.int32,
        )
        return {
            "embeddings": embeddings,
            "train_embeddings": train_embeddings,
            "train_features": train_features,
            "inactive_speakers": inactive,
            "segmentations": segmentations,
            "speaker_counts": speaker_counts,
            "initial_clusters": initial_clusters,
            "initial_cluster_ids": initial_cluster_ids,
            "override_centroid_weights": override_weights,
            "use_override": np.asarray(case_size == 2, dtype=np.bool_),
            "constrained": np.asarray([case_size == 1], dtype=np.bool_),
        }
    raise ValueError(f"Unsupported component: {component!r}")


def compare_sessions(
    source: Path,
    target: Path,
    metadata: dict,
    rtol: float,
    atol: float,
    plda_atol: float | None = None,
) -> dict[str, float]:
    source_session = create_session(source)
    target_session = create_session(target)
    source_inputs = source_session.get_inputs()
    target_inputs = target_session.get_inputs()
    source_outputs = source_session.get_outputs()
    target_outputs = target_session.get_outputs()

    def signature(values: list) -> list[tuple[str, str, tuple]]:
        return [
            (
                value.name,
                value.type,
                tuple(
                    dimension if isinstance(dimension, int) else None
                    for dimension in value.shape
                ),
            )
            for value in values
        ]

    source_signature = signature(source_inputs)
    target_signature = signature(target_inputs)
    if source_signature != target_signature:
        raise RuntimeError(
            f"Input signature changed for {target.name}: "
            f"{source_signature} != {target_signature}"
        )
    source_output_signature = signature(source_outputs)
    target_output_signature = signature(target_outputs)
    if source_output_signature != target_output_signature:
        raise RuntimeError(
            f"Output signature changed for {target.name}: "
            f"{source_output_signature} != {target_output_signature}"
        )

    component = metadata["component"]
    maxima = {"max_abs": 0.0, "mean_abs": 0.0, "max_rel": 0.0}
    mean_errors = []
    input_types = {value.name: value.type for value in source_inputs}
    for case_size in (1, 2):
        feeds = random_inputs(
            component, metadata, case_size, input_types
        )
        source_values = source_session.run(None, feeds)
        target_values = target_session.run(None, feeds)
        for output, source_value, target_value in zip(
            source_outputs, source_values, target_values
        ):
            if source_value.shape != target_value.shape:
                raise RuntimeError(
                    f"Optimized {target.name} changed {output.name!r} shape "
                    f"for case {case_size}: {source_value.shape} != "
                    f"{target_value.shape}."
                )
            if source_value.dtype.kind in "biu":
                if not np.array_equal(source_value, target_value):
                    raise RuntimeError(
                        f"Optimized {target.name} changed a discrete output "
                        f"for case {case_size}."
                    )
                continue
            if source_value.size == 0:
                continue
            absolute = np.abs(source_value - target_value)
            relative = absolute / np.maximum(np.abs(source_value), 1e-7)
            finite_absolute = absolute[np.isfinite(absolute)]
            finite_relative = relative[np.isfinite(relative)]
            if finite_absolute.size:
                maxima["max_abs"] = max(
                    maxima["max_abs"], float(finite_absolute.max())
                )
                mean_errors.append(float(finite_absolute.mean()))
            if finite_relative.size:
                maxima["max_rel"] = max(
                    maxima["max_rel"], float(finite_relative.max())
                )
            output_atol = (
                plda_atol
                if output.name == "train_features" and plda_atol is not None
                else atol
            )
            if not np.allclose(
                source_value,
                target_value,
                rtol=rtol,
                atol=output_atol,
                equal_nan=True,
            ):
                raise RuntimeError(
                    f"Optimized {target.name} failed parity for case {case_size}: "
                    f"max_abs={absolute.max():.6g}, max_rel={relative.max():.6g}."
                )
    maxima["mean_abs"] = float(np.mean(mean_errors)) if mean_errors else 0.0
    return maxima


def optimize_one(
    source: Path,
    target: Path,
    no_shape_infer: bool,
    enable_fp16: bool,
    rtol: float,
    atol: float,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Source ONNX model not found: {source}")
    metadata = read_component_metadata(source)
    source_size = source.stat().st_size
    target.parent.mkdir(parents=True, exist_ok=True)

    with atomic_model_path(target, prefix="community1_onnx_") as temporary_path:
        simplify_onnx(
            source,
            temporary_path,
            no_shape_infer=no_shape_infer,
        )
        converted_to_fp16 = enable_fp16 and metadata["component"] in {
            "frontend",
        }
        if converted_to_fp16:
            convert_model_to_fp16(temporary_path)
        preserve_model_properties(source, temporary_path)
        validation = compare_sessions(
            source,
            temporary_path,
            metadata,
            rtol=rtol,
            atol=atol,
            plda_atol=FP16_PLDA_ATOL if converted_to_fp16 else None,
        )

    target_size = target.stat().st_size
    reduction = 100.0 * (source_size - target_size) / source_size
    print(
        f"{source.name}: {source_size / 1_048_576:.2f} MiB -> "
        f"{target_size / 1_048_576:.2f} MiB ({reduction:+.2f}% reduction), "
        f"max_abs={validation['max_abs']:.6g}, "
        f"mean_abs={validation['mean_abs']:.6g}"
    )


def main() -> None:
    source_dir = SOURCE_DIR.expanduser().resolve()
    output_dir = OUTPUT_DIR.expanduser().resolve()
    if source_dir == output_dir:
        raise ValueError("source_dir and output_dir must be different directories.")
    output_dir.mkdir(parents=True, exist_ok=True)
    removed_constituents = remove_split_models(output_dir)
    if removed_constituents:
        print(f"Removed stale split models: {removed_constituents}")

    for model_name in MODELS:
        optimize_one(
            source_dir / model_name,
            output_dir / model_name,
            no_shape_infer=NO_SHAPE_INFER,
            enable_fp16=ENABLE_FP16,
            rtol=FP16_RTOL if ENABLE_FP16 else RTOL,
            atol=FP16_ATOL if ENABLE_FP16 else ATOL,
        )
    metadata_path = publish_metadata_carrier(source_dir, output_dir)
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()