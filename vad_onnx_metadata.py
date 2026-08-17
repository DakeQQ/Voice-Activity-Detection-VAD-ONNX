import gc
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path

import onnx


REQUIRED_VAD_METADATA_KEYS = (
    "vad_metadata_version",
    "producer",
    "model_name",
    "task",
    "model_family",
    "dynamic_axes",
    "opset",
    "input_audio_dtype",
    "input_sample_rate",
    "model_sample_rate",
    "output_frame_shift_seconds",
)

REQUIRED_SMART_TURN_METADATA_KEYS = (
    "smart_turn_metadata_version",
    "producer",
    "model_name",
    "task",
    "model_family",
    "dynamic_batch",
    "opset",
    "input_audio_dtype",
    "input_sample_rate",
    "model_sample_rate",
    "input_audio_length",
    "context_window_seconds",
    "normalization_embedded",
    "n_fft",
    "hop_length_samples",
    "n_mels",
    "n_frames",
    "num_inputs",
    "num_outputs",
)

COMMUNITY1_METADATA_MODEL_NAME = "community1_Metadata.onnx"
COMMUNITY1_MANIFEST_METADATA_KEY = "pyannote.community1.manifest"
REQUIRED_COMMUNITY1_METADATA_KEYS = (
    "community1_metadata_version",
    "producer",
    "model_name",
    "model_id",
    "task",
    "model_family",
    "opset",
    "device_pipeline",
    "merged_pipeline",
    "onnx_stage_count",
    "host_stages",
    "input_audio_dtype",
    "logical_integer_dtype",
    COMMUNITY1_MANIFEST_METADATA_KEY,
)

CONFIGURABLE_METADATA_KEYS = frozenset({
    "complete_threshold",
    "decibel_thres",
    "detect_mode",
    "do_extend",
    "extend_speech_frame",
    "fusion_threshold_seconds",
    "look_ahead_seconds",
    "lookahead_time_end_point",
    "lookback_time_start_point",
    "max_dynamic_audio_seconds",
    "max_end_silence_time",
    "max_single_segment_time",
    "max_speech_duration_seconds",
    "max_speech_frame",
    "max_start_silence_time",
    "merge_silence_frame",
    "min_silence_duration_ms",
    "min_silence_frame",
    "min_speech_duration_ms",
    "min_speech_duration_seconds",
    "min_speech_frame",
    "negative_threshold",
    "noise_frame_num_used_for_snr",
    "normalize_audio_default",
    "normalize_target_rms",
    "pad_start_frame",
    "sil_to_speech_time_thres",
    "silence_score",
    "smooth_window_size",
    "snr_thres",
    "speaking_score",
    "speech_2_noise_ratio",
    "speech_noise_thres",
    "speech_pad_ms",
    "speech_to_sil_time_thres",
    "stream_chunk_ms",
    "stream_chunk_samples",
    "event_thresholds",
    "window_size_ms",
})


def metadata_path_for_model(model_path):
    path = Path(model_path)
    return path.with_name(path.stem + "_Metadata.onnx")


def community1_metadata_path(model_dir):
    return Path(model_dir) / COMMUNITY1_METADATA_MODEL_NAME


def build_model_metadata(*sections):
    metadata = {}
    for section in sections:
        if not section:
            continue
        for key, value in section.items():
            if value is None:
                continue
            if isinstance(value, bool):
                value = "1" if value else "0"
            elif isinstance(value, (list, tuple)):
                value = ",".join(str(item) for item in value)
            else:
                value = str(value)
            metadata[str(key)] = value
    return metadata


def static_model_metadata(metadata):
    metadata = build_model_metadata(metadata)
    excluded_keys = set(CONFIGURABLE_METADATA_KEYS)
    if metadata.get("dynamic_axes", "").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        excluded_keys.add("input_audio_length")
    return {
        key: value
        for key, value in metadata.items()
        if key not in excluded_keys
    }


def read_onnx_metadata(model_path):
    model = onnx.load(str(model_path), load_external_data=False)
    metadata = {prop.key: prop.value for prop in model.metadata_props}
    del model
    gc.collect()
    return metadata


def write_onnx_metadata(model_path, metadata):
    model = onnx.load(str(model_path), load_external_data=False)
    metadata = static_model_metadata(metadata)
    del model.metadata_props[:]
    for key, value in metadata.items():
        model.metadata_props.add(key=key, value=value)
    onnx.save(model, str(model_path))
    del model
    gc.collect()


def clear_onnx_metadata(model_path):
    model = onnx.load(str(model_path), load_external_data=False)
    if model.metadata_props:
        del model.metadata_props[:]
        onnx.save(model, str(model_path))
    del model
    gc.collect()


def export_metadata_carrier(metadata_path, metadata, opset):
    metadata = static_model_metadata(metadata)
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    static_value = onnx.helper.make_tensor(
        name="static_metadata_value",
        data_type=onnx.TensorProto.INT64,
        dims=(1,),
        vals=(0,),
    )
    graph = onnx.helper.make_graph(
        nodes=[
            onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=["static_metadata"],
                value=static_value,
            )
        ],
        name="vad_metadata",
        inputs=[],
        outputs=[
            onnx.helper.make_tensor_value_info(
                "static_metadata",
                onnx.TensorProto.INT64,
                (1,),
            )
        ],
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="vad_onnx_metadata",
        opset_imports=[onnx.helper.make_opsetid("", int(opset))],
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(metadata_path))
    del model
    gc.collect()
    write_onnx_metadata(metadata_path, metadata)


def export_community1_metadata_carrier(metadata_path, manifest, producer):
    metadata = {
        "community1_metadata_version": manifest["format_version"],
        "producer": producer,
        "model_name": "community1",
        "model_id": manifest["model_id"],
        "task": "speaker_diarization",
        "model_family": "pyannote_community1",
        "opset": manifest["opset"],
        "device_pipeline": manifest["device_pipeline"],
        "merged_pipeline": manifest["merged_pipeline"],
        "onnx_stage_count": manifest["onnx_stage_count"],
        "host_stages": manifest["host_stages"],
        "input_audio_dtype": manifest["input_dtype"],
        "logical_integer_dtype": manifest["logical_integer_dtype"],
        COMMUNITY1_MANIFEST_METADATA_KEY: json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ),
    }
    export_metadata_carrier(metadata_path, metadata, manifest["opset"])


def stamp_export_metadata(model_path, metadata, opset):
    metadata = static_model_metadata(metadata)
    metadata_path = metadata_path_for_model(model_path)
    export_metadata_carrier(metadata_path, metadata, opset)
    clear_onnx_metadata(model_path)
    return metadata_path


def read_source_metadata(main_model_path):
    main_model_path = Path(main_model_path)
    metadata_path = metadata_path_for_model(main_model_path)
    if metadata_path.exists():
        return read_onnx_metadata(metadata_path)
    return read_onnx_metadata(main_model_path) if main_model_path.exists() else {}


def copy_metadata_model(src_main_model_path, dst_main_model_path):
    src_metadata_path = metadata_path_for_model(src_main_model_path)
    if not src_metadata_path.exists():
        return False
    dst_metadata_path = metadata_path_for_model(dst_main_model_path)
    dst_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f'.{dst_metadata_path.stem}_',
        dir=dst_metadata_path.parent,
    ) as directory:
        staged_metadata_path = Path(directory) / dst_metadata_path.name
        shutil.copy2(src_metadata_path, staged_metadata_path)
        original_mode = stat.S_IMODE(staged_metadata_path.stat().st_mode)
        try:
            staged_metadata_path.chmod(original_mode | stat.S_IWUSR)
            write_onnx_metadata(
                staged_metadata_path,
                read_onnx_metadata(staged_metadata_path),
            )
        finally:
            staged_metadata_path.chmod(original_mode)
        os.replace(staged_metadata_path, dst_metadata_path)
    return True


def preserve_optimized_metadata(src_main_model_path, dst_main_model_path, metadata):
    clear_onnx_metadata(dst_main_model_path)
    if metadata:
        if not copy_metadata_model(src_main_model_path, dst_main_model_path):
            dst_model = onnx.load(str(dst_main_model_path), load_external_data=False)
            opset = next(
                item.version
                for item in dst_model.opset_import
                if item.domain in ("", "ai.onnx")
            )
            del dst_model
            gc.collect()
            export_metadata_carrier(
                metadata_path_for_model(dst_main_model_path),
                metadata,
                opset,
            )
    else:
        print(
            f"WARNING: No ONNX metadata found in {src_main_model_path} or "
            f"{metadata_path_for_model(src_main_model_path)}. "
            "Re-export the model to get metadata."
        )


def _parse_bool(value, key):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Metadata key {key} must be a boolean encoded as 1/0, got {value!r}.")


class MetadataReader:
    def __init__(self, metadata):
        self.metadata = dict(metadata or {})

    def string(self, key, required=False):
        value = self.metadata.get(key)
        if value is None or value == "":
            if required:
                raise KeyError(
                    f"Required metadata key {key} is missing. "
                    "Re-export with the matching Export_*.py script."
                )
            return None
        return value

    def required_int(self, key):
        return int(self.string(key, required=True))

    def required_float(self, key):
        return float(self.string(key, required=True))

    def required_bool(self, key):
        return _parse_bool(self.string(key, required=True), key)


def parse_community1_metadata(metadata):
    reader = MetadataReader(metadata)
    for key in REQUIRED_COMMUNITY1_METADATA_KEYS:
        reader.string(key, required=True)
    try:
        manifest = json.loads(
            reader.string(COMMUNITY1_MANIFEST_METADATA_KEY, required=True)
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid Community-1 manifest metadata: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("Community-1 manifest metadata must be a JSON object.")

    direct_values = {
        "format_version": reader.required_int("community1_metadata_version"),
        "model_id": reader.string("model_id", required=True),
        "opset": reader.required_int("opset"),
        "device_pipeline": reader.required_bool("device_pipeline"),
        "merged_pipeline": reader.required_bool("merged_pipeline"),
        "onnx_stage_count": reader.required_int("onnx_stage_count"),
        "input_dtype": reader.string("input_audio_dtype", required=True),
        "logical_integer_dtype": reader.string(
            "logical_integer_dtype", required=True
        ),
    }
    for key, expected in direct_values.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Community-1 manifest {key}={manifest.get(key)!r} does not "
                f"match carrier metadata {expected!r}."
            )
    host_stages = ",".join(str(item) for item in manifest.get("host_stages", ()))
    if host_stages != reader.string("host_stages", required=True):
        raise ValueError("Community-1 host stages do not match carrier metadata.")
    models = manifest.get("models")
    if not isinstance(models, dict) or len(models) != direct_values["onnx_stage_count"]:
        raise ValueError("Community-1 manifest has an invalid models section.")
    return manifest


def load_runtime_metadata(
    model_path,
    make_session,
    required_keys=REQUIRED_VAD_METADATA_KEYS,
):
    metadata_path = metadata_path_for_model(model_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Required metadata model is missing: {metadata_path}. "
            "Re-export with the matching Export_*.py script."
        )
    metadata_session = make_session(str(metadata_path))
    reader = MetadataReader(metadata_session.get_modelmeta().custom_metadata_map)
    for key in required_keys:
        reader.string(key, required=True)
    return reader


def validate_vad_metadata(reader, session, expected_inputs=1, expected_outputs=3):
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != expected_inputs or len(inputs[0].shape) != 3:
        raise ValueError(
            f"The VAD ONNX model must have {expected_inputs} input(s) with a rank-3 audio input first."
        )
    if len(outputs) != expected_outputs:
        raise ValueError(f"The VAD ONNX model must have {expected_outputs} output(s).")

    metadata_dtype = reader.string("input_audio_dtype", required=True).lower()
    expected_dtype = (
        metadata_dtype
        if metadata_dtype.startswith("tensor(")
        else f"tensor({metadata_dtype})"
    )
    audio_input = inputs[0]
    if audio_input.type.lower() != expected_dtype:
        raise ValueError(
            f"ONNX input dtype {audio_input.type} does not match metadata "
            f"input_audio_dtype={metadata_dtype}."
        )

    metadata_dynamic = reader.required_bool("dynamic_axes")
    input_length = audio_input.shape[-1]
    model_dynamic = not isinstance(input_length, int)
    if model_dynamic != metadata_dynamic:
        raise ValueError(
            f"ONNX dynamic input state {model_dynamic} does not match metadata "
            f"dynamic_axes={metadata_dynamic}."
        )
    if isinstance(input_length, int):
        metadata_length = reader.required_int("input_audio_length")
    if isinstance(input_length, int) and input_length != metadata_length:
        raise ValueError(
            f"ONNX input length {input_length} does not match metadata "
            f"input_audio_length={metadata_length}."
        )