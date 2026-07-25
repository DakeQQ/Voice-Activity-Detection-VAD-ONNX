import gc
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from onnx import compose
from onnxruntime.transformers.optimizer import optimize_model
from onnxslim import slim
import onnx
import numpy as np
import torch

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "vad_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from vad_onnx_metadata import stamp_export_metadata


script_dir = Path(__file__).resolve().parent
output_onnx_model = script_dir / "silero_vad_optimized.onnx"
source_onnx_model = script_dir / "silero_vad.onnx"

SILERO_VAD_REPOSITORY = "snakers4/silero-vad"
SILERO_VAD_MODEL_REPOSITORY_PATH = "src/silero_vad/data/silero_vad.onnx"
GITHUB_API_URL = "https://api.github.com"
GITHUB_API_HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "Silero-VAD-ONNX-Exporter/1.0",
    "X-GitHub-Api-Version": "2022-11-28",
}


IN_AUDIO_DTYPE = "F32"  # ["F16", "F32", "INT16"]; float inputs use normalized PCM in [-1, 1].
QUANT_TO_FP16 = False
UPGRADE_OPSET = 20

_INPUT_AUDIO_DTYPES = {
    "F16": (onnx.TensorProto.FLOAT16, torch.float16, "float16"),
    "F32": (onnx.TensorProto.FLOAT, torch.float32, "float"),
    "INT16": (onnx.TensorProto.INT16, torch.int16, "int16"),
}
IN_AUDIO_DTYPE = IN_AUDIO_DTYPE.upper()
if IN_AUDIO_DTYPE not in _INPUT_AUDIO_DTYPES:
    raise ValueError(
        f"Unsupported IN_AUDIO_DTYPE={IN_AUDIO_DTYPE!r}; "
        f"expected one of {tuple(_INPUT_AUDIO_DTYPES)}."
    )

# Model interface constants.
MODEL_SAMPLE_RATE = 16000           # Native model rate; do not change.
IN_SAMPLE_RATE = 16000              # External audio rate; accepts any positive sample rate.
MODEL_WINDOW_SIZE_SAMPLES = 512
MODEL_CONTEXT_SIZE_SAMPLES = 64
if IN_SAMPLE_RATE <= 0:
    raise ValueError("IN_SAMPLE_RATE must be positive.")
WINDOW_SIZE_SAMPLES = round(
    MODEL_WINDOW_SIZE_SAMPLES * IN_SAMPLE_RATE / MODEL_SAMPLE_RATE)
CONTEXT_SIZE_SAMPLES = round(
    MODEL_CONTEXT_SIZE_SAMPLES * IN_SAMPLE_RATE / MODEL_SAMPLE_RATE)
if WINDOW_SIZE_SAMPLES < 1 or CONTEXT_SIZE_SAMPLES < 1:
    raise ValueError("IN_SAMPLE_RATE is too low for the Silero input window.")


def git_blob_sha1_file(path):
    path = Path(path)
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with open(path, "rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_github_json(url):
    request = urllib.request.Request(url, headers=GITHUB_API_HEADERS)
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Failed to fetch GitHub metadata from {url}") from last_error


def resolve_latest_source_model():
    release_url = f"{GITHUB_API_URL}/repos/{SILERO_VAD_REPOSITORY}/releases/latest"
    release = fetch_github_json(release_url)
    tag = release.get("tag_name") if isinstance(release, dict) else None
    if not isinstance(tag, str) or not tag:
        raise RuntimeError("The latest Silero VAD release has no valid tag_name.")

    query = urllib.parse.urlencode({"ref": tag})
    model_metadata_url = (
        f"{GITHUB_API_URL}/repos/{SILERO_VAD_REPOSITORY}/contents/"
        f"{SILERO_VAD_MODEL_REPOSITORY_PATH}?{query}"
    )
    model_metadata = fetch_github_json(model_metadata_url)
    if not isinstance(model_metadata, dict) or model_metadata.get("type") != "file":
        raise RuntimeError(f"Silero VAD release {tag} has no ONNX model file.")

    download_url = model_metadata.get("download_url")
    blob_sha1 = model_metadata.get("sha")
    size = model_metadata.get("size")
    if (
        not isinstance(download_url, str)
        or not isinstance(blob_sha1, str)
        or len(blob_sha1) != 40
        or not isinstance(size, int)
        or size <= 0
    ):
        raise RuntimeError(f"Silero VAD release {tag} has invalid model metadata.")

    return {
        "version": tag.removeprefix("v"),
        "tag": tag,
        "download_url": download_url,
        "git_blob_sha1": blob_sha1,
        "size": size,
    }


def ensure_source_model(model_path):
    model_path = Path(model_path)
    source = resolve_latest_source_model()
    expected_hash = source["git_blob_sha1"]
    if model_path.exists() and git_blob_sha1_file(model_path) == expected_hash:
        print(f"Using Silero VAD {source['version']} from {model_path}")
        return model_path, source

    action = "Updating" if model_path.exists() else "Downloading"
    print(f"{action} Silero VAD to {source['version']} from {source['download_url']}")
    temporary_path = model_path.with_suffix(model_path.suffix + ".download")
    try:
        request = urllib.request.Request(
            source["download_url"],
            headers=GITHUB_API_HEADERS,
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    with open(temporary_path, "wb") as model_file:
                        while chunk := response.read(1024 * 1024):
                            model_file.write(chunk)
                break
            except (OSError, urllib.error.URLError):
                temporary_path.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        actual_size = temporary_path.stat().st_size
        actual_hash = git_blob_sha1_file(temporary_path)
        if actual_size != source["size"] or actual_hash != expected_hash:
            raise RuntimeError(
                "Downloaded Silero source model checksum mismatch: "
                f"expected {source['size']} bytes / {expected_hash}, "
                f"got {actual_size} bytes / {actual_hash}."
            )
        temporary_path.replace(model_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return model_path, source


def adapt_audio_input_dtype(model_path, input_dtype_name):
    model = onnx.load(str(model_path), load_external_data=False)
    audio_input = next(value for value in model.graph.input if value.name == "input")
    original_dtype = audio_input.type.tensor_type.elem_type
    target_dtype = _INPUT_AUDIO_DTYPES[input_dtype_name][0]
    if target_dtype != original_dtype:
        internal_input = "input_model_dtype"
        cast_output = internal_input
        adapter_nodes = []

        def redirect_consumers(graph):
            for node in graph.node:
                for index, name in enumerate(node.input):
                    if name == audio_input.name:
                        node.input[index] = internal_input
                for attribute in node.attribute:
                    if attribute.type == onnx.AttributeProto.GRAPH:
                        redirect_consumers(attribute.g)
                    elif attribute.type == onnx.AttributeProto.GRAPHS:
                        for subgraph in attribute.graphs:
                            redirect_consumers(subgraph)

        redirect_consumers(model.graph)

        if target_dtype == onnx.TensorProto.INT16:
            cast_output = "input_pcm_cast"
        adapter_nodes.append(onnx.helper.make_node(
            "Cast",
            [audio_input.name],
            [cast_output],
            to=original_dtype,
            name="CastAudioInput",
        ))
        if target_dtype == onnx.TensorProto.INT16:
            scale_dtype = (
                np.float16
                if original_dtype == onnx.TensorProto.FLOAT16
                else np.float32
            )
            scale_name = "input_pcm_scale"
            model.graph.initializer.append(onnx.numpy_helper.from_array(
                np.asarray(1.0 / 32768.0, dtype=scale_dtype),
                name=scale_name,
            ))
            adapter_nodes.append(onnx.helper.make_node(
                "Mul",
                [cast_output, scale_name],
                [internal_input],
                name="ScalePCMInput",
            ))
        audio_input.type.tensor_type.elem_type = target_dtype
        for node in reversed(adapter_nodes):
            model.graph.node.insert(0, node)

    onnx.checker.check_model(model)
    onnx.save(model, str(model_path), save_as_external_data=False)
    del model
    gc.collect()


class SileroInputAdapter(torch.nn.Module):
    def __init__(self, model_audio_dtype):
        super().__init__()
        self.model_audio_dtype = model_audio_dtype
        self.scale_pcm = IN_AUDIO_DTYPE == "INT16"

    def forward(self, audio, state, sample_rate):
        audio = audio.float()
        if self.scale_pcm:
            audio = audio * (1.0 / 32768.0)
        audio = torch.nn.functional.interpolate(
            audio.unsqueeze(1),
            size=MODEL_WINDOW_SIZE_SAMPLES + MODEL_CONTEXT_SIZE_SAMPLES,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        audio = audio.to(self.model_audio_dtype)
        model_sample_rate = sample_rate - sample_rate + MODEL_SAMPLE_RATE
        return audio, state, model_sample_rate


def prepend_input_resampler(model_path):
    model_path = Path(model_path)
    silero_model = onnx.load(str(model_path), load_external_data=False)
    inputs = {value.name: value for value in silero_model.graph.input}
    model_audio_type = inputs["input"].type.tensor_type.elem_type
    model_state_type = inputs["state"].type.tensor_type.elem_type
    torch_dtypes = {
        onnx.TensorProto.FLOAT: torch.float32,
        onnx.TensorProto.FLOAT16: torch.float16,
    }
    if model_audio_type not in torch_dtypes or model_state_type not in torch_dtypes:
        raise TypeError("Silero model audio and state must use float16 or float32.")

    input_torch_dtype = _INPUT_AUDIO_DTYPES[IN_AUDIO_DTYPE][1]
    external_input_length = WINDOW_SIZE_SAMPLES + CONTEXT_SIZE_SAMPLES
    adapter = SileroInputAdapter(torch_dtypes[model_audio_type]).eval()
    model_opset = next(
        item.version
        for item in silero_model.opset_import
        if item.domain in ("", "ai.onnx")
    )
    with tempfile.TemporaryDirectory(prefix="silero_resampler_") as temporary_dir:
        adapter_path = Path(temporary_dir) / "adapter.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                adapter,
                (
                    torch.zeros((1, external_input_length), dtype=input_torch_dtype),
                    torch.zeros((2, 1, 128), dtype=torch_dtypes[model_state_type]),
                    torch.tensor(IN_SAMPLE_RATE, dtype=torch.int64),
                ),
                str(adapter_path),
                input_names=["input", "state", "sr"],
                output_names=["model_input", "model_state", "model_sr"],
                dynamic_axes={
                    "input": {0: "batch", 1: "input_samples"},
                    "state": {1: "batch"},
                    "model_input": {0: "batch"},
                    "model_state": {1: "batch"},
                },
                opset_version=model_opset,
                dynamo=False,
            )
        adapter_model = onnx.load(str(adapter_path), load_external_data=False)

    if [value.name for value in adapter_model.graph.input] != ["input", "state", "sr"]:
        raise RuntimeError("Silero rate adapter did not preserve its three-input contract.")
    adapter_model.ir_version = silero_model.ir_version
    merged = compose.merge_models(
        adapter_model,
        silero_model,
        io_map=[
            ("model_input", "input"),
            ("model_state", "state"),
            ("model_sr", "sr"),
        ],
        prefix1="rate_adapter/",
        producer_name="Silero VAD with input resampler",
    )
    for external_name in ("input", "state", "sr"):
        prefixed_name = f"rate_adapter/{external_name}"
        for value in merged.graph.input:
            if value.name == prefixed_name:
                value.name = external_name
        for node in merged.graph.node:
            for index, name in enumerate(node.input):
                if name == prefixed_name:
                    node.input[index] = external_name

    onnx.checker.check_model(merged)
    onnx.save(merged, str(model_path), save_as_external_data=False)
    del silero_model
    del adapter_model
    del merged
    gc.collect()


model_path, source_model = ensure_source_model(source_onnx_model)


if UPGRADE_OPSET > 0:
    model = onnx.load(model_path)
    model = onnx.version_converter.convert_version(model, UPGRADE_OPSET)
    onnx.save(model, output_onnx_model, save_as_external_data=False)
    model_path = output_onnx_model
    del model
    gc.collect()


# onnxslim 1st
slim(
    model=str(model_path),
    output_model=str(output_onnx_model),
    no_shape_infer=False,   # False for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=False,
    verbose=False,
    dtype='fp16' if QUANT_TO_FP16 else 'fp32'
)


# transformers.optimizer
model = optimize_model(str(output_onnx_model),
                       use_gpu=False,
                       opt_level=2,
                       num_heads=0,
                       hidden_size=0,
                       verbose=False,
                       model_type='bert')
model.save_model_to_file(output_onnx_model, use_external_data_format=False)
del model
gc.collect()


# onnxslim 2nd
slim(
    model=str(output_onnx_model),
    output_model=str(output_onnx_model),
    no_shape_infer=False,   # False for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=False,
    verbose=False,
    dtype='fp16' if QUANT_TO_FP16 else 'fp32'
)

if IN_SAMPLE_RATE == MODEL_SAMPLE_RATE:
    adapt_audio_input_dtype(output_onnx_model, IN_AUDIO_DTYPE)
else:
    prepend_input_resampler(output_onnx_model)

model = onnx.load(output_onnx_model, load_external_data=False)
audio_input = next(value for value in model.graph.input if value.name == "input")
audio_dtype = {
    onnx.TensorProto.FLOAT: "float",
    onnx.TensorProto.FLOAT16: "float16",
    onnx.TensorProto.INT16: "int16",
}[audio_input.type.tensor_type.elem_type]
model_opset = next(
    item.version for item in model.opset_import if item.domain in ("", "ai.onnx")
)
del model
gc.collect()

frame_seconds = WINDOW_SIZE_SAMPLES / IN_SAMPLE_RATE
model_metadata = {
    "vad_metadata_version": 1,
    "producer": Path(__file__).name,
    "model_name": "Silero_VAD",
    "task": "voice_activity_detection",
    "model_family": "silero_vad",
    "source_model_version": source_model["version"],
    "source_model_tag": source_model["tag"],
    "source_model_url": source_model["download_url"],
    "source_model_git_blob_sha1": source_model["git_blob_sha1"],
    "dynamic_axes": True,
    "opset": model_opset,
    "input_audio_dtype": audio_dtype,
    "input_sample_rate": IN_SAMPLE_RATE,
    "model_sample_rate": MODEL_SAMPLE_RATE,
    "input_audio_length": WINDOW_SIZE_SAMPLES,
    "output_frame_shift_seconds": frame_seconds,
    "window_size_samples": WINDOW_SIZE_SAMPLES,
    "context_size_samples": CONTEXT_SIZE_SAMPLES,
    "state_layers": 2,
    "state_size": 128,
    "supported_sample_rates": (IN_SAMPLE_RATE,),
    "num_inputs": 3,
    "num_outputs": 2,
}
metadata_path = stamp_export_metadata(
    output_onnx_model,
    model_metadata,
    model_opset,
)
print(f"Metadata saved to: {metadata_path}")

inference_script = Path(__file__).resolve().with_name("Inference_Silero_VAD_ONNX.py")
print(f"\nStarting inference demo with {inference_script.name}")
subprocess.run([sys.executable, str(inference_script)], cwd=script_dir, check=True)
