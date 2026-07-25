"""Simplify and validate the NVIDIA MarbleNet deployment graph."""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "vad_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from Optimize_ONNX_Common import (
    atomic_model_path,
    convert_graph_to_fp16,
    file_sha256,
    simplify_onnx,
)
from vad_onnx_metadata import preserve_optimized_metadata, read_source_metadata


DEFAULT_FOLDER = Path(__file__).resolve().parent / "NVIDIA_MarbleNet_ONNX"
OPTIMIZED_FOLDER = Path(__file__).resolve().parent / "NVIDIA_MarbleNet_Optimized"
SOURCE_ONNX_MODEL = DEFAULT_FOLDER / "NVIDIA_MarbleNet.onnx"
FINAL_ONNX_MODEL = OPTIMIZED_FOLDER / "NVIDIA_MarbleNet.onnx"
BASELINE_ONNX_MODEL = DEFAULT_FOLDER / "NVIDIA_MarbleNet_source_baseline_raw.onnx"

ENABLE_FP16 = False  # Convert internal float compute to FP16; preserve public I/O types.

FP32_RTOL = 1e-5
FP32_ATOL = 1e-6
FP16_RTOL = 1e-2
FP16_ATOL = 1e-2
FP16_OP_BLOCK_LIST = ["Softmax"]

_ORT_AUDIO_DTYPES = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(int16)": np.int16,
}


def inspect_model(model_path):
    model_path = Path(model_path)
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    histogram = Counter(node.op_type for node in model.graph.node)
    return {
        "path": str(model_path),
        "sha256": file_sha256(model_path),
        "bytes": model_path.stat().st_size,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "value_info": len(inferred.graph.value_info),
        "domains": sorted({node.domain or "ai.onnx" for node in model.graph.node}),
        "operators": dict(sorted(histogram.items())),
        "inputs": [value.name for value in model.graph.input],
        "outputs": [value.name for value in model.graph.output],
        "opsets": [(item.domain or "ai.onnx", item.version) for item in model.opset_import],
    }


def validate_runtime(source_path, final_path, rtol, atol, precision):
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CPUExecutionProvider is unavailable")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sessions = [
        ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        for path in (source_path, final_path)
    ]
    source_inputs = [
        (item.name, item.type, item.shape) for item in sessions[0].get_inputs()
    ]
    final_inputs = [
        (item.name, item.type, item.shape) for item in sessions[1].get_inputs()
    ]
    source_outputs = [
        (item.name, item.type, item.shape) for item in sessions[0].get_outputs()
    ]
    final_outputs = [
        (item.name, item.type, item.shape) for item in sessions[1].get_outputs()
    ]
    if source_inputs != final_inputs or source_outputs != final_outputs:
        raise RuntimeError(f"{precision} optimization changed the ONNX Runtime interface")

    audio_metadata = sessions[0].get_inputs()[0]
    audio_dtype = _ORT_AUDIO_DTYPES.get(audio_metadata.type)
    if audio_dtype is None:
        raise TypeError(f"Unsupported MarbleNet audio input type: {audio_metadata.type}")
    generator = np.random.default_rng(20260721)
    aggregate = {"max_abs": 0.0, "mean_abs": 0.0, "max_rel": 0.0}
    mean_errors = []
    configured_length = audio_metadata.shape[-1]
    audio_lengths = (
        (configured_length,)
        if isinstance(configured_length, int)
        else (16000, 32000, 32001)
    )
    for audio_length in audio_lengths:
        shape = (1, 1, audio_length)
        if audio_dtype == np.int16:
            audio = generator.integers(
                -32768, 32768, size=shape, dtype=np.int16
            )
        else:
            audio = generator.uniform(-1.0, 1.0, size=shape).astype(audio_dtype)
        source_values = sessions[0].run(None, {audio_metadata.name: audio})
        final_values = sessions[1].run(None, {audio_metadata.name: audio})
        for metadata, source_value, final_value in zip(
            sessions[0].get_outputs(), source_values, final_values
        ):
            if source_value.shape != final_value.shape or source_value.dtype != final_value.dtype:
                raise RuntimeError(f"Runtime contract changed for {metadata.name!r}")
            if np.issubdtype(source_value.dtype, np.integer):
                matches = np.array_equal(source_value, final_value)
                absolute = np.abs(source_value.astype(np.int64) - final_value.astype(np.int64))
                relative = absolute.astype(np.float64)
            else:
                matches = np.allclose(
                    source_value,
                    final_value,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                )
                absolute = np.abs(source_value - final_value)
                relative = absolute / np.maximum(np.abs(source_value), 1e-7)
            if not matches:
                raise RuntimeError(
                    f"{precision} output {metadata.name!r} exceeded tolerance at length "
                    f"{audio_length}: max_abs={float(absolute.max()):.6g}, "
                    f"max_rel={float(relative.max()):.6g}"
                )
            aggregate["max_abs"] = max(aggregate["max_abs"], float(absolute.max()))
            aggregate["max_rel"] = max(aggregate["max_rel"], float(relative.max()))
            mean_errors.append(float(absolute.mean()))
    aggregate["mean_abs"] = float(np.mean(mean_errors))
    validated_cases = len(audio_lengths)
    return validated_cases, aggregate


def optimize_model(source_path, final_path, use_fp16=ENABLE_FP16):
    source_path = Path(source_path).resolve()
    final_path = Path(final_path).resolve()
    source_metadata = read_source_metadata(source_path)
    source_digest = file_sha256(source_path)
    converted_initializers = 0
    precision = "FP16" if use_fp16 else "FP32"
    rtol = FP16_RTOL if use_fp16 else FP32_RTOL
    atol = FP16_ATOL if use_fp16 else FP32_ATOL

    with atomic_model_path(
        final_path,
        prefix=f"marblenet_{precision.lower()}_",
    ) as staged_path:
        simplify_onnx(source_path, staged_path, no_shape_infer=True)
        if use_fp16:
            model = onnx.load(staged_path, load_external_data=True)
            model, converted_initializers = convert_graph_to_fp16(
                model,
                force_fp16_initializers=True,
                op_block_list=FP16_OP_BLOCK_LIST,
            )
            onnx.save(model, staged_path, save_as_external_data=False)
        validated_cases, metrics = validate_runtime(
            source_path,
            staged_path,
            rtol=rtol,
            atol=atol,
            precision=precision,
        )
        if file_sha256(source_path) != source_digest:
            raise RuntimeError(f"Source MarbleNet model changed during {precision} optimization")
    preserve_optimized_metadata(source_path, final_path, source_metadata)
    return converted_initializers, validated_cases, metrics


def main():
    source_report = inspect_model(SOURCE_ONNX_MODEL)
    if source_report["inputs"] != ["audio"]:
        raise RuntimeError(f"Unexpected source inputs: {source_report['inputs']}")
    if source_report["outputs"] != ["score_silence", "score_active", "signal_len"]:
        raise RuntimeError(f"Unexpected source outputs: {source_report['outputs']}")
    if source_report["domains"] != ["ai.onnx"]:
        raise RuntimeError(f"Unexpected source operator domains: {source_report['domains']}")

    forbidden = {"ATen", "BatchNormalization", "Identity", "MatMul", "Pad", "Reshape"}
    present = sorted(forbidden.intersection(source_report["operators"]))
    if present:
        raise RuntimeError(f"Avoidable or fallback operators remain: {present}")

    converted_initializers, validated_cases, metrics = optimize_model(
        SOURCE_ONNX_MODEL,
        FINAL_ONNX_MODEL,
    )
    final_report = inspect_model(FINAL_ONNX_MODEL)
    print(json.dumps({"source": source_report, "final": final_report}, indent=2))

    if ENABLE_FP16:
        print(
            f"FP16_PASS: converted {converted_initializers} initializers and "
            f"validated {validated_cases} cases; max_abs={metrics['max_abs']:.6g}, "
            f"mean_abs={metrics['mean_abs']:.6g}, max_rel={metrics['max_rel']:.6g}."
        )
    else:
        print(
            f"ONNXSLIM_PASS: validated {validated_cases} cases; "
            f"max_abs={metrics['max_abs']:.6g}, "
            f"mean_abs={metrics['mean_abs']:.6g}, "
            f"max_rel={metrics['max_rel']:.6g}."
        )

    if BASELINE_ONNX_MODEL.exists():
        baseline_report = inspect_model(BASELINE_ONNX_MODEL)
        print(json.dumps({"preserved_pre_change_baseline": baseline_report}, indent=2))

    if ENABLE_FP16:
        print(f"FP16 model written: {FINAL_ONNX_MODEL}")
    else:
        print(f"Simplified FP32 model written: {FINAL_ONNX_MODEL}")


if __name__ == "__main__":
    main()
