"""Apply three targeted rewrites and onnxslim to the FSMN-VAD graph.

The pass replaces decomposed LFR edge padding with standard Pad, removes four
cache-output Reshapes that exist only to repair exporter shape metadata, and
fuses five Conv-Relu pairs with ONNX Runtime FusedConv. Onnxslim then removes
the remaining Constant nodes from the staged deployment graph. The raw model
remains immutable. FusedConv requires ONNX Runtime 1.27.0 with
CPUExecutionProvider and does not require an external custom-operator library.
An opt-in mixed-FP16 mode keeps the raw-PCM frontend in FP32 and converts the
encoder while preserving the public input, output, and cache types.
"""

import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference
from onnxslim import slim

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "vad_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from Optimize_ONNX_Common import (
    convert_graph_to_fp16,
    file_sha256,
    graph_interface,
    graph_maps,
    operator_histogram,
    tensor_shape,
)
from vad_onnx_metadata import preserve_optimized_metadata, read_source_metadata


MICROSOFT_DOMAIN = "com.microsoft"
MICROSOFT_OPSET = 1
MINIMUM_ORT_RELEASE = (1, 27, 0)
MODEL_FOLDER = Path(__file__).resolve().parent / "FSMN_VAD_ONNX"
RAW_ONNX_MODEL = MODEL_FOLDER / "FSMN_source_optimized_raw.onnx"
FINAL_ONNX_MODEL = MODEL_FOLDER / "FSMN.onnx"
OVERWRITE = True

ENABLE_FP16 = False  # May affect VAD accuracy; the INT16 spectral frontend stays FP32.

FP16_RTOL = 1e-2
FP16_ATOL = 1e-2
FP16_OP_BLOCK_LIST = ["Softmax"]
FP16_NODE_PREFIXES = ("/frontend/",)
INT64_MAX = np.iinfo(np.int64).max
LFR_PADS_NAME = "fsmn_lfr_edge_pads"
LFR_PAD_NODE_NAME = "/encoder/LFRPad"
FRONTEND_CHANNEL_AXIS_NAME = "fsmn_frontend_channel_axis"
CACHE_SLICE_CONTROLS = {
    "fsmn_cache_slice_starts": [-19],
    "fsmn_cache_slice_ends": [INT64_MAX],
    "fsmn_cache_slice_axes": [2],
    "fsmn_cache_slice_steps": [1],
}
EXPECTED_INTERFACE = {
    "inputs": (
        ("audio", TensorProto.INT16, (1, None)),
        ("in_cache_0", TensorProto.FLOAT, (1, 128, 19, 1)),
        ("in_cache_1", TensorProto.FLOAT, (1, 128, 19, 1)),
        ("in_cache_2", TensorProto.FLOAT, (1, 128, 19, 1)),
        ("in_cache_3", TensorProto.FLOAT, (1, 128, 19, 1)),
    ),
    "outputs": (
        ("scores", TensorProto.FLOAT, (1, None, 248)),
        ("decibels", TensorProto.FLOAT, (1, None)),
        ("out_cache_0", TensorProto.FLOAT, (1, 128, 19, 1)),
        ("out_cache_1", TensorProto.FLOAT, (1, 128, 19, 1)),
        ("out_cache_2", TensorProto.FLOAT, (1, 128, 19, 1)),
        ("out_cache_3", TensorProto.FLOAT, (1, 128, 19, 1)),
    ),
}
EXPECTED_WEIGHTS = {
    "encoder.in_linear2.linear.weight": (250, 140, 1, 1),
    "encoder.fsmn.0.affine.linear.weight": (250, 128, 1, 1),
    "encoder.fsmn.1.affine.linear.weight": (250, 128, 1, 1),
    "encoder.fsmn.2.affine.linear.weight": (250, 128, 1, 1),
    "encoder.fsmn.3.affine.linear.weight": (250, 128, 1, 1),
}
EXPECTED_CONV_ATTRIBUTES = {
    "dilations": [1, 1],
    "group": 1,
    "kernel_shape": [1, 1],
    "pads": [0, 0, 0, 0],
    "strides": [1, 1],
}
EXPECTED_RAW_OPERATORS = {
    ("ai.onnx", "Add"): 1,
    ("ai.onnx", "Cast"): 1,
    ("ai.onnx", "Clip"): 1,
    ("ai.onnx", "Concat"): 5,
    ("ai.onnx", "Constant"): 35,
    ("ai.onnx", "Conv"): 18,
    ("ai.onnx", "Log"): 2,
    ("ai.onnx", "Mul"): 3,
    ("ai.onnx", "Relu"): 5,
    ("ai.onnx", "Reshape"): 4,
    ("ai.onnx", "Slice"): 6,
    ("ai.onnx", "Softmax"): 1,
    ("ai.onnx", "Split"): 1,
    ("ai.onnx", "Squeeze"): 2,
    ("ai.onnx", "Transpose"): 1,
    ("ai.onnx", "Unsqueeze"): 2,
}
EXPECTED_REWRITTEN_OPERATORS = {
    ("ai.onnx", "Add"): 1,
    ("ai.onnx", "Cast"): 1,
    ("ai.onnx", "Clip"): 1,
    ("ai.onnx", "Concat"): 4,
    ("ai.onnx", "Constant"): 5,
    ("ai.onnx", "Conv"): 13,
    ("ai.onnx", "Log"): 2,
    ("ai.onnx", "Mul"): 3,
    ("ai.onnx", "Pad"): 1,
    ("ai.onnx", "Slice"): 4,
    ("ai.onnx", "Softmax"): 1,
    ("ai.onnx", "Split"): 1,
    ("ai.onnx", "Squeeze"): 2,
    ("ai.onnx", "Transpose"): 1,
    ("ai.onnx", "Unsqueeze"): 2,
    (MICROSOFT_DOMAIN, "FusedConv"): 5,
}
EXPECTED_FINAL_OPERATORS = {
    operator: count
    for operator, count in EXPECTED_REWRITTEN_OPERATORS.items()
    if operator != ("ai.onnx", "Constant")
}


def validate_graph_interface(model):
    interface = graph_interface(model)
    for section in ("inputs", "outputs"):
        actual_values = interface[section]
        expected_values = EXPECTED_INTERFACE[section]
        if len(actual_values) != len(expected_values):
            raise ValueError(
                f"Expected {len(expected_values)} graph {section}, got {len(actual_values)}"
            )
        for actual, expected in zip(actual_values, expected_values):
            actual_name, actual_type, actual_shape = actual
            expected_name, expected_type, expected_shape = expected
            if actual_name != expected_name or actual_type != expected_type:
                raise ValueError(
                    f"Unexpected graph {section[:-1]} {actual_name!r}: "
                    f"expected {expected_name!r} with type {expected_type}"
                )
            if len(actual_shape) != len(expected_shape) or any(
                expected_dimension is not None and actual_dimension != expected_dimension
                for actual_dimension, expected_dimension in zip(actual_shape, expected_shape)
            ):
                raise ValueError(
                    f"Unexpected shape for graph {section[:-1]} {actual_name!r}: "
                    f"{actual_shape}; expected {expected_shape}"
                )

    audio_length = interface["inputs"][0][2][1]
    score_frames = interface["outputs"][0][2][1]
    decibel_frames = interface["outputs"][1][2][1]
    if isinstance(audio_length, str):
        if (audio_length, score_frames, decibel_frames) != (
            "samples",
            "frames",
            "frames",
        ):
            raise ValueError(
                "Dynamic graph dimensions must be audio='samples' and outputs='frames'"
            )
    elif isinstance(audio_length, int):
        if audio_length < 400:
            raise ValueError(f"Static audio length must be at least 400, got {audio_length}")
        expected_frames = (audio_length - 400) // 160 + 1
        if score_frames != expected_frames or decibel_frames != expected_frames:
            raise ValueError(
                f"Static graph with {audio_length} audio samples must produce "
                f"{expected_frames} frames, got scores={score_frames}, "
                f"decibels={decibel_frames}"
            )
    else:
        raise ValueError(
            f"Audio time dimension must be static or named 'samples', got {audio_length!r}"
        )
    return interface


def attributes(node):
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def require_value(value_info, name, shape):
    value = value_info.get(name)
    if value is None:
        raise ValueError(f"Shape inference did not produce metadata for {name!r}")
    tensor = value.type.tensor_type
    if tensor.elem_type != TensorProto.FLOAT or len(tensor.shape.dim) != len(shape):
        raise ValueError(f"Expected float32 rank-{len(shape)} tensor for {name!r}")
    actual_shape = tensor_shape(value)
    for actual, expected in zip(actual_shape, shape):
        if expected is not None and actual != expected:
            raise ValueError(f"Unexpected shape for {name!r}: {actual_shape}")


def constant_array(node):
    if node.domain not in ("", "ai.onnx") or node.op_type != "Constant" or node.input:
        raise ValueError(f"Expected a standard Constant node, got {node.name!r}")
    node_attributes = attributes(node)
    if set(node_attributes) != {"value"}:
        raise ValueError(f"Unexpected Constant attributes at {node.name!r}")
    value = node_attributes["value"]
    if not isinstance(value, onnx.TensorProto):
        raise ValueError(f"Constant {node.name!r} does not contain a tensor")
    return numpy_helper.to_array(value)


def require_constant(producers, consumers, tensor_name, expected):
    node = producers.get(tensor_name)
    if node is None:
        raise ValueError(f"Missing Constant producer for {tensor_name!r}")
    actual = constant_array(node)
    expected = np.asarray(expected, dtype=np.int64)
    if actual.dtype != np.int64 or not np.array_equal(actual, expected):
        raise ValueError(
            f"Unexpected Constant {node.name!r}: {actual.dtype} {actual.tolist()}; "
            f"expected int64 {expected.tolist()}"
        )
    if len(consumers.get(tensor_name, [])) != 1:
        raise ValueError(f"Constant output {tensor_name!r} is not exclusively consumed")
    return node


def require_slice_controls(node, producers, consumers, starts, ends, axes, steps):
    if node.domain not in ("", "ai.onnx") or node.op_type != "Slice":
        raise ValueError(f"Expected a standard Slice node, got {node.name!r}")
    if len(node.input) != 5 or len(node.output) != 1 or attributes(node):
        raise ValueError(f"Unexpected Slice signature at {node.name!r}")
    expected_values = (starts, ends, axes, steps)
    return tuple(
        require_constant(producers, consumers, tensor_name, expected)
        for tensor_name, expected in zip(node.input[1:], expected_values)
    )


def validate_raw_graph(model):
    validate_graph_interface(model)
    if any(node.domain == MICROSOFT_DOMAIN and node.op_type == "FusedConv" for node in model.graph.node):
        raise ValueError("Input model is already rewritten; provide the immutable raw export")
    if any(node.domain not in ("", "ai.onnx") for node in model.graph.node):
        raise ValueError("Raw graph contains an unexpected custom-domain node")

    standard_opsets = [item.version for item in model.opset_import if item.domain in ("", "ai.onnx")]
    if standard_opsets != [20]:
        raise ValueError(f"Expected exactly ai.onnx opset 20, got {standard_opsets}")
    if any(item.domain == MICROSOFT_DOMAIN for item in model.opset_import):
        raise ValueError(f"Raw graph already imports the {MICROSOFT_DOMAIN!r} domain")
    if operator_histogram(model) != EXPECTED_RAW_OPERATORS:
        raise ValueError(
            f"Raw operator histogram changed: {operator_histogram(model)}; "
            f"expected {EXPECTED_RAW_OPERATORS}"
        )
    if len(model.graph.initializer) != 26:
        raise ValueError(f"Expected 26 raw initializers, got {len(model.graph.initializer)}")


def find_fusion_pairs(model, value_info, initializers, consumers):
    graph_outputs = {output.name for output in model.graph.output}

    pairs = []
    matched_weights = set()
    for conv in model.graph.node:
        if conv.domain not in ("", "ai.onnx") or conv.op_type != "Conv":
            continue
        if len(conv.input) != 3 or len(conv.output) != 1:
            continue
        weight_name, bias_name = conv.input[1:]
        if weight_name not in EXPECTED_WEIGHTS:
            continue

        users = consumers.get(conv.output[0], [])
        if len(users) != 1:
            raise ValueError(f"Expected one consumer for candidate Conv output {conv.output[0]!r}")
        relu = users[0]
        if relu.domain not in ("", "ai.onnx") or relu.op_type != "Relu":
            raise ValueError(f"Expected Relu after candidate Conv {conv.name!r}")
        if len(relu.input) != 1 or len(relu.output) != 1 or relu.input[0] != conv.output[0]:
            raise ValueError(f"Malformed Conv-Relu topology at {conv.name!r}")
        if conv.output[0] in graph_outputs:
            raise ValueError(f"Candidate intermediate {conv.output[0]!r} is a graph output")
        if attributes(conv) != EXPECTED_CONV_ATTRIBUTES:
            raise ValueError(f"Unexpected Conv attributes at {conv.name!r}: {attributes(conv)}")

        weight = initializers.get(weight_name)
        bias = initializers.get(bias_name)
        if weight is None or bias is None:
            raise ValueError(f"Candidate Conv {conv.name!r} does not use immutable weight and bias")
        weight_array = numpy_helper.to_array(weight)
        bias_array = numpy_helper.to_array(bias)
        expected_weight_shape = EXPECTED_WEIGHTS[weight_name]
        if weight_array.dtype != np.float32 or weight_array.shape != expected_weight_shape:
            raise ValueError(
                f"Unexpected {weight_name!r}: {weight_array.dtype} {weight_array.shape}; "
                f"expected float32 {expected_weight_shape}"
            )
        if bias_array.dtype != np.float32 or bias_array.shape != (expected_weight_shape[0],):
            raise ValueError(f"Unexpected bias initializer {bias_name!r}")
        if not np.isfinite(weight_array).all() or not np.isfinite(bias_array).all():
            raise ValueError(f"Non-finite initializer in candidate Conv {conv.name!r}")

        require_value(value_info, conv.input[0], (1, None, None, 1))
        require_value(value_info, conv.output[0], (1, 250, None, 1))
        require_value(value_info, relu.output[0], (1, 250, None, 1))
        pairs.append((conv, relu))
        matched_weights.add(weight_name)

    if len(pairs) != len(EXPECTED_WEIGHTS) or matched_weights != set(EXPECTED_WEIGHTS):
        raise ValueError(
            f"Expected exactly {len(EXPECTED_WEIGHTS)} candidate pairs for "
            f"{sorted(EXPECTED_WEIGHTS)}, found {len(pairs)} for {sorted(matched_weights)}"
        )
    return pairs


def find_lfr_padding(model, value_info, initializers, producers, consumers):
    first_convs = [
        node
        for node in model.graph.node
        if node.domain in ("", "ai.onnx")
        and node.op_type == "Conv"
        and len(node.input) == 3
        and node.input[1] == "encoder.in_linear1.linear.weight"
    ]
    if len(first_convs) != 1:
        raise ValueError(f"Expected one first encoder Conv, found {len(first_convs)}")
    first_conv = first_convs[0]
    weight = initializers.get(first_conv.input[1])
    if weight is None:
        raise ValueError("First encoder Conv weight is not immutable")
    weight_array = numpy_helper.to_array(weight)
    if weight_array.dtype != np.float32 or weight_array.shape != (140, 80, 5, 1):
        raise ValueError(f"Unexpected first encoder Conv weight: {weight_array.shape}")

    concat = producers.get(first_conv.input[0])
    if concat is None or concat.domain not in ("", "ai.onnx") or concat.op_type != "Concat":
        raise ValueError("First encoder Conv is not fed by the expected LFR Concat")
    if attributes(concat) != {"axis": 2} or len(concat.input) != 5 or len(concat.output) != 1:
        raise ValueError("Unexpected LFR Concat signature")
    left_name, left_repeat, source_name, right_name, right_repeat = concat.input
    if left_name != left_repeat or right_name != right_repeat:
        raise ValueError("LFR Concat does not repeat each edge frame exactly twice")
    if consumers.get(concat.output[0], []) != [first_conv]:
        raise ValueError("LFR Concat output has an unexpected consumer")

    left = producers.get(left_name)
    right = producers.get(right_name)
    if left is None or right is None or left is right:
        raise ValueError("LFR edge Slice producers are missing or ambiguous")
    if left.input[0] != source_name or right.input[0] != source_name:
        raise ValueError("LFR edge Slices do not share the padded source tensor")
    if consumers.get(left_name, []) != [concat, concat] or consumers.get(right_name, []) != [concat, concat]:
        raise ValueError("LFR edge Slice outputs have unexpected consumers")
    if sorted(id(node) for node in consumers.get(source_name, [])) != sorted(
        (id(left), id(right), id(concat))
    ):
        raise ValueError("LFR source tensor has unexpected consumers")

    left_constants = require_slice_controls(
        left,
        producers,
        consumers,
        [0],
        [1],
        [2],
        [1],
    )
    right_constants = require_slice_controls(
        right,
        producers,
        consumers,
        [-1],
        [INT64_MAX],
        [2],
        [1],
    )
    require_value(value_info, source_name, (1, 80, None, 1))
    require_value(value_info, left.output[0], (1, 80, None, 1))
    require_value(value_info, right.output[0], (1, 80, None, 1))
    require_value(value_info, concat.output[0], (1, 80, None, 1))

    all_tensor_names = {
        value.name
        for value in (*model.graph.input, *model.graph.output, *model.graph.value_info)
    }
    all_tensor_names.update(initializer.name for initializer in model.graph.initializer)
    all_tensor_names.update(output for node in model.graph.node for output in node.output)
    if LFR_PADS_NAME in all_tensor_names:
        raise ValueError(f"Cannot insert collision-free initializer {LFR_PADS_NAME!r}")
    if LFR_PAD_NODE_NAME in {node.name for node in model.graph.node if node.name}:
        raise ValueError(f"Cannot insert collision-free node {LFR_PAD_NODE_NAME!r}")

    pad = helper.make_node(
        "Pad",
        [source_name, LFR_PADS_NAME],
        list(concat.output),
        name=LFR_PAD_NODE_NAME,
        mode="edge",
    )
    removed = (left, right, concat, *left_constants, *right_constants)
    stale_tensors = {
        left.output[0],
        right.output[0],
        *(node.output[0] for node in (*left_constants, *right_constants)),
    }
    pads = numpy_helper.from_array(
        np.asarray([0, 0, 2, 0, 0, 0, 2, 0], dtype=np.int64),
        name=LFR_PADS_NAME,
    )
    return concat, pad, removed, stale_tensors, pads


def find_cache_output_rewrites(value_info, producers, consumers):
    regions = []
    control_node_ids = set()
    for index in range(4):
        output_name = f"out_cache_{index}"
        reshape = producers.get(output_name)
        if reshape is None or reshape.domain not in ("", "ai.onnx") or reshape.op_type != "Reshape":
            raise ValueError(f"Expected Reshape producer for {output_name!r}")
        if len(reshape.input) != 2 or len(reshape.output) != 1:
            raise ValueError(f"Unexpected Reshape signature for {output_name!r}")
        if attributes(reshape) != {"allowzero": 0}:
            raise ValueError(f"Unexpected Reshape attributes for {output_name!r}")
        if consumers.get(output_name, []):
            raise ValueError(f"Graph output {output_name!r} has an unexpected node consumer")

        slice_node = producers.get(reshape.input[0])
        if slice_node is None or consumers.get(slice_node.output[0], []) != [reshape]:
            raise ValueError(f"Cache Slice for {output_name!r} is not exclusively reshaped")
        controls = require_slice_controls(
            slice_node,
            producers,
            consumers,
            [-19],
            [INT64_MAX],
            [2],
            [1],
        )
        if any(id(node) in control_node_ids for node in controls):
            raise ValueError("Cache Slice control Constant nodes are unexpectedly shared in the raw graph")
        control_node_ids.update(id(node) for node in controls)
        concat = producers.get(slice_node.input[0])
        if concat is None or concat.domain not in ("", "ai.onnx") or concat.op_type != "Concat":
            raise ValueError(f"Cache Slice for {output_name!r} is not fed by Concat")
        if attributes(concat) != {"axis": 2} or len(concat.input) != 2:
            raise ValueError(f"Unexpected cache Concat for {output_name!r}")
        if concat.input[0] != f"in_cache_{index}":
            raise ValueError(f"Cache Concat {index} does not preserve its recurrent input")

        shape_constant = require_constant(
            producers,
            consumers,
            reshape.input[1],
            [1, 128, 19, 1],
        )
        require_value(value_info, concat.output[0], (1, 128, None, 1))
        require_value(value_info, slice_node.output[0], (1, 128, None, 1))
        regions.append((slice_node, reshape, shape_constant, controls))
    if len(control_node_ids) != 16:
        raise ValueError(f"Expected 16 cache Slice control Constant nodes, got {len(control_node_ids)}")
    return regions


def find_frontend_axis_dedup(model, producers, consumers):
    unsqueeze_matches = []
    squeeze_matches = []
    for node in model.graph.node:
        if node.domain not in ("", "ai.onnx") or len(node.input) != 2:
            continue
        if node.op_type == "Unsqueeze":
            data_producer = producers.get(node.input[0])
            if (
                data_producer is not None
                and data_producer.domain in ("", "ai.onnx")
                and data_producer.op_type == "Cast"
                and list(data_producer.input) == ["audio"]
                and attributes(data_producer) == {"to": TensorProto.FLOAT}
            ):
                unsqueeze_matches.append(node)
        elif node.op_type == "Squeeze":
            data_producer = producers.get(node.input[0])
            if (
                data_producer is not None
                and data_producer.domain in ("", "ai.onnx")
                and data_producer.op_type == "Conv"
                and len(data_producer.input) == 3
                and data_producer.input[1:] == ["frontend.energy_kernel", "frontend.energy_bias"]
            ):
                squeeze_matches.append(node)
    if len(unsqueeze_matches) != 1 or len(squeeze_matches) != 1:
        raise ValueError(
            f"Expected one frontend channel Unsqueeze and Squeeze, got "
            f"{len(unsqueeze_matches)} and {len(squeeze_matches)}"
        )

    matched_nodes = (unsqueeze_matches[0], squeeze_matches[0])
    constants = tuple(
        require_constant(producers, consumers, node.input[1], [1])
        for node in matched_nodes
    )
    if constants[0] is constants[1]:
        raise ValueError("Frontend channel-axis Constants are unexpectedly shared in the raw graph")
    if any(consumers.get(node.output[0], []) != [matched] for node, matched in zip(constants, matched_nodes)):
        raise ValueError("Frontend channel-axis Constant has an unexpected consumer")
    return matched_nodes, constants


def uses_external_data(model):
    return any(
        initializer.data_location == TensorProto.EXTERNAL or initializer.external_data
        for initializer in model.graph.initializer
    )


def fp16_node_block_list(model):
    blocked_nodes = [
        node.name
        for node in model.graph.node
        if node.name and node.name.startswith(FP16_NODE_PREFIXES)
    ]
    if not blocked_nodes:
        raise RuntimeError("Cannot identify the FP32 spectral frontend for FP16 conversion")
    return blocked_nodes


def validate_runtime(raw_path, final_path, use_fp16=False):
    version_match = re.match(r"^(\d+)\.(\d+)\.(\d+)", ort.__version__)
    minimum_version = ".".join(str(part) for part in MINIMUM_ORT_RELEASE)
    if (
        version_match is None
        or tuple(int(part) for part in version_match.groups()) < MINIMUM_ORT_RELEASE
    ):
        raise RuntimeError(
            f"This rewrite requires ONNX Runtime {minimum_version} or newer, "
            f"but {ort.__version__} is installed"
        )
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
        for path in (raw_path, final_path)
    ]
    if [item.name for item in sessions[0].get_inputs()] != [
        item.name for item in sessions[1].get_inputs()
    ] or [item.name for item in sessions[0].get_outputs()] != [
        item.name for item in sessions[1].get_outputs()
    ]:
        raise RuntimeError("Staged runtime interface differs from the raw model")

    generator = np.random.default_rng(20260721)
    audio_dimension = sessions[0].get_inputs()[0].shape[1]
    cases = (
        ((audio_dimension, False), (audio_dimension, True))
        if isinstance(audio_dimension, int)
        else ((400, False), (401, True), (16000, False), (16123, True))
    )
    for audio_length, populated_cache in cases:
        inputs = {
            "audio": generator.integers(
                -32768,
                32768,
                size=(1, audio_length),
                dtype=np.int16,
            )
        }
        for index in range(4):
            inputs[f"in_cache_{index}"] = (
                generator.standard_normal((1, 128, 19, 1), dtype=np.float32)
                if populated_cache
                else np.zeros((1, 128, 19, 1), dtype=np.float32)
            )
        raw_outputs, final_outputs = (session.run(None, inputs) for session in sessions)
        for metadata, raw_output, final_output in zip(
            sessions[0].get_outputs(),
            raw_outputs,
            final_outputs,
        ):
            if raw_output.shape != final_output.shape or raw_output.dtype != final_output.dtype:
                raise RuntimeError(f"Runtime contract changed for {metadata.name!r}")
            outputs_match = (
                np.allclose(
                    raw_output,
                    final_output,
                    rtol=FP16_RTOL,
                    atol=FP16_ATOL,
                    equal_nan=True,
                )
                if use_fp16
                else np.array_equal(raw_output, final_output, equal_nan=True)
            )
            if not outputs_match:
                difference = np.abs(raw_output - final_output)
                finite = difference[np.isfinite(difference)]
                max_error = float(finite.max()) if finite.size else float("inf")
                raise RuntimeError(
                    f"Runtime output {metadata.name!r} changed for audio length "
                    f"{audio_length}: max_abs={max_error}"
                )
    return len(cases)


def save_checked(
    model,
    raw_path,
    output_path,
    external_data,
    overwrite,
    use_fp16,
    expected_interface,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    external_path = output_path.with_name(output_path.name + ".data")
    if not overwrite and (output_path.exists() or external_path.exists()):
        raise FileExistsError(f"Refusing to overwrite final artifact {output_path}")

    with tempfile.TemporaryDirectory(prefix="fsmn_fusion_", dir=output_path.parent) as directory:
        temporary_path = Path(directory) / output_path.name
        save_options = {}
        if external_data:
            save_options = {
                "save_as_external_data": True,
                "all_tensors_to_one_file": True,
                "location": temporary_path.name + ".data",
                "size_threshold": 0,
            }
        onnx.save_model(model, temporary_path, **save_options)
        slim(
            model=str(temporary_path),
            output_model=str(temporary_path),
            no_shape_infer=True,
            skip_fusion_patterns=False,
            no_constant_folding=False,
            save_as_external_data=external_data,
            verbose=False,
        )
        staged = onnx.load(temporary_path, load_external_data=True)
        onnx.checker.check_model(staged, check_custom_domain=False)
        if graph_interface(staged) != expected_interface:
            raise RuntimeError("Staged final graph changed the public interface")
        if not use_fp16:
            if len(staged.graph.node) != 42:
                raise RuntimeError(
                    f"Expected 42 nodes after onnxslim, got {len(staged.graph.node)}"
                )
            if len(staged.graph.initializer) != 36:
                raise RuntimeError(
                    "Expected 36 initializers after onnxslim, "
                    f"got {len(staged.graph.initializer)}"
                )
            if operator_histogram(staged) != EXPECTED_FINAL_OPERATORS:
                raise RuntimeError(
                    f"Final operator histogram changed: {operator_histogram(staged)}; "
                    f"expected {EXPECTED_FINAL_OPERATORS}"
                )
        if sum(
            node.op_type == "FusedConv" and node.domain == MICROSOFT_DOMAIN
            for node in staged.graph.node
        ) != 5:
            raise RuntimeError("Onnxslim did not preserve exactly five FusedConv nodes")
        validated_cases = validate_runtime(
            raw_path,
            temporary_path,
            use_fp16=use_fp16,
        )

        if overwrite:
            output_path.unlink(missing_ok=True)
            external_path.unlink(missing_ok=True)
        os.replace(temporary_path, output_path)
        temporary_external = temporary_path.with_name(temporary_path.name + ".data")
        if temporary_external.exists():
            os.replace(temporary_external, external_path)
    return validated_cases, len(staged.graph.node), len(staged.graph.initializer)


def optimize_graph(raw_path, final_path, overwrite=False, use_fp16=ENABLE_FP16):
    source_metadata = read_source_metadata(raw_path)
    raw_path = Path(raw_path).resolve()
    final_path = Path(final_path).resolve()
    if raw_path == final_path:
        raise ValueError("Raw and final paths must be distinct")
    raw_digest = file_sha256(raw_path)
    model = onnx.load(raw_path, load_external_data=True)
    original_interface = graph_interface(model)
    original_node_count = len(model.graph.node)
    external_data = uses_external_data(model)
    validate_raw_graph(model)
    inferred = shape_inference.infer_shapes(model)
    value_info = {
        value.name: value
        for value in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)
    }
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    producers, consumers = graph_maps(model)
    pairs = find_fusion_pairs(model, value_info, initializers, consumers)
    concat, pad, padding_nodes, padding_tensors, pads_initializer = find_lfr_padding(
        model,
        value_info,
        initializers,
        producers,
        consumers,
    )
    cache_regions = find_cache_output_rewrites(value_info, producers, consumers)
    frontend_axis_nodes, frontend_axis_constants = find_frontend_axis_dedup(
        model,
        producers,
        consumers,
    )

    all_tensor_names = {
        value.name
        for value in (*model.graph.input, *model.graph.output, *model.graph.value_info)
    }
    all_tensor_names.update(initializer.name for initializer in model.graph.initializer)
    all_tensor_names.update(output for node in model.graph.node for output in node.output)
    inserted_names = {*CACHE_SLICE_CONTROLS, FRONTEND_CHANNEL_AXIS_NAME}
    collisions = inserted_names & all_tensor_names
    if collisions:
        raise ValueError(f"Cannot insert collision-free cache Slice controls: {sorted(collisions)}")

    replacement_by_id = {}
    removed_ids = set()
    stale_value_info = set(padding_tensors)
    existing_node_names = {node.name for node in model.graph.node if node.name}
    for conv, relu in pairs:
        fused_name = conv.name + "_FusedRelu"
        if not conv.name or fused_name in existing_node_names:
            raise ValueError(f"Cannot generate a collision-free fused name for {conv.name!r}")
        fused = helper.make_node(
            "FusedConv",
            list(conv.input),
            list(relu.output),
            name=fused_name,
            domain=MICROSOFT_DOMAIN,
            activation="Relu",
            **attributes(conv),
        )
        replacement_by_id[id(conv)] = fused
        removed_ids.update((id(conv), id(relu)))
        stale_value_info.add(conv.output[0])

    replacement_by_id[id(concat)] = pad
    removed_ids.update(id(node) for node in padding_nodes)
    for slice_node, reshape, shape_constant, control_nodes in cache_regions:
        old_slice_output = slice_node.output[0]
        slice_node.output[0] = reshape.output[0]
        for input_index, control_name in enumerate(CACHE_SLICE_CONTROLS, start=1):
            slice_node.input[input_index] = control_name
        removed_ids.update((id(reshape), id(shape_constant), *(id(node) for node in control_nodes)))
        stale_value_info.update(
            (
                old_slice_output,
                shape_constant.output[0],
                *(node.output[0] for node in control_nodes),
            )
        )
    for node in frontend_axis_nodes:
        node.input[1] = FRONTEND_CHANNEL_AXIS_NAME
    removed_ids.update(id(node) for node in frontend_axis_constants)
    stale_value_info.update(node.output[0] for node in frontend_axis_constants)

    rewritten_nodes = []
    for node in model.graph.node:
        if id(node) in replacement_by_id:
            rewritten_nodes.append(replacement_by_id[id(node)])
        elif id(node) not in removed_ids:
            rewritten_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    model.graph.initializer.append(pads_initializer)
    model.graph.initializer.extend(
        numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=name)
        for name, value in CACHE_SLICE_CONTROLS.items()
    )
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.asarray([1], dtype=np.int64),
            name=FRONTEND_CHANNEL_AXIS_NAME,
        )
    )

    original_value_info_count = len(model.graph.value_info)
    retained_value_info = [
        value for value in model.graph.value_info if value.name not in stale_value_info
    ]
    deleted_value_info_count = original_value_info_count - len(retained_value_info)
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_value_info)
    model.opset_import.append(helper.make_opsetid(MICROSOFT_DOMAIN, MICROSOFT_OPSET))

    if graph_interface(model) != original_interface:
        raise RuntimeError("Rewrite changed the public graph interface")
    if len(model.graph.node) != 47 or original_node_count != 88:
        raise RuntimeError(f"Unexpected node transition: {original_node_count} -> {len(model.graph.node)}")
    if len(model.graph.initializer) != 32:
        raise RuntimeError(f"Expected 32 final initializers, got {len(model.graph.initializer)}")
    if operator_histogram(model) != EXPECTED_REWRITTEN_OPERATORS:
        raise RuntimeError(
            f"Rewritten operator histogram changed: {operator_histogram(model)}; "
            f"expected {EXPECTED_REWRITTEN_OPERATORS}"
        )
    if sum(node.op_type == "FusedConv" and node.domain == MICROSOFT_DOMAIN for node in model.graph.node) != 5:
        raise RuntimeError("Final graph does not contain exactly five FusedConv nodes")
    if any(id(node) in removed_ids for node in model.graph.node):
        raise RuntimeError("A replaced node remained in the final graph")

    onnx.checker.check_model(model, check_custom_domain=False)
    converted_initializer_count = 0
    if use_fp16:
        model, converted_initializer_count = convert_graph_to_fp16(
            model,
            force_fp16_initializers=False,
            op_block_list=FP16_OP_BLOCK_LIST,
            node_block_list=fp16_node_block_list(model),
        )
    validated_cases, final_node_count, final_initializer_count = save_checked(
        model,
        raw_path,
        final_path,
        external_data,
        overwrite,
        use_fp16,
        original_interface,
    )
    preserve_optimized_metadata(raw_path, final_path, source_metadata)
    if file_sha256(raw_path) != raw_digest:
        raise RuntimeError("Raw model changed while the final graph was being written")
    print(f"Matched Conv-Relu pairs: {len(pairs)}")
    print(f"Inserted {MICROSOFT_DOMAIN}::FusedConv nodes: {len(pairs)}")
    print(f"Rewired outputs: {', '.join(relu.output[0] for _, relu in pairs)}")
    print("Replaced LFR edge region: 2 Slice + 1 Concat + 8 Constant -> 1 Pad")
    print(f"Rewired cache Slice outputs: {len(cache_regions)}")
    print("Deduplicated cache Slice controls: 16 Constant nodes -> 4 shared initializers")
    print("Deduplicated frontend channel axis: 2 Constant nodes -> 1 shared initializer")
    if use_fp16:
        print(f"Converted float initializers to FP16: {converted_initializer_count}")
    else:
        print(f"Transformed initializers: 0; added initializers: 6 ({LFR_PADS_NAME} + 5 shared controls)")
    print(
        f"Applied onnxslim: {len(model.graph.node)} -> {final_node_count} nodes; "
        f"{len(model.graph.initializer)} -> {final_initializer_count} initializers"
    )
    print(f"Initializers retained for FusedConv: {len(pairs) * 2}")
    print(f"Deleted Conv nodes: {len(pairs)}; deleted Relu nodes: {len(pairs)}")
    print("Deleted cache-only Reshape nodes: 4; deleted cache-shape Constant nodes: 4")
    print(f"Deleted stale value_info entries: {deleted_value_info_count}")
    print(f"Nodes: {original_node_count} -> {final_node_count}")
    print(f"Added opset import: {MICROSOFT_DOMAIN} {MICROSOFT_OPSET}")
    print(f"Internal precision: {'mixed FP16/FP32' if use_fp16 else 'FP32'}")
    print(
        f"Runtime validation: {validated_cases} bit-exact cases on "
        f"ONNX Runtime {ort.__version__} CPUExecutionProvider with optimizations disabled"
    )
    print(f"Raw model left unchanged: {raw_path}")
    print(f"Raw SHA-256 unchanged: {raw_digest}")
    print(f"Final model written: {final_path}")


def main():
    optimize_graph(
        RAW_ONNX_MODEL,
        FINAL_ONNX_MODEL,
        OVERWRITE,
        use_fp16=ENABLE_FP16,
    )


if __name__ == "__main__":
    main()
