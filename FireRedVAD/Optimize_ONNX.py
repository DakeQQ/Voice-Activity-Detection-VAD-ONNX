"""Apply narrowly targeted FusedConv rewrites and onnxslim to FireRedVAD.

PyTorch's standard ONNX exporter cannot encode Conv with a fused activation or
residual input. This pass replaces only verified Conv-Relu, Conv-Sigmoid, and
memory Conv-Add pairs with com.microsoft::FusedConv. The operator is built into
ONNX Runtime's CPUExecutionProvider and needs no custom-operator library.
Onnxslim then removes redundant Cast and Constant nodes from the staged graph.
An opt-in switch converts internal floating-point compute to FP16 while
preserving the public input and output types.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "vad_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from Optimize_ONNX_Common import (
    atomic_model_path,
    convert_graph_to_fp16,
    file_sha256,
    graph_interface,
    graph_maps,
    metadata_properties,
    operator_histogram,
    simplify_onnx,
    tensor_shape,
)
from vad_onnx_metadata import metadata_path_for_model, read_source_metadata


MICROSOFT_DOMAIN = "com.microsoft"
MICROSOFT_OPSET = 1
ORT_VERSION = ort.__version__
MODEL_FOLDER = Path(__file__).resolve().parent / "FireRedVAD_ONNX"
OPTIMIZED_FOLDER = Path(__file__).resolve().parent / "FireRedVAD_Optimized"

ENABLE_FP16 = False  # Convert internal float compute to FP16; preserve public I/O types.

# (maximum absolute error, maximum mean absolute error, minimum cosine similarity)
FP16_OUTPUT_LIMITS = {
    "probs": (7.5e-2, 3e-2, 0.998),
    "caches_out": (16.0, 3e-1, 0.995),
}

MODEL_TYPES_BY_TASK = {
    "voice_activity_detection": "vad",
    "audio_event_detection": "aed",
    "streaming_voice_activity_detection": "stream_vad",
}
EXPECTED_INITIALIZERS = 39
STANDARD_CONV_ATTRIBUTES = {
    "dilations": [1],
    "group": 1,
    "kernel_shape": [1],
    "pads": [0, 0],
    "strides": [1],
}
EXPECTED_RAW_OPERATORS = {
    "offline": {
        ("ai.onnx", "Add"): 8,
        ("ai.onnx", "Cast"): 1,
        ("ai.onnx", "Clip"): 1,
        ("ai.onnx", "Constant"): 2,
        ("ai.onnx", "Conv"): 28,
        ("ai.onnx", "Log"): 1,
        ("ai.onnx", "Mul"): 1,
        ("ai.onnx", "Relu"): 10,
        ("ai.onnx", "Sigmoid"): 1,
        ("ai.onnx", "Split"): 1,
    },
    "stream_vad": {
        ("ai.onnx", "Add"): 16,
        ("ai.onnx", "Cast"): 1,
        ("ai.onnx", "Clip"): 1,
        ("ai.onnx", "Concat"): 10,
        ("ai.onnx", "Constant"): 15,
        ("ai.onnx", "Conv"): 28,
        ("ai.onnx", "Gather"): 8,
        ("ai.onnx", "Log"): 1,
        ("ai.onnx", "Mul"): 1,
        ("ai.onnx", "Relu"): 10,
        ("ai.onnx", "Sigmoid"): 1,
        ("ai.onnx", "Slice"): 1,
        ("ai.onnx", "Split"): 1,
        ("ai.onnx", "Unsqueeze"): 1,
    },
}
EXPECTED_REWRITTEN_OPERATORS = {
    "offline": {
        ("ai.onnx", "Add"): 1,
        ("ai.onnx", "Cast"): 1,
        ("ai.onnx", "Clip"): 1,
        ("ai.onnx", "Constant"): 2,
        ("ai.onnx", "Conv"): 10,
        ("ai.onnx", "Log"): 1,
        ("ai.onnx", "Mul"): 1,
        ("ai.onnx", "Split"): 1,
        (MICROSOFT_DOMAIN, "FusedConv"): 18,
    },
    "stream_vad": {
        ("ai.onnx", "Add"): 16,
        ("ai.onnx", "Cast"): 1,
        ("ai.onnx", "Clip"): 1,
        ("ai.onnx", "Concat"): 10,
        ("ai.onnx", "Constant"): 15,
        ("ai.onnx", "Conv"): 26,
        ("ai.onnx", "Gather"): 8,
        ("ai.onnx", "Log"): 1,
        ("ai.onnx", "Mul"): 1,
        ("ai.onnx", "Relu"): 9,
        ("ai.onnx", "Slice"): 1,
        ("ai.onnx", "Split"): 1,
        ("ai.onnx", "Unsqueeze"): 1,
        (MICROSOFT_DOMAIN, "FusedConv"): 2,
    },
}
EXPECTED_FINAL_OPERATORS = {
    "offline": {
        ("ai.onnx", "Add"): 1,
        ("ai.onnx", "Clip"): 1,
        ("ai.onnx", "Conv"): 10,
        ("ai.onnx", "Log"): 1,
        ("ai.onnx", "Mul"): 1,
        ("ai.onnx", "Split"): 1,
        (MICROSOFT_DOMAIN, "FusedConv"): 18,
    },
    "stream_vad": {
        ("ai.onnx", "Add"): 16,
        ("ai.onnx", "Clip"): 1,
        ("ai.onnx", "Concat"): 10,
        ("ai.onnx", "Conv"): 26,
        ("ai.onnx", "Gather"): 8,
        ("ai.onnx", "Log"): 1,
        ("ai.onnx", "Mul"): 1,
        ("ai.onnx", "Relu"): 9,
        ("ai.onnx", "Slice"): 1,
        ("ai.onnx", "Split"): 1,
        ("ai.onnx", "Unsqueeze"): 1,
        (MICROSOFT_DOMAIN, "FusedConv"): 2,
    },
}
EXPECTED_FINAL_INITIALIZER_TYPES = {
    "offline": Counter({TensorProto.FLOAT: 40, TensorProto.INT64: 1}),
    "stream_vad": Counter({TensorProto.FLOAT: 40, TensorProto.INT64: 13}),
}
EXPECTED_FP16_CASTS = {"offline": 2, "stream_vad": 4}
EXPECTED_RESIDUAL_PAIRS = {"offline": 7, "stream_vad": 8}


def attributes(node):
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def value_info_map(model):
    inferred = shape_inference.infer_shapes(model)
    return {
        value.name: value
        for value in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)
    }


def require_float_rank3(values, tensor_name, channel_count=None):
    value = values.get(tensor_name)
    if value is None:
        raise ValueError(f"Shape inference did not describe {tensor_name!r}")
    tensor_type = value.type.tensor_type
    if tensor_type.elem_type != TensorProto.FLOAT or len(tensor_type.shape.dim) != 3:
        raise ValueError(f"Expected float32 rank-3 tensor for {tensor_name!r}")
    shape = tensor_shape(value)
    if shape[0] != 1 or (channel_count is not None and shape[1] != channel_count):
        raise ValueError(f"Unexpected shape for {tensor_name!r}: {shape}")
    return shape


def validate_interface(model, model_type):
    inputs = graph_interface(model)["inputs"]
    outputs = graph_interface(model)["outputs"]
    audio_types = {
        TensorProto.FLOAT16,
        TensorProto.FLOAT,
        TensorProto.INT16,
    }
    if model_type == "stream_vad":
        if len(inputs) != 2 or len(outputs) != 2:
            raise ValueError("Stream-VAD must have two inputs and two outputs")
        audio, caches = inputs
        probs, caches_out = outputs
        if audio[0] != "audio" or audio[1] not in audio_types or audio[2][:2] != (1, 1):
            raise ValueError(f"Unexpected Stream-VAD audio input: {audio}")
        if isinstance(audio[2][2], int):
            raise ValueError("Stream-VAD audio length must remain dynamic")
        if caches != ("caches_in", TensorProto.FLOAT, (8, 1, 128, 19)):
            raise ValueError(f"Unexpected Stream-VAD cache input: {caches}")
        if probs[0] != "probs" or probs[1] != TensorProto.FLOAT or probs[2][:2] != (1, 1):
            raise ValueError(f"Unexpected Stream-VAD probability output: {probs}")
        if isinstance(probs[2][2], int):
            raise ValueError("Stream-VAD probability length must remain dynamic")
        if caches_out[0] != "caches_out" or caches_out[1] != TensorProto.FLOAT:
            raise ValueError(f"Unexpected Stream-VAD cache output: {caches_out}")
        if caches_out[2][:3] != (8, 1, 128):
            raise ValueError(f"Unexpected Stream-VAD cache output shape: {caches_out[2]}")
    else:
        output_channels = 3 if model_type == "aed" else 1
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                f"Unexpected {model_type} interface: inputs={inputs}, outputs={outputs}"
            )
        audio = inputs[0]
        probs = outputs[0]
        if (
            audio[0] != "audio"
            or audio[1] not in audio_types
            or len(audio[2]) != 3
            or audio[2][:2] != (1, 1)
            or not isinstance(audio[2][2], (int, str))
        ):
            raise ValueError(f"Unexpected {model_type} audio input: {audio}")
        if (
            probs[0] != "probs"
            or probs[1] != TensorProto.FLOAT
            or len(probs[2]) != 3
            or probs[2][:2] != (1, output_channels)
            or not isinstance(probs[2][2], (int, str))
        ):
            raise ValueError(f"Unexpected {model_type} probability output: {probs}")


def validate_raw_graph(model, model_type):
    validate_interface(model, model_type)
    if any(node.domain not in ("", "ai.onnx") for node in model.graph.node):
        raise ValueError("Raw graph contains a non-standard-domain node")
    standard_opsets = [
        item.version for item in model.opset_import if item.domain in ("", "ai.onnx")
    ]
    if standard_opsets != [20] or len(model.opset_import) != 1:
        raise ValueError("Expected exactly one ai.onnx opset-20 import")
    initializer_names = [item.name for item in model.graph.initializer]
    if len(initializer_names) != len(set(initializer_names)):
        raise ValueError("Raw graph contains duplicate initializer names")


def validate_rewritten_graph(model, model_type):
    profile = "stream_vad" if model_type == "stream_vad" else "offline"
    validate_interface(model, model_type)
    if operator_histogram(model) != EXPECTED_REWRITTEN_OPERATORS[profile]:
        raise ValueError(
            f"Unexpected rewritten operator inventory: {operator_histogram(model)}"
        )
    opsets = {(item.domain or "ai.onnx", item.version) for item in model.opset_import}
    if opsets != {("ai.onnx", 20), (MICROSOFT_DOMAIN, MICROSOFT_OPSET)}:
        raise ValueError(f"Unexpected optimized opset imports: {sorted(opsets)}")
    if len(model.graph.initializer) != EXPECTED_INITIALIZERS:
        raise ValueError(
            f"Expected {EXPECTED_INITIALIZERS} initializers, "
            f"got {len(model.graph.initializer)}"
        )
    if any(item.data_type != TensorProto.FLOAT for item in model.graph.initializer):
        raise ValueError("Expected every rewritten initializer to be float32")
    onnx.checker.check_model(model, check_custom_domain=False)


def validate_rewritten_fp16_graph(model, model_type):
    profile = "stream_vad" if model_type == "stream_vad" else "offline"
    validate_interface(model, model_type)
    actual_operators = Counter(operator_histogram(model))
    expected_operators = Counter(EXPECTED_REWRITTEN_OPERATORS[profile])
    actual_casts = actual_operators.pop(("ai.onnx", "Cast"), 0)
    expected_casts = expected_operators.pop(("ai.onnx", "Cast"), 0)
    if actual_operators != expected_operators or actual_casts < expected_casts:
        raise ValueError(
            f"Unexpected FP16 operator inventory: {operator_histogram(model)}"
        )
    opsets = {(item.domain or "ai.onnx", item.version) for item in model.opset_import}
    if opsets != {("ai.onnx", 20), (MICROSOFT_DOMAIN, MICROSOFT_OPSET)}:
        raise ValueError(f"Unexpected FP16 opset imports: {sorted(opsets)}")
    if len(model.graph.initializer) != EXPECTED_INITIALIZERS:
        raise ValueError(
            f"Expected {EXPECTED_INITIALIZERS} FP16 initializers, "
            f"got {len(model.graph.initializer)}"
        )
    if any(item.data_type != TensorProto.FLOAT16 for item in model.graph.initializer):
        raise ValueError("Expected every rewritten initializer to be float16")
    onnx.checker.check_model(model, check_custom_domain=False)


def validate_final_graph(model, model_type, use_fp16=False):
    profile = "stream_vad" if model_type == "stream_vad" else "offline"
    validate_interface(model, model_type)
    actual_operators = Counter(operator_histogram(model))
    expected_operators = Counter(EXPECTED_FINAL_OPERATORS[profile])
    if use_fp16:
        cast_count = actual_operators.pop(("ai.onnx", "Cast"), 0)
        if (
            actual_operators != expected_operators
            or cast_count != EXPECTED_FP16_CASTS[profile]
        ):
            raise ValueError(
                f"Unexpected slimmed FP16 operator inventory: "
                f"{operator_histogram(model)}"
            )
    elif actual_operators != expected_operators:
        raise ValueError(
            f"Unexpected slimmed operator inventory: {operator_histogram(model)}"
        )

    opsets = {(item.domain or "ai.onnx", item.version) for item in model.opset_import}
    if opsets != {("ai.onnx", 20), (MICROSOFT_DOMAIN, MICROSOFT_OPSET)}:
        raise ValueError(f"Unexpected slimmed opset imports: {sorted(opsets)}")
    expected_initializer_types = EXPECTED_FINAL_INITIALIZER_TYPES[profile].copy()
    if use_fp16:
        expected_initializer_types[TensorProto.FLOAT16] = (
            expected_initializer_types.pop(TensorProto.FLOAT)
        )
    initializer_types = Counter(
        item.data_type for item in model.graph.initializer
    )
    if initializer_types != expected_initializer_types:
        raise ValueError(
            f"Unexpected slimmed initializer types: {initializer_types}; "
            f"expected {expected_initializer_types}"
        )
    onnx.checker.check_model(model, check_custom_domain=False)


def model_type_from_metadata(model_path):
    metadata = read_source_metadata(model_path)
    task = metadata.get("task")
    try:
        return MODEL_TYPES_BY_TASK[task]
    except KeyError as error:
        raise ValueError(
            f"Cannot determine the FireRed model type for {model_path}: "
            f"unsupported or missing metadata task {task!r}"
        ) from error


def require_initializer(initializers, name, expected_shape):
    tensor = initializers.get(name)
    if tensor is None:
        raise ValueError(f"Missing immutable initializer {name!r}")
    array = numpy_helper.to_array(tensor)
    if array.dtype != np.float32 or array.shape != expected_shape:
        raise ValueError(
            f"Unexpected initializer {name!r}: {array.dtype} {array.shape}; "
            f"expected float32 {expected_shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"Initializer {name!r} contains NaN or Inf")
    return array


def find_activation_pairs(model, model_type, values, initializers, consumers):
    graph_outputs = {value.name for value in model.graph.output}
    pairs = []
    activation_counts = Counter()
    weight_shapes = Counter()
    matched_weights = set()
    for conv in model.graph.node:
        if conv.domain not in ("", "ai.onnx") or conv.op_type != "Conv":
            continue
        if len(conv.input) != 3 or len(conv.output) != 1:
            continue
        users = consumers.get(conv.output[0], [])
        if len(users) != 1 or users[0].domain not in ("", "ai.onnx"):
            continue
        activation = users[0]
        if activation.op_type not in ("Relu", "Sigmoid"):
            continue
        if (
            len(activation.input) != 1
            or len(activation.output) != 1
            or activation.input[0] != conv.output[0]
            or attributes(activation)
        ):
            raise ValueError(f"Malformed activation after {conv.name!r}")
        if attributes(conv) != STANDARD_CONV_ATTRIBUTES:
            raise ValueError(f"Unexpected pointwise Conv attributes at {conv.name!r}")

        weight_tensor = initializers.get(conv.input[1])
        if weight_tensor is None:
            raise ValueError(f"Pointwise Conv {conv.name!r} has a mutable weight")
        weight_shape = tuple(weight_tensor.dims)
        weight = require_initializer(initializers, conv.input[1], weight_shape)
        if weight.ndim != 3 or weight.shape[2] != 1 or weight.shape[1] not in (80, 128, 256):
            raise ValueError(f"Unexpected pointwise weight shape at {conv.name!r}: {weight.shape}")
        require_initializer(initializers, conv.input[2], (weight.shape[0],))
        require_float_rank3(values, conv.input[0], weight.shape[1])
        require_float_rank3(values, conv.output[0], weight.shape[0])
        require_float_rank3(values, activation.output[0], weight.shape[0])
        if conv.output[0] in graph_outputs:
            raise ValueError(f"Intermediate Conv output {conv.output[0]!r} is public")
        if activation.op_type == "Sigmoid" and activation.output[0] not in graph_outputs:
            raise ValueError("The matched Sigmoid is not the public probability output")
        if conv.input[1] in matched_weights:
            raise ValueError(f"Initializer {conv.input[1]!r} is matched more than once")

        pairs.append((conv, activation))
        activation_counts[activation.op_type] += 1
        weight_shapes[weight_shape] += 1
        matched_weights.add(conv.input[1])

    output_channels = 3 if model_type == "aed" else 1
    expected_shapes = Counter(
        {
            (256, 80, 1): 1,
            (128, 256, 1): 1,
            (256, 128, 1): 8,
            (output_channels, 256, 1): 1,
        }
    )
    if activation_counts != Counter({"Relu": 10, "Sigmoid": 1}):
        raise ValueError(f"Expected 10 Relu and 1 Sigmoid pair, got {activation_counts}")
    if weight_shapes != expected_shapes:
        raise ValueError(f"Unexpected activation weight inventory: {weight_shapes}")
    if len(pairs) != 11 or len(matched_weights) != 11:
        raise ValueError("Activation fusion candidates are incomplete or ambiguous")
    return pairs


def find_residual_pairs(model, model_type, values, initializers, consumers):
    profile = "stream_vad" if model_type == "stream_vad" else "offline"
    if profile == "stream_vad":
        expected_weight_shape = (128, 1, 20)
        expected_attributes = {
            "dilations": [1],
            "group": 128,
            "kernel_shape": [20],
            "pads": [0, 0],
            "strides": [1],
        }
    else:
        expected_weight_shape = (128, 1, 40)
        expected_attributes = {
            "auto_pad": b"SAME_UPPER",
            "dilations": [1],
            "group": 128,
            "kernel_shape": [40],
            "strides": [1],
        }

    pairs = []
    matched_weights = set()
    for conv in model.graph.node:
        if conv.domain not in ("", "ai.onnx") or conv.op_type != "Conv":
            continue
        if len(conv.input) != 2 or len(conv.output) != 1:
            continue
        weight = initializers.get(conv.input[1])
        if weight is None or tuple(weight.dims) != expected_weight_shape:
            continue
        users = consumers.get(conv.output[0], [])
        if len(users) != 1 or users[0].domain not in ("", "ai.onnx"):
            continue
        add = users[0]
        if add.op_type != "Add":
            continue
        if len(add.input) != 2 or len(add.output) != 1 or attributes(add):
            raise ValueError(f"Malformed residual Add after {conv.name!r}")
        if attributes(conv) != expected_attributes:
            raise ValueError(f"Unexpected memory Conv attributes at {conv.name!r}")
        require_initializer(initializers, conv.input[1], expected_weight_shape)
        other_inputs = [name for name in add.input if name != conv.output[0]]
        if len(other_inputs) != 1:
            raise ValueError(f"Residual Add at {add.name!r} is ambiguous")
        residual_name = other_inputs[0]
        require_float_rank3(values, conv.input[0], 128)
        require_float_rank3(values, conv.output[0], 128)
        require_float_rank3(values, residual_name, 128)
        require_float_rank3(values, add.output[0], 128)
        if conv.input[1] in matched_weights:
            raise ValueError(f"Memory initializer {conv.input[1]!r} is matched twice")
        pairs.append((conv, add, residual_name))
        matched_weights.add(conv.input[1])

    expected_count = EXPECTED_RESIDUAL_PAIRS[profile]
    if len(pairs) != expected_count or len(matched_weights) != expected_count:
        raise ValueError(
            f"Expected exactly {expected_count} memory Conv-Add pairs, got {len(pairs)}"
        )
    return pairs


def unique_node_name(existing_names, source_name, suffix):
    if not source_name:
        raise ValueError("Cannot generate an auditable fused name from an unnamed Conv")
    candidate = source_name + suffix
    if candidate in existing_names:
        raise ValueError(f"Cannot generate collision-free node name {candidate!r}")
    existing_names.add(candidate)
    return candidate


def rewrite_graph(model, model_type):
    original_interface = graph_interface(model)
    original_initializer_names = tuple(item.name for item in model.graph.initializer)
    original_histogram = Counter(operator_histogram(model))
    original_metadata = metadata_properties(model)
    values = value_info_map(model)
    _, consumers = graph_maps(model)
    initializers = {item.name: item for item in model.graph.initializer}
    all_activation_pairs = find_activation_pairs(
        model, model_type, values, initializers, consumers
    )
    all_residual_pairs = find_residual_pairs(
        model, model_type, values, initializers, consumers
    )
    if model_type == "stream_vad":
        sigmoid_pairs = [
            pair for pair in all_activation_pairs if pair[1].op_type == "Sigmoid"
        ]
        if len(sigmoid_pairs) != 1:
            raise ValueError("Expected one public Stream-VAD Sigmoid pair")
        output_conv = sigmoid_pairs[0][0]
        preceding_pairs = [
            pair
            for pair in all_activation_pairs
            if pair[1].op_type == "Relu" and pair[1].output[0] == output_conv.input[0]
        ]
        if len(preceding_pairs) != 1:
            raise ValueError("Expected one post-cache DNN Conv-Relu pair")
        activation_pairs = [preceding_pairs[0], sigmoid_pairs[0]]
        residual_pairs = []
    else:
        activation_pairs = all_activation_pairs
        residual_pairs = all_residual_pairs

    replacements = {}
    removed_ids = set()
    stale_value_names = set()
    existing_names = {node.name for node in model.graph.node if node.name}
    activation_outputs = []
    residual_outputs = []

    for conv, activation in activation_pairs:
        fused = helper.make_node(
            "FusedConv",
            list(conv.input),
            list(activation.output),
            name=unique_node_name(existing_names, conv.name, f"_Fused{activation.op_type}"),
            domain=MICROSOFT_DOMAIN,
            activation=activation.op_type,
            **attributes(conv),
        )
        replacements[id(conv)] = fused
        removed_ids.update((id(conv), id(activation)))
        stale_value_names.add(conv.output[0])
        activation_outputs.extend(activation.output)

    for conv, add, residual_name in residual_pairs:
        fused = helper.make_node(
            "FusedConv",
            [conv.input[0], conv.input[1], "", residual_name],
            list(add.output),
            name=unique_node_name(existing_names, conv.name, "_FusedResidual"),
            domain=MICROSOFT_DOMAIN,
            **attributes(conv),
        )
        replacements[id(conv)] = fused
        removed_ids.update((id(conv), id(add)))
        stale_value_names.add(conv.output[0])
        residual_outputs.extend(add.output)

    rewritten_nodes = []
    for node in model.graph.node:
        replacement = replacements.get(id(node))
        if replacement is not None:
            rewritten_nodes.append(replacement)
        elif id(node) not in removed_ids:
            rewritten_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)

    retained_value_info = [
        value for value in model.graph.value_info if value.name not in stale_value_names
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_value_info)
    model.opset_import.append(helper.make_opsetid(MICROSOFT_DOMAIN, MICROSOFT_OPSET))

    if graph_interface(model) != original_interface:
        raise RuntimeError("Rewrite changed graph input/output names, types, or shapes")
    if tuple(item.name for item in model.graph.initializer) != original_initializer_names:
        raise RuntimeError("Rewrite changed initializer names or ordering")
    if metadata_properties(model) != original_metadata:
        raise RuntimeError("Rewrite changed model metadata")
    if len(model.graph.initializer) != len(original_initializer_names):
        raise RuntimeError("Rewrite unexpectedly added or removed an initializer")
    expected_histogram = original_histogram.copy()
    for _, activation in activation_pairs:
        expected_histogram[("ai.onnx", "Conv")] -= 1
        expected_histogram[("ai.onnx", activation.op_type)] -= 1
    for _, _, _ in residual_pairs:
        expected_histogram[("ai.onnx", "Conv")] -= 1
        expected_histogram[("ai.onnx", "Add")] -= 1
    expected_histogram[(MICROSOFT_DOMAIN, "FusedConv")] += (
        len(activation_pairs) + len(residual_pairs)
    )
    expected_histogram = +expected_histogram
    if Counter(operator_histogram(model)) != expected_histogram:
        raise RuntimeError(
            f"Final operator histogram changed: {operator_histogram(model)}; "
            f"expected {dict(expected_histogram)}"
        )
    if len(model.opset_import) != 2:
        raise RuntimeError("Final graph has unexpected opset imports")
    onnx.checker.check_model(model, check_custom_domain=False)
    return (
        activation_outputs,
        residual_outputs,
        len(stale_value_names),
        len(all_activation_pairs),
        len(all_residual_pairs),
    )


def comparison_metrics(expected, actual):
    difference = np.abs(expected - actual)
    finite = difference[np.isfinite(difference)]
    max_abs = float(finite.max()) if finite.size else 0.0
    mean_abs = float(finite.mean()) if finite.size else 0.0
    denominator = np.maximum(np.abs(expected), np.float32(1e-10))
    relative = difference / denominator
    finite_relative = relative[np.isfinite(relative)]
    max_relative = float(finite_relative.max()) if finite_relative.size else 0.0
    expected_flat = expected[np.isfinite(expected)].astype(np.float64, copy=False).ravel()
    actual_flat = actual[np.isfinite(actual)].astype(np.float64, copy=False).ravel()
    denominator_cosine = np.linalg.norm(expected_flat) * np.linalg.norm(actual_flat)
    if denominator_cosine == 0.0:
        cosine = 1.0 if np.array_equal(expected_flat, actual_flat) else 0.0
    else:
        cosine = float(np.dot(expected_flat, actual_flat) / denominator_cosine)
    return max_abs, mean_abs, max_relative, cosine


def validate_outputs(raw_session, final_session, inputs, case_name, use_fp16):
    raw_outputs = raw_session.run(None, inputs)
    final_outputs = final_session.run(None, inputs)
    metrics = []
    for metadata, raw_output, final_output in zip(
        raw_session.get_outputs(), raw_outputs, final_outputs
    ):
        if raw_output.shape != final_output.shape or raw_output.dtype != final_output.dtype:
            raise RuntimeError(f"Runtime contract changed for {metadata.name!r} in {case_name}")
        if not np.array_equal(np.isnan(raw_output), np.isnan(final_output)):
            raise RuntimeError(f"NaN behavior changed for {metadata.name!r} in {case_name}")
        if not np.array_equal(np.isinf(raw_output), np.isinf(final_output)):
            raise RuntimeError(f"Inf behavior changed for {metadata.name!r} in {case_name}")
        max_abs, mean_abs, max_relative, cosine = comparison_metrics(
            raw_output, final_output
        )
        if use_fp16:
            limits = FP16_OUTPUT_LIMITS.get(metadata.name)
            if limits is None:
                raise RuntimeError(
                    f"No FP16 validation limits are defined for {metadata.name!r}"
                )
            max_abs_limit, mean_abs_limit, cosine_limit = limits
            if (
                max_abs > max_abs_limit
                or mean_abs > mean_abs_limit
                or cosine < cosine_limit
            ):
                raise RuntimeError(
                    f"FP16 output {metadata.name!r} exceeded quality limits in "
                    f"{case_name}: max_abs={max_abs} (limit {max_abs_limit}), "
                    f"mean_abs={mean_abs} (limit {mean_abs_limit}), "
                    f"cosine={cosine} (minimum {cosine_limit})"
                )
        elif not np.allclose(
            raw_output,
            final_output,
            rtol=1e-5,
            atol=1e-5,
            equal_nan=True,
        ):
            raise RuntimeError(
                f"Runtime output {metadata.name!r} changed in {case_name}: "
                f"max_abs={max_abs}, max_relative={max_relative}"
            )
        metrics.append((metadata.name, max_abs, mean_abs, max_relative, cosine))
    return metrics


def validate_runtime(raw_path, final_path, model_type, use_fp16=False):
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
        raise RuntimeError("Final runtime interface differs from the raw graph")

    generator = np.random.default_rng(20260721)
    metadata = read_source_metadata(raw_path)
    input_sample_rate = int(metadata.get("input_sample_rate", 16000))
    model_sample_rate = int(metadata.get("model_sample_rate", 16000))
    window_length = int(metadata.get("window_length_samples", 400))
    nominal_audio_length = int(metadata.get("input_audio_length", input_sample_rate))
    minimum_audio_length = max(
        1,
        int(np.ceil(window_length * input_sample_rate / model_sample_rate)),
    )
    audio_metadata = sessions[0].get_inputs()[0]
    audio_dtype = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(int16)": np.int16,
    }.get(audio_metadata.type)
    if audio_dtype is None:
        raise TypeError(f"Unsupported FireRed audio input type: {audio_metadata.type}")

    def make_audio(length, mode):
        shape = (1, 1, length)
        if mode == "silence":
            return np.zeros(shape, dtype=audio_dtype)
        if audio_dtype == np.int16:
            audio = generator.integers(
                -32768, 32768, size=shape, dtype=np.int16)
        else:
            audio = generator.uniform(-1.0, 1.0, size=shape).astype(audio_dtype)
        if mode == "impulse":
            audio.fill(0)
            audio[0, 0, length // 2] = 32767 if audio_dtype == np.int16 else 1.0
        return audio

    cases = []
    if model_type == "stream_vad":
        cache_metadata = sessions[0].get_inputs()[1]
        cache_shape = tuple(cache_metadata.shape)
        if not all(isinstance(dimension, int) for dimension in cache_shape):
            raise ValueError(f"Stream-VAD cache shape must be static: {cache_shape}")
        cache_dtype = {
            "tensor(float16)": np.float16,
            "tensor(float)": np.float32,
        }.get(cache_metadata.type)
        if cache_dtype is None:
            raise TypeError(f"Unsupported Stream-VAD cache type: {cache_metadata.type}")
        stream_lengths = (
            minimum_audio_length,
            minimum_audio_length + 1,
            max(minimum_audio_length, nominal_audio_length),
            max(minimum_audio_length, nominal_audio_length + input_sample_rate // 100),
            max(minimum_audio_length, input_sample_rate),
        )
        for audio_length, populated_cache in zip(
            stream_lengths,
            (False, True, False, True, True),
        ):
            cases.append(
                (
                    f"length={audio_length},populated_cache={populated_cache}",
                    {
                        "audio": make_audio(audio_length, "random"),
                        "caches_in": (
                            generator.standard_normal(cache_shape).astype(cache_dtype)
                            if populated_cache
                            else np.zeros(cache_shape, dtype=cache_dtype)
                        ),
                    },
                )
            )
    else:
        configured_length = audio_metadata.shape[-1]
        if not isinstance(configured_length, int):
            configured_length = max(minimum_audio_length, nominal_audio_length)
        cases = [
            ("random", {"audio": make_audio(configured_length, "random")}),
            ("silence", {"audio": make_audio(configured_length, "silence")}),
            ("impulse", {"audio": make_audio(configured_length, "impulse")}),
        ]

    aggregate = {}
    for case_name, inputs in cases:
        for output_name, max_abs, mean_abs, max_relative, cosine in validate_outputs(
            sessions[0], sessions[1], inputs, case_name, use_fp16
        ):
            previous = aggregate.get(output_name, (0.0, 0.0, 0.0, 1.0))
            aggregate[output_name] = (
                max(previous[0], max_abs),
                max(previous[1], mean_abs),
                max(previous[2], max_relative),
                min(previous[3], cosine),
            )
    return len(cases), aggregate


def publish_model(
    model,
    source_path,
    final_path,
    model_type,
    *,
    final_is_fp16,
    compare_as_fp16,
    expected_interface,
    expected_metadata,
    source_digest,
    prefix,
):
    with atomic_model_path(final_path, prefix=prefix) as staged_path:
        onnx.save(model, staged_path, save_as_external_data=False)
        staged_model = simplify_onnx(staged_path, no_shape_infer=True)
        validate_final_graph(staged_model, model_type, use_fp16=final_is_fp16)
        if graph_interface(staged_model) != expected_interface:
            raise RuntimeError("Onnxslim changed the public graph interface")
        if metadata_properties(staged_model) != expected_metadata:
            raise RuntimeError("Onnxslim changed model metadata")
        validated_cases, metrics = validate_runtime(
            source_path,
            staged_path,
            model_type,
            use_fp16=compare_as_fp16,
        )
        if file_sha256(source_path) != source_digest:
            raise RuntimeError("Source model changed while optimization was running")
    metadata_path_for_model(final_path).unlink(missing_ok=True)
    return staged_model, validated_cases, metrics


def print_runtime_metrics(metrics):
    for output_name, values in metrics.items():
        print(
            f"Runtime {output_name}: max_abs={values[0]:.3e}, "
            f"max_mean_abs={values[1]:.3e}, max_relative={values[2]:.3e}, "
            f"min_cosine={values[3]:.12f}"
        )


def optimize_graph(
    raw_path,
    final_path,
    model_type,
    remove_source=True,
    use_fp16=ENABLE_FP16,
):
    raw_path = Path(raw_path).resolve()
    final_path = Path(final_path).resolve()
    if model_type not in ("vad", "aed", "stream_vad"):
        raise ValueError(f"Unsupported model type {model_type!r}")
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)

    raw_digest = file_sha256(raw_path)
    model = onnx.load(raw_path, load_external_data=True)
    expected_interface = graph_interface(model)
    expected_metadata = metadata_properties(model)
    validate_raw_graph(model, model_type)
    original_node_count = len(model.graph.node)
    (
        activation_outputs,
        residual_outputs,
        stale_value_count,
        candidate_activation_count,
        candidate_residual_count,
    ) = rewrite_graph(model, model_type)
    validate_rewritten_graph(model, model_type)
    if use_fp16:
        model, _ = convert_graph_to_fp16(
            model,
            force_fp16_initializers=True,
        )
        validate_rewritten_fp16_graph(model, model_type)

    staged_model, validated_cases, metrics = publish_model(
        model,
        raw_path,
        final_path,
        model_type,
        final_is_fp16=use_fp16,
        compare_as_fp16=use_fp16,
        expected_interface=expected_interface,
        expected_metadata=expected_metadata,
        source_digest=raw_digest,
        prefix="firered_fusedconv_",
    )
    final_node_count = len(staged_model.graph.node)
    final_initializer_count = len(staged_model.graph.initializer)

    raw_metadata_path = metadata_path_for_model(raw_path)
    if remove_source and raw_path != final_path:
        raw_path.unlink()
        raw_metadata_path.unlink(missing_ok=True)
    print(f"Matched activation candidates: {candidate_activation_count} (10 Relu, 1 Sigmoid)")
    print(f"Matched memory Conv-Add candidates: {candidate_residual_count}")
    print(f"Fused activation pairs: {len(activation_outputs)}")
    print(f"Fused memory Conv-Add pairs: {len(residual_outputs)}")
    print(
        f"Inserted {MICROSOFT_DOMAIN}::FusedConv nodes: "
        f"{len(activation_outputs) + len(residual_outputs)}"
    )
    print(f"Rewired activation outputs: {', '.join(activation_outputs)}")
    print(f"Rewired residual outputs: {', '.join(residual_outputs)}")
    print(
        f"Deleted standard Conv nodes: "
        f"{len(activation_outputs) + len(residual_outputs)}"
    )
    print(f"Deleted Relu/Sigmoid nodes: {len(activation_outputs)}")
    print(f"Deleted residual Add nodes: {len(residual_outputs)}")
    if use_fp16:
        print(f"Converted float initializers to FP16: {EXPECTED_INITIALIZERS}")
    else:
        print("FusedConv rewrite transformed, added, or deleted initializers: 0")
    print(
        f"Applied onnxslim: {len(model.graph.node)} -> {final_node_count} nodes; "
        f"{len(model.graph.initializer)} -> {final_initializer_count} initializers"
    )
    print(f"Deleted stale value_info entries: {stale_value_count}")
    print(f"Nodes: {original_node_count} -> {final_node_count}")
    print(f"Added opset import: {MICROSOFT_DOMAIN} {MICROSOFT_OPSET}")
    print(f"Internal precision: {'FP16' if use_fp16 else 'FP32'}")
    print_runtime_metrics(metrics)
    print(
        f"Validated {validated_cases} raw/final cases on ONNX Runtime "
        f"{ORT_VERSION} CPUExecutionProvider with optimizations disabled"
    )
    if raw_path == final_path:
        print(f"Optimized model replaced in place: {final_path}")
    elif remove_source:
        print(f"Removed source model after validated replacement: {raw_path}")
    else:
        print(f"Preserved source model: {raw_path}")
    print(f"Final model written: {final_path}")


def optimize_exported_models(
    model_folder=MODEL_FOLDER,
    optimized_folder=OPTIMIZED_FOLDER,
    use_fp16=ENABLE_FP16,
):
    model_folder = Path(model_folder).resolve()
    optimized_folder = Path(optimized_folder).resolve()
    if model_folder == optimized_folder:
        raise ValueError("Source and optimized model folders must be different")
    discovered_paths = sorted(
        path
        for path in model_folder.glob("*.onnx")
        if not path.stem.endswith("_Metadata")
    )
    if not discovered_paths:
        raise FileNotFoundError(f"No ONNX models found in {model_folder}")

    raw_paths = [path for path in discovered_paths if path.stem.endswith("_raw")]
    raw_targets = {
        path.with_name(f"{path.stem.removesuffix('_raw')}.onnx")
        for path in raw_paths
    }
    jobs = [
        (
            path,
            optimized_folder / f"{path.stem.removesuffix('_raw')}.onnx",
        )
        for path in raw_paths
    ]
    jobs.extend(
        (path, optimized_folder / path.name)
        for path in discovered_paths
        if path not in raw_paths and path not in raw_targets
    )
    final_paths = sorted({final_path for _, final_path in jobs})
    if len(final_paths) != len(jobs):
        raise ValueError("Multiple exported models map to the same final ONNX path")

    optimized_folder.mkdir(parents=True, exist_ok=True)
    print(f"Using installed ONNX Runtime {ORT_VERSION}")
    print(f"Found {len(jobs)} exported model(s) in {model_folder}")
    print(f"Writing optimized models to {optimized_folder}")
    for source_path, final_path in jobs:
        model_type = model_type_from_metadata(source_path)
        model = onnx.load(source_path, load_external_data=True)
        already_optimized = any(
            node.domain == MICROSOFT_DOMAIN and node.op_type == "FusedConv"
            for node in model.graph.node
        )
        if already_optimized:
            source_digest = file_sha256(source_path)
            expected_interface = graph_interface(model)
            expected_metadata = metadata_properties(model)
            floating_initializer_types = {
                item.data_type
                for item in model.graph.initializer
                if item.data_type in (TensorProto.FLOAT, TensorProto.FLOAT16)
            }
            if floating_initializer_types == {TensorProto.FLOAT16}:
                source_is_fp16 = True
            elif floating_initializer_types == {TensorProto.FLOAT}:
                source_is_fp16 = False
            else:
                raise ValueError(
                    "Expected one floating-point initializer precision, got "
                    f"{floating_initializer_types}"
                )
            source_is_slimmed = not any(
                node.domain in ("", "ai.onnx") and node.op_type == "Constant"
                for node in model.graph.node
            )
            if source_is_slimmed:
                validate_final_graph(model, model_type, use_fp16=source_is_fp16)
            elif source_is_fp16:
                validate_rewritten_fp16_graph(model, model_type)
            else:
                validate_rewritten_graph(model, model_type)
            convert_to_fp16 = use_fp16 and not source_is_fp16
            if convert_to_fp16:
                model, _ = convert_graph_to_fp16(
                    model,
                    force_fp16_initializers=True,
                )
            final_is_fp16 = source_is_fp16 or convert_to_fp16
            pre_slim_node_count = len(model.graph.node)
            pre_slim_initializer_count = len(model.graph.initializer)
            staged_model, validated_cases, metrics = publish_model(
                model,
                source_path,
                final_path,
                model_type,
                final_is_fp16=final_is_fp16,
                compare_as_fp16=convert_to_fp16,
                expected_interface=expected_interface,
                expected_metadata=expected_metadata,
                source_digest=source_digest,
                prefix="firered_validated_",
            )
            print(
                f"Applied onnxslim: {pre_slim_node_count} -> "
                f"{len(staged_model.graph.node)} nodes; "
                f"{pre_slim_initializer_count} -> "
                f"{len(staged_model.graph.initializer)} initializers"
            )
            print_runtime_metrics(metrics)
            if convert_to_fp16:
                print(
                    f"Converted to FP16 and validated {validated_cases} case(s): "
                    f"{final_path}"
                )
            elif source_is_fp16:
                print(
                    f"Slimmed FP16 graph and validated {validated_cases} case(s): "
                    f"{final_path}"
                )
            else:
                print(
                    f"Slimmed FP32 graph and validated {validated_cases} case(s): "
                    f"{final_path}"
                )
        else:
            optimize_graph(
                source_path,
                final_path,
                model_type,
                remove_source=False,
                use_fp16=use_fp16,
            )

    remaining_source_models = sorted(
        path
        for path in model_folder.glob("*.onnx")
        if not path.stem.endswith("_Metadata")
    )
    if remaining_source_models != discovered_paths:
        raise RuntimeError(
            f"Source model inventory changed unexpectedly: {remaining_source_models}"
        )
    remaining_optimized_models = sorted(
        path
        for path in optimized_folder.glob("*.onnx")
        if not path.stem.endswith("_Metadata")
    )
    if remaining_optimized_models != final_paths:
        raise RuntimeError(
            "Expected only final deployment models in the optimized folder, "
            f"found: {remaining_optimized_models}"
        )
    print(f"Optimized and validated {len(final_paths)} model(s)")


def main():
    optimize_exported_models()


if __name__ == "__main__":
    main()
