"""Shared ONNX optimization pipeline for VAD export scripts."""

from __future__ import annotations

import gc
import hashlib
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import onnx
import onnx.version_converter
from onnx import TensorProto
from onnxruntime.quantization import QuantType, quant_utils, quantize_dynamic
from onnxslim import slim

from vad_onnx_metadata import preserve_optimized_metadata, read_source_metadata


NodeSelector = list[str] | Callable[[str], list[str] | None] | None
IntValue = int | Callable[[str], int]
NoShapeInfer = bool | Literal["auto"]
SymbolicShapeInfer = bool | Literal["auto"]

_DYNAMIC_METHODS = {"DYNAMIC", "DYNAMIC_Q8"}
_VALID_METHODS = {"F32", "F16", *_DYNAMIC_METHODS}
_DYNAMIC_WEIGHT_TYPES = {"QUINT8": QuantType.QUInt8, "QINT8": QuantType.QInt8}
_DEFAULT_F16_OP_BLOCK_LIST = [
    "DynamicQuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeMatMul",
    "Range",
    "MatMulIntegerToFloat",
]


def tensor_shape(value_info) -> tuple[int | str | None, ...]:
    """Return an ONNX tensor shape without discarding symbolic dimensions."""
    dimensions = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return tuple(dimensions)


def graph_interface(model) -> dict[str, tuple[tuple[str, int, tuple], ...]]:
    """Describe public graph names, element types, and shapes."""
    def describe(values):
        return tuple(
            (value.name, value.type.tensor_type.elem_type, tensor_shape(value))
            for value in values
        )

    return {
        "inputs": describe(model.graph.input),
        "outputs": describe(model.graph.output),
    }


def operator_histogram(model) -> dict[tuple[str, str], int]:
    """Count graph operators by normalized domain and type."""
    return dict(
        Counter((node.domain or "ai.onnx", node.op_type) for node in model.graph.node)
    )


def graph_maps(model) -> tuple[dict[str, object], dict[str, list]]:
    """Build validated tensor producer and consumer maps."""
    producers = {}
    consumers = {}
    for node in model.graph.node:
        for output_name in node.output:
            if output_name in producers:
                raise ValueError(f"Tensor {output_name!r} has multiple producers")
            producers[output_name] = node
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    return producers, consumers


def file_sha256(path: str | Path) -> str:
    """Hash a model without loading the complete file into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_properties(model) -> tuple[tuple[str, str], ...]:
    return tuple((item.key, item.value) for item in model.metadata_props)


def preserve_model_properties(source_path: str | Path, target_path: str | Path) -> None:
    """Restore model-level provenance and metadata after a graph optimizer runs."""
    source_model = onnx.load(source_path, load_external_data=False)
    target_model = onnx.load(target_path, load_external_data=False)
    target_model.producer_name = source_model.producer_name
    target_model.producer_version = source_model.producer_version
    target_model.domain = source_model.domain
    target_model.model_version = source_model.model_version
    target_model.doc_string = source_model.doc_string
    del target_model.metadata_props[:]
    for item in source_model.metadata_props:
        target_model.metadata_props.add(key=item.key, value=item.value)
    onnx.checker.check_model(target_model)
    onnx.save(target_model, target_path)


@contextmanager
def atomic_model_path(final_path: str | Path, prefix: str) -> Generator[Path]:
    """Yield a temporary model path and publish it only after the body succeeds."""
    final_path = Path(final_path).resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=prefix, dir=final_path.parent) as directory:
        staged_path = Path(directory) / final_path.name
        yield staged_path
        if not staged_path.is_file():
            raise RuntimeError(f"Optimization did not create staged model {staged_path}")
        os.replace(staged_path, final_path)


def simplify_onnx(
    source_path: str | Path,
    target_path: str | Path | None = None,
    *,
    no_shape_infer: bool,
) -> onnx.ModelProto:
    """Run the standard onnxslim pass used by validated deployment graphs."""
    target_path = Path(source_path if target_path is None else target_path)
    slim(
        model=str(source_path),
        output_model=str(target_path),
        no_shape_infer=no_shape_infer,
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=False,
        verbose=False,
    )
    return onnx.load(target_path, load_external_data=True)


@dataclass
class Plan:
    """Per-model optimization recipe; ``None`` fields inherit config defaults."""

    method: str = "F32"  # F32 | F16 | DYNAMIC | DYNAMIC_Q8
    num_heads: IntValue = 0
    hidden_size: IntValue = 0
    opt_level: int | None = None
    only_onnxruntime: bool | None = None
    optimize: bool = True
    transformer: bool = True
    fp16: bool = False
    external: bool | None = None
    first_slim_no_shape_infer: NoShapeInfer | None = None
    second_slim_no_shape_infer: NoShapeInfer | None = None
    fp16_symbolic_shape_infer: SymbolicShapeInfer | None = None
    upgrade_opset: int | None = None
    nodes_to_exclude: NodeSelector = None
    nodes_to_include: NodeSelector = None
    dynamic_weight_type: str | None = None
    per_channel: bool | None = None
    reduce_range: bool | None = None
    default_tensor_type: int | None = None
    f16_node_block_list: list[str] | None = None
    f16_op_block_list: list[str] | None = None


@dataclass
class OptimizerConfig:
    """Global defaults shared by one thin ``Optimize_ONNX.py`` wrapper."""

    original_folder_path: str
    optimized_folder_path: str
    model_plans: dict[str, Plan]
    force_external_data: bool = False
    upgrade_opset: int = 0
    skip_missing: bool = True
    optimizer_level: int = 2
    optimizer_model_type: str = "bert"
    optimizer_only_onnxruntime: bool = False
    optimizer_fusion_options: dict | None = None
    first_slim_no_shape_infer: NoShapeInfer = False
    second_slim_no_shape_infer: NoShapeInfer | None = None
    slim_skip_fusion_patterns: object = False
    slim_skip_optimizations: object = None
    slim_size_threshold: int | None = None
    slim_no_constant_folding: bool = False
    dynamic_weight_type: str = "QInt8"
    dynamic_per_channel: bool = True
    dynamic_reduce_range: bool = False
    dynamic_default_tensor_type: int | None = None
    dynamic_nodes_to_exclude: NodeSelector = None
    dynamic_nodes_to_include: NodeSelector = None
    f16_keep_io_types: bool = False
    f16_force_initializers: bool = True
    f16_min_positive_val: float = 1e-7
    f16_max_finite_val: float = 32767.0
    f16_symbolic_shape_infer: SymbolicShapeInfer = True
    f16_node_block_list: list[str] | None = None
    f16_op_block_list: list[str] | None = None


@dataclass
class ResolvedPlan:
    method: str
    num_heads: IntValue
    hidden_size: IntValue
    opt_level: int | None
    only_onnxruntime: bool
    optimize: bool
    transformer: bool
    fp16: bool
    external: bool
    first_slim_no_shape_infer: NoShapeInfer
    second_slim_no_shape_infer: NoShapeInfer | None
    fp16_symbolic_shape_infer: SymbolicShapeInfer
    upgrade_opset: int
    nodes_to_exclude: NodeSelector
    nodes_to_include: NodeSelector
    dynamic_weight_type: str
    per_channel: bool
    reduce_range: bool
    default_tensor_type: int | None
    f16_node_block_list: list[str] | None
    f16_op_block_list: list[str]


def _pick(value, default):
    return default if value is None else value


def _uses_fp16(plan: Plan) -> bool:
    return bool(plan.fp16 or plan.method.upper() == "F16")


def resolve_plan(plan: Plan, config: OptimizerConfig) -> ResolvedPlan:
    return ResolvedPlan(
        method=plan.method.upper(),
        num_heads=plan.num_heads,
        hidden_size=plan.hidden_size,
        opt_level=plan.opt_level,
        only_onnxruntime=_pick(plan.only_onnxruntime, config.optimizer_only_onnxruntime),
        optimize=plan.optimize,
        transformer=plan.transformer,
        fp16=plan.fp16,
        external=_pick(plan.external, config.force_external_data),
        first_slim_no_shape_infer=_pick(plan.first_slim_no_shape_infer, config.first_slim_no_shape_infer),
        second_slim_no_shape_infer=_pick(plan.second_slim_no_shape_infer, config.second_slim_no_shape_infer),
        fp16_symbolic_shape_infer=_pick(plan.fp16_symbolic_shape_infer, config.f16_symbolic_shape_infer),
        upgrade_opset=config.upgrade_opset if plan.upgrade_opset is None else plan.upgrade_opset,
        nodes_to_exclude=_pick(plan.nodes_to_exclude, config.dynamic_nodes_to_exclude),
        nodes_to_include=_pick(plan.nodes_to_include, config.dynamic_nodes_to_include),
        dynamic_weight_type=_pick(plan.dynamic_weight_type, config.dynamic_weight_type).upper(),
        per_channel=_pick(plan.per_channel, config.dynamic_per_channel),
        reduce_range=_pick(plan.reduce_range, config.dynamic_reduce_range),
        default_tensor_type=_pick(plan.default_tensor_type, config.dynamic_default_tensor_type),
        f16_node_block_list=_pick(plan.f16_node_block_list, config.f16_node_block_list),
        f16_op_block_list=_pick(plan.f16_op_block_list, config.f16_op_block_list)
        or list(_DEFAULT_F16_OP_BLOCK_LIST),
    )


def validate_plan(name: str, plan: ResolvedPlan) -> None:
    if plan.method not in _VALID_METHODS:
        raise ValueError(
            f"[{name}] unknown method {plan.method!r}; use 'F32', 'F16', "
            "'DYNAMIC', or 'DYNAMIC_Q8'."
        )
    if plan.method in _DYNAMIC_METHODS and plan.dynamic_weight_type not in _DYNAMIC_WEIGHT_TYPES:
        raise ValueError(f"[{name}] unknown dynamic_weight_type; choose 'QUInt8' or 'QInt8'.")


def _remove_external_files(model_path: str) -> None:
    for path in (model_path, model_path + ".data"):
        if os.path.exists(path):
            os.remove(path)


def _save_model(model, model_path: str, external: bool) -> None:
    _remove_external_files(model_path)
    if external:
        onnx.save(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(model_path) + ".data",
        )
    else:
        onnx.save(model, model_path)


def _iter_all_data_tensors(graph):
    yield from graph.initializer
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            yield from attribute.tensors
            if attribute.HasField("g"):
                yield from _iter_all_data_tensors(attribute.g)
            for subgraph in attribute.graphs:
                yield from _iter_all_data_tensors(subgraph)


def _retarget_external_location(model_path: str, new_location: str) -> None:
    model = onnx.load(model_path, load_external_data=False)
    for tensor in _iter_all_data_tensors(model.graph):
        if tensor.data_location == TensorProto.EXTERNAL:
            for entry in tensor.external_data:
                if entry.key == "location":
                    entry.value = new_location
    onnx.save(model, model_path)
    del model
    gc.collect()


def resave(src_path: str, dst_path: str, external: bool) -> None:
    model = onnx.load(src_path)
    _save_model(model, dst_path, external)
    del model
    gc.collect()


def _shape_value_is_dynamic(value) -> bool:
    return value is None or isinstance(value, str)


def detect_dynamic_axes(model_path: str) -> bool:
    try:
        import onnxruntime

        session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        try:
            return any(
                _shape_value_is_dynamic(dimension)
                for metadata in session.get_inputs()
                for dimension in metadata.shape
            )
        finally:
            del session
            gc.collect()
    except Exception:
        model = onnx.load(model_path, load_external_data=False)
        try:
            for value_info in model.graph.input:
                for dimension in value_info.type.tensor_type.shape.dim:
                    if dimension.dim_param or not dimension.HasField("dim_value"):
                        return True
            return False
        finally:
            del model
            gc.collect()


def _resolve_no_shape_infer(setting: NoShapeInfer, dynamic_axes: bool) -> bool:
    return dynamic_axes if setting == "auto" else bool(setting)


def _resolve_symbolic_shape_infer(setting: SymbolicShapeInfer, dynamic_axes: bool) -> bool:
    return not dynamic_axes if setting == "auto" else bool(setting)


def run_onnxslim(
    model_path: str,
    external: bool,
    config: OptimizerConfig,
    no_shape_infer: bool,
) -> None:
    def run_slim() -> None:
        slim(
            model=model_path,
            output_model=model_path,
            no_shape_infer=no_shape_infer,
            skip_fusion_patterns=config.slim_skip_fusion_patterns,
            skip_optimizations=config.slim_skip_optimizations,
            no_constant_folding=config.slim_no_constant_folding,
            size_threshold=config.slim_size_threshold,
            save_as_external_data=external,
            verbose=False,
        )

    data_path = model_path + ".data"
    if not external or not os.path.exists(data_path):
        run_slim()
        return

    stash_path = model_path + ".stash.data"
    if os.path.exists(stash_path):
        os.remove(stash_path)
    os.replace(data_path, stash_path)
    _retarget_external_location(model_path, os.path.basename(stash_path))
    try:
        run_slim()
    except BaseException:
        if not os.path.exists(data_path):
            os.replace(stash_path, data_path)
            _retarget_external_location(model_path, os.path.basename(data_path))
        raise
    finally:
        if os.path.exists(stash_path):
            os.remove(stash_path)


def build_fusion_options(config: OptimizerConfig):
    if not config.optimizer_fusion_options:
        return None
    from onnxruntime.transformers.fusion_options import FusionOptions

    options = FusionOptions(config.optimizer_model_type)
    for key, value in config.optimizer_fusion_options.items():
        setattr(options, key, value)
    return options


def deduplicate_node_names(graph) -> int:
    used_names: set[str] = set()
    next_name_suffix: dict[str, int] = {}
    used_values = {item.name for item in graph.input}
    used_values.update(item.name for item in graph.initializer)
    next_value_suffix: dict[str, int] = {}
    remap: dict[str, str] = {}
    renamed = 0

    for node in graph.node:
        for index, name in enumerate(node.input):
            if name in remap:
                node.input[index] = remap[name]

        if node.name:
            if node.name not in used_names:
                used_names.add(node.name)
            else:
                base_name = node.name
                suffix = next_name_suffix.get(base_name, 1)
                while f"{base_name}_{suffix}" in used_names:
                    suffix += 1
                node.name = f"{base_name}_{suffix}"
                used_names.add(node.name)
                next_name_suffix[base_name] = suffix + 1
                renamed += 1

        for index, output in enumerate(node.output):
            if not output:
                continue
            if output not in used_values:
                used_values.add(output)
                continue
            suffix = next_value_suffix.get(output, 1)
            while f"{output}_{suffix}" in used_values:
                suffix += 1
            new_output = f"{output}_{suffix}"
            node.output[index] = new_output
            used_values.add(new_output)
            next_value_suffix[output] = suffix + 1
            remap[output] = new_output
            renamed += 1
    return renamed


def convert_graph_to_fp16(
    model: onnx.ModelProto,
    *,
    force_fp16_initializers: bool,
    op_block_list: list[str] | None = None,
    node_block_list: list[str] | None = None,
    use_symbolic_shape_infer: bool = False,
    check_custom_domain: bool = False,
) -> tuple[onnx.ModelProto, int]:
    """Convert internal compute to FP16 and verify immutable model contracts."""
    from onnxruntime.transformers.onnx_model import OnnxModel

    original_interface = graph_interface(model)
    original_initializer_names = tuple(item.name for item in model.graph.initializer)
    original_initializer_types = {
        item.name: item.data_type for item in model.graph.initializer
    }
    original_metadata = metadata_properties(model)

    wrapped_model = OnnxModel(model)
    wrapped_model.convert_float_to_float16(
        keep_io_types=True,
        force_fp16_initializers=force_fp16_initializers,
        use_symbolic_shape_infer=use_symbolic_shape_infer,
        max_finite_val=32767.0,
        min_positive_val=1e-7,
        op_block_list=op_block_list,
        node_block_list=node_block_list,
    )
    renamed = deduplicate_node_names(wrapped_model.model.graph)
    if renamed:
        print(f"  Renamed {renamed} duplicate node/value names after FP16 conversion.")
    wrapped_model.topological_sort()
    converted = wrapped_model.model

    if graph_interface(converted) != original_interface:
        raise RuntimeError("FP16 conversion changed the public graph interface")
    if tuple(item.name for item in converted.graph.initializer) != original_initializer_names:
        raise RuntimeError("FP16 conversion changed initializer names or ordering")
    if metadata_properties(converted) != original_metadata:
        raise RuntimeError("FP16 conversion changed model metadata")

    converted_initializers = 0
    for item in converted.graph.initializer:
        original_type = original_initializer_types[item.name]
        if original_type == TensorProto.FLOAT:
            if item.data_type == TensorProto.FLOAT16:
                converted_initializers += 1
            elif item.data_type != TensorProto.FLOAT:
                raise RuntimeError(
                    f"FP16 conversion gave float initializer {item.name!r} "
                    f"unexpected type {item.data_type}"
                )
        elif item.data_type != original_type:
            raise RuntimeError(
                f"FP16 conversion changed non-float initializer {item.name!r}"
            )
    if converted_initializers == 0:
        raise RuntimeError("FP16 conversion produced no float16 initializers")
    onnx.checker.check_model(converted, check_custom_domain=check_custom_domain)
    return converted, converted_initializers


def _resolve_int(value: IntValue, src_path: str) -> int:
    return int(value(src_path)) if callable(value) else int(value)


def _resolve_nodes(selector: NodeSelector, src_path: str) -> list[str] | None:
    nodes = selector(src_path) if callable(selector) else selector
    return nodes or None


def optimize_onnx_model(
    model_path: str,
    plan: ResolvedPlan,
    config: OptimizerConfig,
    src_path: str,
    use_fp16: bool,
    external: bool,
    dynamic_axes: bool,
) -> None:
    from onnxruntime.transformers.optimizer import optimize_model

    opt_level = config.optimizer_level if plan.opt_level is None else plan.opt_level
    kwargs = {
        "use_gpu": False,
        "opt_level": opt_level,
        "num_heads": _resolve_int(plan.num_heads, src_path),
        "hidden_size": _resolve_int(plan.hidden_size, src_path),
        "optimization_options": build_fusion_options(config),
        "model_type": config.optimizer_model_type,
        "only_onnxruntime": plan.only_onnxruntime,
        "verbose": False,
    }

    if opt_level == 0 and plan.only_onnxruntime:
        from onnxruntime.transformers.onnx_model import OnnxModel

        model = OnnxModel(onnx.load(model_path))
    else:
        model = optimize_model(model_path, **kwargs)
    if use_fp16:
        model.convert_float_to_float16(
            keep_io_types=config.f16_keep_io_types,
            force_fp16_initializers=config.f16_force_initializers,
            use_symbolic_shape_infer=_resolve_symbolic_shape_infer(
                plan.fp16_symbolic_shape_infer,
                dynamic_axes,
            ),
            max_finite_val=config.f16_max_finite_val,
            min_positive_val=config.f16_min_positive_val,
            op_block_list=plan.f16_op_block_list,
            node_block_list=_resolve_nodes(plan.f16_node_block_list, src_path),
        )
        if not any(
            tensor.data_type == TensorProto.FLOAT16
            for tensor in _iter_all_data_tensors(model.model.graph)
        ):
            raise RuntimeError("FP16 conversion produced no float16 model tensors")
        renamed = deduplicate_node_names(model.model.graph)
        if renamed:
            print(f"  Renamed {renamed} duplicate node names after float16 conversion.")
    model.save_model_to_file(model_path, use_external_data_format=external)
    del model
    gc.collect()


def quantize_dynamic_int8(
    src_path: str,
    dst_path: str,
    plan: ResolvedPlan,
    external: bool,
) -> None:
    extra_options = {
        "ActivationSymmetric": False,
        "WeightSymmetric": False,
        "EnableSubgraph": True,
        "ForceQuantizeNoInputCheck": False,
        "MatMulConstBOnly": True,
    }
    if plan.default_tensor_type is not None:
        extra_options["DefaultTensorType"] = plan.default_tensor_type
    print(
        f"  Quantizing weights (dynamic INT8, {plan.dynamic_weight_type}, "
        f"per_channel={plan.per_channel}, reduce_range={plan.reduce_range})..."
    )
    model = quant_utils.load_model_with_shape_infer(Path(src_path))
    quantize_dynamic(
        model_input=model,
        model_output=dst_path,
        per_channel=plan.per_channel,
        reduce_range=plan.reduce_range,
        weight_type=_DYNAMIC_WEIGHT_TYPES[plan.dynamic_weight_type],
        extra_options=extra_options,
        nodes_to_quantize=_resolve_nodes(plan.nodes_to_include, src_path),
        nodes_to_exclude=_resolve_nodes(plan.nodes_to_exclude, src_path),
        use_external_data_format=external,
    )
    del model
    gc.collect()


def upgrade_opset_version(model_path: str, version: int, external: bool) -> None:
    print(f"  Upgrading opset to {version}...")
    try:
        model = onnx.version_converter.convert_version(onnx.load(model_path), version)
        _save_model(model, model_path, external)
        del model
        gc.collect()
    except Exception as exc:
        print(f"  Opset upgrade failed: {exc}. Keeping current version.")
        resave(model_path, model_path, external)


def get_model_paths(config: OptimizerConfig, name: str) -> tuple[str, str]:
    return (
        os.path.join(config.original_folder_path, f"{name}.onnx"),
        os.path.join(config.optimized_folder_path, f"{name}.onnx"),
    )


def process_model(name: str, plan: ResolvedPlan, config: OptimizerConfig) -> bool:
    src_path, dst_path = get_model_paths(config, name)
    if not os.path.exists(src_path):
        message = f"  Skipping - file not found: {src_path}"
        if config.skip_missing:
            print(message)
            return False
        raise FileNotFoundError(message)

    source_metadata = read_source_metadata(src_path)
    _remove_external_files(dst_path)
    external = plan.external
    use_fp16 = plan.fp16 or plan.method == "F16"
    dynamic_axes = detect_dynamic_axes(src_path)
    first_no_shape = _resolve_no_shape_infer(plan.first_slim_no_shape_infer, dynamic_axes)
    second_setting = plan.second_slim_no_shape_infer
    if second_setting is None:
        second_setting = plan.first_slim_no_shape_infer
    second_no_shape = _resolve_no_shape_infer(second_setting, dynamic_axes)

    print(f"  Source: {src_path}")
    print(f"  Output: {dst_path}")
    if dynamic_axes:
        print("  Dynamic axes detected; using configured dynamic-safe shape inference.")

    resave(src_path, dst_path, external)
    if plan.optimize:
        print("  Simplifying (onnxslim, pass 1)...")
        run_onnxslim(dst_path, external, config, no_shape_infer=first_no_shape)

    if plan.method in _DYNAMIC_METHODS:
        quantize_dynamic_int8(dst_path, dst_path, plan, external)

    if plan.optimize and (plan.transformer or use_fp16):
        print("  Optimizing (ORT transformers optimizer)...")
        optimize_onnx_model(
            dst_path,
            plan,
            config,
            src_path,
            use_fp16,
            external,
            dynamic_axes,
        )
        print("  Simplifying (onnxslim, pass 2)...")
        run_onnxslim(dst_path, external, config, no_shape_infer=second_no_shape)

    if plan.upgrade_opset > 0:
        upgrade_opset_version(dst_path, plan.upgrade_opset, external)

    if not external and os.path.exists(dst_path + ".data"):
        os.remove(dst_path + ".data")
    preserve_optimized_metadata(src_path, dst_path, source_metadata)
    return True


def run_optimizer(config: OptimizerConfig) -> None:
    os.makedirs(config.optimized_folder_path, exist_ok=True)
    resolved = {
        name: resolve_plan(plan, config)
        for name, plan in config.model_plans.items()
    }
    for name, plan in resolved.items():
        validate_plan(name, plan)

    for name in resolved:
        _, dst_path = get_model_paths(config, name)
        _remove_external_files(dst_path)

    if any(_uses_fp16(plan) for plan in config.model_plans.values()):
        print(
            "FP16 conversion is enabled explicitly. Validate each model with "
            "realistic audio before deployment."
        )

    processed = 0
    for name, plan in resolved.items():
        suffix = " + FP16" if plan.fp16 else ""
        print(f"\n{'=' * 60}\nProcessing: {name}  [{plan.method}{suffix}]\n{'=' * 60}")
        processed += int(process_model(name, plan, config))

    print(f"\n--- Processed {processed} model(s) successfully! ---")
