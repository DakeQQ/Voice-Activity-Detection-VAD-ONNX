"""Merge Community-1's device stages around the host-side AHC boundary."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, helper, numpy_helper


METADATA_KEY = "pyannote.community1"
MODEL_ID = "pyannote/speaker-diarization-community-1"
SEGMENTATION_EMBEDDING_MODEL_NAME = "segmentation_embedding.onnx"
VBX_CLUSTERING_DIARIZATION_MODEL_NAME = "vbx_clustering_diarization.onnx"
SPLIT_MODEL_NAMES = {
    "segmentation": "segmentation.onnx",
    "embedding": "embedding.onnx",
    "vbx": "vbx.onnx",
    "clustering": "clustering.onnx",
    "reconstruction": "reconstruction.onnx",
}
EXACT_CSE_OPERATOR_TYPES = frozenset({"Cast", "Constant", "Shape"})


def _load_component(model_dir: Path, component: str) -> onnx.ModelProto:
    path = model_dir / SPLIT_MODEL_NAMES[component]
    if not path.is_file():
        raise FileNotFoundError(path)
    model = onnx.load(path, load_external_data=True)
    raw_metadata = {
        entry.key: entry.value for entry in model.metadata_props
    }.get(METADATA_KEY)
    if raw_metadata is None:
        raise ValueError(f"{path.name} has no {METADATA_KEY!r} metadata.")
    metadata = json.loads(raw_metadata)
    if metadata.get("component") != component:
        raise ValueError(
            f"Expected {component!r} metadata in {path.name}, got "
            f"{metadata.get('component')!r}."
        )
    if metadata.get("model_id") != MODEL_ID:
        raise ValueError(
            f"Unexpected model_id in {path.name}: {metadata.get('model_id')!r}."
        )
    return model


def _component_metadata(model: onnx.ModelProto) -> dict[str, Any]:
    raw = {entry.key: entry.value for entry in model.metadata_props}[METADATA_KEY]
    return json.loads(raw)


def _prefixed(model: onnx.ModelProto, prefix: str) -> onnx.ModelProto:
    import onnx.compose

    return onnx.compose.add_prefix(
        model,
        prefix,
        rename_nodes=True,
        rename_edges=True,
        rename_inputs=True,
        rename_outputs=True,
        rename_initializers=True,
        rename_value_infos=True,
    )


def _merge_models_no_check(
    first: onnx.ModelProto,
    second: onnx.ModelProto,
    io_map: list[tuple[str, str]],
) -> onnx.ModelProto:
    """Merge prefixed models without mutating either source model."""
    source_by_target = {target: source for source, target in io_map}
    mapped_sources = set(source_by_target.values())
    mapped_targets = set(source_by_target)

    merged = onnx.ModelProto()
    merged.ir_version = max(first.ir_version, second.ir_version)
    merged.producer_name = "Merge_Pyannote_Community1.py"
    merged.graph.name = f"{first.graph.name}_{second.graph.name}_merged"

    opsets: dict[str, int] = {}
    for model in (first, second):
        for opset in model.opset_import:
            opsets[opset.domain] = max(
                opsets.get(opset.domain, 0), opset.version
            )
    for domain, version in sorted(opsets.items()):
        merged.opset_import.add(domain=domain, version=version)

    seen_inputs: set[str] = set()
    inputs = list(first.graph.input) + [
        value
        for value in second.graph.input
        if value.name not in mapped_targets
    ]
    for value in inputs:
        if value.name not in seen_inputs:
            merged.graph.input.append(value)
            seen_inputs.add(value.name)

    initializers: dict[str, TensorProto] = {}
    for initializer in list(first.graph.initializer) + list(
        second.graph.initializer
    ):
        existing = initializers.get(initializer.name)
        if existing is None:
            initializers[initializer.name] = initializer
        elif existing.SerializeToString() != initializer.SerializeToString():
            raise RuntimeError(
                f"Initializer collision with different data: {initializer.name}"
            )
    merged.graph.initializer.extend(initializers.values())

    merged.graph.node.extend(first.graph.node)
    second_start = len(merged.graph.node)
    merged.graph.node.extend(second.graph.node)
    for node in merged.graph.node[second_start:]:
        for index, name in enumerate(node.input):
            node.input[index] = source_by_target.get(name, name)

    seen_value_info = set(seen_inputs) | set(initializers)
    value_infos = list(first.graph.value_info) + list(second.graph.value_info)
    for value in value_infos:
        if value.name not in seen_value_info:
            merged.graph.value_info.append(value)
            seen_value_info.add(value.name)

    seen_outputs: set[str] = set()
    outputs = [
        value
        for value in first.graph.output
        if value.name not in mapped_sources
    ] + list(second.graph.output)
    for value in outputs:
        if value.name not in seen_outputs:
            merged.graph.output.append(value)
            seen_outputs.add(value.name)
    return merged


def _value_info_by_name(
    model: onnx.ModelProto,
) -> dict[str, onnx.ValueInfoProto]:
    values = (
        list(model.graph.input)
        + list(model.graph.output)
        + list(model.graph.value_info)
    )
    return {value.name: value for value in values}


def _set_graph_outputs(
    model: onnx.ModelProto, output_names: list[str]
) -> None:
    by_name = _value_info_by_name(model)
    missing = [name for name in output_names if name not in by_name]
    if missing:
        raise RuntimeError(f"Merged graph has no value_info for: {missing}")
    del model.graph.output[:]
    model.graph.output.extend(by_name[name] for name in output_names)


def _rename_value(model: onnx.ModelProto, old: str, new: str) -> None:
    if old == new:
        return
    for collection in (
        model.graph.input,
        model.graph.output,
        model.graph.value_info,
        model.graph.initializer,
    ):
        for value in collection:
            if value.name == old:
                value.name = new
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new
        for index, name in enumerate(node.output):
            if name == old:
                node.output[index] = new


def _replace_node_input(
    model: onnx.ModelProto, old: str, new: str
) -> None:
    replaced = False
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new
                replaced = True
    if not replaced:
        raise RuntimeError(f"No node consumes merged value {old!r}.")


def _remove_graph_input(model: onnx.ModelProto, name: str) -> None:
    remaining = [value for value in model.graph.input if value.name != name]
    if len(remaining) == len(model.graph.input):
        raise RuntimeError(f"Merged graph has no input named {name!r}.")
    del model.graph.input[:]
    model.graph.input.extend(remaining)


def _order_graph_inputs(
    model: onnx.ModelProto, input_names: list[str]
) -> None:
    by_name = {value.name: value for value in model.graph.input}
    if set(by_name) != set(input_names):
        raise RuntimeError(
            f"Unexpected merged inputs: {sorted(by_name)} != {sorted(input_names)}"
        )
    del model.graph.input[:]
    model.graph.input.extend(by_name[name] for name in input_names)


def _deduplicate_exact_nodes(model: onnx.ModelProto) -> dict[str, int]:
    """Share exact, single-output metadata and conversion expressions."""
    public_outputs = {value.name for value in model.graph.output}
    canonical_outputs: dict[tuple, str] = {}
    replacements: dict[str, str] = {}
    retained_nodes: list[onnx.NodeProto] = []
    removed: Counter[str] = Counter()

    def resolve(name: str) -> str:
        while name in replacements:
            name = replacements[name]
        return name

    for node in model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = resolve(name)
        if (
            node.op_type not in EXACT_CSE_OPERATOR_TYPES
            or len(node.output) != 1
            or not node.output[0]
            or node.output[0] in public_outputs
        ):
            retained_nodes.append(node)
            continue

        signature = (
            node.domain,
            node.op_type,
            tuple(node.input),
            tuple(
                sorted(
                    (attribute.name, attribute.SerializeToString())
                    for attribute in node.attribute
                )
            ),
        )
        canonical_output = canonical_outputs.get(signature)
        if canonical_output is None:
            canonical_outputs[signature] = node.output[0]
            retained_nodes.append(node)
            continue

        replacements[node.output[0]] = canonical_output
        removed[node.op_type] += 1

    del model.graph.node[:]
    model.graph.node.extend(retained_nodes)
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = resolve(name)

    retained_value_info = [
        value
        for value in model.graph.value_info
        if value.name not in replacements
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_value_info)

    stale_inputs = sorted(
        {
            name
            for node in model.graph.node
            for name in node.input
            if name in replacements
        }
    )
    if stale_inputs:
        raise RuntimeError(
            f"Exact CSE left stale node inputs: {stale_inputs}."
        )
    return dict(sorted(removed.items()))


def _prune_unreachable_nodes(model: onnx.ModelProto) -> dict[str, int]:
    before = Counter(node.op_type for node in model.graph.node)
    required_values = {value.name for value in model.graph.output}
    retained_nodes: list[onnx.NodeProto] = []
    for node in reversed(model.graph.node):
        if any(output in required_values for output in node.output):
            retained_nodes.append(node)
            required_values.update(name for name in node.input if name)
    retained_nodes.reverse()
    del model.graph.node[:]
    model.graph.node.extend(retained_nodes)

    graph_inputs = {value.name for value in model.graph.input}
    retained_initializers = [
        value
        for value in model.graph.initializer
        if value.name in required_values or value.name in graph_inputs
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained_initializers)

    available_values = (
        graph_inputs
        | {value.name for value in model.graph.output}
        | {value.name for value in model.graph.initializer}
        | {
            output
            for node in model.graph.node
            for output in node.output
            if output
        }
    )
    retained_value_info = [
        value
        for value in model.graph.value_info
        if value.name in available_values
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_value_info)

    after = Counter(node.op_type for node in model.graph.node)
    return dict(sorted((before - after).items()))


def _use_implicit_lstm_zero_states(model: onnx.ModelProto) -> dict[str, Any]:
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    lstm_nodes = [node for node in model.graph.node if node.op_type == "LSTM"]
    if len(lstm_nodes) != 4:
        raise RuntimeError(
            f"Expected four segmentation LSTMs, found {len(lstm_nodes)}."
        )

    state_names: list[str] = []
    zero_sources: set[str] = set()
    for node in lstm_nodes:
        if len(node.input) < 7 or not node.input[5] or not node.input[6]:
            raise RuntimeError(
                f"LSTM {node.name!r} has no explicit initial zero states."
            )
        for state_name in node.input[5:7]:
            state_names.append(state_name)
            producer = producers.get(state_name)
            if (
                producer is None
                or producer.op_type != "Slice"
                or not producer.input
                or not producer.input[0]
            ):
                raise RuntimeError(
                    f"LSTM state {state_name!r} is not produced by Slice."
                )
            zero_sources.add(producer.input[0])

    if len(set(state_names)) != 8 or len(zero_sources) != 1:
        raise RuntimeError(
            "Expected eight unique LSTM states sliced from one zero tensor."
        )
    zero_source = next(iter(zero_sources))
    zero_producer = producers.get(zero_source)
    if zero_producer is None or zero_producer.op_type != "ConstantOfShape":
        raise RuntimeError(
            f"LSTM state source {zero_source!r} is not ConstantOfShape."
        )
    value_attributes = [
        attribute
        for attribute in zero_producer.attribute
        if attribute.name == "value"
    ]
    if len(value_attributes) != 1:
        raise RuntimeError("LSTM ConstantOfShape has no unique value attribute.")
    zero_value = helper.get_attribute_value(value_attributes[0])
    if (
        zero_value.data_type != TensorProto.FLOAT
        or not bool((numpy_helper.to_array(zero_value) == 0.0).all())
    ):
        raise RuntimeError("LSTM ConstantOfShape is not float32 zero.")

    for node in lstm_nodes:
        node.input[5] = ""
        node.input[6] = ""
    return {
        "lstm_count": len(lstm_nodes),
        "removed_nodes": _prune_unreachable_nodes(model),
    }


def _graph_statistics(model: onnx.ModelProto) -> dict[str, Any]:
    operators: Counter[str] = Counter()

    def visit_graph(graph: onnx.GraphProto) -> None:
        for node in graph.node:
            operators[node.op_type] += 1
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    visit_graph(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for child in attribute.graphs:
                        visit_graph(child)

    visit_graph(model.graph)
    return {
        "node_count": sum(operators.values()),
        "initializer_count": len(model.graph.initializer),
        "operator_histogram": dict(sorted(operators.items())),
    }


def _stamp_and_save(
    model: onnx.ModelProto,
    path: Path,
    metadata: dict[str, Any],
) -> None:
    model.producer_name = "Merge_Pyannote_Community1.py"
    model.doc_string = (
        "Merged device stage of pyannote/speaker-diarization-community-1. "
        "SciPy AHC and optional KMeans remain on the host."
    )
    metadata["graph"] = _graph_statistics(model)
    del model.metadata_props[:]
    entry = model.metadata_props.add()
    entry.key = METADATA_KEY
    entry.value = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    path.with_name(path.name + ".data").unlink(missing_ok=True)
    onnx.save(model, path)
    onnx.checker.check_model(onnx.load(path, load_external_data=False))


def _build_frontend(
    segmentation: onnx.ModelProto,
    embedding: onnx.ModelProto,
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    segmentation_metadata = _component_metadata(segmentation)
    embedding_metadata = _component_metadata(embedding)
    merged = _merge_models_no_check(
        _prefixed(segmentation, "segmentation_"),
        _prefixed(embedding, "embedding_"),
        [
            ("segmentation_chunks", "embedding_chunks"),
            (
                "segmentation_embedding_masks",
                "embedding_embedding_masks",
            ),
            (
                "segmentation_active_speakers",
                "embedding_active_speakers",
            ),
        ],
    )
    _set_graph_outputs(
        merged,
        [
            "segmentation_segmentations",
            "segmentation_inactive_speakers",
            "segmentation_speaker_counts",
            "embedding_embeddings",
            "embedding_train_embeddings",
            "embedding_train_features",
            "embedding_normalized_train_embeddings",
        ],
    )
    renames = {
        "segmentation_waveform": "waveform",
        "segmentation_segmentations": "segmentations",
        "segmentation_inactive_speakers": "inactive_speakers",
        "segmentation_speaker_counts": "speaker_counts",
        "embedding_embeddings": "embeddings",
        "embedding_train_embeddings": "train_embeddings",
        "embedding_train_features": "train_features",
        "embedding_normalized_train_embeddings": (
            "normalized_train_embeddings"
        ),
    }
    for old, new in renames.items():
        _rename_value(merged, old, new)
    _order_graph_inputs(merged, ["waveform"])
    implicit_lstm_zero_state = _use_implicit_lstm_zero_states(merged)
    exact_cse_removed = _deduplicate_exact_nodes(merged)
    merged.graph.name = "community1_frontend"

    metadata = {
        "component": "frontend",
        "pipeline_stage": 1,
        "format_version": 4,
        "model_id": MODEL_ID,
        "merged_components": ["segmentation", "embedding"],
        "host_boundary_after": "normalized_train_embeddings_to_ahc",
        "output_names": [value.name for value in merged.graph.output],
        "sample_rate": int(segmentation_metadata["sample_rate"]),
        "input_dtype": segmentation_metadata["input_dtype"],
        "logical_integer_dtype": "int32",
        "chunk_duration": float(segmentation_metadata["chunk_duration"]),
        "chunk_step": float(segmentation_metadata["chunk_step"]),
        "num_samples": int(segmentation_metadata["num_samples"]),
        "num_frames": int(segmentation_metadata["num_frames"]),
        "mask_num_frames": int(embedding_metadata["mask_num_frames"]),
        "embedding_dimension": int(
            embedding_metadata["embedding_dimension"]
        ),
        "plda_dimension": int(embedding_metadata["plda_dimension"]),
        "implicit_lstm_zero_state": implicit_lstm_zero_state,
        "exact_cse_removed": exact_cse_removed,
    }
    return merged, metadata


def _vbx_selector_nodes() -> list[onnx.NodeProto]:
    zero = helper.make_tensor(
        "selector_zero_value", TensorProto.INT64, [1], [0]
    )
    axis = helper.make_tensor(
        "selector_axis_value", TensorProto.INT64, [1], [1]
    )
    return [
        helper.make_node(
            "Shape",
            ["vbx_centroid_weights"],
            ["vbx_cluster_count"],
            name="vbx_cluster_count",
            start=1,
            end=2,
        ),
        helper.make_node(
            "Shape",
            ["override_centroid_weights"],
            ["override_cluster_count"],
            name="override_cluster_count",
            start=1,
            end=2,
        ),
        helper.make_node(
            "Constant", [], ["selector_zero"], name="selector_zero", value=zero
        ),
        helper.make_node(
            "Constant", [], ["selector_axis"], name="selector_axis", value=axis
        ),
        helper.make_node(
            "Add",
            ["vbx_cluster_count", "override_cluster_count"],
            ["combined_cluster_count"],
            name="combined_cluster_count",
        ),
        helper.make_node(
            "Where",
            ["use_override", "vbx_cluster_count", "selector_zero"],
            ["selected_cluster_start"],
            name="selected_cluster_start",
        ),
        helper.make_node(
            "Where",
            ["use_override", "combined_cluster_count", "vbx_cluster_count"],
            ["selected_cluster_end"],
            name="selected_cluster_end",
        ),
        helper.make_node(
            "Concat",
            ["vbx_centroid_weights", "override_centroid_weights"],
            ["combined_centroid_weights"],
            name="combine_centroid_weights",
            axis=1,
        ),
        helper.make_node(
            "Slice",
            [
                "combined_centroid_weights",
                "selected_cluster_start",
                "selected_cluster_end",
                "selector_axis",
            ],
            ["selected_centroid_weights"],
            name="select_centroid_weights",
        ),
    ]


def _build_backend(
    vbx: onnx.ModelProto,
    clustering: onnx.ModelProto,
    reconstruction: onnx.ModelProto,
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    vbx_metadata = _component_metadata(vbx)
    clustering_metadata = _component_metadata(clustering)
    reconstruction_metadata = _component_metadata(reconstruction)
    clustering_and_reconstruction = _merge_models_no_check(
        _prefixed(clustering, "clustering_"),
        _prefixed(reconstruction, "reconstruction_"),
        [
            (
                "clustering_assignments",
                "reconstruction_assignments",
            )
        ],
    )
    prefixed_vbx = _prefixed(vbx, "vbx_")
    _rename_value(prefixed_vbx, "vbx_train_features", "train_features")
    _rename_value(prefixed_vbx, "vbx_initial_clusters", "initial_clusters")
    _rename_value(
        prefixed_vbx, "vbx_initial_cluster_ids", "initial_cluster_ids"
    )
    vbx_node_count = len(prefixed_vbx.graph.node)
    merged = _merge_models_no_check(
        prefixed_vbx, clustering_and_reconstruction, []
    )
    _set_graph_outputs(
        merged,
        [
            "clustering_centroids",
            "reconstruction_regular_diarization",
            "reconstruction_exclusive_diarization",
        ],
    )
    _replace_node_input(
        merged,
        "clustering_centroid_weights",
        "selected_centroid_weights",
    )
    _remove_graph_input(merged, "clustering_centroid_weights")
    _replace_node_input(
        merged,
        "reconstruction_centroid_weights",
        "selected_centroid_weights",
    )
    _remove_graph_input(merged, "reconstruction_centroid_weights")

    renames = {
        "clustering_embeddings": "embeddings",
        "clustering_train_embeddings": "train_embeddings",
        "clustering_inactive_speakers": "inactive_speakers",
        "clustering_constrained": "constrained",
        "reconstruction_segmentations": "segmentations",
        "reconstruction_speaker_counts": "speaker_counts",
        "clustering_centroids": "centroids",
        "reconstruction_regular_diarization": "regular_diarization",
        "reconstruction_exclusive_diarization": "exclusive_diarization",
    }
    for old, new in renames.items():
        _rename_value(merged, old, new)

    existing_inputs = {value.name for value in merged.graph.input}
    selector_inputs = [
        helper.make_tensor_value_info(
            "override_centroid_weights", TensorProto.FLOAT, [None, None]
        ),
        helper.make_tensor_value_info("use_override", TensorProto.BOOL, []),
    ]
    for value in selector_inputs:
        if value.name not in existing_inputs:
            merged.graph.input.append(value)
    existing_value_info = {
        value.name for value in merged.graph.value_info
    }
    if "selected_centroid_weights" not in existing_value_info:
        merged.graph.value_info.append(
            helper.make_tensor_value_info(
                "selected_centroid_weights",
                TensorProto.FLOAT,
                [None, None],
            )
        )
    nodes = list(merged.graph.node)
    nodes[vbx_node_count:vbx_node_count] = _vbx_selector_nodes()
    del merged.graph.node[:]
    merged.graph.node.extend(nodes)
    _order_graph_inputs(
        merged,
        [
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
        ],
    )
    exact_cse_removed = _deduplicate_exact_nodes(merged)
    merged.graph.name = "community1_backend"

    metadata = {
        "component": "backend",
        "pipeline_stage": 2,
        "format_version": 4,
        "model_id": MODEL_ID,
        "merged_components": ["vbx", "clustering", "reconstruction"],
        "host_boundary_before": "ahc_initial_clusters_and_cluster_ids",
        "supports_centroid_weight_override": True,
        "logical_integer_dtype": "int32",
        "output_names": [value.name for value in merged.graph.output],
        "embedding_dimension": int(
            clustering_metadata["embedding_dimension"]
        ),
        "plda_dimension": int(vbx_metadata["plda_dimension"]),
        "chunk_duration": float(reconstruction_metadata["chunk_duration"]),
        "chunk_step": float(reconstruction_metadata["chunk_step"]),
        "num_frames": int(reconstruction_metadata["num_frames"]),
        "num_local_speakers": int(
            reconstruction_metadata["num_local_speakers"]
        ),
        "receptive_field": reconstruction_metadata["receptive_field"],
        "exact_cse_removed": exact_cse_removed,
    }
    return merged, metadata


def build_merged_models(model_dir: Path) -> dict[str, dict[str, Any]]:
    """Build the two runtime graphs from the five exported constituents."""
    model_dir = Path(model_dir)
    components = {
        component: _load_component(model_dir, component)
        for component in SPLIT_MODEL_NAMES
    }
    frontend, frontend_metadata = _build_frontend(
        components["segmentation"], components["embedding"]
    )
    backend, backend_metadata = _build_backend(
        components["vbx"],
        components["clustering"],
        components["reconstruction"],
    )

    outputs = {
        "frontend": {
            "path": model_dir / SEGMENTATION_EMBEDDING_MODEL_NAME,
            "metadata": frontend_metadata,
        },
        "backend": {
            "path": model_dir / VBX_CLUSTERING_DIARIZATION_MODEL_NAME,
            "metadata": backend_metadata,
        },
    }
    _stamp_and_save(frontend, outputs["frontend"]["path"], frontend_metadata)
    _stamp_and_save(backend, outputs["backend"]["path"], backend_metadata)
    return outputs


def remove_split_models(model_dir: Path) -> list[str]:
    """Remove constituent graphs and their private external-data files."""
    model_dir = Path(model_dir)
    removed: list[str] = []
    for filename in SPLIT_MODEL_NAMES.values():
        path = model_dir / filename
        if not path.is_file():
            continue
        model = onnx.load(path, load_external_data=False)
        external_locations = {
            entry.value
            for initializer in model.graph.initializer
            if initializer.data_location == TensorProto.EXTERNAL
            for entry in initializer.external_data
            if entry.key == "location"
        }
        path.unlink()
        removed.append(path.name)
        for location in sorted(external_locations):
            external_path = model_dir / location
            if external_path.is_file():
                external_path.unlink()
                removed.append(external_path.name)
        conventional_data_path = path.with_name(path.name + ".data")
        if conventional_data_path.is_file():
            conventional_data_path.unlink()
            removed.append(conventional_data_path.name)
    return removed


if __name__ == "__main__":
    build_merged_models(Path(__file__).resolve().parent / "Pyannote_Community1_ONNX")