"""Targeted rewrites for legacy-exporter limitations in DFSMN VAD."""

from collections import Counter, defaultdict
from pathlib import Path
import tempfile

import onnx
from onnx import shape_inference


def _interface_signature(model):
    return (
        tuple(value.SerializeToString() for value in model.graph.input),
        tuple(value.SerializeToString() for value in model.graph.output),
        tuple((item.domain, item.version) for item in model.opset_import),
    )


def _op_histogram(model):
    return dict(sorted(Counter(
        node.op_type for node in model.graph.node).items()))


def rewrite_nkf_initializer_identities(
        raw_model_path, final_model_path, *, frames, model_batch, n_freq,
        filter_order, fc_dim, rnn_dim):
    """Remove only validated NKF initializer aliases from a static graph."""
    raw_model_path = Path(raw_model_path).expanduser().resolve()
    final_model_path = Path(final_model_path).expanduser().resolve()
    if raw_model_path == final_model_path:
        raise ValueError('Raw and final ONNX paths must be distinct.')
    if not raw_model_path.is_file():
        raise FileNotFoundError(raw_model_path)
    if min(
            frames, model_batch, n_freq, filter_order, fc_dim, rnn_dim) <= 0:
        raise ValueError('All NKF static dimensions must be positive integers.')

    model = onnx.load(str(raw_model_path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    original_interface = _interface_signature(model)
    before_histogram = _op_histogram(model)
    before_nodes = len(model.graph.node)
    initializers = {
        tensor.name: tensor for tensor in model.graph.initializer}
    graph_outputs = {value.name for value in model.graph.output}
    consumers = defaultdict(list)
    for node in model.graph.node:
        for input_index, input_name in enumerate(node.input):
            if input_name:
                consumers[input_name].append((node, input_index))

    gru_aliases = 2 * frames - 1
    linear_aliases = frames - 1
    expected = Counter({
        ('GRU', 1, (1, 3 * rnn_dim, fc_dim)): 2 * gru_aliases,
        ('GRU', 2, (1, 3 * rnn_dim, rnn_dim)): 2 * gru_aliases,
        ('GRU', 3, (1, 6 * rnn_dim)): 2 * gru_aliases,
        ('MatMul', 1, (2 * filter_order + 1, fc_dim)): 2 * linear_aliases,
        ('MatMul', 1, (rnn_dim, fc_dim)): 2 * linear_aliases,
        ('MatMul', 1, (fc_dim, filter_order)): 2 * linear_aliases,
    })
    targeted_signatures = set(expected)
    expected_identity_count = 6 * gru_aliases + 6 * linear_aliases

    identities = []
    replacements = {}
    observed = Counter()
    for node in model.graph.node:
        if node.op_type != 'Identity':
            continue
        if node.domain or len(node.input) != 1 or len(node.output) != 1:
            raise ValueError(
                f'Unexpected Identity signature: {node.name or node.output[0]}')
        if node.attribute:
            raise ValueError(
                f'Identity has unexpected attributes: '
                f'{node.name or node.output[0]}')

        source_name, output_name = node.input[0], node.output[0]
        initializer = initializers.get(source_name)
        if initializer is None:
            continue
        shape = tuple(int(dimension) for dimension in initializer.dims)
        output_consumers = consumers.get(output_name, ())
        consumer_signatures = [
            (consumer.op_type, input_index, shape)
            for consumer, input_index in output_consumers
        ]
        targeted = [
            signature in targeted_signatures
            for signature in consumer_signatures
        ]
        if not any(targeted):
            continue
        if not all(targeted):
            raise ValueError(
                f'NKF initializer alias {node.name or output_name} mixes '
                f'targeted and untargeted consumers: {consumer_signatures}')
        if initializer.data_type != onnx.TensorProto.FLOAT:
            raise ValueError(
                f'NKF initializer alias is not FLOAT: '
                f'{node.name or output_name}')
        if output_name in graph_outputs:
            raise ValueError(
                f'Refusing graph-output Identity: {node.name or output_name}')
        if not output_consumers:
            raise ValueError(
                f'NKF initializer alias has no consumers: '
                f'{node.name or output_name}')
        observed.update(consumer_signatures)
        identities.append(node)
        replacements[output_name] = source_name

    if not identities:
        raise ValueError(
            'No NKF initializer aliases found; input appears already rewritten.')
    if len(identities) != expected_identity_count:
        raise ValueError(
            f'Expected {expected_identity_count} NKF initializer aliases, '
            f'found {len(identities)}.')
    if observed != expected:
        raise ValueError(
            'NKF initializer alias topology mismatch.\n'
            f'Expected: {dict(expected)}\nObserved: {dict(observed)}')

    identity_ids = {id(node) for node in identities}
    retained_nodes = []
    rewired_inputs = 0
    for node in model.graph.node:
        if id(node) in identity_ids:
            continue
        for input_index, input_name in enumerate(node.input):
            replacement = replacements.get(input_name)
            if replacement is not None:
                node.input[input_index] = replacement
                rewired_inputs += 1
        retained_nodes.append(node)
    if rewired_inputs != sum(expected.values()):
        raise AssertionError(
            f'Expected {sum(expected.values())} rewired inputs, '
            f'got {rewired_inputs}.')

    del model.graph.node[:]
    model.graph.node.extend(retained_nodes)
    if _interface_signature(model) != original_interface:
        raise ValueError(
            'Graph interface or opset imports changed during NKF rewrite.')
    onnx.checker.check_model(model, full_check=True)
    inferred = shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(inferred, full_check=True)

    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            prefix=final_model_path.name + '.', suffix='.tmp',
            dir=final_model_path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        onnx.save_model(model, str(temporary_path))
        temporary_path.replace(final_model_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        'raw_nodes': before_nodes,
        'final_nodes': len(model.graph.node),
        'raw_op_histogram': before_histogram,
        'final_op_histogram': _op_histogram(model),
        'identity_rewrites': len(identities),
        'rewired_inputs': rewired_inputs,
    }