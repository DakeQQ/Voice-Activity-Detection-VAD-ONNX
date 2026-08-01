import atexit
import gc
import hashlib
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import torch
import torch.nn.functional as F
import torchaudio
import lightning.pytorch.loggers as lightning_loggers

if not hasattr(lightning_loggers, 'NeptuneLogger'):
    class _UnavailableNeptuneLogger:
        pass

    lightning_loggers.NeptuneLogger = _UnavailableNeptuneLogger

# nv_one_logger releases bundled with some NeMo builds narrow Lightning's
# Optional[bool] annotation to bool, which the overrides package rejects.
from lightning.pytorch import Trainer as LightningTrainer

if LightningTrainer.save_checkpoint.__annotations__.get('weights_only') == Optional[bool]:
    LightningTrainer.save_checkpoint.__annotations__['weights_only'] = bool

import nemo.collections.asr as nemo_asr
from STFT_Process import STFT_Process

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "vad_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from vad_onnx_metadata import (
    export_metadata_carrier,
    metadata_path_for_model,
    stamp_export_metadata,
)

script_dir = Path(__file__).resolve().parent

# Export settings
HOME_PATH      = Path.home()
model_path     = HOME_PATH / "Downloads" / "frame_vad_multilingual_marblenet_v2.0.nemo"
export_folder  = script_dir / "NVIDIA_MarbleNet_ONNX"
DYNAMIC_AXES   = False
IN_SAMPLE_RATE = 16000  # External audio rate; accepts any positive sample rate.
OPSET          = 20
IN_AUDIO_DTYPE = 'F32'  # ['F16', 'F32', 'INT16']; float inputs use normalized PCM in [-1, 1].

if IN_SAMPLE_RATE <= 0:
    raise ValueError('IN_SAMPLE_RATE must be positive.')

_INPUT_AUDIO_DTYPES = {
    'F16': (torch.float16, 'float16'),
    'F32': (torch.float32, 'float'),
    'INT16': (torch.int16, 'int16'),
}
IN_AUDIO_DTYPE = IN_AUDIO_DTYPE.upper()
if IN_AUDIO_DTYPE not in _INPUT_AUDIO_DTYPES:
    raise ValueError(
        f"Unsupported IN_AUDIO_DTYPE={IN_AUDIO_DTYPE!r}; "
        f"expected one of {tuple(_INPUT_AUDIO_DTYPES)}."
    )
IN_TORCH_DTYPE, IN_ONNX_DTYPE = _INPUT_AUDIO_DTYPES[IN_AUDIO_DTYPE]

# Fixed MarbleNet frontend parameters
SAMPLE_RATE    = 16000
WINDOW_TYPE    = 'hann_sym'
N_MELS         = 80
NFFT_STFT      = 512
WINDOW_LENGTH  = 400
HOP_LENGTH     = 160
PRE_EMPHASIZE  = 0.97

onnx_model_A = export_folder / "NVIDIA_MarbleNet.onnx"
onnx_model_raw = export_folder / "NVIDIA_MarbleNet_raw.onnx"
onnx_model_Metadata = metadata_path_for_model(onnx_model_A)
export_audio_length = IN_SAMPLE_RATE * 2   # For DYNAMIC_AXES=False, fixed-length export, 2 seconds of audio


def load_nvidia_vad_model(model_file):
    cuda_is_available = torch.cuda.is_available
    try:
        torch.cuda.is_available = lambda: False
        return nemo_asr.models.EncDecSpeakerLabelModel.restore_from(
            restore_path=model_file,
            map_location=torch.device('cpu'),
            strict=False,
        ).float().eval()
    finally:
        torch.cuda.is_available = cuda_is_available


_SHAPE_OPS = {'Shape', 'Size', 'Gather', 'GatherElements', 'GatherND', 'Range', 'ConstantOfShape', 'Expand', 'Tile'}
_LAYOUT_OPS = {'Transpose', 'Reshape', 'Flatten', 'Squeeze', 'Unsqueeze'}
_INDEX_OPS = {'Gather', 'GatherElements', 'GatherND', 'Slice', 'Split'}
_ELEMENTWISE_OPS = {
    'Add', 'Sub', 'Mul', 'Div', 'Neg', 'Pow', 'Where', 'Equal', 'Less',
    'Greater', 'Clip', 'Relu', 'Sigmoid', 'Tanh', 'Exp', 'Log', 'Sqrt',
}


def _onnx_shape(value_info):
    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_param:
            shape.append(dim.dim_param)
        elif dim.HasField('dim_value'):
            shape.append(dim.dim_value)
        else:
            shape.append(None)
    return shape


def audit_onnx_model(model_file, require_optimized_graph):
    model_file = Path(model_file)
    model = onnx.load(model_file)
    onnx.checker.check_model(model)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)

    input_type = {
        'int16': onnx.TensorProto.INT16,
        'float16': onnx.TensorProto.FLOAT16,
        'float': onnx.TensorProto.FLOAT,
    }[IN_ONNX_DTYPE]
    expected_input_shape = [1, 1, 'audio_len'] if DYNAMIC_AXES else [1, 1, export_audio_length]
    expected_output_shape = [1, 'signal_len', 1] if DYNAMIC_AXES else None

    if [value.name for value in model.graph.input] != ['audio']:
        raise RuntimeError(f"Unexpected ONNX inputs in {model_file}: {[value.name for value in model.graph.input]}")
    if [value.name for value in model.graph.output] != ['score_silence', 'score_active', 'signal_len']:
        raise RuntimeError(f"Unexpected ONNX outputs in {model_file}: {[value.name for value in model.graph.output]}")
    if model.graph.input[0].type.tensor_type.elem_type != input_type:
        raise RuntimeError(f"Unexpected audio dtype in {model_file}.")
    if _onnx_shape(model.graph.input[0]) != expected_input_shape:
        raise RuntimeError(f"Unexpected audio shape in {model_file}: {_onnx_shape(model.graph.input[0])}")

    for output in model.graph.output[:2]:
        if output.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise RuntimeError(f"Unexpected score dtype in {model_file}: {output.name}")
        if expected_output_shape is not None and _onnx_shape(output) != expected_output_shape:
            raise RuntimeError(f"Unexpected score shape in {model_file}: {output.name} {_onnx_shape(output)}")
    signal_len = model.graph.output[2]
    if signal_len.type.tensor_type.elem_type != onnx.TensorProto.INT32 or _onnx_shape(signal_len) != [1]:
        raise RuntimeError(f"Unexpected signal_len contract in {model_file}.")

    domains = {node.domain or 'ai.onnx' for node in model.graph.node}
    if domains != {'ai.onnx'}:
        raise RuntimeError(f"Unexpected operator domains in {model_file}: {sorted(domains)}")

    histogram = Counter(node.op_type for node in model.graph.node)
    if require_optimized_graph:
        forbidden = {'ATen', 'BatchNormalization', 'Identity', 'MatMul', 'Pad', 'Reshape'}
        present = sorted(op_type for op_type in forbidden if histogram[op_type])
        if present:
            raise RuntimeError(f"Avoidable or fallback operators remain in {model_file}: {present}")
        if histogram['Mul'] != 1 or histogram['Softmax'] != 1 or histogram['Transpose'] != 1:
            raise RuntimeError(f"Unexpected optimized frontend/classifier pattern in {model_file}: {dict(histogram)}")

        initializer_shapes = {initializer.name: list(initializer.dims) for initializer in model.graph.initializer}
        expected_shapes = {
            'stft_model.stft_kernel': [2 * (NFFT_STFT // 2 + 1), 1, 398],
            'mel_weight': [N_MELS, NFFT_STFT // 2 + 1, 1],
            'mel_bias': [N_MELS],
            'decoder_weight': [2, 128, 1],
            'decoder_bias': [2],
        }
        for name, expected_shape in expected_shapes.items():
            if initializer_shapes.get(name) != expected_shape:
                raise RuntimeError(
                    f"Unexpected initializer {name} in {model_file}: "
                    f"{initializer_shapes.get(name)} != {expected_shape}"
                )

    initializer_bytes = sum(
        np.asarray(onnx.numpy_helper.to_array(initializer)).nbytes
        for initializer in model.graph.initializer
    )
    report = {
        'sha256': hashlib.sha256(model_file.read_bytes()).hexdigest(),
        'bytes': model_file.stat().st_size,
        'nodes': len(model.graph.node),
        'initializers': len(model.graph.initializer),
        'initializer_bytes': initializer_bytes,
        'histogram': dict(sorted(histogram.items())),
        'categories': {
            'conv': histogram['Conv'],
            'matmul': histogram['MatMul'],
            'normalization': histogram['BatchNormalization'] + histogram['LayerNormalization'],
            'reduction': sum(histogram[op_type] for op_type in ('ReduceMean', 'ReduceSum', 'ReduceMax')),
            'cast': histogram['Cast'],
            'shape': sum(histogram[op_type] for op_type in _SHAPE_OPS),
            'layout': sum(histogram[op_type] for op_type in _LAYOUT_OPS),
            'indexing': sum(histogram[op_type] for op_type in _INDEX_OPS),
            'elementwise': sum(histogram[op_type] for op_type in _ELEMENTWISE_OPS),
        },
    }
    del model
    return report


def print_graph_report(label, report):
    print(
        f"  {label}: nodes={report['nodes']} initializers={report['initializers']} "
        f"bytes={report['bytes']} sha256={report['sha256']}"
    )
    print(f"    categories={report['categories']}")
    print(f"    operators={report['histogram']}")


def assert_graph_payload_equal(raw_model_file, final_model_file):
    raw_model = onnx.load(raw_model_file)
    final_model = onnx.load(final_model_file)
    if raw_model.graph.SerializeToString() != final_model.graph.SerializeToString():
        raise RuntimeError("Published final model changed the immutable raw ONNX graph.")
    if [item.SerializeToString() for item in raw_model.opset_import] != [
        item.SerializeToString() for item in final_model.opset_import
    ]:
        raise RuntimeError("Published final model changed raw ONNX opset imports.")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BatchNorm Folding Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def fold_bn_into_conv1d(conv: torch.nn.Conv1d, bn: torch.nn.BatchNorm1d) -> torch.nn.Conv1d:
    """
    Fold a BatchNorm1d into the preceding Conv1d in eval mode.

    Produces a new Conv1d with:
        new_weight = conv_weight * (gamma / sqrt(var + eps))
        new_bias   = (conv_bias - mean) * (gamma / sqrt(var + eps)) + beta

    This eliminates the BatchNorm entirely from the ONNX graph.
    """
    assert not bn.training, "BatchNorm must be in eval mode for folding."
    with torch.no_grad():
        gamma = bn.weight                   # (out_channels,)
        beta = bn.bias                      # (out_channels,)
        mean = bn.running_mean              # (out_channels,)
        var = bn.running_var                # (out_channels,)
        eps = bn.eps

        # Compute per-channel scale factor
        std_inv = torch.rsqrt(var + eps)    # 1 / sqrt(var + eps)
        scale = gamma * std_inv             # (out_channels,)

        # Fold into conv weight: (out_ch, in_ch/groups, kernel_size)
        new_weight = conv.weight * scale.reshape(-1, 1, 1)

        # Fold into conv bias
        if conv.bias is not None:
            new_bias = (conv.bias - mean) * scale + beta
        else:
            new_bias = (-mean) * scale + beta

        # Create new Conv1d with folded parameters
        new_conv = torch.nn.Conv1d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size[0],
            stride=conv.stride[0],
            padding=conv.padding[0],
            dilation=conv.dilation[0],
            groups=conv.groups,
            bias=True,
            padding_mode=conv.padding_mode,
        )
        new_conv.weight = torch.nn.Parameter(new_weight)
        new_conv.bias = torch.nn.Parameter(new_bias)
        return new_conv


def fold_encoder_batchnorms(encoder_module):
    """
    Recursively fold all BatchNorm1d layers into preceding Conv1d within
    the NeMo encoder structure (JasperBlocks with mconv and res lists).

    Mutates the encoder in-place. Returns count of folded BN layers.
    """
    folded_count = 0

    def fold_in_module_list(module_list):
        """Fold BN in a ModuleList where Conv1d precedes BatchNorm1d."""
        nonlocal folded_count
        modules = list(module_list)
        i = 0
        while i < len(modules) - 1:
            current = modules[i]
            next_mod = modules[i + 1]

            # Case 1: MaskedConv1d (has .conv attribute) followed by BatchNorm1d
            if hasattr(current, 'conv') and isinstance(current.conv, torch.nn.Conv1d):
                if isinstance(next_mod, torch.nn.BatchNorm1d):
                    current.conv = fold_bn_into_conv1d(current.conv, next_mod)
                    module_list[i + 1] = torch.nn.Identity()
                    folded_count += 1
            # Case 2: Direct Conv1d followed by BatchNorm1d
            elif isinstance(current, torch.nn.Conv1d):
                if isinstance(next_mod, torch.nn.BatchNorm1d):
                    module_list[i] = fold_bn_into_conv1d(current, next_mod)
                    module_list[i + 1] = torch.nn.Identity()
                    folded_count += 1
            i += 1

    # Iterate over encoder blocks (JasperBlock structure)
    for block in encoder_module.encoder:
        # Fold BN in main convolution path (mconv)
        if hasattr(block, 'mconv'):
            fold_in_module_list(block.mconv)
        # Fold BN in residual projection path (res)
        if hasattr(block, 'res') and block.res:
            for res_branch in block.res:
                if isinstance(res_branch, torch.nn.ModuleList):
                    fold_in_module_list(res_branch)
                elif isinstance(res_branch, torch.nn.Sequential):
                    fold_in_module_list(res_branch)

    return folded_count


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Optimized NVIDIA_VAD Wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class NVIDIA_VAD_Optimized(torch.nn.Module):
    """
    Ultra-optimized MarbleNet VAD wrapper for ONNX export.

    All deterministic computation is precomputed in __init__() as registered
    buffers. Forward path uses only ONNXRuntime-friendly tensor operations.

    Optimizations vs original:
            - Pre-emphasis + int16 scaling -> one Conv1d
            - Zero-only STFT taps removed; real/imag power squared before splitting
            - Mel projection + epsilon -> one bias-fused 1x1 Conv
            - Decoder Linear + bias -> one channel-first 1x1 Conv
            - BatchNorm -> folded into Conv1d weights (eliminates all BN nodes)
      - No runtime constant construction in forward()
      - No Python-level control flow in forward()
    """

    def __init__(self, nvidia_vad, stft_model, nfft_stft, n_mels, sample_rate, pre_emphasis, in_sample_rate=16000):
        super().__init__()

        self.encoder = nvidia_vad.encoder
        self.stft_model = stft_model

        # ── Sample rate interpolation (resample input to model's native 16000 Hz) ──
        self.in_sample_rate_scale = in_sample_rate / 16000.0
        self.model_rate_scale = 1.0 / self.in_sample_rate_scale
        self.resample_before = self.in_sample_rate_scale > 1.0
        self.resample_after = self.in_sample_rate_scale < 1.0

        # ── Fuse mel projection + epsilon into a 1x1 Conv ────────────────
        fbank = torchaudio.functional.melscale_fbanks(
            nfft_stft // 2 + 1, 0, sample_rate // 2, n_mels, sample_rate, 'slaney', 'slaney'
        ).transpose(0, 1)
        self.register_buffer('mel_weight', fbank.unsqueeze(-1))
        self.register_buffer('mel_bias', torch.full((n_mels,), 1e-07, dtype=torch.float32))

        # ── Input scaling + pre-emphasis as a single Conv1d kernel ────────
        # INT16 input uses raw PCM counts and is scaled by 1/32768. F16/F32
        # input is already normalized PCM, matching the denoiser contract.
        # Pre-emphasis: y[n] = x[n] - α*x[n-1], y[0] = x[0]
        # As Conv1d with kernel [-α, 1] and left-pad-by-1 (constant 0):
        #   y[0] = 0*(-α) + x[0]*1 = x[0]  ✓
        #   y[n] = x[n-1]*(-α) + x[n]*1 = x[n] - α*x[n-1]  ✓
        # Fold the selected input scale into the kernel (linearity of convolution).
        input_scale = 1.0 / 32768.0 if IN_AUDIO_DTYPE == 'INT16' else 1.0
        pre_emph_kernel = torch.tensor(
            [[[-pre_emphasis * input_scale, input_scale]]],
            dtype=torch.float32
        )  # shape: (1, 1, 2) — single-channel Conv1d kernel
        self.register_buffer('pre_emph_kernel', pre_emph_kernel)
        self.register_buffer('pre_emph_zero', torch.zeros(1, 1, 1, dtype=torch.float32))

        # ── Export the final Linear as a channel-first 1x1 Conv ───────────
        decoder = nvidia_vad.decoder.layer0
        if not isinstance(decoder, torch.nn.Linear) or decoder.out_features != 2:
            raise TypeError("Expected a two-class Linear decoder for MarbleNet VAD.")
        self.register_buffer('decoder_weight', decoder.weight.detach().unsqueeze(-1).clone())
        self.register_buffer('decoder_bias', decoder.bias.detach().clone())

        # ── Disable masks in encoder (no masking computation needed) ──────
        for block in self.encoder.encoder:
            for layer in block.mconv:
                layer.use_mask = False
            if block.res:
                for res_branch in block.res:
                    for layer in res_branch:
                        layer.use_mask = False

        # ── Fold all BatchNorm1d into Conv1d ──────────────────────────────
        fold_count = fold_encoder_batchnorms(self.encoder)
        print(f"  Folded {fold_count} BatchNorm layers into Conv1d.")

    def forward(self, audio: torch.Tensor) -> tuple:
        """
        Forward pass — optimized for ONNX export.

        Input:
            audio: configured F16/F32/INT16 tensor, shape (1, 1, audio_len)

        Output:
            score_silence: float32, shape (1, signal_len, 1)
            score_active:  float32, shape (1, signal_len, 1)
            signal_len:    int32, shape (1,) — number of valid output frames
        """
        # ── Step 1: input scaling + pre-emphasis in single Conv1d ─────────
        # Resample to model's native 16000 Hz if input rate is higher
        audio = audio.float()
        if self.resample_before:
            audio = F.interpolate(
                audio,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        # Explicit concat avoids the legacy exporter's large constant-Pad subgraph.
        audio = torch.cat((self.pre_emph_zero, audio), dim=2)
        audio = F.conv1d(audio, self.pre_emph_kernel)
        # Resample to model's native 16000 Hz if input rate is lower
        if self.resample_after:
            audio = F.interpolate(
                audio,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )

        # ── Step 2: STFT power spectrum (packed real/imag square) ─────────
        power_spec = self.stft_model(audio)

        # ── Step 3: Fused mel projection + epsilon → Log ─────────────────
        mel_features = F.conv1d(power_spec, self.mel_weight, self.mel_bias).log()

        # ── Step 4: Encoder (BN already folded into Conv1d) ───────────────
        feature_length = torch._shape_as_tensor(mel_features)[-1:]
        encoded, _ = self.encoder(
            audio_signal=mel_features,
            length=feature_length,
        )

        # ── Step 5: Fused decoder projection + Softmax + Split ────────────
        logits = F.conv1d(encoded, self.decoder_weight, self.decoder_bias)
        score = torch.softmax(logits.transpose(1, 2), dim=-1)
        score_silence, score_active = torch.split(score, [1, 1], dim=-1)
        signal_len = (torch._shape_as_tensor(score)[1].int() - 1).unsqueeze(0).to(dtype=torch.int32)

        return score_silence, score_active, signal_len


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Export
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("Ultra-Optimized NVIDIA MarbleNet VAD Export")
print("=" * 70)

print("\n[1/4] Loading pretrained model...")
with torch.inference_mode():
    custom_stft = STFT_Process(
        model_type='stft_power',
        n_fft=NFFT_STFT,
        hop_len=HOP_LENGTH,
        win_length=WINDOW_LENGTH,
        max_frames=0,
        window_type=WINDOW_TYPE,
        center_pad=True,
        pad_mode='constant'
    ).eval()
    model_raw = load_nvidia_vad_model(model_path)

    # ── Build optimized model ─────────────────────────────────────────
    print("\n[2/4] Building optimized model (folding BN, registering buffers)...")
    optimized_model = NVIDIA_VAD_Optimized(
        model_raw, custom_stft, NFFT_STFT, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, IN_SAMPLE_RATE
    ).eval()

    # ── Export to ONNX ────────────────────────────────────────────────
    print("\n[3/4] Exporting immutable source-optimized raw ONNX...")
    audio_dummy = torch.ones((1, 1, export_audio_length), dtype=IN_TORCH_DTYPE)
    export_folder.mkdir(parents=True, exist_ok=True)
    atexit.register(onnx_model_raw.unlink, missing_ok=True)

    dynamic_axes = {
        'audio': {2: 'audio_len'},
        'score_silence': {1: 'signal_len'},
        'score_active': {1: 'signal_len'},
    } if DYNAMIC_AXES else None

    torch.onnx.export(
        optimized_model,
        (audio_dummy,),
        onnx_model_raw,
        export_params=True,
        input_names=['audio'],
        output_names=['score_silence', 'score_active', 'signal_len'],
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False,
    )

    print("\n[4/4] Auditing raw ONNX and publishing final model...")
    raw_report = audit_onnx_model(onnx_model_raw, require_optimized_graph=True)
    print_graph_report('source-optimized raw', raw_report)

    shutil.copyfile(onnx_model_raw, onnx_model_A)

    del optimized_model
    del audio_dummy
    gc.collect()

print(f"\nImmutable raw export: {onnx_model_raw}")
print(f"Published final model: {onnx_model_A}")
print("Source model published unchanged.")

model_metadata = {
    "vad_metadata_version": 1,
    "producer": Path(__file__).name,
    "model_name": "NVIDIA_MarbleNet",
    "task": "voice_activity_detection",
    "model_family": "nvidia_marblenet",
    "dynamic_axes": DYNAMIC_AXES,
    "opset": OPSET,
    "input_audio_dtype": IN_ONNX_DTYPE,
    "input_sample_rate": IN_SAMPLE_RATE,
    "model_sample_rate": SAMPLE_RATE,
    "input_audio_length": export_audio_length,
    "output_frame_shift_seconds": 320 / SAMPLE_RATE,
}
raw_metadata_path = metadata_path_for_model(onnx_model_raw)
export_metadata_carrier(
    raw_metadata_path,
    model_metadata,
    OPSET,
)
stamp_export_metadata(onnx_model_A, model_metadata, OPSET)
print(f"Raw metadata saved to: {raw_metadata_path}")
print(f"Published metadata saved to: {onnx_model_Metadata}")

final_report = audit_onnx_model(onnx_model_A, require_optimized_graph=True)
print_graph_report('published final', final_report)
assert_graph_payload_equal(onnx_model_raw, onnx_model_A)

inference_script = script_dir / "Inference_NVIDIA_MarbleNet_VAD_ONNX.py"
print(f"\nStarting inference demo with {inference_script.name} using: {export_folder.resolve()}")
subprocess.run(
    [sys.executable, str(inference_script), str(export_folder.resolve())],
    cwd=script_dir,
    check=True,
)
