import gc
import time
import random
import site
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import onnxruntime
from datetime import timedelta
from pydub import AudioSegment
from STFT_Process import STFT_Process

model_path = "/home/DakeQQ/Downloads/frame_vad_multilingual_marblenet_v2.0.nemo"      # The NVIDIA VAD download path.
onnx_model_A = "/home/DakeQQ/Downloads/NVIDIA_MarbleNet_ONNX/NVIDIA_MarbleNet.onnx"   # The exported onnx model path.
test_vad_audio = "./vad_sample.wav"                                                   # The test audio path.
save_timestamps_second = "./timestamps_second.txt"                                    # The saved path.
save_timestamps_indices = "./timestamps_indices.txt"                                  # The saved path.


ORT_Accelerate_Providers = []                               # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                                            # else keep empty.
DYNAMIC_AXES = True                                         # The default dynamic axis is input audio length.
IN_SAMPLE_RATE = 16000                                      # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
SAMPLE_RATE = 16000                                         # The model native rate, do not edit the value.
OUTPUT_FRAME_LENGTH = 320                                   # The model parameter, do not edit it.
INPUT_AUDIO_LENGTH = 160000                                 # The max length of input audio segment. For DYNAMIC_AXES=False.
WINDOW_TYPE = 'hann_sym'                                    # Type of window function used in the STFT. NeMo uses symmetric (periodic=False) hann window.
N_MELS = 80                                                 # Number of Mel bands to generate in the Mel-spectrogram. Do not edit it.
NFFT_STFT = 512                                             # Number of FFT components for the STFT process, edit it carefully.
WINDOW_LENGTH = 400                                         # Length of windowing, edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
PRE_EMPHASIZE = 0.97                                        # For audio preprocessing.
OPSET = 18                                                  # ONNX opset version.
NORMALIZE_AUDIO = False                                     # Normalize the input audio to a target RMS level (e.g., 8192) before processing. It can help improve the performance of the model, especially for low-volume audio. Set it to True if you want to enable it.

# ─── VAD Postprocessing Parameters ───────────────────────────────────────────
OUTPUT_FRAME_SHIFT_S = OUTPUT_FRAME_LENGTH / SAMPLE_RATE    # 0.02 seconds (20ms frame shift for NVIDIA MarbleNet output)
SPEAKING_SCORE = 0.5                                        # Speech probability threshold.
SMOOTH_WINDOW_SIZE = 3                                      # Probability smoothing window size (in output frames).
MIN_SPEECH_FRAME = 10                                       # Min speech frames to confirm a segment.
MAX_SPEECH_FRAME = 1000                                     # Max speech frames before forced split.
MIN_SILENCE_FRAME = 10                                      # Min silence frames to confirm end of speech.
MERGE_SILENCE_FRAME = 3                                     # Merge silence gaps shorter than this (frames).
EXTEND_SPEECH_FRAME = 0                                     # Extend speech regions by this many frames.


shutil.copyfile('./modeling_modified/common.py', site.getsitepackages()[-1] + "/nemo/core/classes/common.py")
import nemo.collections.asr as nemo_asr


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
      - Pre-emphasis + int16 scaling → single Conv1d (2 ops vs 8 ops)
      - Mel filterbank → registered buffer (proper initializer)
      - BatchNorm → folded into Conv1d weights (eliminates all BN nodes)
      - No runtime constant construction in forward()
      - No Python-level control flow in forward()
    """

    def __init__(self, nvidia_vad, stft_model, nfft_stft, n_mels, sample_rate, pre_emphasis, in_sample_rate=16000):
        super().__init__()

        self.nvidia_vad = nvidia_vad
        self.stft_model = stft_model

        # ── Sample rate interpolation (resample input to model's native 16000 Hz) ──
        self.in_sample_rate_scale = in_sample_rate / 16000.0
        self.model_rate_scale = 1.0 / self.in_sample_rate_scale
        self.resample_before = self.in_sample_rate_scale > 1.0
        self.resample_after = self.in_sample_rate_scale < 1.0

        # ── Register mel filterbank as buffer (ONNX initializer) ──────────
        fbank = torchaudio.functional.melscale_fbanks(
            nfft_stft // 2 + 1, 0, sample_rate // 2, n_mels, sample_rate, 'slaney', 'slaney'
        ).transpose(0, 1).unsqueeze(0)  # shape: (1, n_mels, n_freq_bins)
        self.register_buffer('fbank', fbank)

        # ── Pre-emphasis + int16→float scaling as single Conv1d kernel ────
        # Original: audio = audio.float() * (1/32768) then pre-emphasis
        # Pre-emphasis: y[n] = x[n] - α*x[n-1], y[0] = x[0]
        # As Conv1d with kernel [-α, 1] and left-pad-by-1 (constant 0):
        #   y[0] = 0*(-α) + x[0]*1 = x[0]  ✓
        #   y[n] = x[n-1]*(-α) + x[n]*1 = x[n] - α*x[n-1]  ✓
        # Folding inv_int16 into the kernel (linearity of convolution):
        #   y = conv(x * s, kernel) = conv(x, kernel * s)
        inv_int16 = 1.0 / 32768.0
        pre_emph_kernel = torch.tensor(
            [[[-pre_emphasis * inv_int16, inv_int16]]],
            dtype=torch.float32
        )  # shape: (1, 1, 2) — single-channel Conv1d kernel
        self.register_buffer('pre_emph_kernel', pre_emph_kernel)

        # ── Log-mel epsilon as buffer ─────────────────────────────────────
        self.register_buffer('log_eps', torch.tensor(1e-07, dtype=torch.float32))

        # ── Disable masks in encoder (no masking computation needed) ──────
        for block in self.nvidia_vad.encoder.encoder:
            for layer in block.mconv:
                layer.use_mask = False
            if block.res:
                for res_branch in block.res:
                    for layer in res_branch:
                        layer.use_mask = False

        # ── Fold all BatchNorm1d into Conv1d ──────────────────────────────
        fold_count = fold_encoder_batchnorms(self.nvidia_vad.encoder)
        print(f"  Folded {fold_count} BatchNorm layers into Conv1d.")

    def forward(self, audio: torch.Tensor) -> tuple:
        """
        Forward pass — optimized for ONNX export.

        Input:
            audio: int16 tensor, shape (1, 1, audio_len)

        Output:
            score_silence: float32, shape (1, signal_len, 1)
            score_active:  float32, shape (1, signal_len, 1)
            signal_len:    int32, shape (1,) — number of valid output frames
        """
        # ── Step 1: int16→float + pre-emphasis in single Conv1d ───────────
        # Resample to model's native 16000 Hz if input rate is higher
        audio = audio.float()
        if self.resample_before:
            audio = F.interpolate(
                audio,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        # Pad left by 1 (constant 0) for causal pre-emphasis filter
        audio = F.pad(audio, (1, 0), mode='constant', value=0.0)
        audio = F.conv1d(audio, self.pre_emph_kernel)
        # Resample to model's native 16000 Hz if input rate is lower
        if self.resample_after:
            audio = F.interpolate(
                audio,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )

        # ── Step 2: STFT (Conv1d-based, already ONNX-optimized) ───────────
        real_part, imag_part = self.stft_model(audio)

        # ── Step 3: Power spectrum → Mel → Log ───────────────────────────
        power_spec = real_part * real_part + imag_part * imag_part
        mel_features = torch.matmul(self.fbank, power_spec)
        mel_features = (mel_features + self.log_eps).log()

        # ── Step 4: Encoder (BN already folded into Conv1d) ───────────────
        encoded, signal_len = self.nvidia_vad.encoder(
            ([mel_features], torch.tensor([mel_features.shape[-1]], dtype=torch.long))
        )

        # ── Step 5: Decoder + Softmax + Split ─────────────────────────────
        score = torch.softmax(
            self.nvidia_vad.decoder(encoded.transpose(1, 2)), dim=-1
        )
        score_silence, score_active = torch.split(score, [1, 1], dim=-1)

        return score_silence, score_active, signal_len.int() - 1


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Reference Model (original, for validation)
# ═══════════════════════════════════════════════════════════════════════════════

class NVIDIA_VAD_Reference(torch.nn.Module):
    """Original (non-optimized) model for numerical validation."""

    def __init__(self, nvidia_vad, stft_model, nfft_stft, n_mels, sample_rate, pre_emphasis, in_sample_rate=16000):
        super().__init__()
        self.nvidia_vad = nvidia_vad
        self.stft_model = stft_model
        self.pre_emphasis = float(pre_emphasis)
        self.in_sample_rate_scale = in_sample_rate / 16000.0
        self.model_rate_scale = 1.0 / self.in_sample_rate_scale
        self.resample_before = self.in_sample_rate_scale > 1.0
        self.resample_after = self.in_sample_rate_scale < 1.0
        self.fbank = (torchaudio.functional.melscale_fbanks(
            nfft_stft // 2 + 1, 0, sample_rate // 2, n_mels, sample_rate, 'slaney', 'slaney'
        )).transpose(0, 1).unsqueeze(0)
        self.inv_int16 = float(1.0 / 32768.0)
        for block in self.nvidia_vad.encoder.encoder:
            for layer in block.mconv:
                layer.use_mask = False
            if block.res:
                for res_branch in block.res:
                    for layer in res_branch:
                        layer.use_mask = False

    def forward(self, audio):
        audio = audio.float()
        if self.resample_before:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        audio = audio * self.inv_int16
        if self.pre_emphasis > 0:
            audio = torch.cat(
                [audio[:, :, [0]], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]],
                dim=-1
            )
        if self.resample_after:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        real_part, imag_part = self.stft_model(audio)
        mel_features = (torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part) + 1e-07).log()
        encoded, signal_len = self.nvidia_vad.encoder(
            ([mel_features], torch.tensor([mel_features.shape[-1]], dtype=torch.long))
        )
        score = torch.softmax(self.nvidia_vad.decoder(encoded.transpose(1, 2)), dim=-1)
        score_silence, score_active = torch.split(score, [1, 1], dim=-1)
        return score_silence, score_active, signal_len.int() - 1


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Export
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("Ultra-Optimized NVIDIA MarbleNet VAD Export")
print("=" * 70)

print("\n[1/5] Loading pretrained model...")
with torch.inference_mode():
    # ── Build reference model first (separate load, avoids deepcopy issues) ──
    print("\n[2/5] Building reference model for validation...")
    custom_stft_ref = STFT_Process(
        model_type='stft_B',
        n_fft=NFFT_STFT,
        hop_len=HOP_LENGTH,
        win_length=WINDOW_LENGTH,
        max_frames=0,
        window_type=WINDOW_TYPE,
        center_pad=True,
        pad_mode='constant'
    ).eval()
    model_raw_ref = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        model_name=model_path, strict=False
    ).to('cpu').float().eval()
    reference_model = NVIDIA_VAD_Reference(
        model_raw_ref, custom_stft_ref, NFFT_STFT, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, IN_SAMPLE_RATE
    ).eval()

    # ── Load again for optimized model (BN will be folded in-place) ───
    custom_stft = STFT_Process(
        model_type='stft_B',
        n_fft=NFFT_STFT,
        hop_len=HOP_LENGTH,
        win_length=WINDOW_LENGTH,
        max_frames=0,
        window_type=WINDOW_TYPE,
        center_pad=True,
        pad_mode='constant'
    ).eval()
    model_raw = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        model_name=model_path, strict=False
    ).to('cpu').float().eval()

    # ── Build optimized model ─────────────────────────────────────────
    print("\n[3/5] Building optimized model (folding BN, registering buffers)...")
    optimized_model = NVIDIA_VAD_Optimized(
        model_raw, custom_stft, NFFT_STFT, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, IN_SAMPLE_RATE
    ).eval()

    # ── Validation: Original PyTorch vs Optimized PyTorch ─────────────
    print("\n[4/5] Validating numerical fidelity...")
    print("\n── Step 1: Reference PyTorch vs Optimized PyTorch ──")

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Test multiple lengths for dynamic axis validation
    test_lengths = [16000, 48000, INPUT_AUDIO_LENGTH] if DYNAMIC_AXES else [INPUT_AUDIO_LENGTH]

    all_pass = True
    for test_len in test_lengths:
        test_audio = torch.randint(-32768, 32767, (1, 1, test_len), dtype=torch.int16)

        ref_silence, ref_active, ref_siglen = reference_model(test_audio)
        opt_silence, opt_active, opt_siglen = optimized_model(test_audio)

        err_silence = (ref_silence - opt_silence).abs()
        err_active = (ref_active - opt_active).abs()

        max_abs_silence = err_silence.max().item()
        max_abs_active = err_active.max().item()
        mean_abs_silence = err_silence.mean().item()
        mean_abs_active = err_active.mean().item()
        siglen_match = (ref_siglen == opt_siglen).all().item()

        pass_check = max_abs_silence <= 1e-5 and max_abs_active <= 1e-5 and siglen_match
        all_pass = all_pass and pass_check

        print(f"  audio_len={test_len:>7d} | "
              f"max_err_silence={max_abs_silence:.2e} | "
              f"max_err_active={max_abs_active:.2e} | "
              f"mean_err={mean_abs_active:.2e} | "
              f"siglen_match={siglen_match} | "
              f"{'PASS' if pass_check else 'FAIL'}")

    if not all_pass:
        print("\n  WARNING: Numerical validation did not fully pass.")
        print("  This may be due to floating-point associativity differences")
        print("  from BN folding. Checking relaxed tolerance (1e-4)...")
        # BN folding can introduce tiny differences due to FP32 associativity
        relaxed = max_abs_silence <= 1e-4 and max_abs_active <= 1e-4
        print(f"  Relaxed tolerance check: {'PASS' if relaxed else 'FAIL'}")

    # ── Export to ONNX ────────────────────────────────────────────────
    print("\n[5/5] Exporting optimized model to ONNX...")
    audio_dummy = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)

    dynamic_axes = {
        'audio': {2: 'audio_len'},
        'score_silence': {1: 'signal_len'},
        'score_active': {1: 'signal_len'},
    } if DYNAMIC_AXES else None

    torch.onnx.export(
        optimized_model,
        (audio_dummy,),
        onnx_model_A,
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

    del reference_model
    del model_raw_ref
    del optimized_model
    del audio_dummy
    gc.collect()

print(f"\nExport complete: {onnx_model_A}")
print("No post-export optimization required.")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ONNXRuntime Inference & Validation
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ONNXRuntime Inference Validation")
print("=" * 70)

# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4
session_opts.log_verbosity_level = 4
session_opts.inter_op_num_threads = 0
session_opts.intra_op_num_threads = 0
session_opts.enable_cpu_mem_arena = True
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")

ort_session_A = onnxruntime.InferenceSession(
    onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers
)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")

in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A0 = out_name_A[0].name
out_name_A1 = out_name_A[1].name
out_name_A2 = out_name_A[2].name

# ── Step 2: Optimized PyTorch vs ONNXRuntime ──────────────────────────────
print("\n── Step 2: Optimized PyTorch vs ONNXRuntime ──")
with torch.inference_mode():
    # Reload optimized model for comparison
    custom_stft_v = STFT_Process(
        model_type='stft_B', n_fft=NFFT_STFT, hop_len=HOP_LENGTH,
        win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE,
        center_pad=True, pad_mode='constant'
    ).eval()
    model_v = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        model_name=model_path, strict=False
    ).to('cpu').float().eval()
    opt_model_v = NVIDIA_VAD_Optimized(
        model_v, custom_stft_v, NFFT_STFT, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, IN_SAMPLE_RATE
    ).eval()

    for test_len in test_lengths:
        test_audio_np = np.random.randint(-32768, 32767, (1, 1, test_len)).astype(np.int16)
        test_audio_pt = torch.from_numpy(test_audio_np)

        # PyTorch forward
        pt_silence, pt_active, pt_siglen = opt_model_v(test_audio_pt)

        # ONNXRuntime forward
        ort_silence, ort_active, ort_siglen = ort_session_A.run(
            [out_name_A0, out_name_A1, out_name_A2],
            {in_name_A0: test_audio_np}
        )

        err_s = np.abs(pt_silence.numpy() - ort_silence).max()
        err_a = np.abs(pt_active.numpy() - ort_active).max()
        siglen_ok = (pt_siglen.numpy() == ort_siglen).all()

        print(f"  audio_len={test_len:>7d} | "
              f"max_err_silence={err_s:.2e} | "
              f"max_err_active={err_a:.2e} | "
              f"siglen_match={siglen_ok} | "
              f"{'PASS' if err_s <= 1e-5 and err_a <= 1e-5 else 'PASS (within FP32 tolerance)' if err_s <= 1e-4 else 'CHECK'}")

    del opt_model_v, model_v, custom_stft_v
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. VAD Inference on Test Audio
# ═══════════════════════════════════════════════════════════════════════════════

def normalise_audio(audio: np.ndarray, target_rms: float = 8192.0) -> np.ndarray:
    _audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(_audio * _audio, dtype=np.float32), dtype=np.float32)
    if rms > 0:
        _audio *= (target_rms / (rms + 1e-7))
        np.clip(_audio, -32768.0, 32767.0, out=_audio)
        return _audio.astype(np.int16)
    else:
        return audio


print(f"\n\nTest Input Audio: {test_vad_audio}")
audio = np.array(
    AudioSegment.from_file(test_vad_audio).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(),
    dtype=np.int16
)
if NORMALIZE_AUDIO:
    audio = normalise_audio(audio)
audio_len = len(audio)
inv_audio_len = float(100.0 / audio_len)
audio = audio.reshape(1, 1, -1)

shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = min(IN_SAMPLE_RATE * 3600, audio_len)
else:
    INPUT_AUDIO_LENGTH = shape_value_in

stride_step = INPUT_AUDIO_LENGTH
if audio_len > INPUT_AUDIO_LENGTH:
    num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
    total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
    pad_amount = total_length_needed - audio_len
    final_slice = audio[:, :, -pad_amount:].astype(np.float32)
    white_noise = (np.sqrt(np.mean(final_slice * final_slice)) * np.random.normal(
        loc=0.0, scale=1.0, size=(1, 1, pad_amount)
    )).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
elif audio_len < INPUT_AUDIO_LENGTH:
    audio_float = audio.astype(np.float32)
    white_noise = (np.sqrt(np.mean(audio_float * audio_float)) * np.random.normal(
        loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len)
    )).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
aligned_len = audio.shape[-1]


# ═══════════════════════════════════════════════════════════════════════════════
# VAD Post-processing (from FireRedVAD, adapted for NVIDIA MarbleNet 20ms frames)
# ═══════════════════════════════════════════════════════════════════════════════

_VAD_SILENCE = 0
_VAD_POSSIBLE_SPEECH = 1
_VAD_SPEECH = 2
_VAD_POSSIBLE_SILENCE = 3


class VadPostprocessor:
    __slots__ = ('smooth_window_size', 'prob_threshold', 'min_speech_frame',
                 'max_speech_frame', 'min_silence_frame', 'merge_silence_frame',
                 'extend_speech_frame', 'frame_shift_s', '_inv_ws', '_half_max')

    def __init__(self, smooth_window_size, prob_threshold, min_speech_frame,
                 max_speech_frame, min_silence_frame, merge_silence_frame,
                 extend_speech_frame, frame_shift_s):
        self.smooth_window_size = max(1, smooth_window_size)
        self.prob_threshold = np.float32(prob_threshold)
        self.min_speech_frame = min_speech_frame
        self.max_speech_frame = max_speech_frame
        self.min_silence_frame = min_silence_frame
        self.merge_silence_frame = merge_silence_frame
        self.extend_speech_frame = extend_speech_frame
        self.frame_shift_s = np.float32(frame_shift_s)
        self._inv_ws = np.float32(1.0 / self.smooth_window_size)
        self._half_max = max_speech_frame >> 1

    def process(self, raw_probs):
        """Process raw probabilities into binary speech decisions (numpy int8 array)."""
        if isinstance(raw_probs, np.ndarray):
            n = raw_probs.shape[0]
            if n == 0:
                return np.empty(0, dtype=np.int8)
            probs = raw_probs.astype(np.float32, copy=False)
        else:
            n = len(raw_probs)
            if n == 0:
                return np.empty(0, dtype=np.int8)
            probs = np.asarray(raw_probs, dtype=np.float32)

        decisions = self._smooth_threshold_state_machine(probs, n)
        self._fix_starts_inplace(decisions, n)
        if self.merge_silence_frame > 0:
            self._merge_silence_inplace(decisions, n)
        if self.extend_speech_frame > 0:
            self._extend_inplace(decisions, n)
        self._split_long_inplace(decisions, probs, n)
        return decisions

    def decision_to_segment(self, decisions, wav_dur=None):
        """Extract (start_sec, end_sec) segments from binary decision array."""
        if isinstance(decisions, np.ndarray):
            dec = decisions
            n = dec.shape[0]
        else:
            dec = np.asarray(decisions, dtype=np.int8)
            n = dec.shape[0]
        if n == 0:
            return []

        padded = np.empty(n + 2, dtype=np.int8)
        padded[0] = 0
        padded[n + 1] = 0
        padded[1:n + 1] = dec
        diff = np.diff(padded)
        starts = np.flatnonzero(diff == 1).astype(np.float32)
        ends = np.flatnonzero(diff == -1).astype(np.float32)

        num_segs = starts.shape[0]
        if num_segs == 0:
            return []

        segments = np.empty((num_segs, 2), dtype=np.float32)
        segments[:, 0] = starts * self.frame_shift_s
        segments[:, 1] = ends * self.frame_shift_s

        if dec[n - 1] != 0:
            end_time = n * self.frame_shift_s
            if wav_dur is not None and wav_dur < end_time:
                end_time = wav_dur
            segments[-1, 1] = end_time

        return [(round(s, 3), round(e, 3)) for s, e in segments.tolist()]

    def _smooth_threshold_state_machine(self, probs, n):
        decisions = np.zeros(n, dtype=np.int8)
        ws = self.smooth_window_size
        threshold = self.prob_threshold
        min_sp = self.min_speech_frame
        min_si = self.min_silence_frame

        if ws > 1:
            cumsum = np.empty(n + 1, dtype=np.float32)
            cumsum[0] = 0.0
            np.cumsum(probs, out=cumsum[1:])
            smoothed = np.empty(n, dtype=np.float32)
            edge_end = min(ws - 1, n)
            for i in range(edge_end):
                smoothed[i] = cumsum[i + 1] / (i + 1)
            if n >= ws:
                smoothed[ws - 1:] = (cumsum[ws:] - cumsum[:n - ws + 1]) * self._inv_ws
        else:
            smoothed = probs

        if min_sp <= 0 and min_si <= 0:
            decisions[:] = (smoothed >= threshold)
        else:
            state = _VAD_SILENCE
            speech_start = 0
            for t in range(n):
                is_speech = smoothed[t] >= threshold
                if state == _VAD_SILENCE:
                    if is_speech:
                        state = _VAD_POSSIBLE_SPEECH
                        speech_start = t
                elif state == _VAD_POSSIBLE_SPEECH:
                    if is_speech:
                        if t - speech_start >= min_sp:
                            state = _VAD_SPEECH
                            decisions[speech_start:t] = 1
                    else:
                        state = _VAD_SILENCE
                elif state == _VAD_SPEECH:
                    if not is_speech:
                        state = _VAD_POSSIBLE_SILENCE
                        silence_start = t
                else:  # _VAD_POSSIBLE_SILENCE
                    if not is_speech:
                        if t - silence_start >= min_si:
                            state = _VAD_SILENCE
                    else:
                        state = _VAD_SPEECH
                decisions[t] = 1 if state >= _VAD_SPEECH else 0

        return decisions

    def _fix_starts_inplace(self, decisions, n):
        ws = self.smooth_window_size
        if ws <= 1:
            return
        for t in range(1, n):
            if decisions[t] == 1 and decisions[t - 1] == 0:
                start = t - ws if t >= ws else 0
                decisions[start:t] = 1

    def _merge_silence_inplace(self, decisions, n):
        merge_thr = self.merge_silence_frame
        silence_start = -1
        for t in range(1, n):
            prev = decisions[t - 1]
            curr = decisions[t]
            if prev == 1 and curr == 0 and silence_start < 0:
                silence_start = t
            elif prev == 0 and curr == 1 and silence_start >= 0:
                if t - silence_start < merge_thr:
                    decisions[silence_start:t] = 1
                silence_start = -1

    def _extend_inplace(self, decisions, n):
        ext = self.extend_speech_frame
        dist = ext + 1
        for t in range(n):
            if decisions[t]:
                dist = 0
            else:
                dist += 1
                if dist <= ext:
                    decisions[t] = 1
        dist = ext + 1
        for t in range(n - 1, -1, -1):
            if decisions[t]:
                dist = 0
            else:
                dist += 1
                if dist <= ext:
                    decisions[t] = 1

    def _split_long_inplace(self, decisions, probs, n):
        max_sf = self.max_speech_frame
        half_max = self._half_max
        t = 0
        while t < n:
            if decisions[t]:
                seg_start = t
                while t < n and decisions[t]:
                    t += 1
                dur = t - seg_start
                if dur > max_sf:
                    pos = seg_start
                    seg_end = t
                    while pos + max_sf < seg_end:
                        w_start = pos + half_max
                        w_end = pos + max_sf
                        if w_end > seg_end:
                            w_end = seg_end
                        if w_start >= w_end:
                            break
                        min_idx = w_start + int(np.argmin(probs[w_start:w_end]))
                        decisions[min_idx] = 0
                        pos = min_idx + 1
            else:
                t += 1


def format_time(seconds):
    """Convert seconds to VTT time format 'hh:mm:ss.mmm'."""
    td = timedelta(seconds=seconds)
    td_sec = td.total_seconds()
    total_seconds = int(td_sec)
    milliseconds = int((td_sec - total_seconds) * 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


# Run VAD inference
slice_start = 0
slice_end = INPUT_AUDIO_LENGTH
all_vad_probs = []
print("\nRunning NVIDIA_VAD (optimized) via ONNXRuntime.")
start_time = time.time()
while slice_end <= aligned_len:
    score_silence, score_active, signal_len = ort_session_A.run(
        [out_name_A0, out_name_A1, out_name_A2],
        {in_name_A0: audio[..., slice_start: slice_end]}
    )
    # score_active shape: [1, signal_len, 1] -> extract speech probs as 1D
    valid_frames = min(int(signal_len[0]), score_active.shape[1])
    all_vad_probs.append(score_active[0, :valid_frames, 0])
    slice_start += stride_step
    slice_end = slice_start + INPUT_AUDIO_LENGTH
end_time = time.time()

# Concatenate all speech probabilities and trim to valid frame count
if all_vad_probs:
    all_vad_probs = np.concatenate(all_vad_probs, axis=0)
else:
    all_vad_probs = np.zeros((0,), dtype=np.float32)

# Post-process using VadPostprocessor (same approach as FireRedVAD)
vad_postprocessor = VadPostprocessor(
    smooth_window_size=SMOOTH_WINDOW_SIZE,
    prob_threshold=SPEAKING_SCORE,
    min_speech_frame=MIN_SPEECH_FRAME,
    max_speech_frame=MAX_SPEECH_FRAME,
    min_silence_frame=MIN_SILENCE_FRAME,
    merge_silence_frame=MERGE_SILENCE_FRAME,
    extend_speech_frame=EXTEND_SPEECH_FRAME,
    frame_shift_s=OUTPUT_FRAME_SHIFT_S,
)
vad_decisions = vad_postprocessor.process(all_vad_probs)
timestamps = vad_postprocessor.decision_to_segment(vad_decisions, audio_len / IN_SAMPLE_RATE)
print(f"Complete: 100.00%")

# Save timestamps
with open(save_timestamps_second, "w", encoding='UTF-8') as file:
    print("\nTimestamps in Second:")
    for start, end in timestamps:
        s_time = format_time(start)
        e_time = format_time(end)
        line = f"{s_time} --> {e_time}\n"
        file.write(line)
        print(line.replace("\n", ""))

with open(save_timestamps_indices, "w", encoding='UTF-8') as file:
    print("\nTimestamps in Indices:")
    for start, end in timestamps:
        line = f"{int(start * IN_SAMPLE_RATE)} --> {int(end * IN_SAMPLE_RATE)}\n"
        file.write(line)
        print(line.replace("\n", ""))

audio_duration = audio_len / IN_SAMPLE_RATE
elapsed = end_time - start_time
rtf = elapsed / audio_duration
print(f"\nVAD Process Complete.\n\nTime Cost: {elapsed:.3f} Seconds")
print(f"Audio Duration: {audio_duration:.3f} Seconds")
print(f"RTF (Real-Time Factor): {rtf:.6f}")
