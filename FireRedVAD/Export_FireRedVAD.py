import gc
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
import numpy as np
import onnx
from Optimize_ONNX import optimize_graph
from STFT_Process import STFT_Process

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "vad_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from vad_onnx_metadata import write_onnx_metadata


script_dir              = Path(__file__).resolve().parent
download_path           = Path.home() / "Downloads" / "FireRedVAD"                          # The FireRedVAD directory
model_path_vad          = download_path / "VAD"                                               # The FireRedVAD VAD model directory (contains model.pth.tar and cmvn.ark).
model_path_aed          = download_path / "AED"                                               # The FireRedVAD AED model directory (contains model.pth.tar and cmvn.ark).
model_path_stream_vad   = download_path / "Stream-VAD"                                        # The FireRedVAD Stream-VAD model directory (contains model.pth.tar and cmvn.ark).
export_folder           = script_dir / "FireRedVAD_ONNX"
onnx_model_vad          = export_folder / "FireRedVAD.onnx"                                  # The exported VAD onnx model path.
onnx_model_aed          = export_folder / "FireRedAED.onnx"                                  # The exported AED onnx model path.
onnx_model_stream_vad   = export_folder / "FireRedStreamVAD.onnx"                            # The exported Stream-VAD onnx model path.


# ─── ONNX Export Settings ─────────────────────────────────────────────────────
EXPORT_VAD = True                       # Export the VAD model.
EXPORT_AED = True                       # Export the AED model.
EXPORT_STREAM_VAD = True                # Export the Stream-VAD model.
DYNAMIC_AXES = False                    # Static graph (no dynamic axes). Set True only if variable-length input is required.
OPSET = 20                              # ONNX opset version.
IN_AUDIO_DTYPE = 'F32'                  # ['F16', 'F32', 'INT16']; float inputs use normalized PCM in [-1, 1].

_INPUT_AUDIO_DTYPES = {
    'F16': (torch.float16, 'tensor(float16)'),
    'F32': (torch.float32, 'tensor(float)'),
    'INT16': (torch.int16, 'tensor(int16)'),
}
IN_AUDIO_DTYPE = IN_AUDIO_DTYPE.upper()
if IN_AUDIO_DTYPE not in _INPUT_AUDIO_DTYPES:
    raise ValueError(
        f"Unsupported IN_AUDIO_DTYPE={IN_AUDIO_DTYPE!r}; "
        f"expected one of {tuple(_INPUT_AUDIO_DTYPES)}."
    )
IN_TORCH_DTYPE, IN_ONNX_DTYPE = _INPUT_AUDIO_DTYPES[IN_AUDIO_DTYPE]
IN_METADATA_DTYPE = IN_ONNX_DTYPE.removeprefix('tensor(').removesuffix(')')


IN_SAMPLE_RATE = 16000                  # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
if IN_SAMPLE_RATE <= 0:
    raise ValueError('IN_SAMPLE_RATE must be positive.')

# ─── Audio & STFT Parameters (do not edit) ────────────────────────────────────
SAMPLE_RATE = 16000                     # The model native rate, do not edit the value.
N_MELS = 80                             # Mel filterbank bands.
NFFT = 400                              # FFT size (frame_length_ms * sample_rate / 1000).
HOP_LENGTH = 160                        # Samples between frames (frame_shift_ms * sample_rate / 1000).
WINDOW_LENGTH = 400                     # Window length in samples (frame_length_ms * sample_rate / 1000).
WINDOW_TYPE = 'povey'                   # Kaldi window: 'povey' = hann^0.85.
PRE_EMPHASIZE = 0.97                    # Pre-emphasis coefficient (Kaldi default).
INPUT_AUDIO_LENGTH = 16000              # Fixed input segment length for static export.

# ─── Stream-VAD Fixed Chunk ──────────────────────────────────────────────────
STREAM_CHUNK_MS = 160                   # [80, 160, 200] Fixed streaming chunk size (ms).
STREAM_CHUNK_SAMPLES = int(IN_SAMPLE_RATE * STREAM_CHUNK_MS / 1000)  # 2560 samples at IN_SAMPLE_RATE.

# ═══════════════════════════════════════════════════════════════════════════════
# CMVN Loader (reads Kaldi-format cmvn.ark)
# ═══════════════════════════════════════════════════════════════════════════════

def load_cmvn(cmvn_file):
    """
    Load Kaldi CMVN statistics from an ark file.
    Returns: (means, inverse_std_variances) as torch tensors of shape (dim,).
    """
    import kaldiio
    stats = torch.from_numpy(np.array(kaldiio.load_mat(cmvn_file), copy=True)).float()
    assert stats.shape[0] == 2
    dim = stats.shape[-1] - 1
    count = stats[0, dim]
    assert count >= 1
    floor = 1e-20
    means = torch.zeros(dim, dtype=torch.float32)
    inv_std = torch.zeros(dim, dtype=torch.float32)
    for d in range(dim):
        mean = stats[0, d] / count
        means[d] = mean
        variance = (stats[1, d] / count) - mean * mean
        if variance < floor:
            variance = floor
        inv_std[d] = 1.0 / math.sqrt(variance.item())
    return means, inv_std


def build_kaldi_mel_filterbank(n_fft, n_mels, sample_rate, low_freq=20.0, high_freq=0.0):
    """
    Build a Kaldi-compatible mel filterbank matrix.
    Kaldi mel scale: linear below 1000 Hz, logarithmic above.
    Returns: torch.Tensor of shape [n_mels, n_fft//2+1]
    """
    if high_freq <= 0:
        high_freq = sample_rate / 2.0 + high_freq

    def mel(f):
        """Kaldi mel scale (not HTK)."""
        if f < 1000.0:
            return f
        else:
            return 1000.0 + 1000.0 * math.log(f / 1000.0) / math.log(2.0)

    def inv_mel(m):
        """Inverse Kaldi mel scale."""
        if m < 1000.0:
            return m
        else:
            return 1000.0 * math.exp((m - 1000.0) * math.log(2.0) / 1000.0)

    num_fft_bins = n_fft // 2 + 1
    mel_low = mel(low_freq)
    mel_high = mel(high_freq)

    # Center frequencies of mel bins
    mel_centers = torch.linspace(mel_low, mel_high, n_mels + 2)
    hz_centers = torch.tensor([inv_mel(m.item()) for m in mel_centers], dtype=torch.float32)

    # FFT bin frequencies
    fft_freqs = torch.linspace(0, sample_rate / 2.0, num_fft_bins)

    # Build triangular filterbank
    filterbank = torch.zeros(n_mels, num_fft_bins, dtype=torch.float32)
    for i in range(n_mels):
        lower = hz_centers[i]
        center = hz_centers[i + 1]
        upper = hz_centers[i + 2]
        for j in range(num_fft_bins):
            freq = fft_freqs[j]
            if lower <= freq <= center and center > lower:
                filterbank[i, j] = (freq - lower) / (center - lower)
            elif center < freq <= upper and upper > center:
                filterbank[i, j] = (upper - freq) / (upper - center)

    return filterbank


# ═══════════════════════════════════════════════════════════════════════════════
# Optimized FSMN / DFSMN Model Components (Channel-First Conv1d Architecture)
#
# Optimizations applied:
#   - All Linear layers replaced with Conv1d(kernel_size=1) to keep data in
#     channel-first [N, C, T] format throughout, eliminating all internal permutes.
#   - Lookback filter uses left-only padding (F.pad + Conv1d(padding=0)) to
#     eliminate the post-conv slice operation.
#   - CMVN normalization fused into the first Conv1d layer weights at load time.
#   - Pre-emphasis computed via fixed Conv1d kernel (Pad + Conv = 2 ops total).
#   - Mel features stay in [N, n_mels, T] channel-first format (no permute needed).
#   - Single permute at the output only ([N, odim, T] -> [N, T, odim]).
# ═══════════════════════════════════════════════════════════════════════════════
def pack_fsmn_memory_kernel(lookback_weight, lookahead_weight, n1, s1, n2, s2):
    """Pack lookback, identity, and lookahead into one symmetric depthwise kernel."""
    left_context = (n1 - 1) * s1
    right_context = n2 * s2
    compact_same_padding = right_context in (left_context, left_context + 1)
    if compact_same_padding:
        kernel_size = left_context + right_context + 1
        center = left_context
    else:
        center = max(left_context, right_context)
        kernel_size = 2 * center + 1
    packed = lookback_weight.new_zeros(lookback_weight.shape[0], 1, kernel_size)

    for tap in range(n1):
        offset = -left_context + tap * s1
        packed[:, :, center + offset] = lookback_weight[:, :, tap]

    packed[:, :, center] += 1.0
    if lookahead_weight is not None:
        for tap in range(n2):
            offset = (tap + 1) * s2
            packed[:, :, center + offset] = lookahead_weight[:, :, tap]

    return packed


class FSMN(torch.nn.Module):
    """
    Channel-first FSMN layer. Input/output: [N, P, T].
    Uses left-only padding for causal lookback (eliminates post-conv slice).
    """
    def __init__(self, P, N1, S1, N2=0, S2=0):
        super().__init__()
        self.N1 = N1
        self.S1 = S1
        self.N2 = N2
        self.S2 = S2
        left_context = (N1 - 1) * S1
        right_context = N2 * S2
        compact_same_padding = right_context in (left_context, left_context + 1)
        if compact_same_padding:
            kernel_size = left_context + right_context + 1
            padding = 'same'
        else:
            memory_radius = max(left_context, right_context)
            kernel_size = 2 * memory_radius + 1
            padding = memory_radius
        self.memory_filter = torch.nn.Conv1d(
            in_channels=P, out_channels=P,
            kernel_size=kernel_size, stride=1,
            padding=padding,
            groups=P, bias=False
        )

    def forward(self, x):
        """
        Args:
            x: [N, P, T] - channel-first
        Returns:
            memory: [N, P, T]
        """
        return self.memory_filter(x)


class DFSMNBlock(torch.nn.Module):
    """
    Channel-first DFSMN Block. All ops in [N, C, T] format (no permutes).
    Conv1d(kernel=1) replaces Linear for channel-first processing.
    """
    def __init__(self, H, P, N1, S1, N2=0, S2=0):
        super().__init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Conv1d(P, H, 1, bias=True),
            torch.nn.ReLU()
        )
        self.fc2 = torch.nn.Conv1d(H, P, 1, bias=False)
        self.fsmn = FSMN(P, N1, S1, N2, S2)

    def forward(self, inputs):
        """
        Args:
            inputs: [N, P, T]
        Returns:
            output: [N, P, T]
        """
        h = self.fc1(inputs)
        p = self.fc2(h)
        memory = self.fsmn(p)
        return memory + inputs


class DFSMN(torch.nn.Module):
    """
    Channel-first DFSMN. All processing in [N, C, T] format.
    Conv1d(kernel=1) replaces Linear; no internal permutes.
    """
    def __init__(self, D, R, M, H, P, N1, S1, N2=0, S2=0):
        super().__init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Conv1d(D, H, 1, bias=True),
            torch.nn.ReLU()
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Conv1d(H, P, 1, bias=True),
            torch.nn.ReLU()
        )
        self.fsmn1 = FSMN(P, N1, S1, N2, S2)
        self.fsmns = torch.nn.ModuleList([
            DFSMNBlock(H, P, N1, S1, N2, S2) for _ in range(R - 1)
        ])
        dnn = [torch.nn.Conv1d(P, H, 1, bias=True), torch.nn.ReLU()]
        for _ in range(M - 1):
            dnn += [torch.nn.Conv1d(H, H, 1, bias=True), torch.nn.ReLU()]
        self.dnns = torch.nn.Sequential(*dnn)

    def forward(self, inputs):
        """
        Args:
            inputs: [N, D, T] - channel-first
        Returns:
            output: [N, H, T]
        """
        h = self.fc1(inputs)
        p = self.fc2(h)
        memory = self.fsmn1(p)
        for fsmn_block in self.fsmns:
            memory = fsmn_block(memory)
        return self.dnns(memory)


class DetectModel(torch.nn.Module):
    """
    Channel-first DetectModel. CMVN fused into first layer at load time.
    Input: [N, D, T], Output: [N, odim, T].
    """
    def __init__(self, args):
        super().__init__()
        self.dfsmn = DFSMN(
            args.idim, args.R, args.M, args.H, args.P,
            args.N1, args.S1, args.N2, args.S2
        )
        self.out = torch.nn.Conv1d(args.H, args.odim, 1)

    def forward(self, feat):
        """
        Args:
            feat: [N, D, T] - channel-first features (CMVN pre-fused into weights)
        Returns:
            probs: [N, odim, T] - sigmoid probabilities
        """
        x = self.dfsmn(feat)
        return torch.sigmoid(self.out(x))

    @classmethod
    def from_pretrained(cls, model_dir, cmvn_means=None, cmvn_inv_std=None):
        """
        Load pretrained weights and convert Linear->Conv1d format.
        Optionally fuse CMVN into the first layer weights.
        """
        import os
        model_path = os.path.join(model_dir, "model.pth.tar")
        package = torch.load(model_path, map_location='cpu', weights_only=False)
        model = cls(package["args"])

        new_state = migrate_checkpoint_state(model, package["model_state_dict"])

        # Fuse CMVN into first Conv1d layer: eliminates runtime sub + mul
        # Original: y = (x - means) * inv_std; z = W @ y + b
        # Fused:    z = (W * inv_std) @ x + (b - W @ (means * inv_std))
        if cmvn_means is not None and cmvn_inv_std is not None:
            fc1_w_key = 'dfsmn.fc1.0.weight'  # [H, D, 1]
            fc1_b_key = 'dfsmn.fc1.0.bias'    # [H]
            W = new_state[fc1_w_key]           # [H, D, 1]
            b = new_state[fc1_b_key]           # [H]
            means_scaled = cmvn_means * cmvn_inv_std  # [D]
            new_state[fc1_w_key] = W * cmvn_inv_std.view(1, -1, 1)
            new_state[fc1_b_key] = b - torch.mv(W.squeeze(-1), means_scaled)

        model.load_state_dict(new_state, strict=True)
        model.eval()
        return model


# ═══════════════════════════════════════════════════════════════════════════════
# FireRedVAD Wrapper for ONNX Export
#
# Pre-emphasis is folded into the immutable Conv1d STFT kernel when resampling
# does not follow pre-emphasis. A symmetric one-sample Conv pad plus a trailing
# zero coefficient preserves the original left-only boundary without ONNX Pad.
# CMVN fused into model weights at load time (zero runtime ops).
# All channel-first [N, C, T] format throughout (no permute needed).
#
# ONNX Graph (forward path):
#   Cast(input→float32) → optional PCM scale → Conv(preemphasis+stft) →
#   Mul → Split → Add → Conv(mel_bank, k=1) → Clip → Log → [DFSMN] → Sigmoid
# ═══════════════════════════════════════════════════════════════════════════════

class FireRedLogMelFrontend(torch.nn.Module):
    def __init__(self, nfft, hop_length, win_length, n_mels, sample_rate,
                 pre_emphasis, window_type, in_sample_rate):
        super().__init__()
        self.input_scale = 32768.0 if IN_AUDIO_DTYPE != 'INT16' else None
        self.hop_length = hop_length

        self.in_sample_rate_scale = in_sample_rate / 16000.0
        self.model_rate_scale = 1.0 / self.in_sample_rate_scale
        self.resample_before = self.in_sample_rate_scale > 1.0
        self.resample_after = self.in_sample_rate_scale < 1.0

        stft = STFT_Process(
            model_type='stft_B',
            n_fft=nfft,
            win_length=win_length,
            hop_len=hop_length,
            window_type=window_type,
            center_pad=False,  # SNIP_EDGES=True → no center padding
            pad_mode='constant'
        )
        stft_kernel = stft.stft_kernel.detach()

        fbank = build_kaldi_mel_filterbank(
            n_fft=nfft, n_mels=n_mels, sample_rate=sample_rate,
            low_freq=20.0, high_freq=0.0
        )
        active_frequencies = torch.any(fbank != 0.0, dim=0)
        full_f_bins = nfft // 2 + 1
        real_kernel = stft_kernel[:full_f_bins][active_frequencies]
        imag_kernel = stft_kernel[full_f_bins:][active_frequencies]
        stft_kernel = torch.cat([real_kernel, imag_kernel], dim=0)
        fbank = fbank[:, active_frequencies]
        self.f_bins = int(active_frequencies.sum().item())

        self.fused_preemphasis = not self.resample_after
        if self.fused_preemphasis:
            frontend_kernel = stft_kernel.new_zeros(
                stft_kernel.shape[0], 1, nfft + 2
            )
            frontend_kernel[:, :, :nfft] = -pre_emphasis * stft_kernel
            frontend_kernel[:, :, 1:nfft + 1] += stft_kernel
            if self.input_scale is not None and not self.resample_before:
                frontend_kernel *= self.input_scale
                self.input_scale = None
            self.register_buffer('frontend_kernel', frontend_kernel)
        else:
            preemph_kernel = torch.tensor(
                [[[-pre_emphasis, 1.0, 0.0]]], dtype=torch.float32
            )
            self.register_buffer('preemph_kernel', preemph_kernel)
            self.register_buffer('stft_kernel', stft_kernel.clone())

        self.register_buffer('fbank_conv', fbank.unsqueeze(-1))

    def forward(self, audio):
        audio = audio.float()
        if self.input_scale is not None:
            audio = audio * self.input_scale

        if self.resample_before:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )

        if self.fused_preemphasis:
            spectrum = torch.nn.functional.conv1d(
                audio,
                self.frontend_kernel,
                stride=self.hop_length,
                padding=1,
            )
        else:
            audio = torch.nn.functional.conv1d(
                audio,
                self.preemph_kernel,
                padding=1,
            )
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
            spectrum = torch.nn.functional.conv1d(
                audio,
                self.stft_kernel,
                stride=self.hop_length,
            )

        spectrum = spectrum * spectrum
        real_power, imag_power = torch.split(spectrum, self.f_bins, dim=1)
        power_spectrum = real_power + imag_power
        mel_features = torch.nn.functional.conv1d(power_spectrum, self.fbank_conv)
        return torch.clamp(mel_features, min=1e-07).log()


class FireRedVAD_ONNX(torch.nn.Module):
    """
    ONNX-exportable FireRedVAD/AED model with an optimized log-mel frontend.
    Input: configured F16/F32/INT16 audio [1, 1, audio_len]
    Output: probs [1, odim, signal_len]
    """
    def __init__(self, detect_model, nfft, hop_length, win_length, n_mels,
                 sample_rate, pre_emphasis, window_type, in_sample_rate=16000):
        super(FireRedVAD_ONNX, self).__init__()
        self.detect_model = detect_model
        self.frontend = FireRedLogMelFrontend(
            nfft,
            hop_length,
            win_length,
            n_mels,
            sample_rate,
            pre_emphasis,
            window_type,
            in_sample_rate,
        )

    def forward(self, audio):
        return self.detect_model(self.frontend(audio))


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming FSMN / DFSMN (Channel-First with Cache, Optimized)
#
# Optimizations:
#   - All in [N, C, T] format (zero internal permutes)
#   - Cache concat + Conv1d(padding=0) outputs exactly T frames (no slice)
#   - Conv1d(kernel=1) replaces Linear
# ═══════════════════════════════════════════════════════════════════════════════

class FSMN_Streaming(torch.nn.Module):
    """
    Channel-first streaming FSMN with cache. Input/output: [1, P, T].
    Cache prepend + Conv1d(padding=0) gives exact T output (no slice needed).
    """
    def __init__(self, P, N1, S1):
        super().__init__()
        self.N1 = N1
        self.S1 = S1
        self.lookback_padding = (N1 - 1) * S1
        self.memory_filter = torch.nn.Conv1d(
            in_channels=P, out_channels=P,
            kernel_size=N1, stride=1,
            padding=0, dilation=S1,
            groups=P, bias=False
        )

    def forward(self, x, cache):
        """
        Args:
            x: [1, P, T] - channel-first
            cache: [1, P, lookback_padding]
        Returns:
            memory: [1, P, T]
        """
        # Concat cache + input: [1, P, lookback_padding + T]
        x_with_cache = torch.cat([cache, x], dim=2)

        # Conv1d on [1, P, lookback_padding + T] with padding=0, dilation=S1, kernel=N1
        # Output length = lookback_padding + T - (N1-1)*S1 = T (exact, no slice needed)
        lookback = self.memory_filter(x_with_cache)

        return x + lookback


class DFSMNBlock_Streaming(torch.nn.Module):
    """
    Channel-first streaming DFSMN Block with cache. No permutes.
    """
    def __init__(self, H, P, N1, S1):
        super().__init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Conv1d(P, H, 1, bias=True),
            torch.nn.ReLU()
        )
        self.fc2 = torch.nn.Conv1d(H, P, 1, bias=False)
        self.fsmn = FSMN_Streaming(P, N1, S1)

    def forward(self, inputs, cache):
        """
        Args:
            inputs: [1, P, T]
            cache: [1, P, lookback_padding]
        Returns:
            output: [1, P, T]
            cache_value: [1, P, T]
        """
        h = self.fc1(inputs)
        p = self.fc2(h)
        memory = self.fsmn(p, cache)
        return memory + inputs, p


class DFSMN_Streaming(torch.nn.Module):
    """
    Channel-first streaming DFSMN with explicit cache management.
    """
    def __init__(self, D, R, M, H, P, N1, S1):
        super().__init__()
        self.R = R
        self.fc1 = torch.nn.Sequential(
            torch.nn.Conv1d(D, H, 1, bias=True),
            torch.nn.ReLU()
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Conv1d(H, P, 1, bias=True),
            torch.nn.ReLU()
        )
        self.fsmn1 = FSMN_Streaming(P, N1, S1)
        self.fsmns = torch.nn.ModuleList([
            DFSMNBlock_Streaming(H, P, N1, S1) for _ in range(R - 1)
        ])
        dnn = [torch.nn.Conv1d(P, H, 1, bias=True), torch.nn.ReLU()]
        for _ in range(M - 1):
            dnn += [torch.nn.Conv1d(H, H, 1, bias=True), torch.nn.ReLU()]
        self.dnns = torch.nn.Sequential(*dnn)

    def forward(self, inputs, caches_in):
        """
        Args:
            inputs: [1, D, T] - channel-first
            caches_in: [R, 1, P, lookback_padding]
        Returns:
            output: [1, H, T]
            caches_out: [R, 1, P, lookback_padding]
        """
        h = self.fc1(inputs)
        p = self.fc2(h)
        memory = self.fsmn1(p, caches_in[0])
        cache_values = [p]

        for i, fsmn_block in enumerate(self.fsmns):
            memory, cache_value = fsmn_block(memory, caches_in[i + 1])
            cache_values.append(cache_value)

        output = self.dnns(memory)
        cache_values = torch.cat(cache_values, dim=0).unsqueeze(1)
        caches_out = torch.cat([caches_in, cache_values], dim=3)
        caches_out = caches_out[:, :, :, -self.fsmn1.lookback_padding:]
        return output, caches_out


class DetectModel_Streaming(torch.nn.Module):
    """
    Channel-first streaming DetectModel with cache. CMVN fused into weights.
    """
    def __init__(self, args):
        super().__init__()
        self.dfsmn = DFSMN_Streaming(
            args.idim, args.R, args.M, args.H, args.P,
            args.N1, args.S1
        )
        self.out = torch.nn.Conv1d(args.H, args.odim, 1)
        self.R = args.R
        self.P = args.P
        self.lookback_padding = (args.N1 - 1) * args.S1

    def forward(self, feat, caches_in):
        """
        Args:
            feat: [1, D, T] - channel-first
            caches_in: [R, 1, P, lookback_padding]
        Returns:
            probs: [1, odim, T]
            caches_out: [R, 1, P, lookback_padding]
        """
        x, caches_out = self.dfsmn(feat, caches_in)
        return torch.sigmoid(self.out(x)), caches_out

    @classmethod
    def from_pretrained(cls, model_dir, cmvn_means=None, cmvn_inv_std=None):
        """
        Load pretrained weights, convert Linear->Conv1d, fuse CMVN.
        """
        import os
        model_path = os.path.join(model_dir, "model.pth.tar")
        package = torch.load(model_path, map_location='cpu', weights_only=False)
        model = cls(package["args"])

        new_state = migrate_checkpoint_state(model, package["model_state_dict"])

        # Fuse CMVN into first layer
        if cmvn_means is not None and cmvn_inv_std is not None:
            fc1_w_key = 'dfsmn.fc1.0.weight'
            fc1_b_key = 'dfsmn.fc1.0.bias'
            W = new_state[fc1_w_key]
            b = new_state[fc1_b_key]
            means_scaled = cmvn_means * cmvn_inv_std
            new_state[fc1_w_key] = W * cmvn_inv_std.view(1, -1, 1)
            new_state[fc1_b_key] = b - torch.mv(W.squeeze(-1), means_scaled)

        model.load_state_dict(new_state, strict=True)
        model.eval()
        return model


def migrate_checkpoint_state(model, original_state):
    """Convert the original Linear/FSMN checkpoint into export-optimized weights."""
    converted = {}
    consumed = set()

    for module_name, module in model.named_modules():
        if isinstance(module, FSMN):
            prefix = f'{module_name}.' if module_name else ''
            lookback_key = prefix + 'lookback_filter.weight'
            lookahead_key = prefix + 'lookahead_filter.weight'
            memory_key = prefix + 'memory_filter.weight'
            lookahead_weight = (
                original_state[lookahead_key] if module.N2 > 0 else None
            )
            converted[memory_key] = pack_fsmn_memory_kernel(
                original_state[lookback_key],
                lookahead_weight,
                module.N1,
                module.S1,
                module.N2,
                module.S2,
            )
            consumed.add(lookback_key)
            if lookahead_weight is not None:
                consumed.add(lookahead_key)
        elif isinstance(module, FSMN_Streaming):
            prefix = f'{module_name}.' if module_name else ''
            lookback_key = prefix + 'lookback_filter.weight'
            memory_key = prefix + 'memory_filter.weight'
            converted[memory_key] = original_state[lookback_key]
            consumed.add(lookback_key)

    for key, value in original_state.items():
        if key in consumed:
            continue
        if 'weight' in key and value.dim() == 2:
            converted[key] = value.unsqueeze(-1)
        else:
            converted[key] = value

    return converted


class FireRedStreamVAD_ONNX(torch.nn.Module):
    """
    ONNX-exportable FireRedVAD Streaming model using STFT_Process for the STFT stage.
    Input: configured F16/F32/INT16 audio [1, 1, audio_len],
           caches_in [R, 1, P, lookback_padding]
    Output: probs [1, 1, T], caches_out [R, 1, P, lookback_padding]

    Dynamic axis: audio dim=2 is dynamic because streaming chunk sizes vary.
    This only affects the Conv1d input length — no Shape/Gather/Range ops introduced.
    """
    def __init__(self, detect_model_streaming, nfft, hop_length, win_length,
                 n_mels, sample_rate, pre_emphasis, window_type, in_sample_rate=16000):
        super(FireRedStreamVAD_ONNX, self).__init__()
        self.detect_model = detect_model_streaming
        self.frontend = FireRedLogMelFrontend(
            nfft,
            hop_length,
            win_length,
            n_mels,
            sample_rate,
            pre_emphasis,
            window_type,
            in_sample_rate,
        )

    def forward(self, audio, caches_in):
        """
        Args:
            audio: [1, 1, audio_len] configured F16/F32/INT16 tensor
            caches_in: [R, 1, P, lookback_padding] float32 tensor
        Returns:
            probs: [1, 1, T] float32 tensor
            caches_out: [R, 1, P, lookback_padding] float32 tensor
        """
        return self.detect_model(self.frontend(audio), caches_in)


# ═══════════════════════════════════════════════════════════════════════════════
# Export Process
# ═══════════════════════════════════════════════════════════════════════════════

def build_firered_metadata(model_type):
    if model_type == 'vad':
        model_name = 'FireRedVAD'
        task = 'voice_activity_detection'
        dynamic_axes = DYNAMIC_AXES
        input_audio_length = INPUT_AUDIO_LENGTH
        extra = {}
    elif model_type == 'aed':
        model_name = 'FireRedAED'
        task = 'audio_event_detection'
        dynamic_axes = DYNAMIC_AXES
        input_audio_length = INPUT_AUDIO_LENGTH
        extra = {
            'event_labels': ('speech', 'singing', 'music'),
        }
    elif model_type == 'stream_vad':
        model_name = 'FireRedStreamVAD'
        task = 'streaming_voice_activity_detection'
        dynamic_axes = True
        input_audio_length = STREAM_CHUNK_SAMPLES
        extra = {}
    else:
        raise ValueError(f'Unsupported FireRed model type: {model_type!r}')

    return {
        'vad_metadata_version': 1,
        'producer': Path(__file__).name,
        'model_name': model_name,
        'task': task,
        'model_family': 'fireredvad_dfsmn',
        'dynamic_axes': dynamic_axes,
        'opset': OPSET,
        'input_audio_dtype': IN_METADATA_DTYPE,
        'input_sample_rate': IN_SAMPLE_RATE,
        'model_sample_rate': SAMPLE_RATE,
        'input_audio_length': input_audio_length,
        'output_frame_shift_seconds': HOP_LENGTH / SAMPLE_RATE,
        'output_frame_length_seconds': WINDOW_LENGTH / SAMPLE_RATE,
        'window_length_samples': WINDOW_LENGTH,
        'hop_length_samples': HOP_LENGTH,
        **extra,
    }


def finalize_export(staged_raw_path, final_path, model_type):
    write_onnx_metadata(
        staged_raw_path,
        build_firered_metadata(model_type),
    )
    raw_model = onnx.load(staged_raw_path, load_external_data=True)
    onnx.checker.check_model(raw_model)
    onnx.shape_inference.infer_shapes(raw_model, check_type=True, strict_mode=True)
    del raw_model
    gc.collect()

    optimize_graph(
        staged_raw_path,
        final_path,
        model_type,
    )

    final_model = onnx.load(final_path, load_external_data=True)
    onnx.checker.check_model(final_model, check_custom_domain=False)
    if not final_model.metadata_props:
        raise RuntimeError(f'Embedded metadata is missing from {final_path}.')
    del final_model
    gc.collect()
    print(f'Final deployment model: {final_path}')


def export_model(model_path, final_path, model_type='vad'):
    """Export a FireRedVAD or FireRedAED model to ONNX (static graph, fused pre-emphasis+STFT)."""
    print(f'\nExport {model_type.upper()} start ...')

    with tempfile.TemporaryDirectory(
        prefix=f'firered_{model_type}_raw_',
        dir=export_folder,
    ) as directory, torch.inference_mode():
        staged_raw_path = Path(directory) / f'{Path(final_path).stem}_raw.onnx'
        # Load CMVN
        cmvn_file = os.path.join(model_path, "cmvn.ark")
        cmvn_means, cmvn_inv_std = load_cmvn(cmvn_file)

        # Load DetectModel with CMVN fused into first layer weights
        detect_model = DetectModel.from_pretrained(
            model_path, cmvn_means=cmvn_means, cmvn_inv_std=cmvn_inv_std
        )

        # Build the optimized export wrapper (fused pre-emphasis + STFT, no STFT_Process)
        model = FireRedVAD_ONNX(
            detect_model=detect_model,
            nfft=NFFT,
            hop_length=HOP_LENGTH,
            win_length=WINDOW_LENGTH,
            n_mels=N_MELS,
            sample_rate=SAMPLE_RATE,
            pre_emphasis=PRE_EMPHASIZE,
            window_type=WINDOW_TYPE,
            in_sample_rate=IN_SAMPLE_RATE
        ).eval()

        # Fixed input shape for static graph
        audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=IN_TORCH_DTYPE)

        output_names = ['probs']
        dynamic_axes_dict = {
            'audio': {2: 'audio_len'},
            'probs': {2: 'signal_len'}
        } if DYNAMIC_AXES else None

        # Export to ONNX (static graph by default)
        torch.onnx.export(
            model,
            (audio,),
            staged_raw_path,
            input_names=['audio'],
            output_names=output_names,
            export_params=True,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes_dict,
            opset_version=OPSET,
            training=torch.onnx.TrainingMode.EVAL,
            keep_initializers_as_inputs=False,
            dynamo=False
        )
        del model
        del detect_model
        del audio
        gc.collect()

        finalize_export(staged_raw_path, final_path, model_type)
    print(f'Export {model_type.upper()} done!')


def export_stream_vad_model(model_path, final_path):
    """Export a FireRedVAD Stream-VAD model to ONNX with cache input/output."""
    print(f'\nExport STREAM-VAD start ...')

    with tempfile.TemporaryDirectory(
        prefix='firered_stream_vad_raw_',
        dir=export_folder,
    ) as directory, torch.inference_mode():
        staged_raw_path = Path(directory) / f'{Path(final_path).stem}_raw.onnx'
        # Load CMVN
        cmvn_file = os.path.join(model_path, "cmvn.ark")
        cmvn_means, cmvn_inv_std = load_cmvn(cmvn_file)

        # Load DetectModel in streaming mode with CMVN fused into first layer
        detect_model = DetectModel_Streaming.from_pretrained(
            model_path, cmvn_means=cmvn_means, cmvn_inv_std=cmvn_inv_std
        )
        R = detect_model.R
        P = detect_model.P
        lookback_padding = detect_model.lookback_padding

        # Build the streaming export wrapper (fused pre-emphasis + STFT)
        model = FireRedStreamVAD_ONNX(
            detect_model_streaming=detect_model,
            nfft=NFFT,
            hop_length=HOP_LENGTH,
            win_length=WINDOW_LENGTH,
            n_mels=N_MELS,
            sample_rate=SAMPLE_RATE,
            pre_emphasis=PRE_EMPHASIZE,
            window_type=WINDOW_TYPE,
            in_sample_rate=IN_SAMPLE_RATE
        ).eval()

        # Fixed chunk size + zero-initialized caches
        audio = torch.ones((1, 1, STREAM_CHUNK_SAMPLES), dtype=IN_TORCH_DTYPE)
        caches_in = torch.zeros((R, 1, P, lookback_padding), dtype=torch.float32)

        print(f'  Model info: R={R}, P={P}, lookback_padding={lookback_padding}')
        print(f'  Cache shape: [{R}, 1, {P}, {lookback_padding}]')
        print(f'  Chunk size: {STREAM_CHUNK_SAMPLES} samples ({STREAM_CHUNK_MS}ms)')

        # Stream-VAD uses dynamic audio axis because chunk sizes vary at runtime.
        # Cache dimensions are static (fixed model architecture).
        dynamic_axes_dict = {
            'audio': {2: 'audio_len'},
            'probs': {2: 'signal_len'}
        }

        # Export to ONNX
        torch.onnx.export(
            model,
            (audio, caches_in),
            staged_raw_path,
            input_names=['audio', 'caches_in'],
            output_names=['probs', 'caches_out'],
            export_params=True,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes_dict,
            opset_version=OPSET,
            training=torch.onnx.TrainingMode.EVAL,
            keep_initializers_as_inputs=False,
            dynamo=False
        )
        del model
        del detect_model
        del audio
        del caches_in
        gc.collect()

        finalize_export(staged_raw_path, final_path, 'stream_vad')
    print(f'Export STREAM-VAD done!')
    return R, P, lookback_padding


# ═══════════════════════════════════════════════════════════════════════════════
# Main: Export and Run Inference
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    export_folder.mkdir(parents=True, exist_ok=True)

    if EXPORT_VAD:
        export_model(
            model_path_vad,
            onnx_model_vad,
            model_type='vad',
        )
    if EXPORT_AED:
        export_model(
            model_path_aed,
            onnx_model_aed,
            model_type='aed',
        )
    if EXPORT_STREAM_VAD:
        export_stream_vad_model(
            model_path_stream_vad,
            onnx_model_stream_vad,
        )

    inference_script = script_dir / "Inference_FireRedVAD_ONNX.py"
    print(
        f'\nStarting inference demo with {inference_script.name} using: '
        f'{export_folder.resolve()}'
    )
    subprocess.run(
        [sys.executable, str(inference_script), str(export_folder.resolve())],
        cwd=script_dir,
        check=True,
    )


if __name__ == '__main__':
    main()
