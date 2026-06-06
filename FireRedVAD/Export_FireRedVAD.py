import gc
import math
import time
import torch
import numpy as np
import onnxruntime
from datetime import timedelta
from pydub import AudioSegment
from STFT_Process import STFT_Process


download_path           = "/home/DakeQQ/Downloads/FireRedVAD/"                                # The FireRedVAD directory
model_path_vad          = download_path + "VAD"                                               # The FireRedVAD VAD model directory (contains model.pth.tar and cmvn.ark).
model_path_aed          = download_path + "AED"                                               # The FireRedVAD AED model directory (contains model.pth.tar and cmvn.ark).
model_path_stream_vad   = download_path + "Stream-VAD"                                        # The FireRedVAD Stream-VAD model directory (contains model.pth.tar and cmvn.ark).
onnx_model_vad          = "/home/DakeQQ/Downloads/FireRedVAD_ONNX/FireRedVAD.onnx"            # The exported VAD onnx model path.
onnx_model_aed          = "/home/DakeQQ/Downloads/FireRedVAD_ONNX/FireRedAED.onnx"            # The exported AED onnx model path.
onnx_model_stream_vad   = "/home/DakeQQ/Downloads/FireRedVAD_ONNX/FireRedStreamVAD.onnx"      # The exported Stream-VAD onnx model path.
test_vad_audio          = "./vad_sample.wav"                                                  # The VAD test audio path.
test_aed_audio          = "./vad_sample.wav"                                                  # The AED test audio path.
save_timestamps_second  = "./timestamps_second.txt"                                           # The saved path.
save_timestamps_indices = "./timestamps_indices.txt"                                          # The saved path.


# ─── ONNX Export Settings ─────────────────────────────────────────────────────
EXPORT_VAD = True                       # Export the VAD model.
EXPORT_AED = True                       # Export the AED model.
EXPORT_STREAM_VAD = True                # Export the Stream-VAD model.
DYNAMIC_AXES = False                    # Static graph (no dynamic axes). Set True only if variable-length input is required.
NORMALIZE_AUDIO = False                 # Normalize input audio to target RMS level.
OPSET = 18                              # ONNX opset version.
ORT_Accelerate_Providers = []           # e.g. ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'MIGraphXExecutionProvider']


IN_SAMPLE_RATE = 16000                  # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.

# ─── Audio & STFT Parameters (do not edit) ────────────────────────────────────
SAMPLE_RATE = 16000                     # The model native rate, do not edit the value.
FRAME_SHIFT_MS = 10                     # Frame shift (ms).
FRAME_LENGTH_MS = 25                    # Frame length (ms).
N_MELS = 80                             # Mel filterbank bands.
NFFT = 400                              # FFT size (frame_length_ms * sample_rate / 1000).
HOP_LENGTH = 160                        # Samples between frames (frame_shift_ms * sample_rate / 1000).
WINDOW_LENGTH = 400                     # Window length in samples (frame_length_ms * sample_rate / 1000).
WINDOW_TYPE = 'povey'                   # Kaldi window: 'povey' = hann^0.85.
PRE_EMPHASIZE = 0.97                    # Pre-emphasis coefficient (Kaldi default).
SNIP_EDGES = True                       # First frame starts at sample 0 (no center padding).
INPUT_AUDIO_LENGTH = 16000              # Fixed input segment length for static export.
OUTPUT_FRAME_SHIFT = HOP_LENGTH         # Output frame shift in samples.

# ─── Stream-VAD Fixed Chunk ──────────────────────────────────────────────────
STREAM_CHUNK_MS = 160                   # [80, 160, 200] Fixed streaming chunk size (ms).
STREAM_CHUNK_SAMPLES = int(IN_SAMPLE_RATE * STREAM_CHUNK_MS / 1000)  # 2560 samples at IN_SAMPLE_RATE.
# Dynamic axes justification for Stream-VAD:
#   The streaming model requires dynamic audio_len because:
#   - Different deployment scenarios use different chunk sizes (80ms, 160ms, 200ms).
#   - The last audio chunk in a stream is typically shorter than the standard chunk.
#   - Making it static would require the runtime to always zero-pad, adding latency.
#   Dynamic axis: audio input dim=2 ('audio_len').
#   This introduces no Shape/Gather/Range ops — only affects Conv1d input length.
STREAM_VAD_DYNAMIC_AXES = True          # Stream-VAD keeps dynamic audio axis.

# ─── Derived Constants ────────────────────────────────────────────────────────
FRAME_SHIFT_S = FRAME_SHIFT_MS / 1000.0
FRAME_LENGTH_S = FRAME_LENGTH_MS / 1000.0
FRAME_PER_SECONDS = int(1000 / FRAME_SHIFT_MS)
_FRAME_SHIFT_F32 = np.float32(FRAME_SHIFT_S)
_FRAME_LENGTH_F32 = np.float32(FRAME_LENGTH_S)

# ─── VAD Postprocessing ──────────────────────────────────────────────────────
SPEAKING_SCORE = 0.4                    # Speech probability threshold.
SILENCE_SCORE = 0.6                     # 1 - SPEAKING_SCORE; higher = easier to cut off speech.
MIN_SPEECH_FRAME = 20                   # Min speech frames to confirm a segment.
MAX_SPEECH_FRAME = 2000                 # Max speech frames before forced split.
MIN_SILENCE_FRAME = 20                  # Min silence frames to confirm end of speech.
MERGE_SILENCE_FRAME = 5                 # Merge silence gaps shorter than this (frames).
EXTEND_SPEECH_FRAME = 0                 # Extend speech regions by this many frames.
SMOOTH_WINDOW_SIZE = 5                  # Probability smoothing window size.

# ─── AED (Audio Event Detection) Postprocessing ──────────────────────────────
MIN_EVENT_FRAME = 20                    # Min event frames to confirm a segment.
MAX_EVENT_FRAME = 2000                  # Max event frames before forced split.
SINGING_THRESHOLD = 0.5                 # Singing detection threshold.
MUSIC_THRESHOLD = 0.5                   # Music detection threshold.

# ─── Stream-VAD Settings ─────────────────────────────────────────────────────
STREAM_VAD_THRESHOLD = 0.4              # Speech probability threshold.
PAD_START_FRAME = 5                     # Frames to pad at segment start.
MIN_SPEECH_FRAME_STREAM = 8             # Min speech frames to confirm a segment.
MAX_SPEECH_FRAME_STREAM = 2000          # Max speech frames before forced split.
MIN_SILENCE_FRAME_STREAM = 20           # Min silence frames to confirm end of speech.


# ═══════════════════════════════════════════════════════════════════════════════
# CMVN Loader (reads Kaldi-format cmvn.ark)
# ═══════════════════════════════════════════════════════════════════════════════

def load_cmvn(cmvn_file):
    """
    Load Kaldi CMVN statistics from an ark file.
    Returns: (means, inverse_std_variances) as torch tensors of shape (dim,).
    """
    import kaldiio
    stats = torch.from_numpy(kaldiio.load_mat(cmvn_file)).float()
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
        self.lookback_padding = (N1 - 1) * S1
        # No built-in padding; we apply left-only pad in forward
        self.lookback_filter = torch.nn.Conv1d(
            in_channels=P, out_channels=P,
            kernel_size=N1, stride=1,
            padding=0, dilation=S1,
            groups=P, bias=False
        )
        if self.N2 > 0:
            self.lookahead_padding = (N2 - 1) * S2
            self.lookahead_filter = torch.nn.Conv1d(
                in_channels=P, out_channels=P,
                kernel_size=N2, stride=1,
                padding=0, dilation=S2,
                groups=P, bias=False
            )

    def forward(self, x):
        """
        Args:
            x: [N, P, T] - channel-first
        Returns:
            memory: [N, P, T]
        """
        # Lookback: left-only causal padding -> conv outputs exactly T frames
        lookback = self.lookback_filter(
            torch.nn.functional.pad(x, (self.lookback_padding, 0))
        )
        memory = x + lookback

        # Lookahead (non-streaming only): right-only padding
        if self.N2 > 0:
            T = x.size(2)
            if T > 1:
                # Right-pad for future context
                x_padded = torch.nn.functional.pad(x, (0, self.N2 * self.S2))
                lookahead = self.lookahead_filter(x_padded)
                # Conv output is [N, P, T+S2]; skip first S2 for strictly-future alignment
                memory = memory + lookahead[:, :, self.S2:]

        return memory


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

        # Convert Linear weights [out, in] -> Conv1d weights [out, in, 1]
        orig_state = package["model_state_dict"]
        new_state = {}
        for k, v in orig_state.items():
            if 'lookback_filter' in k or 'lookahead_filter' in k:
                new_state[k] = v  # Already Conv1d [C, 1, K] format
            elif 'weight' in k and v.dim() == 2:
                new_state[k] = v.unsqueeze(-1)  # [out, in] -> [out, in, 1]
            else:
                new_state[k] = v  # bias [out] or other

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
# FireRedVAD Wrapper for ONNX Export (using STFT_Process module)
#
# Uses the shared STFT_Process (Conv1d-based STFT) with separate pre-emphasis.
# CMVN fused into model weights at load time (zero runtime ops).
# All channel-first [N, C, T] format throughout (no permute needed).
#
# ONNX Graph (forward path):
#   Cast(int16→float32) → Pad(1,0) → Conv(preemph) → Conv(stft) → Split →
#   Mul → Mul → Add → Conv(mel_bank, k=1) → Clip → Log → [DFSMN] → Sigmoid
# ═══════════════════════════════════════════════════════════════════════════════

class FireRedVAD_ONNX(torch.nn.Module):
    """
    ONNX-exportable FireRedVAD/AED model using STFT_Process for the STFT stage.
    Input: int16 audio [1, 1, audio_len]
    Output: probs [1, odim, signal_len]
    """
    def __init__(self, detect_model, nfft, hop_length, win_length, n_mels,
                 sample_rate, pre_emphasis, window_type, in_sample_rate=16000):
        super(FireRedVAD_ONNX, self).__init__()
        self.detect_model = detect_model

        # Sample rate interpolation (resample input to model's native 16000 Hz)
        self.in_sample_rate_scale = in_sample_rate / 16000.0
        self.model_rate_scale = 1.0 / self.in_sample_rate_scale
        self.resample_before = self.in_sample_rate_scale > 1.0
        self.resample_after = self.in_sample_rate_scale < 1.0

        # Pre-emphasis kernel: y[t] = x[t] - coeff * x[t-1]
        self.register_buffer(
            'preemph_kernel',
            torch.tensor([[[-pre_emphasis, 1.0]]], dtype=torch.float32)
        )

        # STFT via STFT_Process (Conv1d-based, ONNX-friendly)
        self.stft = STFT_Process(
            model_type='stft_B',
            n_fft=nfft,
            win_length=win_length,
            hop_len=hop_length,
            window_type=window_type,
            center_pad=False,  # SNIP_EDGES=True → no center padding
            pad_mode='constant'
        )

        # Mel filterbank as Conv1d kernel: [n_mels, F_bins, 1]
        fbank = build_kaldi_mel_filterbank(
            n_fft=nfft, n_mels=n_mels, sample_rate=sample_rate,
            low_freq=20.0, high_freq=0.0
        )
        self.register_buffer('fbank_conv', fbank.unsqueeze(-1))

    def forward(self, audio):
        """
        Args:
            audio: [1, 1, audio_len] int16 tensor
        Returns:
            probs: [1, odim, T] float32 tensor (sigmoid probabilities)
        """
        # 1. int16 → float32 (Cast)
        audio = audio.float()

        # 2. Resample to model's native 16000 Hz if input rate is higher
        if self.resample_before:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )

        # 3. Pre-emphasis: y[t] = x[t] - coeff * x[t-1]
        audio = torch.nn.functional.conv1d(torch.nn.functional.pad(audio, (1, 0)), self.preemph_kernel)

        # 4. Resample to model's native 16000 Hz if input rate is lower
        if self.resample_after:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )

        # 3. STFT via STFT_Process (Conv1d, no center padding)
        real_part, imag_part = self.stft(audio)

        # 4. Power spectrum [1, F_bins, T]
        power_spectrum = real_part * real_part + imag_part * imag_part

        # 5. Mel filterbank via Conv1d(kernel_size=1) [1, n_mels, T]
        mel_features = torch.nn.functional.conv1d(power_spectrum, self.fbank_conv)

        # 6. Log with floor (Clip + Log)
        mel_features = torch.clamp(mel_features, min=1e-07).log()

        # 7. Model forward (CMVN fused into first layer weights)
        probs = self.detect_model(mel_features)

        # 8. Output in channel-first format [1, odim, T]
        return probs


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
        self.lookback_filter = torch.nn.Conv1d(
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
            new_cache: [1, P, lookback_padding]
        """
        # Concat cache + input: [1, P, lookback_padding + T]
        x_with_cache = torch.cat([cache, x], dim=2)

        # Update cache: last lookback_padding samples
        new_cache = x_with_cache[:, :, -self.lookback_padding:]

        # Conv1d on [1, P, lookback_padding + T] with padding=0, dilation=S1, kernel=N1
        # Output length = lookback_padding + T - (N1-1)*S1 = T (exact, no slice needed)
        lookback = self.lookback_filter(x_with_cache)

        return x + lookback, new_cache


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
            new_cache: [1, P, lookback_padding]
        """
        h = self.fc1(inputs)
        p = self.fc2(h)
        memory, new_cache = self.fsmn(p, cache)
        return memory + inputs, new_cache


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
        memory, new_cache_0 = self.fsmn1(p, caches_in[0])
        new_caches = [new_cache_0]

        for i, fsmn_block in enumerate(self.fsmns):
            memory, new_cache_i = fsmn_block(memory, caches_in[i + 1])
            new_caches.append(new_cache_i)

        output = self.dnns(memory)
        caches_out = torch.stack(new_caches, dim=0)
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

        # Convert Linear weights [out, in] -> Conv1d weights [out, in, 1]
        orig_state = package["model_state_dict"]
        new_state = {}
        for k, v in orig_state.items():
            if 'lookback_filter' in k or 'lookahead_filter' in k:
                new_state[k] = v
            elif 'weight' in k and v.dim() == 2:
                new_state[k] = v.unsqueeze(-1)
            else:
                new_state[k] = v

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


class FireRedStreamVAD_ONNX(torch.nn.Module):
    """
    ONNX-exportable FireRedVAD Streaming model using STFT_Process for the STFT stage.
    Input: int16 audio [1, 1, audio_len], caches_in [R, 1, P, lookback_padding]
    Output: probs [1, 1, T], caches_out [R, 1, P, lookback_padding]

    Dynamic axis: audio dim=2 is dynamic because streaming chunk sizes vary.
    This only affects the Conv1d input length — no Shape/Gather/Range ops introduced.
    """
    def __init__(self, detect_model_streaming, nfft, hop_length, win_length,
                 n_mels, sample_rate, pre_emphasis, window_type, in_sample_rate=16000):
        super(FireRedStreamVAD_ONNX, self).__init__()
        self.detect_model = detect_model_streaming

        # Sample rate interpolation (resample input to model's native 16000 Hz)
        self.in_sample_rate_scale = in_sample_rate / 16000.0
        self.model_rate_scale = 1.0 / self.in_sample_rate_scale
        self.resample_before = self.in_sample_rate_scale > 1.0
        self.resample_after = self.in_sample_rate_scale < 1.0

        # Pre-emphasis kernel: y[t] = x[t] - coeff * x[t-1]
        self.register_buffer(
            'preemph_kernel',
            torch.tensor([[[-pre_emphasis, 1.0]]], dtype=torch.float32)
        )

        # STFT via STFT_Process (Conv1d-based, ONNX-friendly)
        self.stft = STFT_Process(
            model_type='stft_B',
            n_fft=nfft,
            win_length=win_length,
            hop_len=hop_length,
            window_type=window_type,
            center_pad=False,  # SNIP_EDGES=True → no center padding
            pad_mode='constant'
        )

        # Mel filterbank as Conv1d kernel: [n_mels, F_bins, 1]
        fbank = build_kaldi_mel_filterbank(
            n_fft=nfft, n_mels=n_mels, sample_rate=sample_rate,
            low_freq=20.0, high_freq=sample_rate // 2
        )
        self.register_buffer('fbank_conv', fbank.unsqueeze(-1))

    def forward(self, audio, caches_in):
        """
        Args:
            audio: [1, 1, audio_len] int16 tensor
            caches_in: [R, 1, P, lookback_padding] float32 tensor
        Returns:
            probs: [1, 1, T] float32 tensor
            caches_out: [R, 1, P, lookback_padding] float32 tensor
        """
        # 1. int16 → float32
        audio = audio.float()

        # 2. Resample to model's native 16000 Hz if input rate is higher
        if self.resample_before:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )

        # 3. Pre-emphasis: y[t] = x[t] - coeff * x[t-1]
        audio = torch.nn.functional.conv1d(torch.nn.functional.pad(audio, (1, 0)), self.preemph_kernel)

        # 4. Resample to model's native 16000 Hz if input rate is lower
        if self.resample_after:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )

        # 3. STFT via STFT_Process (Conv1d, no center padding)
        real_part, imag_part = self.stft(audio)

        # 4. Power spectrum
        power_spectrum = real_part * real_part + imag_part * imag_part

        # 5. Mel filterbank via Conv1d(kernel_size=1)
        mel_features = torch.nn.functional.conv1d(power_spectrum, self.fbank_conv)

        # 6. Log
        mel_features = torch.clamp(mel_features, min=1e-07).log()

        # 7. Model forward (CMVN fused, channel-first throughout)
        probs, caches_out = self.detect_model(mel_features, caches_in)

        # 8. Output in channel-first format [1, odim, T]
        return probs, caches_out


# ═══════════════════════════════════════════════════════════════════════════════
# Export Process
# ═══════════════════════════════════════════════════════════════════════════════

def export_model(model_path, onnx_path, model_type='vad'):
    """Export a FireRedVAD or FireRedAED model to ONNX (static graph, fused pre-emphasis+STFT)."""
    import os
    print(f'\nExport {model_type.upper()} start ...')

    with torch.inference_mode():
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
        audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)

        output_names = ['probs']
        dynamic_axes_dict = {
            'audio': {2: 'audio_len'},
            'probs': {2: 'signal_len'}
        } if DYNAMIC_AXES else None

        # Export to ONNX (static graph by default)
        torch.onnx.export(
            model,
            (audio,),
            onnx_path,
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

    print(f'Export {model_type.upper()} done! -> {onnx_path}')


def export_stream_vad_model(model_path, onnx_path):
    """Export a FireRedVAD Stream-VAD model to ONNX with cache input/output."""
    import os
    print(f'\nExport STREAM-VAD start ...')

    with torch.inference_mode():
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
        audio = torch.ones((1, 1, STREAM_CHUNK_SAMPLES), dtype=torch.int16)
        caches_in = torch.zeros((R, 1, P, lookback_padding), dtype=torch.float32)

        print(f'  Model info: R={R}, P={P}, lookback_padding={lookback_padding}')
        print(f'  Cache shape: [{R}, 1, {P}, {lookback_padding}]')
        print(f'  Chunk size: {STREAM_CHUNK_SAMPLES} samples ({STREAM_CHUNK_MS}ms)')

        # Stream-VAD uses dynamic audio axis because chunk sizes vary at runtime.
        # Cache dimensions are static (fixed model architecture).
        dynamic_axes_dict = {
            'audio': {2: 'audio_len'},
            'probs': {2: 'signal_len'}
        } if STREAM_VAD_DYNAMIC_AXES else None

        # Export to ONNX
        torch.onnx.export(
            model,
            (audio, caches_in),
            onnx_path,
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

    print(f'Export STREAM-VAD done! -> {onnx_path}')
    return R, P, lookback_padding


# ═══════════════════════════════════════════════════════════════════════════════
# ONNX Runtime Inference Utilities
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


def valid_frame_count(num_samples: int) -> int:
    """Return the number of valid frames for snip_edges=True fbank extraction."""
    resampled_samples = int(num_samples * SAMPLE_RATE / IN_SAMPLE_RATE)
    if resampled_samples < WINDOW_LENGTH:
        return 0
    return 1 + (resampled_samples - WINDOW_LENGTH) // HOP_LENGTH


# ═══════════════════════════════════════════════════════════════════════════════
# VAD Post-processing (Optimized: pre-allocated numpy, fused ops, C++ portable)
#
# Design principles:
#   - Integer state constants (no enum overhead, direct C++ translation)
#   - numpy int8 arrays for decisions (8x less memory than Python int lists)
#   - Pre-allocated buffers, in-place mutations (zero intermediate allocations)
#   - Fused smooth + threshold + state machine (single pass over probabilities)
#   - Circular buffer for streaming (fixed array, no deque/allocation per frame)
#   - Batch processing to avoid per-frame Python object creation
#   - All algorithms use simple index loops (direct C/C++ translation)
#   - Segment extraction via np.diff (vectorized, no Python iteration)
# ═══════════════════════════════════════════════════════════════════════════════

_VAD_SILENCE = 0
_VAD_POSSIBLE_SPEECH = 1
_VAD_SPEECH = 2
_VAD_POSSIBLE_SILENCE = 3


class VadPostprocessor:
    __slots__ = ('smooth_window_size', 'prob_threshold', 'min_speech_frame',
                 'max_speech_frame', 'min_silence_frame', 'merge_silence_frame',
                 'extend_speech_frame', '_inv_ws', '_half_max')

    def __init__(self, smooth_window_size, prob_threshold, min_speech_frame,
                 max_speech_frame, min_silence_frame, merge_silence_frame,
                 extend_speech_frame):
        self.smooth_window_size = max(1, smooth_window_size)
        self.prob_threshold = np.float32(prob_threshold)
        self.min_speech_frame = min_speech_frame
        self.max_speech_frame = max_speech_frame
        self.min_silence_frame = min_silence_frame
        self.merge_silence_frame = merge_silence_frame
        self.extend_speech_frame = extend_speech_frame
        # Pre-compute reciprocal for smoothing division
        self._inv_ws = np.float32(1.0 / self.smooth_window_size)
        self._half_max = max_speech_frame >> 1

    def process(self, raw_probs):
        """Process raw probabilities into binary speech decisions (numpy int8 array).
        Accepts numpy array or list. Returns numpy int8 array."""
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

        # Fused: smooth → threshold → state machine (single allocation for decisions)
        decisions = self._smooth_threshold_state_machine(probs, n)

        # Fix smooth window start: extend rising edges leftward (in-place)
        self._fix_starts_inplace(decisions, n)

        # Merge short silence gaps (in-place)
        if self.merge_silence_frame > 0:
            self._merge_silence_inplace(decisions, n)

        # Extend speech regions in both directions (in-place, two-pass O(n))
        if self.extend_speech_frame > 0:
            self._extend_inplace(decisions, n)

        # Split segments exceeding max_speech_frame (in-place)
        self._split_long_inplace(decisions, probs, n)

        return decisions

    def decision_to_segment(self, decisions, wav_dur=None):
        """Extract (start_sec, end_sec) segments from binary decision array.
        Uses np.diff for vectorized edge detection (no Python iteration)."""
        if isinstance(decisions, np.ndarray):
            dec = decisions
            n = dec.shape[0]
        else:
            dec = np.asarray(decisions, dtype=np.int8)
            n = dec.shape[0]
        if n == 0:
            return []

        # Pad with 0 at both ends to detect edges at boundaries
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

        # Vectorized conversion to seconds
        segments = np.empty((num_segs, 2), dtype=np.float32)
        segments[:, 0] = starts * _FRAME_SHIFT_F32
        segments[:, 1] = ends * _FRAME_SHIFT_F32

        # Fix last segment if speech extends to end of audio
        if dec[n - 1] != 0:
            end_time = n * _FRAME_SHIFT_F32 + _FRAME_LENGTH_F32
            if wav_dur is not None and wav_dur < end_time:
                end_time = wav_dur
            segments[-1, 1] = end_time

        # .tolist() converts float32 → Python float; round fixes float32 representation
        return [(round(s, 3), round(e, 3)) for s, e in segments.tolist()]

    def _smooth_threshold_state_machine(self, probs, n):
        """Fused: cumsum-based moving average → threshold → state machine → int8 array.
        Single output allocation. Cumsum smoothing is O(n) with no convolution overhead."""
        decisions = np.zeros(n, dtype=np.int8)
        ws = self.smooth_window_size
        threshold = self.prob_threshold
        min_sp = self.min_speech_frame
        min_si = self.min_silence_frame

        # Compute smoothed probabilities via prefix sum (O(n), cache-friendly)
        if ws > 1:
            cumsum = np.empty(n + 1, dtype=np.float32)
            cumsum[0] = 0.0
            np.cumsum(probs, out=cumsum[1:])
            smoothed = np.empty(n, dtype=np.float32)
            # Edge correction: frames 0..ws-2 use expanding window
            edge_end = min(ws - 1, n)
            for i in range(edge_end):
                smoothed[i] = cumsum[i + 1] / (i + 1)  # assignment truncates to float32
            # Main frames: fixed-width sliding window
            if n >= ws:
                smoothed[ws - 1:] = (cumsum[ws:] - cumsum[:n - ws + 1]) * self._inv_ws
        else:
            smoothed = probs

        # State machine: sequential pass (unavoidable for stateful logic)
        if min_sp <= 0 and min_si <= 0:
            # No state machine needed; direct threshold → binary
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

                # Write decision: 1 if in SPEECH or POSSIBLE_SILENCE
                decisions[t] = 1 if state >= _VAD_SPEECH else 0

        return decisions

    def _fix_starts_inplace(self, decisions, n):
        """Expand each rising edge leftward by smooth_window_size (in-place, O(n))."""
        ws = self.smooth_window_size
        if ws <= 1:
            return
        for t in range(1, n):
            if decisions[t] == 1 and decisions[t - 1] == 0:
                start = t - ws if t >= ws else 0
                decisions[start:t] = 1

    def _merge_silence_inplace(self, decisions, n):
        """Fill silence gaps shorter than merge_silence_frame (in-place, O(n))."""
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
        """Morphological dilation: two-pass O(n), no allocation, C++ portable.
        Forward pass extends rightward; backward pass extends leftward."""
        ext = self.extend_speech_frame
        # Forward pass: extend speech rightward
        dist = ext + 1  # Distance from last speech frame (>ext means no extension)
        for t in range(n):
            if decisions[t]:
                dist = 0
            else:
                dist += 1
                if dist <= ext:
                    decisions[t] = 1
        # Backward pass: extend speech leftward
        dist = ext + 1
        for t in range(n - 1, -1, -1):
            if decisions[t]:
                dist = 0
            else:
                dist += 1
                if dist <= ext:
                    decisions[t] = 1

    def _split_long_inplace(self, decisions, probs, n):
        """Split speech segments exceeding max_speech_frame at lowest-prob points.
        Scans segments directly from array (no float conversion roundtrip)."""
        max_sf = self.max_speech_frame
        half_max = self._half_max
        t = 0
        while t < n:
            if decisions[t]:
                seg_start = t
                while t < n and decisions[t]:
                    t += 1
                # t is now one past segment end
                dur = t - seg_start
                if dur > max_sf:
                    # Find split points within this long segment
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


class StreamVadPostprocessor:
    """Streaming VAD postprocessor with circular buffer smoothing.
    Uses fixed-size array instead of deque. Returns timestamps directly
    via process_batch() to avoid per-frame object allocation."""
    __slots__ = ('smooth_window_size', 'speech_threshold', 'pad_start_frame',
                 'min_speech_frame', 'max_speech_frame', 'min_silence_frame',
                 '_window_buf', '_window_sum', '_window_pos', '_window_count',
                 'frame_cnt', 'state', 'speech_cnt', 'silence_cnt',
                 'hit_max_speech', 'last_speech_start_frame', 'last_speech_end_frame')

    def __init__(self, smooth_window_size, speech_threshold, pad_start_frame,
                 min_speech_frame, max_speech_frame, min_silence_frame):
        self.smooth_window_size = max(1, smooth_window_size)
        self.speech_threshold = np.float32(speech_threshold)
        self.pad_start_frame = max(self.smooth_window_size, pad_start_frame)
        self.min_speech_frame = min_speech_frame
        self.max_speech_frame = max_speech_frame
        self.min_silence_frame = min_silence_frame
        # Pre-allocate circular buffer (fixed size, no dynamic allocation)
        self._window_buf = np.zeros(self.smooth_window_size, dtype=np.float32)
        self._window_sum = np.float32(0.0)
        self._window_pos = 0
        self._window_count = 0
        self.frame_cnt = 0
        self.state = _VAD_SILENCE
        self.speech_cnt = 0
        self.silence_cnt = 0
        self.hit_max_speech = False
        self.last_speech_start_frame = -1
        self.last_speech_end_frame = -1

    def reset(self):
        self.frame_cnt = 0
        self._window_buf[:] = 0.0
        self._window_sum = np.float32(0.0)
        self._window_pos = 0
        self._window_count = 0
        self.state = _VAD_SILENCE
        self.speech_cnt = 0
        self.silence_cnt = 0
        self.hit_max_speech = False
        self.last_speech_start_frame = -1
        self.last_speech_end_frame = -1

    def process_batch(self, raw_probs):
        """Process all frame probabilities and return timestamps directly.
        Avoids per-frame object allocation. Accepts numpy array.
        Returns: list of (start_sec, end_sec) tuples."""
        if isinstance(raw_probs, np.ndarray):
            probs = raw_probs
            n = probs.shape[0]
        else:
            probs = np.asarray(raw_probs, dtype=np.float32)
            n = probs.shape[0]
        if n == 0:
            return []

        inv_fps = 1.0 / FRAME_PER_SECONDS
        timestamps = []
        ws = self.smooth_window_size
        threshold = self.speech_threshold
        min_sp = self.min_speech_frame
        max_sp = self.max_speech_frame
        min_si = self.min_silence_frame
        pad_start = self.pad_start_frame

        # Local references for tight loop
        buf = self._window_buf
        buf_sum = self._window_sum
        buf_pos = self._window_pos
        buf_count = self._window_count
        frame_cnt = self.frame_cnt
        state = self.state
        speech_cnt = self.speech_cnt
        silence_cnt = self.silence_cnt
        hit_max = self.hit_max_speech
        last_start = self.last_speech_start_frame
        last_end = self.last_speech_end_frame

        for t in range(n):
            raw_p = probs[t]
            frame_cnt += 1

            # Circular buffer smoothing (no allocation)
            if ws <= 1:
                smoothed = raw_p
            else:
                old_val = buf[buf_pos]
                buf[buf_pos] = raw_p
                buf_sum += raw_p - old_val
                buf_pos = (buf_pos + 1) % ws
                if buf_count < ws:
                    buf_count += 1
                smoothed = buf_sum / buf_count

            is_speech = 1 if smoothed >= threshold else 0

            # Track speech start/end events
            seg_start_out = -1
            seg_end_out = -1

            if hit_max:
                seg_start_out = frame_cnt
                last_start = frame_cnt
                hit_max = False

            if state == _VAD_SILENCE:
                if is_speech:
                    state = _VAD_POSSIBLE_SPEECH
                    speech_cnt = 1
                else:
                    silence_cnt += 1
                    speech_cnt = 0

            elif state == _VAD_POSSIBLE_SPEECH:
                if is_speech:
                    speech_cnt += 1
                    if speech_cnt >= min_sp:
                        state = _VAD_SPEECH
                        seg_start_out = max(
                            1,
                            frame_cnt - speech_cnt + 1 - pad_start,
                            last_end + 1
                        )
                        last_start = seg_start_out
                        silence_cnt = 0
                else:
                    state = _VAD_SILENCE
                    silence_cnt = 1
                    speech_cnt = 0

            elif state == _VAD_SPEECH:
                speech_cnt += 1
                if is_speech:
                    silence_cnt = 0
                    if speech_cnt >= max_sp:
                        hit_max = True
                        speech_cnt = 0
                        seg_end_out = frame_cnt
                        seg_start_out = last_start
                        last_start = -1
                        last_end = frame_cnt
                else:
                    state = _VAD_POSSIBLE_SILENCE
                    silence_cnt = 1

            else:  # _VAD_POSSIBLE_SILENCE
                speech_cnt += 1
                if is_speech:
                    state = _VAD_SPEECH
                    silence_cnt = 0
                    if speech_cnt >= max_sp:
                        hit_max = True
                        speech_cnt = 0
                        seg_end_out = frame_cnt
                        seg_start_out = last_start
                        last_start = -1
                        last_end = frame_cnt
                else:
                    silence_cnt += 1
                    if silence_cnt >= min_si:
                        state = _VAD_SILENCE
                        seg_end_out = frame_cnt
                        seg_start_out = last_start
                        last_end = frame_cnt
                        last_start = -1
                        speech_cnt = 0

            # Collect timestamp when a segment ends
            if seg_end_out > 0 and seg_start_out > 0:
                s_frame = max(0, seg_start_out - 1)
                e_frame = max(0, seg_end_out - 1)
                timestamps.append((s_frame * inv_fps, e_frame * inv_fps))
            elif seg_start_out > 0 and seg_end_out <= 0:
                pass  # Segment started, waiting for end

        # Handle unterminated segment at end of stream
        if last_start > 0:
            s_frame = max(0, last_start - 1)
            e_frame = frame_cnt - 1
            timestamps.append((s_frame * inv_fps, e_frame * inv_fps))

        # Write back state for continued streaming
        self._window_sum = buf_sum
        self._window_pos = buf_pos
        self._window_count = buf_count
        self.frame_cnt = frame_cnt
        self.state = state
        self.speech_cnt = speech_cnt
        self.silence_cnt = silence_cnt
        self.hit_max_speech = hit_max
        self.last_speech_start_frame = last_start
        self.last_speech_end_frame = last_end

        return timestamps

    def process_one_frame(self, raw_prob):
        """Process single frame. Returns (is_start, is_end, start_frame, end_frame).
        Lightweight tuple return (no dataclass allocation)."""
        self.frame_cnt += 1
        ws = self.smooth_window_size

        # Circular buffer smoothing
        if ws <= 1:
            smoothed = raw_prob
        else:
            old_val = self._window_buf[self._window_pos]
            self._window_buf[self._window_pos] = raw_prob
            self._window_sum += raw_prob - old_val
            self._window_pos = (self._window_pos + 1) % ws
            if self._window_count < ws:
                self._window_count += 1
            smoothed = self._window_sum / self._window_count

        is_speech = 1 if smoothed >= self.speech_threshold else 0

        seg_start_out = -1
        seg_end_out = -1
        is_start = False
        is_end = False

        if self.hit_max_speech:
            is_start = True
            seg_start_out = self.frame_cnt
            self.last_speech_start_frame = self.frame_cnt
            self.hit_max_speech = False

        if self.state == _VAD_SILENCE:
            if is_speech:
                self.state = _VAD_POSSIBLE_SPEECH
                self.speech_cnt = 1
            else:
                self.silence_cnt += 1
                self.speech_cnt = 0

        elif self.state == _VAD_POSSIBLE_SPEECH:
            if is_speech:
                self.speech_cnt += 1
                if self.speech_cnt >= self.min_speech_frame:
                    self.state = _VAD_SPEECH
                    is_start = True
                    seg_start_out = max(
                        1,
                        self.frame_cnt - self.speech_cnt + 1 - self.pad_start_frame,
                        self.last_speech_end_frame + 1
                    )
                    self.last_speech_start_frame = seg_start_out
                    self.silence_cnt = 0
            else:
                self.state = _VAD_SILENCE
                self.silence_cnt = 1
                self.speech_cnt = 0

        elif self.state == _VAD_SPEECH:
            self.speech_cnt += 1
            if is_speech:
                self.silence_cnt = 0
                if self.speech_cnt >= self.max_speech_frame:
                    self.hit_max_speech = True
                    self.speech_cnt = 0
                    is_end = True
                    seg_end_out = self.frame_cnt
                    seg_start_out = self.last_speech_start_frame
                    self.last_speech_start_frame = -1
                    self.last_speech_end_frame = self.frame_cnt
            else:
                self.state = _VAD_POSSIBLE_SILENCE
                self.silence_cnt = 1

        else:  # _VAD_POSSIBLE_SILENCE
            self.speech_cnt += 1
            if is_speech:
                self.state = _VAD_SPEECH
                self.silence_cnt = 0
                if self.speech_cnt >= self.max_speech_frame:
                    self.hit_max_speech = True
                    self.speech_cnt = 0
                    is_end = True
                    seg_end_out = self.frame_cnt
                    seg_start_out = self.last_speech_start_frame
                    self.last_speech_start_frame = -1
                    self.last_speech_end_frame = self.frame_cnt
            else:
                self.silence_cnt += 1
                if self.silence_cnt >= self.min_silence_frame:
                    self.state = _VAD_SILENCE
                    is_end = True
                    seg_end_out = self.frame_cnt
                    seg_start_out = self.last_speech_start_frame
                    self.last_speech_end_frame = self.frame_cnt
                    self.last_speech_start_frame = -1
                    self.speech_cnt = 0

        return (is_start, is_end, seg_start_out, seg_end_out)


def stream_vad_results_to_timestamps(results):
    """Convert streaming results (tuples from process_one_frame) to timestamps.
    Each result is (is_start, is_end, start_frame, end_frame)."""
    inv_fps = 1.0 / FRAME_PER_SECONDS
    timestamps = []
    start = -1
    for frame_idx, (is_start, is_end, sf_start, sf_end) in enumerate(results, 1):
        if is_start:
            start = max(0, sf_start - 1)
        if is_end:
            end = max(0, sf_end - 1)
            timestamps.append((start * inv_fps, end * inv_fps))
            start = -1
    if start >= 0 and results:
        end = len(results) - 1
        timestamps.append((start * inv_fps, end * inv_fps))
    return timestamps


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


# ═══════════════════════════════════════════════════════════════════════════════
# Main: Export + Inference Test
# ═══════════════════════════════════════════════════════════════════════════════

# Export VAD model
if EXPORT_VAD:
    export_model(model_path_vad, onnx_model_vad, model_type='vad')

# Export AED model
if EXPORT_AED:
    export_model(model_path_aed, onnx_model_aed, model_type='aed')

# Export Stream-VAD model
STREAM_VAD_CACHE_INFO = None
if EXPORT_STREAM_VAD:
    STREAM_VAD_CACHE_INFO = export_stream_vad_model(model_path_stream_vad, onnx_model_stream_vad)


# ═══════════════════════════════════════════════════════════════════════════════
# Numerical Validation: Optimized PyTorch vs ONNX Runtime
# ═══════════════════════════════════════════════════════════════════════════════

def validate_export(onnx_path, model_path, model_type='vad'):
    """
    Validate that the exported ONNX model matches the optimized PyTorch model.
    Reports: max_abs_error, max_relative_error, mean_absolute_error, allclose result.
    """
    import os
    print(f'\n{"─"*40}')
    print(f'Validating {model_type.upper()} export: PyTorch vs ONNXRuntime')

    with torch.inference_mode():
        # Rebuild the PyTorch model
        cmvn_file = os.path.join(model_path, "cmvn.ark")
        cmvn_means, cmvn_inv_std = load_cmvn(cmvn_file)
        detect_model = DetectModel.from_pretrained(
            model_path, cmvn_means=cmvn_means, cmvn_inv_std=cmvn_inv_std
        )
        pt_model = FireRedVAD_ONNX(
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

        # Deterministic test input
        np.random.seed(1234)
        torch.manual_seed(1234)
        test_audio = torch.from_numpy(
            np.random.randint(-8000, 8000, size=(1, 1, INPUT_AUDIO_LENGTH), dtype=np.int16)
        )

        # PyTorch forward
        pt_output = pt_model(test_audio).numpy()

        # ONNXRuntime forward
        sess_opts = onnxruntime.SessionOptions()
        sess_opts.log_severity_level = 4
        sess = onnxruntime.InferenceSession(onnx_path, sess_options=sess_opts, providers=[])
        ort_output = sess.run(None, {'audio': test_audio.numpy()})[0]

        # Compare
        abs_diff = np.abs(pt_output - ort_output)
        max_abs = abs_diff.max()
        mean_abs = abs_diff.mean()
        denom = np.maximum(np.abs(pt_output), 1e-10)
        max_rel = (abs_diff / denom).max()
        allclose = np.allclose(pt_output, ort_output, rtol=1e-5, atol=1e-5)

        print(f'  input_shape:        {list(test_audio.shape)}')
        print(f'  output_shape:       {list(pt_output.shape)} (PyTorch) vs {list(ort_output.shape)} (ORT)')
        print(f'  dtype:              {pt_output.dtype}')
        print(f'  max_abs_error:      {max_abs:.2e}')
        print(f'  max_relative_error: {max_rel:.2e}')
        print(f'  mean_absolute_error:{mean_abs:.2e}')
        print(f'  allclose(rtol=1e-5, atol=1e-5): {allclose}')
        print(f'  opset_version:      {OPSET}')
        print(f'  dynamic_axes:       {DYNAMIC_AXES}')

        if not allclose:
            print(f'  WARNING: Tolerance exceeded! Check for numerical precision issues.')

        del pt_model, detect_model, sess
        gc.collect()
    return allclose


# Run validation
if EXPORT_VAD:
    validate_export(onnx_model_vad, model_path_vad, 'vad')
if EXPORT_AED:
    validate_export(onnx_model_aed, model_path_aed, 'aed')


# ═══════════════════════════════════════════════════════════════════════════════
# Run VAD Inference with ONNX Runtime
# ═══════════════════════════════════════════════════════════════════════════════

print('\n\nStart to run FireRedVAD by ONNX Runtime.\n\nNow, loading the model...')

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


ort_session_A = onnxruntime.InferenceSession(onnx_model_vad, sess_options=session_opts, providers=ORT_Accelerate_Providers)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A0 = out_name_A[0].name


# Load the input audio
print(f"\nTest Input Audio: {test_vad_audio}")
audio = np.array(AudioSegment.from_file(test_vad_audio).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
if NORMALIZE_AUDIO:
    audio = normalise_audio(audio)
audio_len = len(audio)
audio = audio.reshape(1, 1, -1)

shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH_RUN = min(IN_SAMPLE_RATE * 3600, audio_len)  # You can adjust it.
else:
    INPUT_AUDIO_LENGTH_RUN = shape_value_in

stride_step = INPUT_AUDIO_LENGTH_RUN
if audio_len > INPUT_AUDIO_LENGTH_RUN:
    num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH_RUN) / stride_step)) + 1
    total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH_RUN
    pad_amount = total_length_needed - audio_len
    final_slice = audio[:, :, -pad_amount:].astype(np.float32)
    white_noise = (np.sqrt(np.mean(final_slice * final_slice)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
elif audio_len < INPUT_AUDIO_LENGTH_RUN:
    audio_float = audio.astype(np.float32)
    white_noise = (np.sqrt(np.mean(audio_float * audio_float)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH_RUN - audio_len))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
aligned_len = audio.shape[-1]

# Start to run FireRedVAD
slice_start = 0
slice_end = INPUT_AUDIO_LENGTH_RUN
all_vad_probs = []
print("\nRunning the FireRedVAD by ONNX Runtime.")
start_time = time.time()
while slice_end <= aligned_len:
    probs = ort_session_A.run(
        [out_name_A0], {in_name_A0: audio[:, :, slice_start: slice_end]})[0]
    all_vad_probs.append(probs[0, 0])
    slice_start += stride_step
    slice_end = slice_start + INPUT_AUDIO_LENGTH_RUN
end_time = time.time()

valid_vad_frames = valid_frame_count(audio_len)
if all_vad_probs:
    all_vad_probs = np.concatenate(all_vad_probs, axis=0)[:valid_vad_frames]
else:
    all_vad_probs = np.zeros((0,), dtype=np.float32)

vad_postprocessor = VadPostprocessor(
    SMOOTH_WINDOW_SIZE,
    SPEAKING_SCORE,
    MIN_SPEECH_FRAME,
    MAX_SPEECH_FRAME,
    MIN_SILENCE_FRAME,
    MERGE_SILENCE_FRAME,
    EXTEND_SPEECH_FRAME,
)
vad_decisions = vad_postprocessor.process(all_vad_probs)
timestamps = vad_postprocessor.decision_to_segment(vad_decisions, audio_len / IN_SAMPLE_RATE)
print(f"Complete: 100.00%")

# Save the timestamps.
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

vad_rtf = (end_time - start_time) / (audio_len / IN_SAMPLE_RATE)
print(f"\nVAD Process Complete.\n\nTime Cost: {end_time - start_time:.3f} Seconds")
print(f"RTF: {vad_rtf:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Run AED (Audio Event Detection) Inference with ONNX Runtime
# ═══════════════════════════════════════════════════════════════════════════════

if EXPORT_AED:
    print('\n\n' + '=' * 70)
    print('Start to run FireRedAED by ONNX Runtime.\n\nNow, loading the model...')

    ort_session_B = onnxruntime.InferenceSession(onnx_model_aed, sess_options=session_opts, providers=ORT_Accelerate_Providers)
    print(f"\nUsable Providers: {ort_session_B.get_providers()}")
    in_name_B = ort_session_B.get_inputs()
    out_name_B = ort_session_B.get_outputs()
    in_name_B0 = in_name_B[0].name
    out_name_B0 = out_name_B[0].name

    # AED event mapping: index -> event name
    IDX2EVENT = {0: "speech", 1: "singing", 2: "music"}
    EVENT_THRESHOLDS = {"speech": SPEAKING_SCORE, "singing": SINGING_THRESHOLD, "music": MUSIC_THRESHOLD}

    # Load the input audio
    print(f"\nTest Input Audio: {test_aed_audio}")
    audio_aed = np.array(AudioSegment.from_file(test_aed_audio).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
    if NORMALIZE_AUDIO:
        audio_aed = normalise_audio(audio_aed)
    audio_aed_len = len(audio_aed)
    audio_aed_dur = audio_aed_len / IN_SAMPLE_RATE
    audio_aed = audio_aed.reshape(1, 1, -1)

    shape_value_in_B = ort_session_B._inputs_meta[0].shape[-1]
    if isinstance(shape_value_in_B, str):
        INPUT_AUDIO_LENGTH_RUN_B = min(IN_SAMPLE_RATE * 3600, audio_aed_len)
    else:
        INPUT_AUDIO_LENGTH_RUN_B = shape_value_in_B

    stride_step_B = INPUT_AUDIO_LENGTH_RUN_B
    if audio_aed_len > INPUT_AUDIO_LENGTH_RUN_B:
        num_windows_B = int(np.ceil((audio_aed_len - INPUT_AUDIO_LENGTH_RUN_B) / stride_step_B)) + 1
        total_length_needed_B = (num_windows_B - 1) * stride_step_B + INPUT_AUDIO_LENGTH_RUN_B
        pad_amount_B = total_length_needed_B - audio_aed_len
        final_slice_B = audio_aed[:, :, -pad_amount_B:].astype(np.float32)
        white_noise_B = (np.sqrt(np.mean(final_slice_B * final_slice_B)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount_B))).astype(audio_aed.dtype)
        audio_aed = np.concatenate((audio_aed, white_noise_B), axis=-1)
    elif audio_aed_len < INPUT_AUDIO_LENGTH_RUN_B:
        audio_aed_float = audio_aed.astype(np.float32)
        white_noise_B = (np.sqrt(np.mean(audio_aed_float * audio_aed_float)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH_RUN_B - audio_aed_len))).astype(audio_aed.dtype)
        audio_aed = np.concatenate((audio_aed, white_noise_B), axis=-1)
    aligned_len_B = audio_aed.shape[-1]

    # Run AED inference
    print("\nRunning the FireRedAED by ONNX Runtime.")
    start_time = time.time()
    all_probs = []
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH_RUN_B
    while slice_end <= aligned_len_B:
        probs_aed = ort_session_B.run(
            [out_name_B0], {in_name_B0: audio_aed[:, :, slice_start: slice_end]})[0]
        # probs_aed shape: [1, 3, T]
        all_probs.append(probs_aed[0])  # [3, T]
        slice_start += stride_step_B
        slice_end = slice_start + INPUT_AUDIO_LENGTH_RUN_B
    end_time = time.time()

    valid_aed_frames = valid_frame_count(audio_aed_len)
    if all_probs:
        all_probs = np.concatenate(all_probs, axis=1)[:, :valid_aed_frames]
    else:
        all_probs = np.zeros((len(IDX2EVENT), 0), dtype=np.float32)

    # Post-process each event type
    event2timestamps = {}
    event2ratio = {}
    event2postprocessor = {
        "speech": VadPostprocessor(
            SMOOTH_WINDOW_SIZE,
            SPEAKING_SCORE,
            MIN_EVENT_FRAME,
            MAX_EVENT_FRAME,
            MIN_SILENCE_FRAME,
            MERGE_SILENCE_FRAME,
            EXTEND_SPEECH_FRAME,
        ),
        "singing": VadPostprocessor(
            SMOOTH_WINDOW_SIZE,
            SINGING_THRESHOLD,
            MIN_EVENT_FRAME,
            MAX_EVENT_FRAME,
            MIN_SILENCE_FRAME,
            MERGE_SILENCE_FRAME,
            EXTEND_SPEECH_FRAME,
        ),
        "music": VadPostprocessor(
            SMOOTH_WINDOW_SIZE,
            MUSIC_THRESHOLD,
            MIN_EVENT_FRAME,
            MAX_EVENT_FRAME,
            MIN_SILENCE_FRAME,
            MERGE_SILENCE_FRAME,
            EXTEND_SPEECH_FRAME,
        ),
    }
    for idx, event in IDX2EVENT.items():
        threshold = EVENT_THRESHOLDS[event]
        postprocessor = event2postprocessor[event]
        event_probs = all_probs[idx]
        decisions = postprocessor.process(event_probs)
        event2timestamps[event] = postprocessor.decision_to_segment(decisions, audio_aed_dur)
        raw_ratio = float(np.mean(event_probs >= threshold)) if len(event_probs) > 0 else 0.0
        event2ratio[event] = round(raw_ratio, 3)

    # Print AED results
    print(f"Complete: 100.00%")
    print(f"\nAED Results:")
    print(f"  Audio duration: {audio_aed_dur:.3f}s")
    print(f"  Event ratios: {event2ratio}")
    for event, ts_list in event2timestamps.items():
        print(f"\n  [{event}] segments ({len(ts_list)}):")
        for start_s, end_s in ts_list:
            print(f"    {format_time(start_s)} --> {format_time(end_s)}")

    aed_rtf = (end_time - start_time) / audio_aed_dur
    print(f"\nAED Process Complete.\n\nTime Cost: {end_time - start_time:.3f} Seconds")
    print(f"RTF: {aed_rtf:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Run Stream-VAD Inference with ONNX Runtime (frame-by-frame with caches)
# ═══════════════════════════════════════════════════════════════════════════════

if EXPORT_STREAM_VAD:
    print('\n\n' + '=' * 70)
    print('Start to run FireRedStreamVAD by ONNX Runtime.\n\nNow, loading the model...')

    ort_session_C = onnxruntime.InferenceSession(onnx_model_stream_vad, sess_options=session_opts, providers=ORT_Accelerate_Providers)
    print(f"\nUsable Providers: {ort_session_C.get_providers()}")
    in_name_C = ort_session_C.get_inputs()
    out_name_C = ort_session_C.get_outputs()
    in_name_C0 = in_name_C[0].name  # audio
    in_name_C1 = in_name_C[1].name  # caches_in
    out_name_C0 = out_name_C[0].name  # probs
    out_name_C1 = out_name_C[1].name  # caches_out

    # Determine cache shape from the model metadata
    cache_shape = ort_session_C._inputs_meta[1].shape
    # cache_shape should be [R, 1, P, lookback_padding]
    R_val = cache_shape[0]
    P_val = cache_shape[2]
    L_val = cache_shape[3]
    print(f"  Cache shape: [{R_val}, 1, {P_val}, {L_val}]")

    # Load the input audio
    print(f"\nTest Input Audio: {test_vad_audio}")
    audio_stream = np.array(AudioSegment.from_file(test_vad_audio).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
    if NORMALIZE_AUDIO:
        audio_stream = normalise_audio(audio_stream)
    audio_stream_len = len(audio_stream)
    audio_stream_dur = audio_stream_len / IN_SAMPLE_RATE

    # Process in chunks (e.g., 160ms = 2560 samples per chunk -> produces ~16 frames)
    # Uses STREAM_CHUNK_MS and STREAM_CHUNK_SAMPLES from top-level config.

    # Initialize caches to zero
    caches = np.zeros((R_val, 1, P_val, L_val), dtype=np.float32)

    # Collect all frame probabilities
    all_stream_probs = []

    print(f"\nRunning the FireRedStreamVAD by ONNX Runtime.")
    print(f"  Audio duration: {audio_stream_dur:.3f}s")
    print(f"  Chunk size: {STREAM_CHUNK_MS}ms ({STREAM_CHUNK_SAMPLES} samples)")
    start_time = time.time()

    # Process audio in streaming chunks
    pos = 0
    while pos < audio_stream_len:
        chunk_end = min(pos + STREAM_CHUNK_SAMPLES, audio_stream_len)
        chunk = audio_stream[pos:chunk_end]

        # Pad last chunk if needed (must be at least WINDOW_LENGTH samples for 1 frame)
        if len(chunk) < WINDOW_LENGTH:
            chunk = np.pad(chunk, (0, WINDOW_LENGTH - len(chunk)), mode='constant')

        chunk_input = chunk.reshape(1, 1, -1).astype(np.int16)

        # Run inference with caches
        probs_out, caches = ort_session_C.run(
            [out_name_C0, out_name_C1],
            {in_name_C0: chunk_input, in_name_C1: caches}
        )
        # probs_out shape: [1, 1, T]
        all_stream_probs.append(probs_out[0, 0])  # [T]

        pos = chunk_end

    end_time = time.time()

    # Concatenate all frame probabilities
    valid_stream_frames = valid_frame_count(audio_stream_len)
    if all_stream_probs:
        all_stream_probs = np.concatenate(all_stream_probs, axis=0)[:valid_stream_frames]
    else:
        all_stream_probs = np.zeros((0,), dtype=np.float32)

    stream_postprocessor = StreamVadPostprocessor(
        SMOOTH_WINDOW_SIZE,
        STREAM_VAD_THRESHOLD,
        PAD_START_FRAME,
        MIN_SPEECH_FRAME_STREAM,
        MAX_SPEECH_FRAME_STREAM,
        MIN_SILENCE_FRAME_STREAM,
    )
    total_frames = len(all_stream_probs)
    stream_timestamps = stream_postprocessor.process_batch(all_stream_probs)

    # Print Stream-VAD results
    print(f"Complete: 100.00%")
    print(f"\nStream-VAD Results:")
    print(f"  Audio duration: {audio_stream_dur:.3f}s")
    print(f"  Total frames: {total_frames}")
    print(f"  Segments detected: {len(stream_timestamps)}")
    print(f"\n  Timestamps in Second:")
    for start_s, end_s in stream_timestamps:
        print(f"    {format_time(start_s)} --> {format_time(end_s)}")

    stream_vad_rtf = (end_time - start_time) / audio_stream_dur
    print(f"\nStream-VAD Process Complete.\n\nTime Cost: {end_time - start_time:.3f} Seconds")
    print(f"RTF: {stream_vad_rtf:.4f}")
