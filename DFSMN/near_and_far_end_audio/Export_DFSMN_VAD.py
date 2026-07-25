import gc
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi

from Rewrite_ONNX_Export_Limitations import (
    rewrite_nkf_initializer_identities,
)

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "vad_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from vad_onnx_metadata import stamp_export_metadata  # noqa: E402


parent_path = Path(__file__).resolve().parent
dfsmn_model_path = Path.home() / "Downloads" / "speech_dfsmn_aec_psm_16k"
light_aec_path = Path.home() / "Downloads" / "SDAEC-main"               # Path keywords select SDAEC, Deep_Echo, or NKF.
onnx_model = parent_path / "DFSMN_VAD_ONNX" / "DFSMN_VAD.onnx"


DYNAMIC_AXES = False
OPSET = 20
INPUT_AUDIO_LENGTH = 32000  # External input length in IN_SAMPLE_RATE samples.
MODEL_SAMPLE_RATE = 16000   # Native model rate; do not change.
IN_SAMPLE_RATE = 16000      # External audio rate; accepts any positive sample rate.
IN_AUDIO_DTYPE = 'F32'      # ['F16', 'F32', 'INT16']; float inputs use normalized PCM in [-1, 1].

if IN_SAMPLE_RATE <= 0:
    raise ValueError('IN_SAMPLE_RATE must be positive.')
MODEL_AUDIO_LENGTH = int(round(
    INPUT_AUDIO_LENGTH * MODEL_SAMPLE_RATE / IN_SAMPLE_RATE))
INPUT_TO_MODEL_SCALE = float(MODEL_SAMPLE_RATE / IN_SAMPLE_RATE)


def infer_light_aec_model(project_path):
    project_path_key = str(project_path).lower().replace('-', '_').replace(' ', '_')
    if 'deep_echo' in project_path_key or (
            'deep' in project_path_key and 'echo' in project_path_key):
        return 'Deep_Echo'
    if 'nkf' in project_path_key:
        return 'NKF'
    if 'sdaec' in project_path_key:
        return 'SDAEC'
    raise ValueError(
        f"Unable to infer LIGHT_AEC_MODEL from light_aec_path: {project_path}. "
        "Expected keywords for 'SDAEC', 'Deep_Echo', or 'NKF'."
    )


LIGHT_AEC_MODEL = infer_light_aec_model(light_aec_path)

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

# Official DFSMN-AEC frontend configuration from dey_mini.yaml.
FBANK_NFFT = 1024
FBANK_WINDOW_LENGTH = 640
FBANK_HOP_LENGTH = 320
FBANK_MELS = 80
FBANK_DITHER = 1.0
PREEMPHASIS = 0.97
FBANK_MAX_FRAMES = 4096 if DYNAMIC_AXES else (
    (MODEL_AUDIO_LENGTH - FBANK_WINDOW_LENGTH) // FBANK_HOP_LENGTH + 1)

# Light-AEC backends are traceable replacements for the native MinDAEC frontend.
SDAEC_NFFT = 319
SDAEC_HOP_LENGTH = 160
SDAEC_ALPHA_ORDER = 10
SDAEC_MAX_FRAMES = 4096 if DYNAMIC_AXES else MODEL_AUDIO_LENGTH // SDAEC_HOP_LENGTH + 1
DEEP_ECHO_NFFT = 319
DEEP_ECHO_HOP_LENGTH = 160
DEEP_ECHO_ORDER = 10
NKF_NFFT = 1024
NKF_HOP_LENGTH = 256
NKF_FILTER_ORDER = 4
NKF_FC_DIM = 18
NKF_RNN_LAYERS = 1
NKF_RNN_DIM = 18
ECHO_ESTIMATE_FACTOR = 1.15

if LIGHT_AEC_MODEL == 'NKF' and DYNAMIC_AXES:
    raise ValueError('NKF requires DYNAMIC_AXES=False because its adaptive loop is statically unrolled.')
if LIGHT_AEC_MODEL == 'NKF' and MODEL_AUDIO_LENGTH < 32000:
    raise ValueError('NKF requires at least 32000 samples at the model sample rate.')
if LIGHT_AEC_MODEL == 'Deep_Echo' and DYNAMIC_AXES:
    raise ValueError(
        'Deep Echo currently requires DYNAMIC_AXES=False because its ISTFT '
        'normalization is precomputed for MODEL_AUDIO_LENGTH.')


def centered_stft_frame_count(audio_length, n_fft, hop_length):
    return (audio_length + 2 * (n_fft // 2) - n_fft) // hop_length + 1


def load_checkpoint(module, path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")
    module.load_state_dict(state_dict, strict=True)
    return module


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model', 'network', 'net'):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break
    if not isinstance(checkpoint, dict):
        raise TypeError('Unsupported checkpoint format. Expected a state_dict-like mapping.')
    for prefix in ('module.', 'model.', 'network.', 'net.'):
        if checkpoint and all(key.startswith(prefix) for key in checkpoint):
            checkpoint = {
                key[len(prefix):]: value for key, value in checkpoint.items()
            }
    return checkpoint


def fold_linear_into_pointwise_conv(linear, conv):
    if conv.kernel_size != (1, 1) or conv.groups != 1:
        raise ValueError('Linear projection folding requires an ungrouped 1x1 Conv2d.')
    if linear.bias is None or conv.bias is None:
        raise ValueError('Linear projection folding requires both affine biases.')
    if conv.in_channels < linear.out_features:
        raise ValueError('Conv2d has fewer input channels than the Linear output.')
    with torch.no_grad():
        conv_weight = conv.weight[:, :, 0, 0].double()
        projected_weight = conv_weight[:, :linear.out_features]
        fused_weight = torch.cat((
            projected_weight @ linear.weight.double(),
            conv_weight[:, linear.out_features:],
        ), dim=1).to(dtype=conv.weight.dtype)
        fused_bias = (
            conv.bias.double() + projected_weight @ linear.bias.double()
        ).to(dtype=conv.bias.dtype)
    return (
        fused_weight.unsqueeze(-1).unsqueeze(-1).contiguous(),
        fused_bias.contiguous(),
    )


def load_kaldi_feature_transform(path):
    text = Path(path).read_text(encoding="utf-8")

    def parse_vector(name):
        section = text.find(name)
        start = text.find("[", section)
        end = text.find("]", start)
        if section < 0 or start < 0 or end < 0:
            raise ValueError(f"Unable to parse {name} from {path}.")
        return torch.from_numpy(
            np.fromstring(text[start + 1:end], dtype=np.float32, sep=" "))

    shift = parse_vector("AddShift")
    scale = parse_vector("Rescale")
    if shift.numel() != FBANK_MELS * 3 or scale.numel() != FBANK_MELS * 3:
        raise ValueError(
            f"Expected {FBANK_MELS * 3} MVN values, got "
            f"shift={shift.numel()} and scale={scale.numel()}.")
    return shift, scale


class AffineTransform(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, inputs):
        return self.linear(inputs)


class UniDeepFsmn(torch.nn.Module):
    def __init__(self, input_dim=512, output_dim=512, lorder=20, hidden_size=512):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, hidden_size)
        self.project = torch.nn.Linear(hidden_size, output_dim, bias=False)
        self.conv1 = torch.nn.Conv2d(
            output_dim,
            output_dim,
            kernel_size=(lorder, 1),
            groups=output_dim,
            bias=False,
        )
        self.register_buffer(
            "left_padding",
            torch.zeros(1, output_dim, lorder - 1, 1),
            persistent=False,
        )

    def forward(self, inputs):
        hidden = self.project(F.relu(self.linear(inputs))).transpose(1, 2)
        padding = self.left_padding[..., 0].expand(hidden.shape[0], -1, -1)
        memory = F.conv1d(
            torch.cat((padding, hidden), dim=2),
            self.conv1.weight[..., 0],
            bias=self.conv1.bias,
            stride=self.conv1.stride[0],
            padding=self.conv1.padding[0],
            dilation=self.conv1.dilation[0],
            groups=self.conv1.groups,
        )
        return inputs + (memory + hidden).transpose(1, 2)


class DFSMNMaskNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = AffineTransform(FBANK_MELS * 3, 512)
        self.deepfsmn = torch.nn.Sequential(
            *(UniDeepFsmn() for _ in range(12)))
        self.linear2 = AffineTransform(512, 321)
        self.linear3 = AffineTransform(512, 1)
        self.vad = True

    def forward(self, features):
        hidden = F.relu(self.linear1(features))
        hidden = self.deepfsmn(hidden)
        return torch.sigmoid(self.linear3(hidden))

    def fold_feature_transform(self, shift, scale):
        with torch.no_grad():
            weight = self.linear1.linear.weight
            bias = self.linear1.linear.bias
            folded_weight = (
                weight.double() * scale.double().unsqueeze(0)).float()
            folded_bias = (
                bias.double()
                + weight.double() @ (shift.double() * scale.double())
            ).float()
            weight.copy_(folded_weight)
            bias.copy_(folded_bias)
        return self


class AlphaPredictor(torch.nn.Module):
    def __init__(self, order):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 1)
        self.linear2 = torch.nn.Linear(order, 1)

    def fold(self):
        temporal_weight = self.linear2.weight
        power_weight = self.linear1.weight[0]
        kernel = torch.stack((
            temporal_weight * power_weight[1],
            temporal_weight * power_weight[0],
        ), dim=1)
        bias = (
            self.linear2.bias
            + temporal_weight.sum(dim=1) * self.linear1.bias[0]
        )
        return kernel.detach(), bias.detach()


class SDAECSTFT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        half_nfft = SDAEC_NFFT // 2
        window = torch.hamming_window(SDAEC_NFFT, periodic=True)
        time = torch.arange(SDAEC_NFFT, dtype=torch.float32).unsqueeze(0)
        frequency = torch.arange(half_nfft + 1, dtype=torch.float32).unsqueeze(1)
        angles = 2.0 * torch.pi * frequency * time / SDAEC_NFFT
        self.register_buffer(
            "analysis_kernel",
            torch.cat((torch.cos(angles), -torch.sin(angles)), dim=0)
            .mul(window)
            .unsqueeze(1),
            persistent=False,
        )
        self.padding = half_nfft

    def forward(self, audio):
        audio = F.pad(audio, (self.padding, self.padding), mode="reflect")
        return F.conv1d(
            audio, self.analysis_kernel, stride=SDAEC_HOP_LENGTH)


class SDAECISTFT(torch.nn.Module):
    def __init__(self, max_frames):
        super().__init__()
        half_nfft = SDAEC_NFFT // 2
        window = torch.hamming_window(SDAEC_NFFT, periodic=True)
        fourier = torch.fft.fft(torch.eye(SDAEC_NFFT, dtype=torch.float32))
        basis = torch.vstack((
            fourier.real[:half_nfft + 1],
            fourier.imag[:half_nfft + 1],
        )).float()
        inverse_basis = torch.linalg.pinv(
            (basis * SDAEC_NFFT) / SDAEC_HOP_LENGTH).T.unsqueeze(1)
        self.register_buffer(
            "inverse_basis",
            inverse_basis * window.view(1, 1, -1),
            persistent=False,
        )

        output_length = SDAEC_NFFT + SDAEC_HOP_LENGTH * (max_frames - 1)
        window_sum = torch.zeros(output_length, dtype=torch.float32)
        window_squared = window.square()
        for frame in range(max_frames):
            start = frame * SDAEC_HOP_LENGTH
            window_sum[start:start + SDAEC_NFFT] += window_squared
        self.register_buffer(
            "window_sum_inverse",
            SDAEC_NFFT / (window_sum * SDAEC_HOP_LENGTH + 1e-7),
            persistent=False,
        )
        self.crop = half_nfft

    def forward(self, spectrum):
        inverse = F.conv_transpose1d(
            spectrum,
            self.inverse_basis,
            stride=SDAEC_HOP_LENGTH,
            padding=self.crop,
        )
        return inverse * self.window_sum_inverse[
            self.crop:self.crop + inverse.shape[-1]]


class PackedSTFT(torch.nn.Module):
    def __init__(self, n_fft, hop_length, window_type):
        super().__init__()
        half_nfft = n_fft // 2
        if window_type == 'hamming':
            window = torch.hamming_window(n_fft, periodic=True)
        elif window_type == 'hann':
            window = torch.hann_window(n_fft, periodic=True)
        else:
            raise ValueError(f'Unsupported backend window: {window_type}')
        time = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
        frequency = torch.arange(half_nfft + 1, dtype=torch.float32).unsqueeze(1)
        angles = 2.0 * torch.pi * frequency * time / n_fft
        self.register_buffer(
            'analysis_kernel',
            torch.cat((torch.cos(angles), -torch.sin(angles)), dim=0)
            .mul(window)
            .unsqueeze(1),
            persistent=False,
        )
        self.hop_length = hop_length
        self.padding = half_nfft

    def forward(self, audio):
        return F.conv1d(
            audio,
            self.analysis_kernel,
            stride=self.hop_length,
            padding=self.padding,
        )


class PackedISTFT(torch.nn.Module):
    def __init__(self, n_fft, hop_length, window_type, frames, output_length):
        super().__init__()
        half_nfft = n_fft // 2
        frequency_bins = half_nfft + 1
        if window_type == 'hamming':
            window = torch.hamming_window(n_fft, periodic=True)
        elif window_type == 'hann':
            window = torch.hann_window(n_fft, periodic=True)
        else:
            raise ValueError(f'Unsupported backend window: {window_type}')

        time = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
        frequency = torch.arange(frequency_bins, dtype=torch.float32).unsqueeze(1)
        angles = 2.0 * torch.pi * frequency * time / n_fft
        scale = torch.full((frequency_bins, 1), 2.0, dtype=torch.float32)
        scale[0] = 1.0
        if n_fft % 2 == 0:
            scale[-1] = 1.0
        inverse_kernel = torch.cat((
            scale * torch.cos(angles) / n_fft,
            scale * -torch.sin(angles) / n_fft,
        ), dim=0).mul(window).unsqueeze(1)
        self.register_buffer('inverse_kernel', inverse_kernel, persistent=False)

        output_start = half_nfft
        output_end = output_start + output_length
        window_sum = F.conv_transpose1d(
            torch.ones(1, 1, frames, dtype=torch.float32),
            window.square().reshape(1, 1, -1),
            stride=hop_length,
        )[..., output_start:output_end]
        if window_sum.shape[-1] != output_length:
            raise ValueError(
                f'ISTFT normalization has {window_sum.shape[-1]} samples; '
                f'expected {output_length}.'
            )
        self.register_buffer(
            'window_sum_inverse', window_sum.reciprocal(), persistent=False)
        self.hop_length = hop_length
        self.output_start = output_start
        self.output_end = output_end

    def forward(self, spectrum):
        inverse = F.conv_transpose1d(
            spectrum, self.inverse_kernel, stride=self.hop_length)
        return (
            inverse[..., self.output_start:self.output_end]
            * self.window_sum_inverse
        )


class ChannelLayerNorm(torch.nn.Module):
    def __init__(self, channels, frequencies):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1, channels, frequencies, 1))
        self.b = torch.nn.Parameter(
            torch.rand(1, channels, frequencies, 1) * 1e-4)
        self.sum_norm_scale = float(channels * frequencies - 1) ** 0.5
        self.sum_norm_eps = 1e-8 * self.sum_norm_scale
        self.scale_folded = False

    def fold_scale(self):
        if not self.scale_folded:
            with torch.no_grad():
                self.w.mul_(self.sum_norm_scale)
            self.scale_folded = True
        return self

    def forward(self, inputs):
        mean = inputs.mean((1, 2), keepdim=True)
        centered = inputs - mean
        if self.scale_folded:
            norm = centered.square().sum((1, 2), keepdim=True).sqrt()
            return centered / (norm + self.sum_norm_eps) * self.w + self.b
        std = centered.std((1, 2), keepdim=True)
        return centered / (std + 1e-8) * self.w + self.b


class ChannelLSTMTime(torch.nn.Module):
    def __init__(self, in_channels, feature_channels, out_channels, num_layers=1,
                 frequencies=160):
        super().__init__()
        self.lstm2 = torch.nn.LSTM(
            in_channels,
            feature_channels,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = torch.nn.Linear(feature_channels, out_channels)
        self.out_channels = out_channels
        self.frequencies = frequencies

    def forward_recurrent(self, inputs):
        batch = inputs.shape[0]
        frames = inputs.shape[-1]
        hidden = inputs.permute(0, 2, 3, 1).reshape(
            batch * self.frequencies, frames, self.lstm2.input_size)
        hidden, _ = self.lstm2(hidden)
        return hidden.reshape(
            batch, self.frequencies, frames, self.linear.in_features)

    def forward(self, inputs):
        hidden = self.linear(self.forward_recurrent(inputs))
        return hidden.permute(0, 3, 1, 2)


class ChannelLSTMFrequency(torch.nn.Module):
    def __init__(self, in_channels, feature_channels, out_channels, frequencies=160):
        super().__init__()
        self.lstm2 = torch.nn.LSTM(
            in_channels,
            feature_channels,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = torch.nn.Linear(feature_channels * 2, out_channels)
        self.out_channels = out_channels
        self.frequencies = frequencies

    def forward_recurrent(self, inputs):
        batch = inputs.shape[0]
        frames = inputs.shape[-1]
        hidden = inputs.permute(0, 3, 2, 1).reshape(
            batch * frames, self.frequencies, self.lstm2.input_size)
        hidden, _ = self.lstm2(hidden)
        return hidden.reshape(
            batch, frames, self.frequencies, self.linear.in_features)

    def forward(self, inputs):
        hidden = self.linear(self.forward_recurrent(inputs))
        return hidden.permute(0, 3, 2, 1)


class CepsUnit(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.frequencies = 81
        self.nfft = SDAEC_NFFT // 2 + 1
        self.ch_lstm_f = ChannelLSTMFrequency(
            channels * 2, channels, channels * 2, frequencies=self.frequencies)
        self.LN = ChannelLayerNorm(channels * 2, self.frequencies)

        half_nfft = self.nfft // 2
        time = torch.arange(self.nfft, dtype=torch.float32).unsqueeze(0)
        frequency = torch.arange(half_nfft + 1, dtype=torch.float32).unsqueeze(1)
        angles = 2.0 * torch.pi * frequency * time / self.nfft
        self.register_buffer(
            "analysis_weight",
            torch.cat((torch.cos(angles), -torch.sin(angles)), dim=0),
            persistent=False,
        )

        fourier = torch.fft.fft(torch.eye(self.nfft, dtype=torch.float32))
        basis = torch.vstack((
            fourier.real[:half_nfft + 1],
            fourier.imag[:half_nfft + 1],
        )).float()
        self.register_buffer(
            "inverse_weight",
            torch.linalg.pinv(basis),
            persistent=False,
        )

    def forward(self, inputs):
        batch = inputs.shape[0]
        frames = inputs.shape[-1]
        rows = inputs.permute(0, 1, 3, 2)
        spectrum = F.linear(rows, self.analysis_weight)
        spectrum = spectrum.reshape(
            batch, self.channels, frames, 2, self.frequencies
        ).permute(0, 3, 1, 4, 2).reshape(
            batch, self.channels * 2, self.frequencies, frames)

        processed = self.ch_lstm_f(self.LN(spectrum))
        processed_real, processed_imag = processed.split(self.channels, dim=1)
        real, imag = spectrum.split(self.channels, dim=1)
        output_real = processed_real * real - processed_imag * imag
        output_imag = processed_real * imag + processed_imag * real

        packed = torch.cat((output_real, output_imag), dim=2)
        inverse = F.linear(
            packed.permute(0, 1, 3, 2), self.inverse_weight)
        return inverse.permute(0, 1, 3, 2)


class CFB(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_gate = torch.nn.Conv2d(in_channels, out_channels, (1, 1))
        self.conv_input = torch.nn.Conv2d(in_channels, out_channels, (1, 1))
        self.conv = torch.nn.Conv2d(
            out_channels, out_channels, (3, 1), padding=(1, 0))
        self.ceps_unit = CepsUnit(out_channels)
        self.LN0 = ChannelLayerNorm(in_channels, 160)
        self.LN1 = ChannelLayerNorm(out_channels, 160)
        self.LN2 = ChannelLayerNorm(out_channels, 160)

    def forward(self, inputs):
        gate = torch.sigmoid(self.conv_gate(self.LN0(inputs)))
        projected = self.conv_input(inputs)
        gated = gate * projected
        output = self.conv(self.LN1(gated))
        return output + self.ceps_unit(self.LN2(projected - gated))


class SDAECNet(torch.nn.Module):
    def __init__(self, max_frames):
        super().__init__()
        channels = 20
        self.in_ch_lstm = ChannelLSTMFrequency(4, channels, channels)
        self.in_conv = torch.nn.Conv2d(4 + channels, channels, (1, 1))
        self.cfb_e1 = CFB(channels, channels)
        self.cfb_e2 = CFB(channels, channels)
        self.cfb_e3 = CFB(channels, channels)
        self.cfb_e4 = CFB(channels, channels)
        self.cfb_e5 = CFB(channels, channels)
        self.ln = ChannelLayerNorm(channels, 160)
        self.ch_lstm = ChannelLSTMTime(
            channels, channels * 2, channels, num_layers=2)
        self.cfb_d5 = CFB(channels, channels)
        self.cfb_d4 = CFB(channels * 2, channels)
        self.cfb_d3 = CFB(channels * 2, channels)
        self.cfb_d2 = CFB(channels * 2, channels)
        self.cfb_d1 = CFB(channels * 2, channels)
        self.out_ch_lstm = ChannelLSTMTime(
            channels * 2, channels, channels * 2)
        self.out_conv = torch.nn.Conv2d(channels * 3, 2, (1, 1))
        self.istft = SDAECISTFT(max_frames)
        self.register_buffer(
            "in_fused_weight", torch.empty(0), persistent=False)
        self.register_buffer(
            "in_fused_bias", torch.empty(0), persistent=False)
        self.register_buffer(
            "out_fused_weight", torch.empty(0), persistent=False)
        self.register_buffer(
            "out_fused_bias", torch.empty(0), persistent=False)

    def fold_for_export(self):
        with torch.no_grad():
            in_conv_weight = self.in_conv.weight[:, :, 0, 0].double()
            in_projected = in_conv_weight[
                :, :self.in_ch_lstm.linear.out_features]
            self.in_fused_weight = torch.cat((
                in_projected @ self.in_ch_lstm.linear.weight.double(),
                in_conv_weight[:, self.in_ch_lstm.linear.out_features:],
            ), dim=1).float().unsqueeze(-1).unsqueeze(-1).contiguous()
            self.in_fused_bias = (
                self.in_conv.bias.double()
                + in_projected @ self.in_ch_lstm.linear.bias.double()
            ).float()

            out_conv_weight = self.out_conv.weight[:, :, 0, 0].double()
            out_projected = out_conv_weight[
                :, :self.out_ch_lstm.linear.out_features]
            self.out_fused_weight = torch.cat((
                out_projected @ self.out_ch_lstm.linear.weight.double(),
                out_conv_weight[:, self.out_ch_lstm.linear.out_features:],
            ), dim=1).float().unsqueeze(-1).unsqueeze(-1).contiguous()
            self.out_fused_bias = (
                self.out_conv.bias.double()
                + out_projected @ self.out_ch_lstm.linear.bias.double()
            ).float()
        for module in self.modules():
            if isinstance(module, ChannelLayerNorm):
                module.fold_scale()
        return self

    def forward(self, inputs):
        encoder_0 = self.in_ch_lstm.forward_recurrent(inputs).permute(0, 3, 2, 1)
        encoder_0 = F.conv2d(
            torch.cat((encoder_0, inputs), dim=1),
            self.in_fused_weight,
            self.in_fused_bias,
        )
        encoder_1 = self.cfb_e1(encoder_0)
        encoder_2 = self.cfb_e2(encoder_1)
        encoder_3 = self.cfb_e3(encoder_2)
        encoder_4 = self.cfb_e4(encoder_3)
        encoder_5 = self.cfb_e5(encoder_4)
        temporal = self.ch_lstm(self.ln(encoder_5))
        decoder_5 = self.cfb_d5(encoder_5 * temporal)
        decoder_4 = self.cfb_d4(torch.cat((encoder_4, decoder_5), dim=1))
        decoder_3 = self.cfb_d3(torch.cat((encoder_3, decoder_4), dim=1))
        decoder_2 = self.cfb_d2(torch.cat((encoder_2, decoder_3), dim=1))
        decoder_1 = self.cfb_d1(torch.cat((encoder_1, decoder_2), dim=1))
        decoder_0 = self.out_ch_lstm.forward_recurrent(
            torch.cat((encoder_0, decoder_1), dim=1)
        ).permute(0, 3, 1, 2)
        spectrum = F.conv2d(
            torch.cat((decoder_0, decoder_1), dim=1),
            self.out_fused_weight,
            self.out_fused_bias,
        )
        return self.istft(spectrum.reshape(
            spectrum.shape[0], SDAEC_NFFT + 1, spectrum.shape[-1]))


class SDAEC(torch.nn.Module):
    def __init__(self, alpha_predictor, network):
        super().__init__()
        self.stft = SDAECSTFT()
        self.network = network
        alpha_kernel, alpha_bias = alpha_predictor.fold()
        self.register_buffer(
            "alpha_kernel", alpha_kernel, persistent=False)
        self.register_buffer(
            "alpha_bias", alpha_bias, persistent=False)
        self.register_buffer(
            "alpha_padding",
            torch.zeros(1, 2, alpha_kernel.shape[-1] - 1),
            persistent=False,
        )

    def forward(self, near_end_audio, far_end_audio):
        batch = near_end_audio.shape[0]
        spectrum = self.stft(torch.cat((near_end_audio, far_end_audio), dim=0))
        spectrum = spectrum.reshape(
            2,
            batch,
            2,
            SDAEC_NFFT // 2 + 1,
            spectrum.shape[-1],
        )
        frame_power = spectrum.square().sum((2, 3)).permute(1, 0, 2)
        mix_comp = spectrum[0]
        far_comp = spectrum[1]
        padding = self.alpha_padding.expand(batch, -1, -1)
        alpha = F.conv1d(
            torch.cat((padding, frame_power), dim=-1),
            self.alpha_kernel,
            self.alpha_bias,
        ).unsqueeze(2).abs()
        return self.network(torch.cat((mix_comp, far_comp * alpha), dim=1))


class DeepEchoLayerNorm(torch.nn.Module):
    def __init__(self, channels, frequencies, sum_scale=1.0):
        super().__init__()
        self.w = torch.nn.Parameter(
            torch.ones(1, channels, frequencies, 1))
        self.b = torch.nn.Parameter(
            torch.rand(1, channels, frequencies, 1) * 1e-4)
        denominator = float(max(channels * frequencies - 1, 1))
        self.weight_scale = denominator ** 0.5
        self.norm_eps = 1e-8 * denominator
        self.sum_scale = float(sum_scale)
        self.sum_scale_squared = float(sum_scale * sum_scale)
        self.scale_folded = False

    def fold_scale(self):
        if not self.scale_folded:
            with torch.no_grad():
                self.w.mul_(self.weight_scale)
            self.scale_folded = True
        return self

    def forward(self, inputs, norm_eps=None):
        centered = inputs - inputs.mean((1, 2), keepdim=True)
        epsilon = self.norm_eps if norm_eps is None else norm_eps
        if self.sum_scale != 1.0:
            centered = centered * self.sum_scale
            epsilon = epsilon * self.sum_scale_squared
        inverse_std = torch.rsqrt(
            centered.square().sum((1, 2), keepdim=True) + epsilon)
        return centered * inverse_std * self.w + self.b


class DeepEchoChannelLSTMTime(torch.nn.Module):
    def __init__(self, in_channels, feature_channels, out_channels,
                 num_layers=1, frequencies=160, static_batch=1,
                 static_frames=0):
        super().__init__()
        self.lstm2 = torch.nn.LSTM(
            in_channels,
            feature_channels,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = torch.nn.Linear(feature_channels, out_channels)
        self.frequencies = frequencies
        self.static_batch = static_batch
        self.static_frames = static_frames
        self.register_buffer(
            'initial_state',
            torch.zeros(
                num_layers,
                static_batch * frequencies,
                feature_channels,
            ),
            persistent=False,
        )

    def forward_recurrent(self, inputs):
        frames = self.static_frames if self.static_frames else inputs.shape[-1]
        hidden = inputs.permute(0, 2, 3, 1).reshape(
            self.static_batch * self.frequencies,
            frames,
            self.lstm2.input_size,
        )
        hidden, _ = self.lstm2(
            hidden, (self.initial_state, self.initial_state))
        return hidden.reshape(
            self.static_batch,
            self.frequencies,
            frames,
            self.linear.in_features,
        )

    def forward(self, inputs):
        hidden = self.linear(self.forward_recurrent(inputs))
        return hidden.permute(0, 3, 1, 2)


class DeepEchoChannelLSTMFrequency(torch.nn.Module):
    def __init__(self, in_channels, feature_channels, out_channels,
                 frequencies=160, static_batch=1, static_frames=0,
                 register_state=True):
        super().__init__()
        self.lstm2 = torch.nn.LSTM(
            in_channels,
            feature_channels,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = torch.nn.Linear(feature_channels * 2, out_channels)
        self.frequencies = frequencies
        self.static_batch = static_batch
        self.static_frames = static_frames
        if static_frames and register_state:
            self.register_buffer(
                'initial_state',
                torch.zeros(
                    2,
                    static_batch * static_frames,
                    feature_channels,
                ),
                persistent=False,
            )

    def forward_recurrent(self, inputs, initial_state=None):
        batch = self.static_batch if self.static_frames else inputs.shape[0]
        frames = self.static_frames if self.static_frames else inputs.shape[-1]
        hidden = inputs.transpose(1, 3).reshape(
            batch * frames,
            self.frequencies,
            self.lstm2.input_size,
        )
        if initial_state is not None:
            hidden, _ = self.lstm2(
                hidden, (initial_state, initial_state))
        elif hasattr(self, 'initial_state'):
            hidden, _ = self.lstm2(
                hidden, (self.initial_state, self.initial_state))
        else:
            hidden, _ = self.lstm2(hidden)
        return hidden.reshape(
            batch,
            frames,
            self.frequencies,
            self.linear.in_features,
        )

    def forward(self, inputs, initial_state=None):
        hidden = self.linear(
            self.forward_recurrent(inputs, initial_state))
        return hidden.transpose(1, 3)


class DeepEchoCepsUnit(torch.nn.Module):
    def __init__(self, channels, static_batch=1, static_frames=0,
                 build_basis=True, external_lstm_state=False):
        super().__init__()
        self.channels = channels
        self.static_batch = static_batch
        self.static_frames = static_frames
        self.frequencies = 81
        self.nfft = DEEP_ECHO_NFFT // 2 + 1
        self.ch_lstm_f = DeepEchoChannelLSTMFrequency(
            channels * 2,
            channels,
            channels * 2,
            frequencies=self.frequencies,
            static_batch=static_batch,
            static_frames=static_frames,
            register_state=not external_lstm_state,
        )
        self.LN = DeepEchoLayerNorm(
            channels * 2, self.frequencies, sum_scale=0.25)
        if self.frequencies != self.nfft // 2 + 1:
            raise ValueError(
                'Deep Echo cepstral transform geometry is inconsistent.')

        if build_basis:
            time = torch.arange(self.nfft, dtype=torch.float32).unsqueeze(0)
            frequency = torch.arange(
                self.frequencies, dtype=torch.float32).unsqueeze(1)
            angles = 2.0 * torch.pi * frequency * time / self.nfft
            self.register_buffer(
                'stft_kernel',
                torch.cat((torch.cos(angles), -torch.sin(angles)), dim=0)
                .unsqueeze(1),
                persistent=False,
            )
            fourier = torch.fft.fft(
                torch.eye(self.nfft, dtype=torch.float32))
            basis = torch.vstack((
                fourier.real[:self.frequencies],
                fourier.imag[:self.frequencies],
            )).float()
            self.register_buffer(
                'inverse_basis',
                torch.linalg.pinv(basis).T.unsqueeze(1),
                persistent=False,
            )

    def forward(self, inputs, norm_eps, stft_kernel, inverse_basis,
                initial_state):
        batch = self.static_batch if self.static_frames else inputs.shape[0]
        frames = self.static_frames if self.static_frames else inputs.shape[-1]
        rows = inputs.transpose(2, 3).reshape(-1, 1, self.nfft)
        spectrum = F.conv1d(rows, stft_kernel, stride=self.nfft)
        spectrum = spectrum.reshape(
            batch,
            self.channels,
            frames,
            2,
            self.frequencies,
        ).permute(0, 3, 1, 4, 2)
        spectrum_real, spectrum_imag = spectrum.split(1, dim=1)

        processed = self.ch_lstm_f(
            self.LN(spectrum.reshape(
                batch,
                self.channels * 2,
                self.frequencies,
                frames,
            ), norm_eps),
            initial_state,
        ).reshape(batch, 2, self.channels, self.frequencies, frames)
        processed_real, processed_imag = processed.split(1, dim=1)
        output_real = processed_real * spectrum_real - processed_imag * spectrum_imag
        output_imag = processed_real * spectrum_imag + processed_imag * spectrum_real

        packed = torch.cat((output_real, output_imag), dim=3)
        packed = packed.permute(0, 1, 2, 4, 3).reshape(
            -1, 2 * self.frequencies, 1)
        inverse = F.conv_transpose1d(
            packed, inverse_basis, stride=self.nfft)
        return inverse.reshape(
            batch, self.channels, frames, self.nfft).transpose(2, 3)


class DeepEchoCFB(torch.nn.Module):
    def __init__(self, in_channels, out_channels, static_batch=1,
                 static_frames=0, build_ceps_basis=True,
                 external_lstm_state=False):
        super().__init__()
        self.conv_gate = torch.nn.Conv2d(in_channels, out_channels, (1, 1))
        self.conv_input = torch.nn.Conv2d(in_channels, out_channels, (1, 1))
        self.conv = torch.nn.Conv2d(
            out_channels, out_channels, (3, 1), padding=(1, 0))
        self.ceps_unit = DeepEchoCepsUnit(
            out_channels,
            static_batch=static_batch,
            static_frames=static_frames,
            build_basis=build_ceps_basis,
            external_lstm_state=external_lstm_state,
        )
        self.LN0 = DeepEchoLayerNorm(in_channels, 160)
        self.LN1 = DeepEchoLayerNorm(out_channels, 160)
        self.LN2 = DeepEchoLayerNorm(out_channels, 160)

    def forward(self, inputs, norm_eps, ceps_norm_eps, stft_kernel,
                inverse_basis, initial_state):
        gate = torch.sigmoid(self.conv_gate(self.LN0(inputs, norm_eps)))
        projected = self.conv_input(inputs)
        gated = gate * projected
        output = self.conv(self.LN1(gated, norm_eps))
        return output + self.ceps_unit(
            self.LN2(projected - gated, norm_eps),
            ceps_norm_eps,
            stft_kernel,
            inverse_basis,
            initial_state,
        )


class DeepEchoNet(torch.nn.Module):
    def __init__(self, custom_istft, order=DEEP_ECHO_ORDER, channels=20,
                 static_batch=1, static_frames=0):
        super().__init__()
        self.order = order
        self.static_batch = static_batch
        self.static_frames = static_frames
        self.n_freq = DEEP_ECHO_NFFT // 2 + 1
        self.custom_istft = custom_istft

        delay_kernel = torch.eye(order, dtype=torch.float32)
        self.register_buffer(
            'delay_bank_kernel',
            delay_kernel.repeat(2, 1).reshape(2 * order, 1, 1, order),
            persistent=False,
        )
        self.register_buffer(
            'delay_pad',
            torch.zeros(static_batch, 2, self.n_freq, order - 1),
            persistent=False,
        )
        self.register_buffer(
            'norm_eps',
            torch.tensor(1e-8 * (channels * self.n_freq - 1)),
            persistent=False,
        )
        self.register_buffer(
            'ceps_norm_eps',
            torch.tensor(1e-8 * (2 * channels * 81 - 1)),
            persistent=False,
        )
        if static_frames:
            self.register_buffer(
                'bidir_initial_state',
                torch.zeros(2, static_batch * static_frames, channels),
                persistent=False,
            )

        self.in_ch_lstm = DeepEchoChannelLSTMFrequency(
            4,
            channels,
            channels,
            static_batch=static_batch,
            static_frames=static_frames,
            register_state=False,
        )
        self.in_conv = torch.nn.Conv2d(4 + channels, channels, (1, 1))
        self.cfb_e1 = DeepEchoCFB(
            channels,
            channels,
            static_batch=static_batch,
            static_frames=static_frames,
            external_lstm_state=True,
        )
        self.ln = DeepEchoLayerNorm(channels, self.n_freq)
        self.ch_lstm = DeepEchoChannelLSTMTime(
            channels,
            channels * 2,
            channels,
            num_layers=2,
            static_batch=static_batch,
            static_frames=static_frames,
        )
        self.cfb_d1 = DeepEchoCFB(
            channels,
            channels,
            static_batch=static_batch,
            static_frames=static_frames,
            build_ceps_basis=False,
            external_lstm_state=True,
        )
        self.out_ch_lstm = DeepEchoChannelLSTMTime(
            channels * 2,
            channels,
            channels * 2,
            static_batch=static_batch,
            static_frames=static_frames,
        )
        self.out_conv = torch.nn.Conv2d(
            channels * 3, order * 2, (1, 1))
        self.io_projections_folded = False

    def fold_for_export(self):
        if self.io_projections_folded:
            return self
        in_weight, in_bias = fold_linear_into_pointwise_conv(
            self.in_ch_lstm.linear, self.in_conv)
        out_weight, out_bias = fold_linear_into_pointwise_conv(
            self.out_ch_lstm.linear, self.out_conv)
        self.register_buffer('in_fused_weight', in_weight, persistent=False)
        self.register_buffer('in_fused_bias', in_bias, persistent=False)
        self.register_buffer('out_fused_weight', out_weight, persistent=False)
        self.register_buffer('out_fused_bias', out_bias, persistent=False)
        for module in self.modules():
            if isinstance(module, DeepEchoLayerNorm):
                module.fold_scale()
            elif isinstance(module, torch.nn.LSTM):
                module.flatten_parameters()
        self.io_projections_folded = True
        return self

    def apply_echo_path(self, far_comp, echo_path):
        frames = self.static_frames if self.static_frames else far_comp.shape[-1]
        delayed_far = F.conv2d(
            torch.cat((self.delay_pad, far_comp), dim=-1),
            self.delay_bank_kernel,
            groups=2,
        ).reshape(
            self.static_batch,
            2,
            self.order,
            self.n_freq,
            frames,
        )
        far_real, far_imag = delayed_far.split(1, dim=1)
        path_real, path_imag = echo_path.split(1, dim=1)
        echo_real = far_real * path_real - far_imag * path_imag
        echo_imag = far_real * path_imag + far_imag * path_real
        return torch.cat((
            echo_real.sum(dim=2),
            echo_imag.sum(dim=2),
        ), dim=1)

    def forward(self, inputs):
        frames = self.static_frames if self.static_frames else inputs.shape[-1]
        initial_state = getattr(self, 'bidir_initial_state', None)
        ceps_basis = self.cfb_e1.ceps_unit
        mix_comp = inputs[:, 0::2]
        far_comp = inputs[:, 1::2]

        encoder_0 = self.in_ch_lstm.forward_recurrent(
            inputs, initial_state).permute(0, 3, 2, 1)
        encoder_0 = F.conv2d(
            torch.cat((encoder_0, inputs), dim=1),
            self.in_fused_weight,
            self.in_fused_bias,
        )
        encoder_1 = self.cfb_e1(
            encoder_0,
            self.norm_eps,
            self.ceps_norm_eps,
            ceps_basis.stft_kernel,
            ceps_basis.inverse_basis,
            initial_state,
        )
        temporal = self.ch_lstm(self.ln(encoder_1, self.norm_eps))
        decoder_1 = self.cfb_d1(
            encoder_1 * temporal,
            self.norm_eps,
            self.ceps_norm_eps,
            ceps_basis.stft_kernel,
            ceps_basis.inverse_basis,
            initial_state,
        )
        decoder_input = torch.cat((encoder_0, decoder_1), dim=1)
        decoder_0 = self.out_ch_lstm.forward_recurrent(
            decoder_input).permute(0, 3, 1, 2)
        output = F.conv2d(
            torch.cat((decoder_0, decoder_1), dim=1),
            self.out_fused_weight,
            self.out_fused_bias,
        )
        echo_path = output.reshape(
            self.static_batch,
            2,
            self.order,
            self.n_freq,
            frames,
        )
        enhanced = mix_comp - self.apply_echo_path(far_comp, echo_path)
        return self.custom_istft(
            enhanced.reshape(
                self.static_batch, 2 * self.n_freq, frames))


class DeepEchoAEC(torch.nn.Module):
    def __init__(self, stft, network):
        super().__init__()
        self.stft = stft
        self.network = network

    def forward(self, near_end_audio, far_end_audio):
        batch = near_end_audio.shape[0]
        spectrum = self.stft(
            torch.cat((near_end_audio, far_end_audio), dim=0))
        spectrum = spectrum.reshape(
            2,
            batch,
            2,
            DEEP_ECHO_NFFT // 2 + 1,
            spectrum.shape[-1],
        )
        inputs = spectrum.permute(1, 2, 0, 3, 4).reshape(
            batch,
            4,
            DEEP_ECHO_NFFT // 2 + 1,
            spectrum.shape[-1],
        )
        return self.network(inputs)


class NKFComplexGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        if num_layers != 1:
            raise ValueError('The NKF checkpoint uses exactly one GRU layer.')
        self.gru_r = torch.nn.GRU(input_size, hidden_size, num_layers)
        self.gru_i = torch.nn.GRU(input_size, hidden_size, num_layers)

    def forward(self, real, imag, state_rr, state_ir, state_ri, state_ii):
        _, state_rr = self.gru_r(real, state_rr)
        _, state_ir = self.gru_r(imag, state_ir)
        _, state_ri = self.gru_i(real, state_ri)
        _, state_ii = self.gru_i(imag, state_ii)
        return (
            state_rr - state_ii,
            state_ri + state_ir,
            state_rr,
            state_ir,
            state_ri,
            state_ii,
        )


class NKFComplexDense(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.linear_real = torch.nn.Linear(input_channels, output_channels)
        self.linear_imag = torch.nn.Linear(input_channels, output_channels)

    def forward(self, real, imag):
        return self.linear_real(real), self.linear_imag(imag)


class NKFComplexPReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu = torch.nn.PReLU()
        self.negative_slope = 0.0
        self.use_cached_weight = False

    def cache_weight(self):
        self.negative_slope = float(self.prelu.weight.detach().item())
        self.use_cached_weight = True
        return self

    def forward(self, real, imag):
        if self.use_cached_weight:
            return (
                F.leaky_relu(real, negative_slope=self.negative_slope),
                F.leaky_relu(imag, negative_slope=self.negative_slope),
            )
        return (
            F.prelu(real, self.prelu.weight),
            F.prelu(imag, self.prelu.weight),
        )


class NKFKGNet(torch.nn.Module):
    def __init__(self, filter_order, fc_dim, rnn_layers, rnn_dim,
                 model_batch, frequencies):
        super().__init__()
        self.filter_order = filter_order
        self.model_batch = model_batch
        self.frequencies = frequencies
        self.branch_batch = model_batch * frequencies
        self.fc_in_dense = NKFComplexDense(2 * filter_order + 1, fc_dim)
        self.fc_in_act = NKFComplexPReLU()
        self.complex_gru = NKFComplexGRU(fc_dim, rnn_dim, rnn_layers)
        self.fc_out_dense1 = NKFComplexDense(rnn_dim, fc_dim)
        self.fc_out_act = NKFComplexPReLU()
        self.fc_out_dense2 = NKFComplexDense(fc_dim, filter_order)

    def cache_export_constants(self):
        self.fc_in_act.cache_weight()
        self.fc_out_act.cache_weight()
        return self

    def forward(self, input_real, input_imag, state_rr, state_ir,
                state_ri, state_ii):
        real, imag = self.fc_in_dense(input_real, input_imag)
        real, imag = self.fc_in_act(real, imag)
        real = real.reshape(1, self.branch_batch, -1)
        imag = imag.reshape(1, self.branch_batch, -1)
        real, imag, state_rr, state_ir, state_ri, state_ii = self.complex_gru(
            real, imag, state_rr, state_ir, state_ri, state_ii)
        real = real.reshape(self.model_batch, self.frequencies, -1)
        imag = imag.reshape(self.model_batch, self.frequencies, -1)
        real, imag = self.fc_out_dense1(real, imag)
        real, imag = self.fc_out_act(real, imag)
        real, imag = self.fc_out_dense2(real, imag)
        return (
            real.reshape(
                self.model_batch, self.frequencies, self.filter_order),
            imag.reshape(
                self.model_batch, self.frequencies, self.filter_order),
            state_rr,
            state_ir,
            state_ri,
            state_ii,
        )


class NKFInner(torch.nn.Module):
    def __init__(self, custom_stft, custom_istft, frames,
                 filter_order=NKF_FILTER_ORDER, fc_dim=NKF_FC_DIM,
                 rnn_layers=NKF_RNN_LAYERS, rnn_dim=NKF_RNN_DIM,
                 model_batch=1):
        super().__init__()
        self.filter_order = filter_order
        self.custom_stft = custom_stft
        self.custom_istft = custom_istft
        self.frames = frames
        self.model_batch = model_batch
        self.frequencies = NKF_NFFT // 2 + 1
        self.branch_batch = model_batch * self.frequencies
        self.kg_net = NKFKGNet(
            filter_order,
            fc_dim,
            rnn_layers,
            rnn_dim,
            model_batch,
            self.frequencies,
        )

        self.register_buffer(
            'h_prior_real_buffer',
            torch.zeros(model_batch, self.frequencies, filter_order),
            persistent=False,
        )
        self.register_buffer(
            'h_prior_imag_buffer',
            torch.zeros(model_batch, self.frequencies, filter_order),
            persistent=False,
        )
        for name in ('h_rr_buffer', 'h_ir_buffer',
                     'h_ri_buffer', 'h_ii_buffer'):
            self.register_buffer(
                name,
                torch.zeros(rnn_layers, self.branch_batch, rnn_dim),
                persistent=False,
            )
        self.register_buffer(
            'zeros_pad',
            torch.zeros(model_batch, self.frequencies, filter_order - 1),
            persistent=False,
        )

    def cache_export_constants(self):
        self.kg_net.cache_export_constants()
        return self

    def forward(self, audio_pair):
        pair = self.custom_stft(audio_pair).reshape(
            2,
            self.model_batch,
            2,
            self.frequencies,
            self.frames,
        )
        reference_real = pair[0, :, 0]
        reference_imag = pair[0, :, 1]
        microphone_real = pair[1, :, 0]
        microphone_imag = pair[1, :, 1]

        h_prior_real = self.h_prior_real_buffer
        h_prior_imag = self.h_prior_imag_buffer
        state_rr = self.h_rr_buffer
        state_ir = self.h_ir_buffer
        state_ri = self.h_ri_buffer
        state_ii = self.h_ii_buffer
        reference_real = torch.cat(
            (self.zeros_pad, reference_real), dim=2)
        reference_imag = torch.cat(
            (self.zeros_pad, reference_imag), dim=2)
        echo_real_frames = []
        echo_imag_frames = []

        delayed_real = reference_real[..., :self.filter_order]
        delayed_imag = reference_imag[..., :self.filter_order]
        microphone_real_frame = microphone_real[..., [0]]
        microphone_imag_frame = microphone_imag[..., [0]]
        input_real = torch.cat(
            (delayed_real, microphone_real_frame, h_prior_real), dim=2)
        input_imag = torch.cat(
            (delayed_imag, microphone_imag_frame, h_prior_imag), dim=2)
        gain_real, gain_imag, state_rr, state_ir, state_ri, state_ii = (
            self.kg_net(
                input_real,
                input_imag,
                state_rr,
                state_ir,
                state_ri,
                state_ii,
            )
        )
        h_post_real = (
            gain_real * microphone_real_frame
            - gain_imag * microphone_imag_frame
        )
        h_post_imag = (
            gain_real * microphone_imag_frame
            + gain_imag * microphone_real_frame
        )
        echo_real_frames.append((
            delayed_real * h_post_real - delayed_imag * h_post_imag
        ).sum(dim=2, keepdim=True))
        echo_imag_frames.append((
            delayed_real * h_post_imag + delayed_imag * h_post_real
        ).sum(dim=2, keepdim=True))

        for frame in range(1, self.frames):
            delayed_real = reference_real[
                ..., frame:frame + self.filter_order]
            delayed_imag = reference_imag[
                ..., frame:frame + self.filter_order]
            microphone_real_frame = microphone_real[..., frame:frame + 1]
            microphone_imag_frame = microphone_imag[..., frame:frame + 1]

            delta_real = h_post_real - h_prior_real
            delta_imag = h_post_imag - h_prior_imag
            h_prior_real, h_post_real = h_post_real, h_prior_real
            h_prior_imag, h_post_imag = h_post_imag, h_prior_imag
            error_real = microphone_real_frame - (
                delayed_real * h_prior_real
                - delayed_imag * h_prior_imag
            ).sum(dim=2, keepdim=True)
            error_imag = microphone_imag_frame - (
                delayed_real * h_prior_imag
                + delayed_imag * h_prior_real
            ).sum(dim=2, keepdim=True)

            input_real = torch.cat(
                (delayed_real, error_real, delta_real), dim=2)
            input_imag = torch.cat(
                (delayed_imag, error_imag, delta_imag), dim=2)
            gain_real, gain_imag, state_rr, state_ir, state_ri, state_ii = (
                self.kg_net(
                    input_real,
                    input_imag,
                    state_rr,
                    state_ir,
                    state_ri,
                    state_ii,
                )
            )
            h_post_real = (
                h_prior_real
                + gain_real * error_real
                - gain_imag * error_imag
            )
            h_post_imag = (
                h_prior_imag
                + gain_real * error_imag
                + gain_imag * error_real
            )
            echo_real_frames.append((
                delayed_real * h_post_real - delayed_imag * h_post_imag
            ).sum(dim=2, keepdim=True))
            echo_imag_frames.append((
                delayed_real * h_post_imag + delayed_imag * h_post_real
            ).sum(dim=2, keepdim=True))

        echo_real = torch.cat(echo_real_frames, dim=2)
        echo_imag = torch.cat(echo_imag_frames, dim=2)
        enhanced = torch.cat((
            microphone_real - echo_real,
            microphone_imag - echo_imag,
        ), dim=1)
        return self.custom_istft(enhanced)


class NKFAEC(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, near_end_audio, far_end_audio):
        return self.network(torch.cat((far_end_audio, near_end_audio), dim=0))


def load_nkf_weights(export_model, original_state_dict):
    mapped = {
        'kg_net.fc_in_dense.linear_real.weight':
            original_state_dict['kg_net.fc_in.0.linear_real.weight'],
        'kg_net.fc_in_dense.linear_real.bias':
            original_state_dict['kg_net.fc_in.0.linear_real.bias'],
        'kg_net.fc_in_dense.linear_imag.weight':
            original_state_dict['kg_net.fc_in.0.linear_imag.weight'],
        'kg_net.fc_in_dense.linear_imag.bias':
            original_state_dict['kg_net.fc_in.0.linear_imag.bias'],
        'kg_net.fc_in_act.prelu.weight':
            original_state_dict['kg_net.fc_in.1.prelu.weight'],
        'kg_net.fc_out_dense1.linear_real.weight':
            original_state_dict['kg_net.fc_out.0.linear_real.weight'],
        'kg_net.fc_out_dense1.linear_real.bias':
            original_state_dict['kg_net.fc_out.0.linear_real.bias'],
        'kg_net.fc_out_dense1.linear_imag.weight':
            original_state_dict['kg_net.fc_out.0.linear_imag.weight'],
        'kg_net.fc_out_dense1.linear_imag.bias':
            original_state_dict['kg_net.fc_out.0.linear_imag.bias'],
        'kg_net.fc_out_act.prelu.weight':
            original_state_dict['kg_net.fc_out.1.prelu.weight'],
        'kg_net.fc_out_dense2.linear_real.weight':
            original_state_dict['kg_net.fc_out.2.linear_real.weight'],
        'kg_net.fc_out_dense2.linear_real.bias':
            original_state_dict['kg_net.fc_out.2.linear_real.bias'],
        'kg_net.fc_out_dense2.linear_imag.weight':
            original_state_dict['kg_net.fc_out.2.linear_imag.weight'],
        'kg_net.fc_out_dense2.linear_imag.bias':
            original_state_dict['kg_net.fc_out.2.linear_imag.bias'],
    }
    for gru_name in ('gru_r', 'gru_i'):
        for parameter in (
                'weight_ih_l0', 'weight_hh_l0',
                'bias_ih_l0', 'bias_hh_l0'):
            key = f'kg_net.complex_gru.{gru_name}.{parameter}'
            mapped[key] = original_state_dict[key]
    missing, unexpected = export_model.load_state_dict(mapped, strict=False)
    trainable_missing = [
        key for key in missing
        if key in dict(export_model.named_parameters())
    ]
    if trainable_missing or unexpected:
        raise ValueError(
            'NKF checkpoint mapping is incomplete: '
            f'missing={trainable_missing}, unexpected={unexpected}'
        )
    return export_model


def build_kaldi_fbank_frontend():
    bins = FBANK_NFFT // 2 + 1
    window = torch.hamming_window(
        FBANK_WINDOW_LENGTH,
        periodic=False,
        alpha=0.54,
        beta=0.46,
        dtype=torch.float64,
    )
    time = torch.arange(FBANK_WINDOW_LENGTH, dtype=torch.float64).unsqueeze(0)
    frequency = torch.arange(bins, dtype=torch.float64).unsqueeze(1)
    angles = 2.0 * torch.pi * frequency * time / FBANK_NFFT
    dft_weight = torch.cat((
        torch.cos(angles) * window.unsqueeze(0),
        -torch.sin(angles) * window.unsqueeze(0),
    ), dim=0)

    transformed = dft_weight.clone()
    transformed[:, 0] *= 1.0 - PREEMPHASIS
    transformed[:, :-1] -= PREEMPHASIS * dft_weight[:, 1:]
    transformed -= transformed.mean(dim=1, keepdim=True)

    mel_banks, _ = kaldi.get_mel_banks(
        FBANK_MELS,
        FBANK_NFFT,
        float(MODEL_SAMPLE_RATE),
        20.0,
        0.0,
        100.0,
        -500.0,
        1.0,
    )
    mel_banks = F.pad(mel_banks.to(torch.float64), (0, 1))
    return transformed.unsqueeze(1).float(), mel_banks.float()


class KaldiFbank(torch.nn.Module):
    def __init__(self, dither):
        super().__init__()
        analysis_weight, mel_banks = build_kaldi_fbank_frontend()
        self.register_buffer(
            "analysis_weight", analysis_weight.squeeze(1), persistent=False)
        self.register_buffer("mel_banks", mel_banks, persistent=False)
        self.register_buffer(
            "pcm_scale",
            torch.tensor(32768.0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "log_floor",
            torch.tensor(torch.finfo(torch.float32).eps),
            persistent=False,
        )
        self.register_buffer(
            "frame_indices",
            torch.arange(FBANK_MAX_FRAMES, dtype=torch.int64).unsqueeze(1)
            * FBANK_HOP_LENGTH
            + torch.arange(FBANK_WINDOW_LENGTH, dtype=torch.int64).unsqueeze(0),
            persistent=False,
        )
        self.dither = float(dither)

    def forward(self, audio_streams):
        audio_streams = audio_streams * self.pcm_scale
        frames = (
            audio_streams.shape[-1] - FBANK_WINDOW_LENGTH
        ) // FBANK_HOP_LENGTH + 1
        indices = self.frame_indices[:frames]
        features = []
        for stream in audio_streams.split(1, dim=1):
            framed = stream.squeeze(0).squeeze(0)[indices]
            if self.dither:
                framed = framed + torch.randn_like(framed) * self.dither
            spectrum = F.linear(framed, self.analysis_weight)
            power = spectrum.reshape(
                frames, 2, FBANK_NFFT // 2 + 1).square().sum(dim=1)
            mel = F.linear(power, self.mel_banks).clamp(min=self.log_floor).log()
            features.append(mel)
        return torch.cat(features, dim=1).unsqueeze(0)


class DFSMNVAD(torch.nn.Module):
    def __init__(self, light_aec, fbank, dfsmn):
        super().__init__()
        self.light_aec = light_aec
        self.fbank = fbank
        self.dfsmn = dfsmn
        self.input_scale = 1.0 / 32768.0 if IN_AUDIO_DTYPE == 'INT16' else None

    def forward(self, near_end_audio, far_end_audio):
        near_end_audio = near_end_audio.float()
        far_end_audio = far_end_audio.float()
        if self.input_scale is not None:
            near_end_audio = near_end_audio * self.input_scale
            far_end_audio = far_end_audio * self.input_scale
        if IN_SAMPLE_RATE != MODEL_SAMPLE_RATE:
            target_length = MODEL_AUDIO_LENGTH if not DYNAMIC_AXES else None
            scale_factor = None if not DYNAMIC_AXES else INPUT_TO_MODEL_SCALE
            near_end_audio = F.interpolate(
                near_end_audio,
                size=target_length,
                scale_factor=scale_factor,
                mode='linear',
                align_corners=False,
            )
            far_end_audio = F.interpolate(
                far_end_audio,
                size=target_length,
                scale_factor=scale_factor,
                mode='linear',
                align_corners=False,
            )

        linear_residual = self.light_aec(near_end_audio, far_end_audio)
        near_end_audio = near_end_audio[..., :linear_residual.shape[-1]]
        echo_estimate = (
            near_end_audio - ECHO_ESTIMATE_FACTOR * linear_residual)

        features = self.fbank(torch.cat((
            near_end_audio,
            linear_residual,
            echo_estimate,
        ), dim=1))
        vad = self.dfsmn(features)
        return vad.squeeze()


def build_light_aec():
    if LIGHT_AEC_MODEL == 'SDAEC':
        alpha_predictor = load_checkpoint(
            AlphaPredictor(SDAEC_ALPHA_ORDER),
            light_aec_path / 'Model' / 'alpha.ckpt',
        ).float().eval()
        network = load_checkpoint(
            SDAECNet(SDAEC_MAX_FRAMES),
            light_aec_path / 'Model' / 'ICCRN.ckpt',
        ).float().eval().fold_for_export()
        return SDAEC(alpha_predictor, network).eval()

    if LIGHT_AEC_MODEL == 'Deep_Echo':
        frames = centered_stft_frame_count(
            MODEL_AUDIO_LENGTH, DEEP_ECHO_NFFT, DEEP_ECHO_HOP_LENGTH)
        stft = PackedSTFT(
            DEEP_ECHO_NFFT, DEEP_ECHO_HOP_LENGTH, 'hamming').eval()
        istft = PackedISTFT(
            DEEP_ECHO_NFFT,
            DEEP_ECHO_HOP_LENGTH,
            'hamming',
            frames,
            MODEL_AUDIO_LENGTH,
        ).eval()
        network = DeepEchoNet(
            istft,
            static_frames=frames if not DYNAMIC_AXES else 0,
        )
        checkpoint_path = light_aec_path / 'model.ckpt'
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f'Missing Deep Echo checkpoint: {checkpoint_path}')
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        missing, _ = network.load_state_dict(
            extract_state_dict(checkpoint), strict=False)
        trainable_parameters = dict(network.named_parameters())
        missing_parameters = [
            key for key in missing if key in trainable_parameters]
        if missing_parameters:
            raise ValueError(
                'Deep Echo checkpoint is missing trainable parameters: '
                f'{missing_parameters}'
            )
        return DeepEchoAEC(
            stft, network.float().eval().fold_for_export()).eval()

    if LIGHT_AEC_MODEL == 'NKF':
        frames = centered_stft_frame_count(
            MODEL_AUDIO_LENGTH, NKF_NFFT, NKF_HOP_LENGTH)
        stft = PackedSTFT(NKF_NFFT, NKF_HOP_LENGTH, 'hann').eval()
        istft = PackedISTFT(
            NKF_NFFT,
            NKF_HOP_LENGTH,
            'hann',
            frames,
            MODEL_AUDIO_LENGTH,
        ).eval()
        network = NKFInner(stft, istft, frames).eval()
        checkpoint_path = light_aec_path / 'src' / 'nkf_epoch70.pt'
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'Missing NKF checkpoint: {checkpoint_path}')
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location='cpu', weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        network = load_nkf_weights(network, checkpoint)
        return NKFAEC(network.float().eval().cache_export_constants()).eval()

    raise ValueError(f'Unsupported LIGHT_AEC_MODEL: {LIGHT_AEC_MODEL}')


def build_model():
    light_aec = build_light_aec()
    dfsmn = load_checkpoint(
        DFSMNMaskNet(),
        dfsmn_model_path / "pytorch_model.bin",
    ).float().eval()
    shift, scale = load_kaldi_feature_transform(
        dfsmn_model_path / "feature.DEY.mvn.txt")
    dfsmn.fold_feature_transform(shift, scale)
    return DFSMNVAD(
        light_aec,
        KaldiFbank(FBANK_DITHER),
        dfsmn,
    ).eval()


def build_export_metadata(opset=OPSET):
    return {
        "vad_metadata_version": 1,
        "producer": Path(__file__).name,
        "model_name": f"DFSMN_VAD_{LIGHT_AEC_MODEL}",
        "task": "voice_activity_detection",
        "model_family": f"dfsmn_{LIGHT_AEC_MODEL.lower()}",
        "light_aec_model": LIGHT_AEC_MODEL,
        "dynamic_axes": DYNAMIC_AXES,
        "opset": opset,
        "input_audio_dtype": IN_ONNX_DTYPE.removeprefix("tensor(").removesuffix(")"),
        "input_sample_rate": IN_SAMPLE_RATE,
        "model_sample_rate": MODEL_SAMPLE_RATE,
        "input_audio_length": INPUT_AUDIO_LENGTH,
        "output_frame_shift_seconds": FBANK_HOP_LENGTH / MODEL_SAMPLE_RATE,
        "output_frame_shift_samples": FBANK_HOP_LENGTH,
        "fbank_window_length_samples": FBANK_WINDOW_LENGTH,
        "num_audio_inputs": 2,
        "num_outputs": 1,
    }


def main():
    print(
        f"Export start with {LIGHT_AEC_MODEL} backend: {light_aec_path}")
    onnx_model.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = build_model()
    near_end_pcm = torch.randint(
        -12000,
        12000,
        (1, 1, INPUT_AUDIO_LENGTH),
        dtype=torch.int16,
    )
    far_end_pcm = torch.randint(
        -12000,
        12000,
        (1, 1, INPUT_AUDIO_LENGTH),
        dtype=torch.int16,
    )
    if IN_AUDIO_DTYPE == 'INT16':
        near_end_audio = near_end_pcm
        far_end_audio = far_end_pcm
    else:
        near_end_audio = (near_end_pcm.float() / 32768.0).to(IN_TORCH_DTYPE)
        far_end_audio = (far_end_pcm.float() / 32768.0).to(IN_TORCH_DTYPE)

    temporary_model_path = None
    export_model_path = onnx_model
    if LIGHT_AEC_MODEL == 'NKF':
        with tempfile.NamedTemporaryFile(
                prefix='DFSMN_VAD.', suffix='.onnx',
                dir=onnx_model.parent, delete=False) as temporary_model:
            temporary_model_path = Path(temporary_model.name)
        export_model_path = temporary_model_path

    try:
        with torch.inference_mode():
            torch.onnx.export(
                model,
                (near_end_audio, far_end_audio),
                export_model_path,
                input_names=["near_end_audio", "far_end_audio"],
                output_names=["vad_results"],
                dynamic_axes={
                    "near_end_audio": {2: "audio_len"},
                    "far_end_audio": {2: "audio_len"},
                    "vad_results": {0: "signal_len"},
                } if DYNAMIC_AXES else None,
                opset_version=OPSET,
                dynamo=False,
            )
        del model
        gc.collect()

        if temporary_model_path is not None:
            frames = centered_stft_frame_count(
                MODEL_AUDIO_LENGTH, NKF_NFFT, NKF_HOP_LENGTH)
            rewrite_summary = rewrite_nkf_initializer_identities(
                temporary_model_path,
                onnx_model,
                frames=frames,
                model_batch=1,
                n_freq=NKF_NFFT // 2 + 1,
                filter_order=NKF_FILTER_ORDER,
                fc_dim=NKF_FC_DIM,
                rnn_dim=NKF_RNN_DIM,
            )
            print(
                'Targeted NKF rewrite: '
                f"{rewrite_summary['raw_nodes']} -> "
                f"{rewrite_summary['final_nodes']} nodes.")
    finally:
        if temporary_model_path is not None:
            temporary_model_path.unlink(missing_ok=True)

    model_metadata = build_export_metadata()
    metadata_path = stamp_export_metadata(onnx_model, model_metadata, OPSET)
    print(f"Metadata saved to: {metadata_path}")
    print(f"Export done: {onnx_model}")
    inference_script = parent_path / "Inference_DFSMN_VAD_ONNX.py"
    print(f"\nStarting inference demo with {inference_script.name}")
    subprocess.run(
        [sys.executable, str(inference_script)],
        cwd=parent_path,
        check=True,
    )


if __name__ == "__main__":
    main()