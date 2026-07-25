"""Export the standalone FunASR FSMN-VAD model to ONNX.

The ONNX graph traces the official tensor pipeline: Kaldi-compatible fbank,
LFR, CMVN, the 248-class FSMN encoder, waveform decibels, and FSMN caches.
"""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "vad_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from vad_onnx_metadata import stamp_export_metadata


MODEL_DIR = Path.home() / "Downloads" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
ONNX_PATH = (
    Path(__file__).resolve().parent
    / "FSMN_VAD_ONNX"
    / "FSMN.onnx"
)
INPUT_AUDIO_LENGTH = 32000  # External input length in IN_SAMPLE_RATE samples.
DYNAMIC_AXES = False

MODEL_SAMPLE_RATE = 16000   # Native model rate; do not change.
IN_SAMPLE_RATE = 16000      # External audio rate; accepts any positive sample rate.
FRAME_LENGTH = 400
FRAME_SHIFT = 160
N_FFT = 512
OPSET_VERSION = 20
IN_AUDIO_DTYPE = "INT16"    # ["F16", "F32", "INT16"]; float inputs use normalized PCM in [-1, 1].

if IN_SAMPLE_RATE <= 0:
    raise ValueError("IN_SAMPLE_RATE must be positive")
MODEL_AUDIO_LENGTH = int(round(
    INPUT_AUDIO_LENGTH * MODEL_SAMPLE_RATE / IN_SAMPLE_RATE))
INPUT_TO_MODEL_SCALE = float(MODEL_SAMPLE_RATE / IN_SAMPLE_RATE)

_INPUT_AUDIO_DTYPES = {
    "F16": torch.float16,
    "F32": torch.float32,
    "INT16": torch.int16,
}
IN_AUDIO_DTYPE = IN_AUDIO_DTYPE.upper()
if IN_AUDIO_DTYPE not in _INPUT_AUDIO_DTYPES:
    raise ValueError(
        f"Unsupported IN_AUDIO_DTYPE={IN_AUDIO_DTYPE!r}; "
        f"expected one of {tuple(_INPUT_AUDIO_DTYPES)}"
    )
IN_TORCH_DTYPE = _INPUT_AUDIO_DTYPES[IN_AUDIO_DTYPE]


class LinearTransform(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=1):
        super().__init__()
        self.linear = nn.Conv2d(input_dim, output_dim, (kernel_size, 1), bias=False)

    def forward(self, inputs):
        return self.linear(inputs)


class AffineTransform(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=1):
        super().__init__()
        self.linear = nn.Conv2d(input_dim, output_dim, (kernel_size, 1))

    def forward(self, inputs):
        return self.linear(inputs)


class FSMNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, lorder, rorder, lstride, rstride):
        super().__init__()
        if rorder != 0:
            raise ValueError("The supported FSMN-VAD checkpoint must be causal (rorder=0)")
        self.output_dim = output_dim
        self.cache_frames = (lorder - 1) * lstride
        self.conv_left = nn.Conv2d(
            input_dim,
            output_dim,
            (lorder, 1),
            dilation=(lstride, 1),
            groups=input_dim,
            bias=False,
        )

    def forward(self, inputs, in_cache):
        left_context = torch.cat((in_cache, inputs), dim=2)
        out_cache = left_context[:, :, -self.cache_frames:, :].reshape(
            1,
            self.output_dim,
            self.cache_frames,
            1,
        )
        outputs = self.conv_left(left_context)
        return outputs, out_cache


class BasicBlock(nn.Module):
    def __init__(self, linear_dim, proj_dim, lorder, rorder, lstride, rstride):
        super().__init__()
        self.linear = LinearTransform(linear_dim, proj_dim)
        self.fsmn_block = FSMNBlock(proj_dim, proj_dim, lorder, rorder, lstride, rstride)
        self.affine = AffineTransform(proj_dim, linear_dim)
        self.relu = nn.ReLU()

    def forward(self, inputs, in_cache):
        outputs, out_cache = self.fsmn_block(self.linear(inputs), in_cache)
        return self.relu(self.affine(outputs)), out_cache


class FSMNEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        input_affine_dim,
        fsmn_layers,
        linear_dim,
        proj_dim,
        lorder,
        rorder,
        lstride,
        rstride,
        output_affine_dim,
        output_dim,
        use_softmax=True,
        lfr_m=5,
    ):
        super().__init__()
        if fsmn_layers != 4:
            raise ValueError(f"The checkpoint must contain four FSMN layers, got {fsmn_layers}")
        if lfr_m != 5 or input_dim != 400 or output_affine_dim != 140:
            raise ValueError("The optimized checkpoint migration requires input_dim=400, lfr_m=5, output_affine_dim=140")
        self.in_linear1 = AffineTransform(input_dim // lfr_m, input_affine_dim, lfr_m)
        self.in_linear2 = AffineTransform(input_affine_dim, linear_dim)
        self.relu = nn.ReLU()
        self.fsmn = nn.ModuleList(
            BasicBlock(linear_dim, proj_dim, lorder, rorder, lstride, rstride)
            for _ in range(fsmn_layers)
        )
        self.out_linear = AffineTransform(linear_dim, output_dim)
        self.softmax = nn.Softmax(dim=1) if use_softmax else nn.Identity()

    def forward(self, inputs, cache_0, cache_1, cache_2, cache_3):
        left = inputs[:, :, :1]
        right = inputs[:, :, -1:]
        inputs = torch.cat((left, left, inputs, right, right), dim=2)
        outputs = self.relu(self.in_linear2(self.in_linear1(inputs)))
        outputs, cache_0 = self.fsmn[0](outputs, cache_0)
        outputs, cache_1 = self.fsmn[1](outputs, cache_1)
        outputs, cache_2 = self.fsmn[2](outputs, cache_2)
        outputs, cache_3 = self.fsmn[3](outputs, cache_3)
        scores = self.softmax(self.out_linear(outputs)).squeeze(-1).transpose(1, 2)
        return scores, cache_0, cache_1, cache_2, cache_3


def create_kaldi_mel_filter(num_mels, n_fft, sample_rate, low_freq=20.0, high_freq=0.0):
    nyquist = 0.5 * sample_rate
    high_freq = high_freq + nyquist if high_freq <= 0.0 else high_freq
    mel_low = 1127.0 * math.log(1.0 + low_freq / 700.0)
    mel_high = 1127.0 * math.log(1.0 + high_freq / 700.0)
    mel_delta = (mel_high - mel_low) / (num_mels + 1)

    bin_index = torch.arange(num_mels, dtype=torch.float32).unsqueeze(1)
    left_mel = mel_low + bin_index * mel_delta
    center_mel = mel_low + (bin_index + 1.0) * mel_delta
    right_mel = mel_low + (bin_index + 2.0) * mel_delta

    fft_bin_width = sample_rate / n_fft
    frequencies = fft_bin_width * torch.arange(n_fft // 2, dtype=torch.float32)
    mel_frequencies = 1127.0 * torch.log1p(frequencies / 700.0).unsqueeze(0)
    up_slope = (mel_frequencies - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel_frequencies) / (right_mel - center_mel)
    filters = torch.clamp(torch.minimum(up_slope, down_slope), min=0.0)
    return F.pad(filters, (0, 1))


def create_kaldi_stft_kernel(frame_length, n_fft, preemphasis):
    dtype = torch.float64
    identity = torch.eye(frame_length, dtype=dtype)
    remove_dc = identity - torch.ones((frame_length, frame_length), dtype=dtype) / frame_length

    preemphasis_matrix = torch.zeros_like(identity)
    preemphasis_matrix[0, 0] = 1.0 - preemphasis
    indices = torch.arange(1, frame_length)
    preemphasis_matrix[indices, indices] = 1.0
    preemphasis_matrix[indices, indices - 1] = -preemphasis

    window = torch.hamming_window(
        frame_length,
        periodic=False,
        alpha=0.54,
        beta=0.46,
        dtype=dtype,
    )
    frequencies = torch.arange(n_fft // 2 + 1, dtype=dtype).unsqueeze(1)
    samples = torch.arange(frame_length, dtype=dtype).unsqueeze(0)
    phase = 2.0 * torch.pi * frequencies * samples / n_fft
    dft_basis = torch.cat((torch.cos(phase), -torch.sin(phase)), dim=0)
    kernel = dft_basis @ torch.diag(window) @ preemphasis_matrix @ remove_dc
    return kernel.float().unsqueeze(1)


class KaldiFbankFrontend(nn.Module):
    def __init__(self, frontend_config):
        super().__init__()
        expected = {
            "fs": MODEL_SAMPLE_RATE,
            "window": "hamming",
            "n_mels": 80,
            "frame_length": 25,
            "frame_shift": 10,
            "dither": 0.0,
            "lfr_m": 5,
            "lfr_n": 1,
        }
        for name, value in expected.items():
            actual = frontend_config.get(name, value)
            if actual != value:
                raise ValueError(f"Unsupported frontend setting {name}={actual!r}; expected {value!r}")

        self.num_mels = expected["n_mels"]
        self.input_scale = 1.0 / 32768.0 if IN_AUDIO_DTYPE == "INT16" else 1.0
        self.register_buffer(
            "stft_kernel",
            create_kaldi_stft_kernel(FRAME_LENGTH, N_FFT, preemphasis=0.97)
            * (32768.0 * self.input_scale),
        )
        self.register_buffer(
            "mel_filter",
            create_kaldi_mel_filter(
                self.num_mels, N_FFT, MODEL_SAMPLE_RATE).unsqueeze(-1),
        )
        self.register_buffer(
            "energy_kernel",
            torch.full((1, 1, FRAME_LENGTH), self.input_scale**2),
        )
        self.register_buffer("energy_bias", torch.tensor([0.000001]))
        self.decibel_scale = 10.0 / math.log(10.0)

    def normalize_audio(self, audio):
        return audio.to(torch.float32) * self.input_scale

    def forward(self, audio):
        waveform = audio.to(torch.float32).unsqueeze(1)
        spectrum = F.conv1d(waveform, self.stft_kernel, stride=FRAME_SHIFT)
        real_power, imaginary_power = torch.split(spectrum.square(), N_FFT // 2 + 1, dim=1)
        power = real_power + imaginary_power
        fbank = F.conv1d(power, self.mel_filter)
        fbank = fbank.clamp_min(torch.finfo(torch.float32).eps).log()

        frame_energy = F.conv1d(
            waveform.square(),
            self.energy_kernel,
            self.energy_bias,
            stride=FRAME_SHIFT,
        )
        decibels = torch.log(frame_energy.squeeze(1)) * self.decibel_scale
        return fbank, decibels


class FSMNVAD(nn.Module):
    def __init__(self, frontend, encoder, model_audio_length,
                 dynamic_axes, input_to_model_scale):
        super().__init__()
        self.frontend = frontend
        self.encoder = encoder
        self.model_audio_length = model_audio_length
        self.dynamic_axes = dynamic_axes
        self.input_to_model_scale = input_to_model_scale

    def forward(self, audio, cache_0, cache_1, cache_2, cache_3):
        if IN_SAMPLE_RATE != MODEL_SAMPLE_RATE:
            audio = F.interpolate(
                audio.to(torch.float32).unsqueeze(1),
                size=self.model_audio_length if not self.dynamic_axes else None,
                scale_factor=(
                    None if not self.dynamic_axes
                    else self.input_to_model_scale
                ),
                mode="linear",
                align_corners=False,
            ).squeeze(1)
        features, decibels = self.frontend(audio)
        scores, cache_0, cache_1, cache_2, cache_3 = self.encoder(
            features.unsqueeze(-1),
            cache_0,
            cache_1,
            cache_2,
            cache_3,
        )
        return scores, decibels, cache_0, cache_1, cache_2, cache_3


def load_cmvn(cmvn_path):
    means = None
    scales = None
    with open(cmvn_path, "r", encoding="utf-8") as cmvn_file:
        lines = cmvn_file.readlines()
    for index, line in enumerate(lines):
        fields = line.split()
        if fields and fields[0] == "<AddShift>":
            means = [float(value) for value in lines[index + 1].split()[3:-1]]
        elif fields and fields[0] == "<Rescale>":
            scales = [float(value) for value in lines[index + 1].split()[3:-1]]
    if means is None or scales is None:
        raise ValueError(f"Invalid Kaldi CMVN file: {cmvn_path}")
    return torch.tensor(means, dtype=torch.float32), torch.tensor(scales, dtype=torch.float32)


def load_config(model_dir):
    config_path = Path(model_dir) / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def migrate_encoder_state(encoder_state, means, scales, lfr_m):
    state = dict(encoder_state)

    def pop_checked(name, shape):
        tensor = state.pop(name)
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"Unexpected checkpoint shape for {name}: {tuple(tensor.shape)} != {shape}")
        return tensor

    first_weight = pop_checked("in_linear1.linear.weight", (140, 400))
    first_bias = pop_checked("in_linear1.linear.bias", (140,))
    if means.shape != scales.shape or means.numel() != first_weight.shape[1]:
        raise ValueError("CMVN dimensions do not match the encoder input")
    scaled_weight = first_weight * scales.unsqueeze(0)
    state["in_linear1.linear.weight"] = (
        scaled_weight.reshape(first_weight.shape[0], lfr_m, -1)
        .transpose(1, 2)
        .unsqueeze(-1)
        .contiguous()
    )
    state["in_linear1.linear.bias"] = first_bias + torch.mv(first_weight, means * scales)

    state["in_linear2.linear.weight"] = state["in_linear2.linear.weight"].reshape(250, 140, 1, 1)
    for layer in range(4):
        for suffix in ("linear.linear.weight", "affine.linear.weight"):
            name = f"fsmn.{layer}.{suffix}"
            state[name] = state[name].unsqueeze(-1).unsqueeze(-1)
        name = f"fsmn.{layer}.fsmn_block.conv_left.weight"
        weight = state[name].clone()
        weight[:, 0, -1, 0].add_(1.0)
        state[name] = weight

    first_output_weight = pop_checked("out_linear1.linear.weight", (140, 250))
    first_output_bias = pop_checked("out_linear1.linear.bias", (140,))
    second_output_weight = pop_checked("out_linear2.linear.weight", (248, 140))
    second_output_bias = pop_checked("out_linear2.linear.bias", (248,))
    state["out_linear.linear.weight"] = torch.mm(
        second_output_weight,
        first_output_weight,
    ).reshape(248, 250, 1, 1)
    state["out_linear.linear.bias"] = second_output_bias + torch.mv(
        second_output_weight,
        first_output_bias,
    )
    return state


def build_model(model_dir, model_audio_length=MODEL_AUDIO_LENGTH,
                dynamic_axes=DYNAMIC_AXES,
                input_to_model_scale=INPUT_TO_MODEL_SCALE):
    model_dir = Path(model_dir)
    config = load_config(model_dir)
    lfr_m = config["frontend_conf"]["lfr_m"]
    encoder = FSMNEncoder(**config["encoder_conf"], lfr_m=lfr_m)
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    encoder_state = {
        name.removeprefix("encoder."): value
        for name, value in checkpoint.items()
        if name.startswith("encoder.")
    }
    means, scales = load_cmvn(model_dir / "am.mvn")
    encoder.load_state_dict(
        migrate_encoder_state(encoder_state, means, scales, lfr_m),
        strict=True,
    )
    frontend = KaldiFbankFrontend(config["frontend_conf"])
    return FSMNVAD(
        frontend,
        encoder,
        model_audio_length,
        dynamic_axes,
        input_to_model_scale,
    ).eval(), config


def make_caches(config, dtype=torch.float32):
    encoder_config = config["encoder_conf"]
    shape = (
        1,
        encoder_config["proj_dim"],
        (encoder_config["lorder"] - 1) * encoder_config["lstride"],
        1,
    )
    return [torch.zeros(shape, dtype=dtype) for _ in range(encoder_config["fsmn_layers"])]


def build_export_metadata(config, input_audio_length, dynamic_axes):
    model_config = config["model_conf"]
    encoder_config = config["encoder_conf"]
    frame_ms = int(model_config.get("frame_in_ms", 10))
    return {
        "vad_metadata_version": 1,
        "producer": Path(__file__).name,
        "model_name": "FunASR_FSMN_VAD",
        "task": "voice_activity_detection",
        "model_family": "fsmn_vad",
        "dynamic_axes": dynamic_axes,
        "opset": OPSET_VERSION,
        "input_audio_dtype": {
            "F16": "float16",
            "F32": "float",
            "INT16": "int16",
        }[IN_AUDIO_DTYPE],
        "input_sample_rate": IN_SAMPLE_RATE,
        "model_sample_rate": MODEL_SAMPLE_RATE,
        "input_audio_length": input_audio_length,
        "output_frame_shift_seconds": FRAME_SHIFT / MODEL_SAMPLE_RATE,
        "output_frame_shift_samples": FRAME_SHIFT,
        "frame_length_samples": FRAME_LENGTH,
        "num_audio_inputs": 1,
        "num_inputs": 1 + int(encoder_config["fsmn_layers"]),
        "num_outputs": 2 + int(encoder_config["fsmn_layers"]),
        "cache_count": encoder_config["fsmn_layers"],
        "cache_channels": encoder_config["proj_dim"],
        "cache_frames": (encoder_config["lorder"] - 1) * encoder_config["lstride"],
        "score_classes": encoder_config["output_dim"],
        "sil_pdf_ids": model_config.get("sil_pdf_ids", [0]),
        "frame_in_ms": frame_ms,
    }


def export_onnx(model_dir, onnx_path, input_audio_length, dynamic_axes):
    input_to_model_scale = float(MODEL_SAMPLE_RATE / IN_SAMPLE_RATE)
    model_audio_length = int(round(
        input_audio_length * MODEL_SAMPLE_RATE / IN_SAMPLE_RATE))
    if model_audio_length < FRAME_LENGTH:
        raise ValueError(
            "The configured input duration must provide at least "
            f"{FRAME_LENGTH} samples at {MODEL_SAMPLE_RATE} Hz"
        )
    model, config = build_model(
        model_dir,
        model_audio_length=model_audio_length,
        dynamic_axes=dynamic_axes,
        input_to_model_scale=input_to_model_scale,
    )
    generator = torch.Generator().manual_seed(0)
    normalized_audio = torch.rand((1, input_audio_length), generator=generator) * 2.0 - 1.0
    if IN_AUDIO_DTYPE == "INT16":
        audio = (normalized_audio * 32768.0).clamp(-32768, 32767).to(IN_TORCH_DTYPE)
    else:
        audio = normalized_audio.to(IN_TORCH_DTYPE)
    caches = make_caches(config)

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    axes = None
    if dynamic_axes:
        axes = {
            "audio": {1: "samples"},
            "scores": {1: "frames"},
            "decibels": {1: "frames"},
        }

    print(f"Exporting {onnx_path} ...")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (audio, *caches),
            str(onnx_path),
            input_names=["audio", "in_cache_0", "in_cache_1", "in_cache_2", "in_cache_3"],
            output_names=["scores", "decibels", "out_cache_0", "out_cache_1", "out_cache_2", "out_cache_3"],
            dynamic_axes=axes,
            opset_version=OPSET_VERSION,
            dynamo=False,
        )

    metadata_path = stamp_export_metadata(
        onnx_path,
        build_export_metadata(config, input_audio_length, dynamic_axes),
        OPSET_VERSION,
    )
    print(f"Export complete. Metadata saved to: {metadata_path}")


def main():
    export_onnx(
        MODEL_DIR,
        ONNX_PATH,
        INPUT_AUDIO_LENGTH,
        dynamic_axes=DYNAMIC_AXES,
    )
    inference_script = Path(__file__).resolve().with_name("Inference_FSMN_VAD_ONNX.py")
    print(f"\nStarting inference demo with {inference_script.name}")
    subprocess.run(
        [sys.executable, str(inference_script)],
        cwd=inference_script.parent,
        check=True,
    )


if __name__ == "__main__":
    main()
