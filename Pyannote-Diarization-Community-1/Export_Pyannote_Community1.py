import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
from scipy.linalg import eigh

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Merge_Pyannote_Community1 import (
    build_merged_models,
    remove_split_models,
)
from STFT_Process import create_stft_kernels
from vad_onnx_metadata import (
    community1_metadata_path,
    export_community1_metadata_carrier,
)


download_path = Path.home() / "Downloads" / "speaker-diarization-community-1"
model_path_segmentation = download_path / "segmentation" / "pytorch_model.bin"
export_folder = script_dir / "Pyannote_Community1_ONNX"
onnx_model_segmentation = export_folder / "segmentation.onnx"
onnx_model_embedding = export_folder / "embedding.onnx"
onnx_model_vbx = export_folder / "vbx.onnx"
onnx_model_clustering = export_folder / "clustering.onnx"
onnx_model_reconstruction = export_folder / "reconstruction.onnx"

OPSET = 20
INPUT_AUDIO_LENGTH = 48000      # Static ONNX window length in samples.
IN_AUDIO_DTYPE = "F32"          # ["F16", "F32", "INT16"]; float inputs use normalized PCM in [-1, 1].

_INPUT_AUDIO_DTYPES = {
    "F16": (torch.float16, onnx.TensorProto.FLOAT16, "float16"),
    "F32": (torch.float32, onnx.TensorProto.FLOAT, "float32"),
    "INT16": (torch.int16, onnx.TensorProto.INT16, "int16"),
}
IN_AUDIO_DTYPE = IN_AUDIO_DTYPE.upper()
if IN_AUDIO_DTYPE not in _INPUT_AUDIO_DTYPES:
    raise ValueError(
        f"Unsupported IN_AUDIO_DTYPE={IN_AUDIO_DTYPE!r}; "
        f"expected one of {tuple(_INPUT_AUDIO_DTYPES)}."
    )
IN_TORCH_DTYPE, IN_ONNX_DTYPE, IN_METADATA_DTYPE = _INPUT_AUDIO_DTYPES[
    IN_AUDIO_DTYPE
]

SAMPLE_RATE = 16_000
SEGMENTATION_DURATION = 10.0  # Checkpoint training chunk duration.
SEGMENTATION_WINDOW_DURATION = INPUT_AUDIO_LENGTH / SAMPLE_RATE
SEGMENTATION_STEP = 1.0
SEGMENTATION_NUM_POWERSET_CLASSES = 7
SEGMENTATION_NUM_SPEAKERS = 3
SEGMENTATION_RECEPTIVE_FIELD = {
    "start": 0.0,
    "duration": 0.0619375,
    "step": 0.016875,
}
POWERSET_MAPPING = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
]

FBANK_FRAME_LENGTH = 400
FBANK_FRAME_SHIFT = 160
FBANK_NFFT = 512
FBANK_NUM_MELS = 80
EMBEDDING_DIMENSION = 256
EMBEDDING_MIN_NUM_SAMPLES = 1680
EMBEDDING_POOL_CHANNELS = 2560


def convolution_output_length(
    input_length: int, kernel_size: int, stride: int = 1
) -> int:
    return (input_length - kernel_size) // stride + 1


def segmentation_frame_count(input_length: int) -> int:
    for kernel_size, stride in (
        (251, 10),
        (3, 3),
        (5, 1),
        (3, 3),
        (5, 1),
        (3, 3),
    ):
        input_length = convolution_output_length(
            input_length, kernel_size, stride
        )
    return input_length


if INPUT_AUDIO_LENGTH < EMBEDDING_MIN_NUM_SAMPLES:
    raise ValueError(
        f"INPUT_AUDIO_LENGTH must be at least {EMBEDDING_MIN_NUM_SAMPLES}."
    )
if not 0.0 < SEGMENTATION_STEP <= SEGMENTATION_WINDOW_DURATION:
    raise ValueError(
        "SEGMENTATION_STEP must be positive and no greater than the "
        "configured export window duration."
    )

SEGMENTATION_NUM_SAMPLES = INPUT_AUDIO_LENGTH
SEGMENTATION_NUM_FRAMES = segmentation_frame_count(INPUT_AUDIO_LENGTH)
FBANK_NUM_FRAMES = convolution_output_length(
    INPUT_AUDIO_LENGTH, FBANK_FRAME_LENGTH, FBANK_FRAME_SHIFT
)
EMBEDDING_FEATURE_NUM_FRAMES = (FBANK_NUM_FRAMES + 7) // 8


class CheckpointProblem(Enum):
    BINARY_CLASSIFICATION = 0
    MONO_LABEL_CLASSIFICATION = 1
    MULTI_LABEL_CLASSIFICATION = 2
    REPRESENTATION = 3
    REGRESSION = 4


class CheckpointResolution(Enum):
    FRAME = 1
    CHUNK = 2


@dataclass
class CheckpointSpecifications:
    problem: CheckpointProblem
    resolution: CheckpointResolution
    duration: float
    min_duration: float | None = None
    warm_up: tuple[float, float] | None = (0.0, 0.0)
    classes: list[str] | None = None
    powerset_max_classes: int | None = None
    permutation_invariant: bool = False


CHECKPOINT_SAFE_GLOBALS = [
    (CheckpointProblem, "pyannote.audio.core.task.Problem"),
    (CheckpointResolution, "pyannote.audio.core.task.Resolution"),
    (CheckpointSpecifications, "pyannote.audio.core.task.Specifications"),
]


def load_checkpoint(path: Path) -> dict:
    with torch.serialization.safe_globals(CHECKPOINT_SAFE_GLOBALS):
        return torch.load(path, map_location="cpu", weights_only=True)


def cast_audio_to_float32(waveform: torch.Tensor) -> torch.Tensor:
    if IN_AUDIO_DTYPE == "F32":
        return waveform
    return waveform.float()


def audio_to_float32(waveform: torch.Tensor) -> torch.Tensor:
    waveform = cast_audio_to_float32(waveform)
    if IN_AUDIO_DTYPE == "INT16":
        waveform = waveform * (1.0 / 32768.0)
    return waveform


class WaveformChunkerONNX(torch.nn.Module):
    def __init__(self, num_samples: int, step_samples: int):
        super().__init__()
        self.num_samples = num_samples
        self.step_samples = step_samples
        self.register_buffer(
            "sample_offsets", torch.arange(num_samples, dtype=torch.int32)
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        audio_len = torch._shape_as_tensor(waveform)[2].to(torch.int32)
        remaining = torch.clamp(audio_len - self.num_samples, min=0)
        num_chunks = torch.div(
            remaining + self.step_samples - 1,
            self.step_samples,
            rounding_mode="floor",
        ) + 1
        starts = torch.arange(
            num_chunks, dtype=torch.int32, device=waveform.device
        ) * self.step_samples
        indices = starts.unsqueeze(1) + self.sample_offsets.unsqueeze(0)
        padded = torch.cat(
            [waveform.reshape(-1), waveform.new_zeros(self.num_samples)], dim=0
        )
        return padded.index_select(
            0, indices.reshape(-1)
        ).reshape(-1, 1, self.num_samples)


class Community1SincNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wav_norm1d = torch.nn.InstanceNorm1d(1, affine=True)
        self.conv1d = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(1, 80, 251, stride=10, bias=False),
                torch.nn.Conv1d(80, 60, 5),
                torch.nn.Conv1d(60, 60, 5),
            ]
        )
        self.pool1d = torch.nn.ModuleList(
            [torch.nn.MaxPool1d(3, stride=3) for _ in range(3)]
        )
        self.norm1d = torch.nn.ModuleList(
            [
                torch.nn.InstanceNorm1d(80, affine=True),
                torch.nn.InstanceNorm1d(60, affine=True),
                torch.nn.InstanceNorm1d(60, affine=True),
            ]
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        output = self.wav_norm1d(waveform)
        for index, (convolution, pooling, normalization) in enumerate(
            zip(self.conv1d, self.pool1d, self.norm1d)
        ):
            output = convolution(output)
            if index == 0:
                output = torch.abs(output)
            output = F.leaky_relu(normalization(pooling(output)))
        return output


class Community1PyanNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sincnet = Community1SincNet()
        self.lstm = torch.nn.LSTM(
            60,
            hidden_size=128,
            num_layers=4,
            bidirectional=True,
            dropout=0.5,
            batch_first=True,
        )
        self.linear = torch.nn.ModuleList(
            [torch.nn.Linear(256, 128), torch.nn.Linear(128, 128)]
        )
        self.classifier = torch.nn.Linear(128, SEGMENTATION_NUM_POWERSET_CLASSES)

    def forward_logits(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = audio_to_float32(waveform)
        output = self.sincnet(waveform).transpose(1, 2)
        output, _ = self.lstm(output)
        for linear in self.linear:
            output = F.leaky_relu(linear(output))
        return self.classifier(output)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.forward_logits(waveform), dim=-1)


class FrameAggregatorONNX(torch.nn.Module):
    def __init__(
        self,
        chunk_duration: float,
        chunk_step: float,
        frame_step: float,
        frames_per_chunk: int,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.chunk_duration_in_frames = chunk_duration / frame_step
        self.chunk_step_in_frames = chunk_step / frame_step
        self.frames_per_chunk = frames_per_chunk
        self.num_classes = num_classes

    def forward(
        self, values: torch.Tensor, average: bool = False
    ) -> torch.Tensor:
        shape = torch._shape_as_tensor(values)
        num_chunks = shape[0]
        num_chunks_float = num_chunks.to(torch.float32)
        num_classes = shape[2] if self.num_classes is None else self.num_classes
        chunk_indices = torch.arange(
            num_chunks_float, dtype=torch.float32, device=values.device
        )
        frame_starts = torch.round(
            chunk_indices * self.chunk_step_in_frames
        ).to(torch.int64)
        frame_indices = frame_starts.unsqueeze(1) + torch.arange(
            self.frames_per_chunk,
            dtype=torch.int64,
            device=values.device,
        ).unsqueeze(0)
        num_output_frames = torch.round(
            (num_chunks_float - 1.0)
            * self.chunk_step_in_frames
            + self.chunk_duration_in_frames
        ).to(torch.int32) + 1
        num_output_frames_shape = num_output_frames.to(torch.int64)
        output = values.new_zeros((num_output_frames_shape, num_classes))
        scatter_indices = frame_indices.reshape(-1, 1).expand(-1, num_classes)
        output.scatter_add_(0, scatter_indices, values.reshape(-1, num_classes))
        if average:
            counts = values.new_zeros((num_output_frames_shape, 1))
            counts.scatter_add_(
                0,
                frame_indices.reshape(-1, 1),
                values.new_ones((num_chunks * self.frames_per_chunk, 1)),
            )
            output = output / torch.clamp(counts, min=1e-12)
        return output


class SegmentationPipelineONNX(torch.nn.Module):
    def __init__(self, segmentation: Community1PyanNet):
        super().__init__()
        step_samples = round(SEGMENTATION_STEP * SAMPLE_RATE)
        self.chunker = WaveformChunkerONNX(SEGMENTATION_NUM_SAMPLES, step_samples)
        self.segmentation = segmentation
        self.aggregator = FrameAggregatorONNX(
            SEGMENTATION_WINDOW_DURATION,
            SEGMENTATION_STEP,
            SEGMENTATION_RECEPTIVE_FIELD["step"],
            SEGMENTATION_NUM_FRAMES,
            num_classes=1,
        )
        self.minimum_clean_frames = math.ceil(
            SEGMENTATION_NUM_FRAMES
            * EMBEDDING_MIN_NUM_SAMPLES
            / SEGMENTATION_NUM_SAMPLES
        )
        self.register_buffer(
            "powerset_mapping",
            torch.tensor(POWERSET_MAPPING, dtype=torch.float32),
        )

    def forward(
        self, waveform: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        chunks = self.chunker(waveform)
        powerset_scores = self.segmentation.forward_logits(chunks)
        powerset_classes = torch.argmax(
            powerset_scores, dim=-1
        ).to(torch.int32)
        segmentations = self.powerset_mapping[powerset_classes]

        active_count = torch.sum(segmentations, dim=2, keepdim=True)
        clean_segmentations = segmentations * (active_count < 2).to(torch.float32)
        masks = segmentations.transpose(1, 2)
        clean_masks = clean_segmentations.transpose(1, 2)
        clean_frame_counts = torch.sum(clean_masks, dim=2)
        use_clean = clean_frame_counts.unsqueeze(2) > float(
            self.minimum_clean_frames
        )
        embedding_masks = torch.where(use_clean, clean_masks, masks)
        active = clean_frame_counts >= 0.2 * SEGMENTATION_NUM_FRAMES
        inactive = torch.sum(segmentations, dim=1) == 0

        speaker_counts = torch.round(
            self.aggregator(active_count, average=True)
        ).to(torch.int32)
        return (
            chunks,
            segmentations,
            embedding_masks,
            active,
            inactive,
            speaker_counts,
        )


def build_sinc_filters(state: dict[str, torch.Tensor]) -> torch.Tensor:
    low_hz = state.pop("sincnet.conv1d.0.filterbank.low_hz_")
    band_hz = state.pop("sincnet.conv1d.0.filterbank.band_hz_")
    window = state.pop("sincnet.conv1d.0.filterbank.window_")
    time = state.pop("sincnet.conv1d.0.filterbank.n_")

    low = 50.0 + torch.abs(low_hz)
    high = torch.clamp(low + 50.0 + torch.abs(band_hz), 50.0, SAMPLE_RATE / 2)
    band = (high - low)[:, 0]
    low_angle = torch.matmul(low, time)
    high_angle = torch.matmul(high, time)

    cosine_left = (
        (torch.sin(high_angle) - torch.sin(low_angle)) / (time / 2.0)
    ) * window
    cosine = torch.cat(
        [cosine_left, 2.0 * band[:, None], cosine_left.flip(1)], dim=1
    )

    sine_left = (
        (torch.cos(low_angle) - torch.cos(high_angle)) / (time / 2.0)
    ) * window
    sine = torch.cat(
        [sine_left, torch.zeros_like(band[:, None]), -sine_left.flip(1)], dim=1
    )
    scale = 2.0 * band[:, None]
    return torch.cat([cosine / scale, sine / scale], dim=0).unsqueeze(1)


def load_segmentation_model(checkpoint: Path) -> Community1PyanNet:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    package = load_checkpoint(checkpoint)
    hyperparameters = package.get("hyper_parameters", {})
    expected = {
        "sincnet": {"stride": 10, "sample_rate": SAMPLE_RATE},
        "lstm": {
            "hidden_size": 128,
            "num_layers": 4,
            "bidirectional": True,
            "monolithic": True,
            "dropout": 0.5,
            "batch_first": True,
        },
        "linear": {"hidden_size": 128, "num_layers": 2},
    }
    for key, value in expected.items():
        if hyperparameters.get(key) != value:
            raise ValueError(
                f"Unsupported segmentation {key} configuration: "
                f"{hyperparameters.get(key)!r}."
            )

    state = dict(package["state_dict"])
    state["sincnet.conv1d.0.weight"] = build_sinc_filters(state)
    model = Community1PyanNet()
    model.load_state_dict(state, strict=True)
    return model.eval()


class StatisticsPooling(torch.nn.Module):
    def forward(
        self, features: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        if weights is None:
            mean = features.mean(dim=-1)
            standard_deviation = features.std(dim=-1, correction=1)
            return torch.cat([mean, standard_deviation], dim=-1)

        if weights.ndim != 2:
            raise ValueError("The exported embedding graph expects 2-D weights.")
        weights = F.interpolate(
            weights.unsqueeze(1), size=features.shape[-1], mode="nearest"
        )
        total = weights.sum(dim=2) + 1e-8
        mean = torch.sum(features * weights, dim=2) / total
        squared_difference = torch.square(features - mean.unsqueeze(2))
        squared_weights = torch.square(weights).sum(dim=2)
        variance = torch.sum(squared_difference * weights, dim=2) / (
            total - squared_weights / total
        )
        return torch.cat([mean, torch.sqrt(variance)], dim=1)


class StaticMaskStatisticsPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        indices = torch.div(
            torch.arange(EMBEDDING_FEATURE_NUM_FRAMES, dtype=torch.int32)
            * SEGMENTATION_NUM_FRAMES,
            EMBEDDING_FEATURE_NUM_FRAMES,
            rounding_mode="floor",
        )
        self.register_buffer("mask_indices", indices)

    def forward(
        self, features: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        weights = weights.index_select(1, self.mask_indices).unsqueeze(1)
        total = weights.sum(dim=2) + 1e-8
        mean = torch.sum(features * weights, dim=2) / total
        squared_difference = torch.square(features - mean.unsqueeze(2))
        squared_weights = torch.square(weights).sum(dim=2)
        variance = torch.sum(squared_difference * weights, dim=2) / (
            total - squared_weights / total
        )
        return torch.cat([mean, torch.sqrt(variance)], dim=1)

    def forward_speaker_masks(
        self, features: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        weights = weights.index_select(2, self.mask_indices).unsqueeze(2)
        features = features.unsqueeze(1)
        total = weights.sum(dim=3) + 1e-8
        mean = torch.sum(features * weights, dim=3) / total
        squared_difference = torch.square(features - mean.unsqueeze(3))
        squared_weights = torch.square(weights).sum(dim=3)
        variance = torch.sum(squared_difference * weights, dim=3) / (
            total - squared_weights / total
        )
        return torch.cat([mean, torch.sqrt(variance)], dim=2)


class ResNetBasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = torch.nn.BatchNorm2d(output_channels)
        self.conv2 = torch.nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = torch.nn.BatchNorm2d(output_channels)
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(output_channels),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = F.relu(self.bn1(self.conv1(features)))
        output = self.bn2(self.conv2(output))
        return F.relu(output + self.shortcut(features))


class Community1ResNet34(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dynamic_audio_len = False
        self.input_channels = 32
        self.conv1 = torch.nn.Conv2d(
            1, 32, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 3, stride=1)
        self.layer2 = self._make_layer(64, 4, stride=2)
        self.layer3 = self._make_layer(128, 6, stride=2)
        self.layer4 = self._make_layer(256, 3, stride=2)
        self.pool = StatisticsPooling()
        self.seg_1 = torch.nn.Linear(5120, EMBEDDING_DIMENSION)

    def _make_layer(
        self, output_channels: int, block_count: int, stride: int
    ) -> torch.nn.Sequential:
        strides = [stride] + [1] * (block_count - 1)
        blocks = []
        for block_stride in strides:
            blocks.append(
                ResNetBasicBlock(
                    self.input_channels, output_channels, stride=block_stride
                )
            )
            self.input_channels = output_channels
        return torch.nn.Sequential(*blocks)

    def _encode_nchw(self, output: torch.Tensor) -> torch.Tensor:
        output = F.relu(self.bn1(self.conv1(output)))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        if self.dynamic_audio_len:
            output = output.flatten(1, 2)
        else:
            output = output.reshape(
                -1, EMBEDDING_POOL_CHANNELS, EMBEDDING_FEATURE_NUM_FRAMES
            )
        return output

    def _forward_nchw(
        self, output: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        return self.seg_1(self.pool(self._encode_nchw(output), weights=weights))

    def forward_speaker_masks(
        self, features: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        encoded = self._encode_nchw(features)
        pooled = self.pool.forward_speaker_masks(encoded, weights)
        return self.seg_1(pooled)

    def forward(
        self, features: torch.Tensor, weights: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if weights is None:
            raise ValueError("Speaker weights are required.")
        output = features.permute(0, 2, 1).unsqueeze(1)
        embedding = self._forward_nchw(output, weights)
        return embedding.new_zeros(()), embedding

    def forward_onnx(
        self, features: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        return self._forward_nchw(features, weights)


class Community1EmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = Community1ResNet34()


def load_embedding_model(checkpoint: Path) -> Community1EmbeddingModel:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    package = load_checkpoint(checkpoint)
    model = Community1EmbeddingModel()
    model.load_state_dict(package["state_dict"], strict=True)
    return model.eval()


def prepare_embedding_model_for_export(
    model: Community1EmbeddingModel,
    dynamic_axes: bool,
) -> Community1EmbeddingModel:
    resnet = model.resnet
    resnet.conv1 = torch.nn.utils.fusion.fuse_conv_bn_eval(
        resnet.conv1, resnet.bn1
    )
    resnet.bn1 = torch.nn.Identity()

    for layer in (resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4):
        for block in layer:
            block.conv1 = torch.nn.utils.fusion.fuse_conv_bn_eval(
                block.conv1, block.bn1
            )
            block.bn1 = torch.nn.Identity()
            block.conv2 = torch.nn.utils.fusion.fuse_conv_bn_eval(
                block.conv2, block.bn2
            )
            block.bn2 = torch.nn.Identity()
            if len(block.shortcut) == 2:
                block.shortcut[0] = torch.nn.utils.fusion.fuse_conv_bn_eval(
                    block.shortcut[0], block.shortcut[1]
                )
                block.shortcut[1] = torch.nn.Identity()

    model.resnet.dynamic_audio_len = dynamic_axes
    if not dynamic_axes:
        model.resnet.pool = StaticMaskStatisticsPooling()
    return model


class KaldiFbankONNX(torch.nn.Module):
    """Fixed, ONNX-friendly equivalent of Community-1's Kaldi fbank frontend."""

    def __init__(self):
        super().__init__()

        basis = create_stft_kernels(
            n_fft=FBANK_NFFT,
            win_length=FBANK_FRAME_LENGTH,
            window_type="hamming_sym",
            frame_length=FBANK_FRAME_LENGTH,
            dtype=torch.float64,
        )

        preemphasis = 0.97
        coefficients = basis.clone()
        coefficients[:, :, 0] = (1.0 - preemphasis) * basis[:, :, 0]
        coefficients[:, :, :-1] -= preemphasis * basis[:, :, 1:]
        coefficients -= coefficients.mean(dim=2, keepdim=True)
        pcm_scale = 1.0 if IN_AUDIO_DTYPE == "INT16" else 32768.0
        self.register_buffer("dft_kernel", coefficients.float() * pcm_scale)

        mel_bank, _ = kaldi.get_mel_banks(
            FBANK_NUM_MELS,
            FBANK_NFFT,
            SAMPLE_RATE,
            20.0,
            0.0,
            100.0,
            -500.0,
            1.0,
        )
        mel_bank = F.pad(mel_bank.float(), (0, 1))
        self.register_buffer("mel_kernel", mel_bank.unsqueeze(-1))

    def _forward_channels_first(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrum = F.conv1d(
            waveform, self.dft_kernel, stride=FBANK_FRAME_SHIFT
        ).square()
        real_power, imag_power = torch.split(
            spectrum, FBANK_NFFT // 2 + 1, dim=1
        )
        power = real_power + imag_power
        features = F.conv1d(power, self.mel_kernel)
        features = torch.clamp(features, min=torch.finfo(torch.float32).eps).log()
        return features - features.mean(dim=2, keepdim=True)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self._forward_channels_first(waveform).transpose(1, 2)

    def forward_onnx(self, waveform: torch.Tensor) -> torch.Tensor:
        return self._forward_channels_first(waveform).unsqueeze(1)


class EmbeddingONNX(torch.nn.Module):
    """Waveform and segmentation-mask boundary around the local ResNet34."""

    def __init__(self, embedding_model: Community1EmbeddingModel):
        super().__init__()
        self.frontend = KaldiFbankONNX()
        self.resnet = embedding_model.resnet

    def forward(
        self, waveform: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        features = self.frontend.forward_onnx(cast_audio_to_float32(waveform))
        return self.resnet.forward_onnx(features, weights)


class PLDAProjectionONNX(torch.nn.Module):
    def __init__(self, directory: Path, lda_dimension: int = 128):
        super().__init__()
        transform_path = directory / "xvec_transform.npz"
        plda_path = directory / "plda.npz"
        if not transform_path.is_file() or not plda_path.is_file():
            raise FileNotFoundError(
                f"PLDA files not found under {directory}; expected "
                "xvec_transform.npz and plda.npz."
            )

        transform_data = np.load(transform_path)
        mean1 = transform_data["mean1"].astype(np.float32)
        mean2 = transform_data["mean2"].astype(np.float32)
        lda = transform_data["lda"].astype(np.float32)

        plda_data = np.load(plda_path)
        plda_mean = plda_data["mu"].astype(np.float32)
        plda_transform = plda_data["tr"]
        plda_psi = plda_data["psi"]
        within = np.linalg.inv(plda_transform.T @ plda_transform)
        between = np.linalg.inv(
            (plda_transform.T / plda_psi) @ plda_transform
        )
        covariance, whitening = eigh(between, within)
        self.psi = covariance[::-1][:lda_dimension].astype(np.float32)
        projection = whitening.T[::-1][:lda_dimension].astype(np.float32)

        lda_input_scale = math.sqrt(lda.shape[0])
        integer_scale = int(lda_input_scale)
        if (
            lda_input_scale != integer_scale
            or integer_scale <= 0
            or integer_scale & (integer_scale - 1)
        ):
            raise ValueError(
                "The PLDA LDA input scale must be an exact power of two."
            )
        lda *= np.float32(lda_input_scale)

        self.register_buffer("mean1", torch.from_numpy(mean1))
        self.register_buffer("mean2", torch.from_numpy(mean2))
        self.register_buffer("lda", torch.from_numpy(lda))
        self.register_buffer("plda_mean", torch.from_numpy(plda_mean))
        self.register_buffer("projection", torch.from_numpy(projection))
        self.lda_output_scale = math.sqrt(lda.shape[1])

    @staticmethod
    def normalize(values: torch.Tensor) -> torch.Tensor:
        return values / torch.linalg.vector_norm(values, dim=-1, keepdim=True)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        centered = self.normalize(embeddings - self.mean1)
        xvectors = self.lda_output_scale * self.normalize(
            torch.matmul(centered, self.lda) - self.mean2
        )
        return torch.matmul(xvectors - self.plda_mean, self.projection.T)


class ClusteringPreprocessONNX(torch.nn.Module):
    def forward(
        self, embeddings: torch.Tensor, active: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = torch.all(torch.isfinite(embeddings), dim=2)
        training_indices = torch.nonzero(
            (active & valid).reshape(-1), as_tuple=False
        ).reshape(-1).to(torch.int32)
        train_embeddings = embeddings.reshape(
            -1, EMBEDDING_DIMENSION
        ).index_select(0, training_indices)
        normalized = train_embeddings / torch.linalg.vector_norm(
            train_embeddings, dim=1, keepdim=True
        )
        return train_embeddings, normalized


class VBxClusteringONNX(torch.nn.Module):
    def __init__(
        self,
        phi: np.ndarray,
        fa: float = 0.07,
        fb: float = 0.8,
        max_iterations: int = 20,
    ):
        super().__init__()
        self.register_buffer("phi", torch.from_numpy(phi.astype(np.float32)))
        self.fa_over_fb = fa / fb
        self.fa = fa
        self.fb = fb
        self.max_iterations = max_iterations
        self.gaussian_dimension_term = float(
            np.float32(self.phi.shape[0])
            * np.float32(math.log(2.0 * math.pi))
        )

    def forward(
        self,
        features: torch.Tensor,
        initial_clusters: torch.Tensor,
        initial_cluster_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        responsibilities = torch.softmax(
            (
                initial_clusters.unsqueeze(1)
                == initial_cluster_ids.unsqueeze(0)
            ).to(features.dtype)
            * 7.0,
            dim=1,
        )
        priors = torch.ones_like(
            initial_cluster_ids, dtype=features.dtype
        )
        priors = priors / torch.sum(priors)
        constant = -0.5 * (
            torch.sum(features * features, dim=1, keepdim=True)
            + self.gaussian_dimension_term
        )
        rho = features * torch.sqrt(self.phi)
        previous_elbo = features.new_zeros(())
        stopped = torch.zeros((), dtype=torch.bool, device=features.device)

        for iteration in range(self.max_iterations):
            inverse_precision = 1.0 / (
                1.0
                + self.fa_over_fb
                * torch.sum(responsibilities, dim=0).unsqueeze(1)
                * self.phi
            )
            alpha = (
                self.fa_over_fb
                * inverse_precision
                * torch.matmul(responsibilities.T, rho)
            )
            log_probability = self.fa * (
                torch.matmul(rho, alpha.T)
                - 0.5
                * torch.matmul(inverse_precision + alpha * alpha, self.phi)
                + constant
            )
            log_priors = torch.log(priors + 1e-8)
            marginal = torch.logsumexp(
                log_probability + log_priors.unsqueeze(0), dim=1
            )
            next_responsibilities = torch.exp(
                log_probability
                + log_priors.unsqueeze(0)
                - marginal.unsqueeze(1)
            )
            next_priors = torch.sum(next_responsibilities, dim=0)
            next_priors = next_priors / torch.sum(next_priors)
            elbo = torch.sum(marginal) + self.fb * 0.5 * torch.sum(
                torch.log(inverse_precision)
                - inverse_precision
                - alpha * alpha
                + 1.0
            )
            update = ~stopped
            responsibilities = torch.where(
                update, next_responsibilities, responsibilities
            )
            priors = torch.where(update, next_priors, priors)
            if iteration > 0:
                stopped = stopped | (elbo - previous_elbo < 1e-4)
            previous_elbo = torch.where(update, elbo, previous_elbo)

        active_clusters = torch.nonzero(
            priors > 1e-7, as_tuple=False
        ).reshape(-1).to(torch.int32)
        return responsibilities.index_select(1, active_clusters), priors


class ClusteringFinalizeONNX(torch.nn.Module):
    def forward(
        self,
        embeddings: torch.Tensor,
        train_embeddings: torch.Tensor,
        inactive: torch.Tensor,
        centroid_weights: torch.Tensor,
        constrained: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight_sums = torch.sum(centroid_weights, dim=0).unsqueeze(1)
        centroids = torch.matmul(
            centroid_weights.T, train_embeddings
        ) / weight_sums

        cosine = torch.matmul(embeddings, centroids.T) / (
            torch.linalg.vector_norm(
                embeddings, dim=2, keepdim=True
            )
            * torch.linalg.vector_norm(
                centroids, dim=1, keepdim=True
            ).T
        )
        scores = 1.0 + cosine
        finite_scores = torch.isfinite(scores)
        finite_minimum = torch.amin(
            torch.where(
                finite_scores,
                scores,
                torch.full_like(scores, float("inf")),
            )
        )
        inactive_score = torch.where(
            torch.isfinite(finite_minimum),
            finite_minimum - 1.0,
            finite_minimum.new_tensor(-1e6),
        )
        scores = torch.where(finite_scores, scores, inactive_score)
        scores = torch.where(
            inactive.unsqueeze(2), inactive_score, scores
        )

        unconstrained_assignments = torch.argmax(
            scores, dim=2
        ).to(torch.int32)
        cluster_ids = torch.cumsum(
            torch.ones_like(weight_sums[:, 0], dtype=torch.int32), dim=0
        ) - 1
        num_clusters = cluster_ids[-1] + 1
        options = torch.cat([cluster_ids, num_clusters.unsqueeze(0)])
        first = options.reshape(-1, 1, 1)
        second = options.reshape(1, -1, 1)
        third = options.reshape(1, 1, -1)
        first_assigned = first < num_clusters
        second_assigned = second < num_clusters
        third_assigned = third < num_clusters
        assigned_count = (
            first_assigned.to(torch.int32)
            + second_assigned.to(torch.int32)
            + third_assigned.to(torch.int32)
        )
        target_count = torch.minimum(
            num_clusters, num_clusters.new_tensor(SEGMENTATION_NUM_SPEAKERS)
        )
        unique = (
            (~first_assigned)
            | (~second_assigned)
            | (first != second)
        ) & (
            (~first_assigned) | (~third_assigned) | (first != third)
        ) & (
            (~second_assigned) | (~third_assigned) | (second != third)
        )
        valid = unique & (assigned_count == target_count)

        padded_scores = torch.cat(
            [scores, scores.new_zeros((*scores.shape[:2], 1))], dim=2
        )
        total_scores = (
            padded_scores[:, 0, :].unsqueeze(2).unsqueeze(3)
            + padded_scores[:, 1, :].unsqueeze(1).unsqueeze(3)
            + padded_scores[:, 2, :].unsqueeze(1).unsqueeze(2)
        )
        total_scores = torch.where(
            valid.unsqueeze(0),
            total_scores,
            total_scores.new_full((), torch.finfo(total_scores.dtype).min),
        )
        best = torch.argmax(
            torch.flatten(total_scores, start_dim=1), dim=1
        ).to(torch.int32)
        option_count = num_clusters + 1
        option_count_squared = option_count * option_count
        first_choice = torch.div(
            best, option_count_squared, rounding_mode="floor"
        )
        remainder = best - first_choice * option_count_squared
        second_choice = torch.div(
            remainder, option_count, rounding_mode="floor"
        )
        third_choice = remainder - second_choice * option_count
        constrained_assignments = torch.stack(
            [first_choice, second_choice, third_choice], dim=1
        )
        constrained_assignments = torch.where(
            constrained_assignments == num_clusters,
            constrained_assignments.new_full((), -2),
            constrained_assignments,
        )
        assignments = torch.where(
            constrained.reshape(()),
            constrained_assignments,
            unconstrained_assignments,
        )
        assignments = torch.where(
            inactive, assignments.new_full((), -2), assignments
        )
        return assignments, centroids


class EmbeddingPipelineONNX(torch.nn.Module):
    def __init__(
        self,
        embedding_model: Community1EmbeddingModel,
        plda_directory: Path,
    ):
        super().__init__()
        self.frontend = KaldiFbankONNX()
        self.resnet = embedding_model.resnet
        self.plda = PLDAProjectionONNX(plda_directory)
        self.clustering_preprocess = ClusteringPreprocessONNX()

    def forward(
        self,
        chunks: torch.Tensor,
        masks: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.frontend.forward_onnx(
            cast_audio_to_float32(chunks)
        )
        embeddings = self.resnet.forward_speaker_masks(features, masks)
        train_embeddings, normalized = self.clustering_preprocess(
            embeddings, active
        )
        return (
            embeddings,
            train_embeddings,
            self.plda(train_embeddings),
            normalized,
        )


class ReconstructionPipelineONNX(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator = FrameAggregatorONNX(
            SEGMENTATION_WINDOW_DURATION,
            SEGMENTATION_STEP,
            SEGMENTATION_RECEPTIVE_FIELD["step"],
            SEGMENTATION_NUM_FRAMES,
        )

    def forward(
        self,
        segmentations: torch.Tensor,
        speaker_counts: torch.Tensor,
        assignments: torch.Tensor,
        centroid_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cluster_ids = torch.cumsum(
            torch.ones_like(centroid_weights[0], dtype=torch.int32), dim=0
        ) - 1
        membership = (
            assignments.unsqueeze(2) == cluster_ids.reshape(1, 1, -1)
        ).to(segmentations.dtype)
        clustered = torch.clamp(
            torch.matmul(segmentations, membership), max=1.0
        )
        clustered = torch.cat(
            [
                clustered,
                torch.zeros_like(segmentations),
            ],
            dim=2,
        )
        activations = self.aggregator(clustered)
        speaker_indices = torch.arange(
            torch._shape_as_tensor(activations)[1],
            dtype=activations.dtype,
            device=activations.device,
        )
        tie_break_scores = (
            activations * (speaker_indices[-1] + 2.0) - speaker_indices
        )
        top_speakers = torch.topk(
            tie_break_scores, SEGMENTATION_NUM_SPEAKERS, dim=1
        ).indices
        ranks = torch.arange(
            SEGMENTATION_NUM_SPEAKERS,
            dtype=speaker_counts.dtype,
            device=speaker_counts.device,
        ).reshape(1, -1)
        selected = ranks < speaker_counts
        regular = torch.zeros_like(activations).scatter(
            1, top_speakers, selected.to(activations.dtype)
        )
        exclusive = torch.zeros_like(activations).scatter(
            1, top_speakers[:, :1], (speaker_counts > 0).to(activations.dtype)
        )
        return regular, exclusive


def find_embedding_checkpoint(model_dir: Path) -> Path:
    expected = model_dir / "embedding" / "pytorch_model.bin"
    if expected.is_file():
        return expected

    legacy = model_dir / " embedding" / "pytorch_model.bin"
    if legacy.is_file():
        print(f"Warning: using snapshot directory with leading space: {legacy.parent}")
        return legacy

    for candidate in model_dir.iterdir():
        if candidate.is_dir() and candidate.name.strip() == "embedding":
            checkpoint = candidate / "pytorch_model.bin"
            if checkpoint.is_file():
                return checkpoint
    raise FileNotFoundError(f"Embedding checkpoint not found under {model_dir}")


def stamp_metadata(path: Path, metadata: dict[str, Any]) -> None:
    model = onnx.load(path)
    model.producer_name = "Standalone Community-1 ONNX exporter"
    model.doc_string = (
        "Device-resident stage of pyannote/speaker-diarization-community-1. "
        "Only AHC initialization, optional forced KMeans, and result formatting "
        "run on the host."
    )
    del model.metadata_props[:]
    entry = model.metadata_props.add()
    entry.key = "pyannote.community1"
    entry.value = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    onnx.checker.check_model(model)
    onnx.save(model, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def graph_statistics(path: Path) -> dict[str, Any]:
    model = onnx.load(path, load_external_data=False)
    operators = Counter(node.op_type for node in model.graph.node)
    return {
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "operator_histogram": dict(sorted(operators.items())),
    }


def audit_exported_graph(
    path: Path,
    expected_inputs: dict[str, int],
    expected_outputs: dict[str, int] | None = None,
    forbidden_operators: set[str] | None = None,
) -> None:
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    onnx.checker.check_model(onnx.shape_inference.infer_shapes(model))

    actual_inputs = {
        value.name: value.type.tensor_type.elem_type for value in model.graph.input
    }
    if actual_inputs != expected_inputs:
        raise RuntimeError(
            f"Unexpected inputs for {path.name}: {actual_inputs} != {expected_inputs}."
        )

    actual_outputs = {
        value.name: value.type.tensor_type.elem_type for value in model.graph.output
    }
    for name, expected_dtype in (expected_outputs or {}).items():
        actual_dtype = actual_outputs.get(name)
        if actual_dtype != expected_dtype:
            raise RuntimeError(
                f"Unexpected {name!r} dtype for {path.name}: "
                f"{actual_dtype} != {expected_dtype}."
            )

    custom_nodes = [
        f"{node.domain or 'ai.onnx'}::{node.op_type}"
        for node in model.graph.node
        if node.domain not in ("", "ai.onnx")
    ]
    if custom_nodes:
        raise RuntimeError(f"Unexpected custom ONNX nodes: {custom_nodes}")

    forbidden = forbidden_operators or set()
    present = sorted(
        forbidden.intersection(node.op_type for node in model.graph.node)
    )
    if present:
        raise RuntimeError(f"Eliminated ONNX operators reappeared: {present}")


def export_segmentation(
    checkpoint: Path,
    output_path: Path,
    opset: int,
    waveform: torch.Tensor,
) -> dict[str, Any]:
    print(f"\nLoading segmentation: {checkpoint}")
    wrapper = SegmentationPipelineONNX(
        load_segmentation_model(checkpoint)
    ).eval()
    output_names = [
        "chunks",
        "segmentations",
        "embedding_masks",
        "active_speakers",
        "inactive_speakers",
        "speaker_counts",
    ]
    dynamic_axes = {
        "waveform": {2: "audio_len"},
        "chunks": {0: "num_chunks"},
        "segmentations": {0: "num_chunks"},
        "embedding_masks": {0: "num_chunks"},
        "active_speakers": {0: "num_chunks"},
        "inactive_speakers": {0: "num_chunks"},
        "speaker_counts": {0: "num_diarization_frames"},
    }
    torch.onnx.export(
        wrapper,
        (waveform,),
        output_path,
        input_names=["waveform"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        export_params=True,
        keep_initializers_as_inputs=False,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False,
    )
    audit_exported_graph(
        output_path,
        {"waveform": IN_ONNX_DTYPE},
        expected_outputs={"speaker_counts": onnx.TensorProto.INT32},
    )

    metadata = {
        "component": "segmentation",
        "pipeline_stage": 1,
        "format_version": 3,
        "model_id": "pyannote/speaker-diarization-community-1",
        "dynamic_audio_length": True,
        "output_names": output_names,
        "sample_rate": SAMPLE_RATE,
        "chunk_duration": SEGMENTATION_WINDOW_DURATION,
        "training_chunk_duration": SEGMENTATION_DURATION,
        "chunk_step": SEGMENTATION_STEP,
        "num_samples": SEGMENTATION_NUM_SAMPLES,
        "num_frames": SEGMENTATION_NUM_FRAMES,
        "num_powerset_classes": SEGMENTATION_NUM_POWERSET_CLASSES,
        "num_local_speakers": SEGMENTATION_NUM_SPEAKERS,
        "powerset_mapping": POWERSET_MAPPING,
        "receptive_field": SEGMENTATION_RECEPTIVE_FIELD,
        "input_dtype": IN_METADATA_DTYPE,
        "logical_integer_dtype": "int32",
    }
    metadata["graph"] = graph_statistics(output_path)
    stamp_metadata(output_path, metadata)
    print(f"Exported {output_path.name}")
    return metadata


def export_embedding(
    checkpoint: Path,
    output_path: Path,
    opset: int,
    chunks: torch.Tensor,
    masks: torch.Tensor,
    active: torch.Tensor,
) -> dict[str, Any]:
    print(f"\nLoading embedding: {checkpoint}")
    model = load_embedding_model(checkpoint)
    wrapper = EmbeddingPipelineONNX(
        prepare_embedding_model_for_export(model, dynamic_axes=False),
        download_path / "plda",
    ).eval()

    output_names = [
        "embeddings",
        "train_embeddings",
        "train_features",
        "normalized_train_embeddings",
    ]
    dynamic_axes = {
        "chunks": {0: "num_chunks"},
        "embedding_masks": {0: "num_chunks"},
        "active_speakers": {0: "num_chunks"},
        "embeddings": {0: "num_chunks"},
        "train_embeddings": {0: "num_training_embeddings"},
        "train_features": {0: "num_training_embeddings"},
        "normalized_train_embeddings": {0: "num_training_embeddings"},
    }
    torch.onnx.export(
        wrapper,
        (chunks, masks, active),
        output_path,
        input_names=["chunks", "embedding_masks", "active_speakers"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        export_params=True,
        keep_initializers_as_inputs=False,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False,
    )
    audit_exported_graph(
        output_path,
        {
            "chunks": IN_ONNX_DTYPE,
            "embedding_masks": onnx.TensorProto.FLOAT,
            "active_speakers": onnx.TensorProto.BOOL,
        },
        forbidden_operators={"BatchNormalization", "Resize"},
    )
    exported_model = onnx.load(output_path, load_external_data=False)
    mask_indices = [
        value
        for value in exported_model.graph.initializer
        if value.name.endswith("mask_indices")
    ]
    if (
        len(mask_indices) != 1
        or mask_indices[0].data_type != onnx.TensorProto.INT32
        or list(mask_indices[0].dims) != [EMBEDDING_FEATURE_NUM_FRAMES]
    ):
        raise RuntimeError(
            "Expected one int32"
            f"[{EMBEDDING_FEATURE_NUM_FRAMES}] mask-index initializer."
        )

    metadata = {
        "component": "embedding",
        "pipeline_stage": 2,
        "format_version": 3,
        "model_id": "pyannote/speaker-diarization-community-1",
        "dynamic_chunk_count": True,
        "output_names": output_names,
        "sample_rate": SAMPLE_RATE,
        "training_chunk_duration": SEGMENTATION_DURATION,
        "num_samples": SEGMENTATION_NUM_SAMPLES,
        "mask_num_frames": SEGMENTATION_NUM_FRAMES,
        "fbank_num_frames": FBANK_NUM_FRAMES,
        "fbank_num_mels": FBANK_NUM_MELS,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "min_num_samples": EMBEDDING_MIN_NUM_SAMPLES,
        "input_dtype": IN_METADATA_DTYPE,
        "plda_dimension": int(wrapper.plda.psi.shape[0]),
        "plda_psi": wrapper.plda.psi.tolist(),
    }
    metadata["graph"] = graph_statistics(output_path)
    stamp_metadata(output_path, metadata)
    print(f"Exported {output_path.name}")
    return metadata


def export_vbx(
    output_path: Path,
    opset: int,
    phi: np.ndarray,
    train_features: torch.Tensor,
    initial_clusters: torch.Tensor,
    initial_cluster_ids: torch.Tensor,
) -> dict[str, Any]:
    print("\nExporting device-side VBx clustering")
    wrapper = VBxClusteringONNX(phi).eval()
    output_names = ["centroid_weights", "speaker_priors"]
    dynamic_axes = {
        "train_features": {0: "num_training_embeddings"},
        "initial_clusters": {0: "num_training_embeddings"},
        "initial_cluster_ids": {0: "num_initial_clusters"},
        "centroid_weights": {
            0: "num_training_embeddings",
            1: "num_active_clusters",
        },
        "speaker_priors": {0: "num_initial_clusters"},
    }
    torch.onnx.export(
        wrapper,
        (train_features, initial_clusters, initial_cluster_ids),
        output_path,
        input_names=[
            "train_features",
            "initial_clusters",
            "initial_cluster_ids",
        ],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        export_params=True,
        keep_initializers_as_inputs=False,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False,
    )
    audit_exported_graph(
        output_path,
        {
            "train_features": onnx.TensorProto.FLOAT,
            "initial_clusters": onnx.TensorProto.INT32,
            "initial_cluster_ids": onnx.TensorProto.INT32,
        },
    )
    metadata = {
        "component": "vbx",
        "pipeline_stage": 3,
        "format_version": 3,
        "model_id": "pyannote/speaker-diarization-community-1",
        "dynamic_training_count": True,
        "dynamic_cluster_count": True,
        "output_names": output_names,
        "plda_dimension": int(phi.shape[0]),
        "fa": wrapper.fa,
        "fb": wrapper.fb,
        "max_iterations": wrapper.max_iterations,
        "prior_threshold": 1e-7,
        "logical_integer_dtype": "int32",
    }
    metadata["graph"] = graph_statistics(output_path)
    stamp_metadata(output_path, metadata)
    print(f"Exported {output_path.name}")
    return metadata


def export_clustering(
    output_path: Path,
    opset: int,
    embeddings: torch.Tensor,
    train_embeddings: torch.Tensor,
    inactive: torch.Tensor,
    centroid_weights: torch.Tensor,
    constrained: torch.Tensor,
) -> dict[str, Any]:
    print("\nExporting device-side clustering finalization")
    wrapper = ClusteringFinalizeONNX().eval()
    output_names = ["assignments", "centroids"]
    dynamic_axes = {
        "embeddings": {0: "num_chunks"},
        "train_embeddings": {0: "num_training_embeddings"},
        "inactive_speakers": {0: "num_chunks"},
        "centroid_weights": {
            0: "num_training_embeddings",
            1: "num_clusters",
        },
        "assignments": {0: "num_chunks"},
        "centroids": {0: "num_clusters"},
    }
    torch.onnx.export(
        wrapper,
        (
            embeddings,
            train_embeddings,
            inactive,
            centroid_weights,
            constrained,
        ),
        output_path,
        input_names=[
            "embeddings",
            "train_embeddings",
            "inactive_speakers",
            "centroid_weights",
            "constrained",
        ],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        export_params=True,
        keep_initializers_as_inputs=False,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False,
    )
    audit_exported_graph(
        output_path,
        {
            "embeddings": onnx.TensorProto.FLOAT,
            "train_embeddings": onnx.TensorProto.FLOAT,
            "inactive_speakers": onnx.TensorProto.BOOL,
            "centroid_weights": onnx.TensorProto.FLOAT,
            "constrained": onnx.TensorProto.BOOL,
        },
        expected_outputs={"assignments": onnx.TensorProto.INT32},
    )
    metadata = {
        "component": "clustering",
        "pipeline_stage": 4,
        "format_version": 3,
        "model_id": "pyannote/speaker-diarization-community-1",
        "dynamic_chunk_count": True,
        "dynamic_training_count": True,
        "dynamic_cluster_count": True,
        "output_names": output_names,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "num_local_speakers": SEGMENTATION_NUM_SPEAKERS,
        "logical_integer_dtype": "int32",
    }
    metadata["graph"] = graph_statistics(output_path)
    stamp_metadata(output_path, metadata)
    print(f"Exported {output_path.name}")
    return metadata


def export_reconstruction(
    output_path: Path,
    opset: int,
    segmentations: torch.Tensor,
    speaker_counts: torch.Tensor,
    assignments: torch.Tensor,
    centroid_weights: torch.Tensor,
) -> dict[str, Any]:
    print("\nExporting device-side diarization reconstruction")
    wrapper = ReconstructionPipelineONNX().eval()
    output_names = ["regular_diarization", "exclusive_diarization"]
    dynamic_axes = {
        "segmentations": {0: "num_chunks"},
        "speaker_counts": {0: "num_diarization_frames"},
        "assignments": {0: "num_chunks"},
        "centroid_weights": {
            0: "num_training_embeddings",
            1: "num_clusters",
        },
        "regular_diarization": {
            0: "num_diarization_frames",
            1: "num_output_speakers",
        },
        "exclusive_diarization": {
            0: "num_diarization_frames",
            1: "num_output_speakers",
        },
    }
    torch.onnx.export(
        wrapper,
        (segmentations, speaker_counts, assignments, centroid_weights),
        output_path,
        input_names=[
            "segmentations",
            "speaker_counts",
            "assignments",
            "centroid_weights",
        ],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        export_params=True,
        keep_initializers_as_inputs=False,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False,
    )
    audit_exported_graph(
        output_path,
        {
            "segmentations": onnx.TensorProto.FLOAT,
            "speaker_counts": onnx.TensorProto.INT32,
            "assignments": onnx.TensorProto.INT32,
            "centroid_weights": onnx.TensorProto.FLOAT,
        },
    )
    metadata = {
        "component": "reconstruction",
        "pipeline_stage": 5,
        "format_version": 3,
        "model_id": "pyannote/speaker-diarization-community-1",
        "dynamic_chunk_count": True,
        "dynamic_speaker_count": True,
        "output_names": output_names,
        "chunk_duration": SEGMENTATION_WINDOW_DURATION,
        "chunk_step": SEGMENTATION_STEP,
        "num_frames": SEGMENTATION_NUM_FRAMES,
        "num_local_speakers": SEGMENTATION_NUM_SPEAKERS,
        "receptive_field": SEGMENTATION_RECEPTIVE_FIELD,
        "logical_integer_dtype": "int32",
    }
    metadata["graph"] = graph_statistics(output_path)
    stamp_metadata(output_path, metadata)
    print(f"Exported {output_path.name}")
    return metadata


def main() -> None:
    export_folder.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(9527)
    waveform = torch.ones(
        1,
        1,
        SEGMENTATION_NUM_SAMPLES + round(0.5 * SAMPLE_RATE),
        dtype=IN_TORCH_DTYPE,
    )
    chunks = torch.ones(
        2, 1, SEGMENTATION_NUM_SAMPLES, dtype=IN_TORCH_DTYPE
    )
    masks = (
        torch.rand(2, SEGMENTATION_NUM_SPEAKERS, SEGMENTATION_NUM_FRAMES)
        > 0.35
    ).float()
    active = torch.tensor(
        [[True, False, True], [True, True, False]], dtype=torch.bool
    )
    train_embeddings = torch.randn(7, EMBEDDING_DIMENSION)
    train_features = torch.randn(7, 128)
    initial_clusters = torch.tensor(
        [0, 0, 1, 1, 2, 2, 2], dtype=torch.int32
    )
    initial_cluster_ids = torch.arange(3, dtype=torch.int32)
    clustering_embeddings = torch.randn(
        2, SEGMENTATION_NUM_SPEAKERS, EMBEDDING_DIMENSION
    )
    inactive = torch.tensor(
        [[False, False, True], [False, True, False]], dtype=torch.bool
    )
    centroid_weights = torch.rand(7, 3)
    constrained = torch.tensor([True], dtype=torch.bool)
    segmentations = (
        torch.rand(2, SEGMENTATION_NUM_FRAMES, SEGMENTATION_NUM_SPEAKERS)
        > 0.5
    ).float()
    num_diarization_frames = round(
        (
            SEGMENTATION_WINDOW_DURATION + SEGMENTATION_STEP
        ) / SEGMENTATION_RECEPTIVE_FIELD["step"]
    ) + 1
    speaker_counts = torch.randint(
        0, 3, (num_diarization_frames, 1), dtype=torch.int32
    )
    assignments = torch.tensor([[0, 1, -2], [1, 0, 2]], dtype=torch.int32)

    export_segmentation(
        model_path_segmentation,
        onnx_model_segmentation,
        OPSET,
        waveform,
    )
    export_embedding(
        find_embedding_checkpoint(download_path),
        onnx_model_embedding,
        OPSET,
        chunks,
        masks,
        active,
    )
    plda = PLDAProjectionONNX(download_path / "plda")
    export_vbx(
        onnx_model_vbx,
        OPSET,
        plda.psi,
        train_features,
        initial_clusters,
        initial_cluster_ids,
    )
    export_clustering(
        onnx_model_clustering,
        OPSET,
        clustering_embeddings,
        train_embeddings,
        inactive,
        centroid_weights,
        constrained,
    )
    export_reconstruction(
        onnx_model_reconstruction,
        OPSET,
        segmentations,
        speaker_counts,
        assignments,
        centroid_weights,
    )

    print("\nMerging the five device stages around the host AHC boundary")
    merged_models = build_merged_models(export_folder)
    removed_constituents = remove_split_models(export_folder)

    manifest = {
        "format_version": 4,
        "model_id": "pyannote/speaker-diarization-community-1",
        "opset": OPSET,
        "device_pipeline": True,
        "merged_pipeline": True,
        "onnx_stage_count": 2,
        "host_stages": ["ahc_initialization", "optional_forced_kmeans"],
        "input_dtype": IN_METADATA_DTYPE,
        "logical_integer_dtype": "int32",
        "models": {
            component: {
                "file": details["path"].name,
                "sha256": sha256(details["path"]),
                **details["metadata"],
            }
            for component, details in merged_models.items()
        },
    }
    if removed_constituents:
        manifest["absorbed_constituents"] = removed_constituents
    metadata_path = community1_metadata_path(export_folder)
    export_community1_metadata_carrier(
        metadata_path,
        manifest,
        producer="Export_Pyannote_Community1.py",
    )
    (export_folder / "community1_onnx.json").unlink(missing_ok=True)
    print(f"\nExport complete: {export_folder}")
    print(f"Metadata: {metadata_path}")
    for component, details in merged_models.items():
        print(f"{component.title()}: {details['path']}")
    if removed_constituents:
        print(f"Removed absorbed constituents: {removed_constituents}")

    inference_script = script_dir / "Inference_Pyannote_Community1_ONNX.py"
    print(
        f"\nStarting inference demo with {inference_script.name} using: "
        f"{export_folder.resolve()}"
    )
    subprocess.run(
        [sys.executable, str(inference_script), str(export_folder.resolve())],
        cwd=script_dir,
        check=True,
    )


if __name__ == "__main__":
    main()
