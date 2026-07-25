#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Turn — export the GLUED end-to-end ONNX model (waveform -> P(complete))
=============================================================================

This script builds an ONNX-exportable log-mel **feature extractor** (a single
static-graph ``torch.nn.Module`` that reimplements ``WhisperFeatureExtractor``
following ``STFT_Process.py``: all constants precomputed as buffers, no
torch.stft / numpy / transformers in the graph) and then **glues** it onto each
downloaded Smart Turn classifier with ``onnx.compose`` to produce fully fused,
end-to-end graphs. ``IN_AUDIO_DTYPE`` selects the waveform input contract:

    audio (batch, 1, IN_SAMPLE_RATE * 8) float16/float32/int16  ->  logits (batch, 1) float32

Float inputs contain normalized PCM in [-1, 1]. INT16 inputs contain signed PCM;
their 2^-15 scale is folded into the waveform-normalization epsilon.

The whole pipeline — raw mono PCM at IN_SAMPLE_RATE -> internal 16 kHz audio -> Whisper log-mel features ->
P(complete) — then runs inside a *single* ONNX Runtime session, with no
intermediate tensor leaving the graph and no Python audio pre-processing at
inference time. The glued models are the deliverable; the bare feature extractor
is only an intermediate and is written to a temporary file.

Feature-extraction parameters (sample rate, FFT size, hop, mel bins, sample and
frame counts and the mel filter bank) are read from the ``WhisperFeatureExtractor``
config — the Whisper-Tiny preprocessor config that ``inference.py`` uses — rather
than hard-coded. The context window (``MAX_SECONDS``, 8 s for Smart Turn v3) is
itself read from the classifier's fixed ``input_features`` frame count, so the
feature extractor always emits exactly the number of frames the classifier expects.

Pipeline (matching ``WhisperFeatureExtractor(chunk_length=MAX_SECONDS)``, torch
path, which is what ``inference.py`` uses with ``do_normalize=True``):

     1. selected input -> float32
     2. zero-mean / unit-variance over the window
         (INT16 2^-15 scale folded into epsilon)
    3. STFT  (Conv1d windowed-DFT via STFT_Process 'stft_B', hann, center, reflect)
    4. drop the last STFT frame               (Whisper: stft[..., :-1])
    5. power spectrum  real^2 + imag^2
    6. mel projection  mel_filters.T @ power
    7. clamp(1e-10).log10()
    8. maximum(log, log.max() - 8)            (per sample)
    9. (log + 4) / 4

Configure ``SMART_TURN_DIR`` (the folder with the downloaded Smart Turn v3 ONNX
classifiers), ``CLASSIFIER_MODELS`` (which classifiers to glue), and
``IN_AUDIO_DTYPE`` below, then run:

    python Export_SmartTurn_Glued.py

Each classifier "<name>.onnx" yields a fused "<name>-glued.onnx" next to it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import compose, version_converter  # ONNX model glue + opset alignment
from transformers import WhisperFeatureExtractor

# Reuse the ONNX-exportable STFT from this repo (follows STFT_Process.py).
from STFT_Process import STFT_Process

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "vad_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from vad_onnx_metadata import stamp_export_metadata


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Configuration  (user settings + Whisper feature-extractor config)
# ═════════════════════════════════════════════════════════════════════════════
# --- User settings -----------------------------------------------------------
# Folder containing the downloaded Smart Turn v3 ONNX classifiers
# (from https://huggingface.co/pipecat-ai/smart-turn-v3).
SMART_TURN_DIR = str(Path.home() / "Downloads" / "smart-turn-v3")

IN_AUDIO_DTYPE = "F32"  # ["F16", "F32", "INT16"]; float inputs use normalized PCM in [-1, 1].
IN_SAMPLE_RATE = 16000   # External audio rate; accepts any positive sample rate.
OPSET_VERSION = 20       # ONNX opset version; torch >= 2.12 requires >= 18.

if IN_SAMPLE_RATE <= 0:
    raise ValueError("IN_SAMPLE_RATE must be positive.")
if OPSET_VERSION < 18:
    raise ValueError("OPSET_VERSION must be at least 18 for torch >= 2.12.")

_INPUT_AUDIO_DTYPES = {
    "F16": (torch.float16, "float16"),
    "F32": (torch.float32, "float"),
    "INT16": (torch.int16, "int16"),
}
IN_AUDIO_DTYPE = IN_AUDIO_DTYPE.upper()
if IN_AUDIO_DTYPE not in _INPUT_AUDIO_DTYPES:
    raise ValueError(
        f"Unsupported IN_AUDIO_DTYPE={IN_AUDIO_DTYPE!r}; "
        f"expected one of {tuple(_INPUT_AUDIO_DTYPES)}."
    )
IN_TORCH_DTYPE, IN_METADATA_DTYPE = _INPUT_AUDIO_DTYPES[IN_AUDIO_DTYPE]

# Classifier ONNX files (inside SMART_TURN_DIR) to fuse the feature extractor
# onto. Each one yields a fused end-to-end graph "<name>-glued.onnx" beside it.
CLASSIFIER_MODELS = [
    "smart-turn-v3.2-gpu.onnx",
    "smart-turn-v3.2-cpu.onnx",
]

# Directory in which to save the fused end-to-end "<name>-glued.onnx" models.
# Leave as None (or "") to write each glued model next to its source classifier
# in SMART_TURN_DIR (the original behaviour); the directory is created if needed.
# Anchored to this script's own folder so the output lands under SmartTurn/ no
# matter which working directory the script is launched from.
GLUED_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "SmartTurn_Glued_ONNX"
)  # None or "" = next to source classifier


def _window_seconds_from_classifier(smart_turn_dir: str, classifier_models: list[str]) -> int:
    """
    Read the context-window length (seconds) from the classifier itself, instead of
    hard-coding it. A Smart Turn classifier's ``input_features`` input has a fixed
    mel-frame count (Smart Turn v3: 800). With Whisper's hop length and sample rate,
    that frame count maps to a single window length, so deriving the window here
    guarantees the feature extractor always emits exactly the frames the classifier
    expects — the value can never drift out of sync with the model.
    """
    base = WhisperFeatureExtractor()  # defaults give sr/hop (independent of chunk_length)
    for name in classifier_models:
        path = os.path.join(smart_turn_dir, name)
        if not os.path.isfile(path):
            continue
        graph = onnx.load(path).graph
        inp = next((i for i in graph.input if i.name == "input_features"), graph.input[0])
        frames = inp.type.tensor_type.shape.dim[-1].dim_value  # (batch, n_mels, n_frames)
        if frames <= 0:
            raise ValueError(f"{name}: classifier has a dynamic frame axis; cannot derive the window.")
        seconds = frames * base.hop_length / base.sampling_rate
        if seconds != int(seconds):
            raise ValueError(f"{name}: {frames} frames -> non-integer window ({seconds} s).")
        return int(seconds)
    raise FileNotFoundError(
        f"None of {classifier_models} found in {smart_turn_dir}/ to read the context window from."
    )


# Smart Turn context window in seconds — NOT a free parameter. It is read from the
# classifier's fixed ``input_features`` frame count (Smart Turn v3: 800 frames -> 8 s)
# so the feature extractor always produces exactly what the classifier consumes;
# a mismatch would otherwise fail the glue step with a shape-inference error. Every
# other feature-extraction parameter comes from the WhisperFeatureExtractor config.
MAX_SECONDS = _window_seconds_from_classifier(SMART_TURN_DIR, CLASSIFIER_MODELS)


@dataclass(frozen=True)
class FeatureConfig:
    """
    Feature-extraction parameters read straight from the ``WhisperFeatureExtractor``
    config (the Whisper-Tiny preprocessor config that ``inference.py`` uses), so the
    audio settings are never hard-coded.
    """

    sampling_rate: int          # 16000
    n_fft: int                  # 400
    hop_length: int             # 160
    n_mels: int                 # 80
    n_samples: int              # MAX_SECONDS * sampling_rate (128000, fixed window)
    n_frames: int               # output frames after Whisper's drop-last (800)
    mel_filters: np.ndarray     # (1 + n_fft // 2, n_mels) = (201, 80)

    @property
    def stft_frames(self) -> int:
        # The STFT emits one extra frame that Whisper drops (stft[..., :-1]).
        return self.n_frames + 1

    @classmethod
    def from_whisper(cls, fe: WhisperFeatureExtractor) -> "FeatureConfig":
        return cls(
            sampling_rate=fe.sampling_rate,
            n_fft=fe.n_fft,
            hop_length=fe.hop_length,
            n_mels=fe.feature_size,
            n_samples=fe.n_samples,
            n_frames=fe.nb_max_frames,
            mel_filters=np.asarray(fe.mel_filters, dtype=np.float32),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Export helpers
# ═════════════════════════════════════════════════════════════════════════════
def glued_output_path(classifier_path: str) -> str:
    """
    Fused-graph path for a classifier: "<name>-glued.onnx". Written to
    ``GLUED_OUTPUT_DIR`` when that is set, otherwise next to the source
    classifier (i.e. in ``SMART_TURN_DIR``).
    """
    base, ext = os.path.splitext(classifier_path)
    if GLUED_OUTPUT_DIR:
        return os.path.join(GLUED_OUTPUT_DIR, f"{os.path.basename(base)}-glued{ext}")
    return f"{base}-glued{ext}"


def input_audio_length(cfg: FeatureConfig) -> int:
    return int(round(cfg.n_samples * IN_SAMPLE_RATE / cfg.sampling_rate))


# ═════════════════════════════════════════════════════════════════════════════
# 3.  The feature-extractor module (static graph, ONNX-exportable)
# ═════════════════════════════════════════════════════════════════════════════
class SmartTurnFeatureExtractor(torch.nn.Module):
    """
    Raw waveform (batch, 1, samples) -> Whisper log-mel features (batch, 80, 800).

    All constants (STFT DFT basis + mel filter bank) are precomputed as buffers.
    The forward path is pure, branch-free tensor ops so it traces to a clean,
    static ONNX graph (only the batch dimension is dynamic).
    """

    def __init__(self, cfg: FeatureConfig):
        super().__init__()
        self.model_audio_length = cfg.n_samples
        self.resample_input = IN_SAMPLE_RATE != cfg.sampling_rate

        # Conv1d windowed-DFT STFT (real + imag), identical framing to the
        # torch.stft used by WhisperFeatureExtractor: center pad + reflect, hann
        # window of length n_fft.
        self.stft = STFT_Process(
            model_type="stft_B",
            n_fft=cfg.n_fft,
            win_length=cfg.n_fft,
            hop_len=cfg.hop_length,
            max_frames=cfg.stft_frames,
            window_type="hann",
            center_pad=True,
            pad_mode="reflect",
        )

        # Mel projection matrix, stored transposed (n_mels, n_freq) so it
        # left-multiplies the power spectrum: (n_mels, n_freq) @ (B, n_freq, T).
        mel_fb = torch.from_numpy(np.ascontiguousarray(cfg.mel_filters.T)).float()
        self.register_buffer("mel_fb", mel_fb)
        input_scale = 1.0 / 32768.0 if IN_AUDIO_DTYPE == "INT16" else 1.0
        self.normalization_eps = 1e-7 / (input_scale * input_scale)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # 1. Convert to float32 and normalize over the whole window. For INT16,
        # the 2^-15 input scale is folded into epsilon because normalization is
        # otherwise invariant to a positive constant input scale.
        if IN_AUDIO_DTYPE != "F32":
            audio = audio.float()
        if self.resample_input:
            audio = torch.nn.functional.interpolate(
                audio,
                size=self.model_audio_length,
                mode="linear",
                align_corners=False,
            )
        mean = audio.mean(dim=-1, keepdim=True)
        centered = audio - mean
        var = (centered * centered).mean(dim=-1, keepdim=True)
        audio = centered / torch.sqrt(var + self.normalization_eps)

        # 2. STFT -> (B, 201, 801) real & imaginary parts
        real, imag = self.stft(audio)

        # 3. drop the last frame (Whisper: stft[..., :-1]) -> (B, 201, 800)
        real = real[..., :-1]
        imag = imag[..., :-1]

        # 4. power spectrum
        power = real * real + imag * imag                      # (B, 201, 800)

        # 5. mel projection
        mel = torch.matmul(self.mel_fb, power)                 # (B, 80, 800)

        # 6. log10 with floor (log-mel floor 1e-10)
        log_mel = torch.clamp(mel, min=1e-10).log10()

        # 7. per-sample dynamic-range clamp to [max - 8, max] (8 dB range)
        max_val = log_mel.amax(dim=(-2, -1), keepdim=True)
        log_mel = torch.maximum(log_mel, max_val - 8.0)

        # 8. scale to roughly [-1, 1]
        log_mel = (log_mel + 4.0) * 0.25
        return log_mel                                         # (B, 80, 800)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Export
# ═════════════════════════════════════════════════════════════════════════════
def export_feature_extractor(model: torch.nn.Module, cfg: FeatureConfig, path: str) -> None:
    dummy_pcm = torch.randint(
        -4000,
        4000,
        (1, 1, input_audio_length(cfg)),
        dtype=torch.int16,
    )
    if IN_AUDIO_DTYPE == "INT16":
        dummy = dummy_pcm
    else:
        dummy = (dummy_pcm.float() * (1.0 / 32768.0)).to(IN_TORCH_DTYPE)
    torch.onnx.export(
        model,
        (dummy,),
        path,
        input_names=["audio"],
        # "input_features" bridges the two graphs (FE output -> classifier input);
        # the exact name is unimportant as long as both sides agree.
        output_names=["input_features"],
        dynamic_axes={"audio": {0: "batch"}, "input_features": {0: "batch"}},
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  feature extractor exported -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Glue  (fuse the feature extractor + each classifier into one ONNX graph)
# ═════════════════════════════════════════════════════════════════════════════════════════
def glue_models(feature_extractor_path: str, classifier_path: str, output_path: str) -> onnx.ModelProto:
    """
    Fuse ``feature_extractor_path`` -> ``classifier_path`` into a single graph
    (audio -> logits) with ``onnx.compose.merge_models`` and save it.

    The two source graphs share no internal tensor/initializer/node names, so the
    merge needs no prefixing. Their IR versions are aligned, and a classifier
    whose default-domain opset differs is converted to the feature extractor's
    configured opset before composition.
    """
    fe = onnx.load(feature_extractor_path)
    clf = onnx.load(classifier_path)

    fe_opset = next(
        item.version for item in fe.opset_import if item.domain in ("", "ai.onnx")
    )
    clf_opset = next(
        item.version for item in clf.opset_import if item.domain in ("", "ai.onnx")
    )
    if clf_opset != fe_opset:
        print(f"    converting classifier opset: {clf_opset} -> {fe_opset}")
        clf = version_converter.convert_version(clf, fe_opset)

    # merge_models requires both models to share an IR version; bump to the max.
    ir_version = max(fe.ir_version, clf.ir_version)
    fe.ir_version = ir_version
    clf.ir_version = ir_version

    glued = compose.merge_models(
        fe,
        clf,
        io_map=[("input_features", "input_features")],
        producer_name="smart-turn-glue",
        doc_string="Smart Turn end-to-end: waveform -> Whisper log-mel -> P(complete)",
    )

    # The int8 classifier carries com.microsoft ops whose schemas the standalone
    # ONNX checker does not know, so shape inference and checking are advisory.
    try:
        glued = onnx.shape_inference.infer_shapes(glued)
    except Exception as exc:  # noqa: BLE001 - advisory only
        print(f"    (shape inference skipped: {exc})")
    try:
        onnx.checker.check_model(glued)
        print("    onnx.checker: OK")
    except Exception as exc:  # noqa: BLE001 - custom-domain ops are expected for int8
        print(f"    onnx.checker: advisory ({type(exc).__name__}: custom-domain ops)")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    onnx.save(glued, output_path)

    g_in = [(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim])
            for i in glued.graph.input]
    g_out = [(o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim])
             for o in glued.graph.output]
    print(f"    glued IO: in={g_in}  out={g_out}")
    print(f"    saved -> {output_path}")
    return glued


def glue_classifiers(fe_path: str, cfg: FeatureConfig) -> list[str]:
    """Glue the feature extractor onto each configured classifier."""
    print("\n═══════════════════════════════════════════════════════════════════")
    print("  Glue: feature extractor  +  classifier  ->  end-to-end (audio -> logits)")
    print("  Bridge tensor: input_features  (FE output -> classifier input)")
    print("═══════════════════════════════════════════════════════════════════")

    produced = []
    for name in CLASSIFIER_MODELS:
        classifier_path = os.path.join(SMART_TURN_DIR, name)
        if not os.path.isfile(classifier_path):
            print(f"── Skipping (not found): {classifier_path}")
            continue

        glued_path = glued_output_path(classifier_path)
        print(f"\n══ {name} ══")
        glued_model = glue_models(fe_path, classifier_path, glued_path)
        model_opset = next(
            item.version
            for item in glued_model.opset_import
            if item.domain in ("", "ai.onnx")
        )
        metadata_path = stamp_export_metadata(
            glued_path,
            {
                "smart_turn_metadata_version": 1,
                "producer": Path(__file__).name,
                "model_name": Path(glued_path).stem,
                "task": "turn_completion_detection",
                "model_family": "smart_turn_v3",
                "dynamic_batch": True,
                "opset": model_opset,
                "input_audio_dtype": IN_METADATA_DTYPE,
                "input_sample_rate": IN_SAMPLE_RATE,
                "model_sample_rate": cfg.sampling_rate,
                "input_audio_length": input_audio_length(cfg),
                "context_window_seconds": MAX_SECONDS,
                "normalization_embedded": True,
                "n_fft": cfg.n_fft,
                "hop_length_samples": cfg.hop_length,
                "n_mels": cfg.n_mels,
                "n_frames": cfg.n_frames,
                "num_inputs": 1,
                "num_outputs": 1,
            },
            model_opset,
        )
        print(f"    metadata -> {metadata_path}")
        produced.append(glued_path)
    return produced


# ═════════════════════════════════════════════════════════════════════════════
# 6.  Main
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    # Read every feature-extraction parameter from the WhisperFeatureExtractor config.
    fe = WhisperFeatureExtractor(chunk_length=MAX_SECONDS)
    cfg = FeatureConfig.from_whisper(fe)

    print(
        f"\nConfig  INPUT_SR={IN_SAMPLE_RATE}, MODEL_SR={cfg.sampling_rate}, "
        f"WINDOW={input_audio_length(cfg)} input samples ({MAX_SECONDS}s), "
        f"NFFT={cfg.n_fft}, "
        f"HOP={cfg.hop_length}, MELS={cfg.n_mels}, FRAMES={cfg.n_frames}, "
        f"INPUT_DTYPE={IN_AUDIO_DTYPE}"
    )
    print(f"Target  glued end-to-end graphs (audio -> logits), opset={OPSET_VERSION}\n")

    # The bare feature extractor is only an intermediate, so it goes to a temp file;
    # the glued end-to-end models are the deliverable and are saved in SMART_TURN_DIR.
    with tempfile.TemporaryDirectory() as tmpdir:
        fe_path = os.path.join(tmpdir, "smart-turn-feature-extractor.onnx")

        with torch.no_grad():
            model = SmartTurnFeatureExtractor(cfg).eval()
            export_feature_extractor(model, cfg, fe_path)

        produced = glue_classifiers(fe_path, cfg)

    print("\n── Summary ──")
    print(f"  glued end-to-end : audio {IN_METADATA_DTYPE} "
          f"[batch, 1, {input_audio_length(cfg)}]  ->  "
          f"logits f32 [batch, 1]")
    if produced:
        print(f"  glued models     : {', '.join(produced)}")
    else:
        print(f"  glued models     : (none — no classifiers found in {SMART_TURN_DIR}/)")
    print(f"  opset            : {OPSET_VERSION}   dynamic_axes: batch")

    inference_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Inference_SmartTurn_Glued ONNX.py",
    )
    exported_folder = os.path.abspath(GLUED_OUTPUT_DIR or SMART_TURN_DIR)
    print(f"\nStarting inference demo with {os.path.basename(inference_script)}")
    subprocess.run(
        [sys.executable, inference_script, exported_folder],
        cwd=os.path.dirname(inference_script),
        check=True,
    )


if __name__ == "__main__":
    main()
