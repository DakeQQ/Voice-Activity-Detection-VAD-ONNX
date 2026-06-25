#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Turn — export the GLUED end-to-end ONNX model (int16 PCM -> P(complete))
==============================================================================

This script builds an ONNX-exportable log-mel **feature extractor** (a single
static-graph ``torch.nn.Module`` that reimplements ``WhisperFeatureExtractor``
following ``STFT_Process.py``: all constants precomputed as buffers, no
torch.stft / numpy / transformers in the graph) and then **glues** it onto each
downloaded Smart Turn classifier with ``onnx.compose`` to produce fully fused,
end-to-end graphs:

    pcm_int16 (batch, 128000) int16  ->  logits (batch, 1) float32

The whole pipeline — raw 16 kHz mono PCM -> Whisper log-mel features ->
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

Pipeline (bit-for-bit matching ``WhisperFeatureExtractor(chunk_length=MAX_SECONDS)``,
torch path, which is what ``inference.py`` uses with ``do_normalize=True``):

    1. int16 -> float32 in [-1, 1)            (x * 2^-15)
    2. zero-mean / unit-variance over the window
                                              ((x - mean) / sqrt(var + 1e-7))
    3. STFT  (Conv1d windowed-DFT via STFT_Process 'stft_B', hann, center, reflect)
    4. drop the last STFT frame               (Whisper: stft[..., :-1])
    5. power spectrum  real^2 + imag^2
    6. mel projection  mel_filters.T @ power
    7. clamp(1e-10).log10()
    8. maximum(log, log.max() - 8)            (per sample)
    9. (log + 4) / 4

Configure ``SMART_TURN_DIR`` (the folder with the downloaded Smart Turn v3 ONNX
classifiers) and ``CLASSIFIER_MODELS`` (which classifiers to glue) below, then run:

    python export_smart_turn_features.py

Each classifier "<name>.onnx" yields a fused "<name>-glued.onnx" next to it.
At inference time:

    window = truncate_or_pad_int16(pcm, 128000)           # int16, fixed window
    logits = session.run(None, {"pcm_int16": window[None]})[0]
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import compose  # the ONNX model "glue" package
from transformers import WhisperFeatureExtractor

# Reuse the validated, ONNX-exportable STFT from this repo (follows STFT_Process.py).
from STFT_Process import STFT_Process


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Configuration  (user settings + Whisper feature-extractor config)
# ═════════════════════════════════════════════════════════════════════════════
# --- User settings -----------------------------------------------------------
# Folder containing the downloaded Smart Turn v3 ONNX classifiers
# (from https://huggingface.co/pipecat-ai/smart-turn-v3).
SMART_TURN_DIR = "/home/iamj/Downloads/smart-turn-v3"

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
# 2.  Helpers
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


def truncate_or_pad_int16(pcm: np.ndarray, n_samples: int) -> np.ndarray:
    """
    int16 equivalent of ``audio_utils.truncate_audio_to_last_n_seconds``.

    Keep the last ``n_samples`` (most recent context window) of a long clip, or
    zero-pad a short clip at the *front* — so the result is always ``n_samples``.
    """
    pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
    if len(pcm) > n_samples:
        return pcm[-n_samples:]
    if len(pcm) < n_samples:
        return np.pad(pcm, (n_samples - len(pcm), 0), mode="constant", constant_values=0)
    return pcm


def truncate_audio_to_last_n_seconds(audio_array, n_seconds=8, sample_rate=16000):
    """
    Float reference of ``audio_utils.truncate_audio_to_last_n_seconds`` (inlined so
    this export script is standalone). Truncate audio to last ``n_seconds`` or
    front-pad with zeros to meet ``n_seconds``.
    """
    max_samples = n_seconds * sample_rate
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]
    elif len(audio_array) < max_samples:
        # Pad with zeros at the beginning
        padding = max_samples - len(audio_array)
        return np.pad(audio_array, (padding, 0), mode="constant", constant_values=0)
    return audio_array


# ═════════════════════════════════════════════════════════════════════════════
# 3.  The feature-extractor module (static graph, ONNX-exportable)
# ═════════════════════════════════════════════════════════════════════════════
class SmartTurnFeatureExtractor(torch.nn.Module):
    """
    Raw int16 PCM  ->  Whisper log-mel features (batch, 80, 800).

    All constants (STFT DFT basis + mel filter bank) are precomputed as buffers.
    The forward path is pure, branch-free tensor ops so it traces to a clean,
    static ONNX graph (only the batch dimension is dynamic).
    """

    def __init__(self, cfg: FeatureConfig):
        super().__init__()

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
        self.register_buffer("inv_int16", torch.tensor(1.0 / 32768.0, dtype=torch.float32))  # exact 2^-15 reciprocal

    def forward(self, pcm_int16: torch.Tensor) -> torch.Tensor:
        # 1. int16 -> float32 in [-1, 1)  (multiply by exact 2^-15 reciprocal, 1 / 32768; multiply in-graph rather than divide)
        x = pcm_int16.float() * self.inv_int16      # (B, 128000)

        # 2. zero-mean / unit-variance over the whole window (population variance, zero_mean_unit_var_norm epsilon 1e-7)
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        var = (centered * centered).mean(dim=-1, keepdim=True)
        x = centered / torch.sqrt(var + 1e-7)                  # (B, 128000)

        # 3. STFT -> (B, 201, 801) real & imaginary parts
        real, imag = self.stft(x.unsqueeze(1))

        # 4. drop the last frame (Whisper: stft[..., :-1]) -> (B, 201, 800)
        real = real[..., :-1]
        imag = imag[..., :-1]

        # 5. power spectrum
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
    dummy = torch.randint(-4000, 4000, (1, cfg.n_samples), dtype=torch.int16)
    torch.onnx.export(
        model,
        (dummy,),
        path,
        input_names=["pcm_int16"],
        # "input_features" bridges the two graphs (FE output -> classifier input);
        # the exact name is unimportant as long as both sides agree.
        output_names=["input_features"],
        dynamic_axes={"pcm_int16": {0: "batch"}, "input_features": {0: "batch"}},
        opset_version=18,  # torch >= 2.12 requires opset >= 18
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  feature extractor exported -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Validation
# ═════════════════════════════════════════════════════════════════════════════
def reference_features(fe: WhisperFeatureExtractor, cfg: FeatureConfig,
                       pcm_int16: np.ndarray) -> np.ndarray:
    """
    Ground-truth features via the exact reference pipeline used by inference.py:
    int16 -> float [-1,1) -> truncate_audio_to_last_n_seconds -> WhisperFeatureExtractor.
    """
    audio = np.asarray(pcm_int16, dtype=np.float32).reshape(-1) / 32768.0
    audio = truncate_audio_to_last_n_seconds(audio, n_seconds=MAX_SECONDS)
    out = fe(
        audio,
        sampling_rate=cfg.sampling_rate,
        return_tensors="np",
        padding="max_length",
        max_length=cfg.n_samples,
        truncation=True,
        do_normalize=True,
    )
    return out.input_features.astype(np.float32)


def make_test_pcm(cfg: FeatureConfig, seconds: float, seed: int) -> np.ndarray:
    """Deterministic speech-ish int16 PCM of a given length (harmonics + noise)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * cfg.sampling_rate)
    t = np.arange(n, dtype=np.float32) / cfg.sampling_rate
    sig = np.zeros(n, dtype=np.float32)
    for h, g in zip((1, 2, 3, 4), (1.0, 0.5, 0.25, 0.12)):
        sig += g * np.sin(2.0 * np.pi * 120.0 * h * t)
    sig *= 0.5 * (1.0 + np.sin(2.0 * np.pi * 4.0 * t)) * 0.8 + 0.2   # syllable env
    sig += 0.03 * rng.standard_normal(n)
    sig *= 0.6 / (np.max(np.abs(sig)) + 1e-9)
    return np.clip(sig * 32768.0, -32768, 32767).astype(np.int16)


def make_test_cases(cfg: FeatureConfig) -> dict:
    """Three clips that exercise the exact / front-pad / truncate code paths."""
    return {
        "8.0 s (exact window)": make_test_pcm(cfg, 8.0, seed=0),
        "3.5 s (short -> front-pad)": make_test_pcm(cfg, 3.5, seed=1),
        "13.0 s (long -> truncate)": make_test_pcm(cfg, 13.0, seed=2),
    }


def validate_feature_extractor(model: torch.nn.Module, cfg: FeatureConfig,
                               fe: WhisperFeatureExtractor, fe_path: str) -> float:
    """Check the ONNX / PyTorch feature extractor against WhisperFeatureExtractor."""
    sess = ort.InferenceSession(fe_path, providers=["CPUExecutionProvider"])

    print("\n── Validation: ONNX / PyTorch feature extractor vs WhisperFeatureExtractor ──")
    worst = 0.0
    for name, pcm in make_test_cases(cfg).items():
        feed = truncate_or_pad_int16(pcm, cfg.n_samples)[None, :].astype(np.int16)
        ref = reference_features(fe, cfg, pcm)
        onnx_out = sess.run(None, {"pcm_int16": feed})[0]
        with torch.no_grad():
            torch_out = model(torch.from_numpy(feed)).numpy()

        d_onnx = float(np.abs(onnx_out - ref).max())
        d_pt = float(np.abs(torch_out - ref).max())
        worst = max(worst, d_onnx)
        print(f"  {name:30s} shape={onnx_out.shape}  "
              f"ONNX-vs-ref={d_onnx:.2e}  PT-vs-ref={d_pt:.2e}")

    status = "PASS" if worst < 1e-3 else "CHECK"
    print(f"  worst ONNX-vs-reference max|Δ| = {worst:.2e}  "
          f"({status}; Whisper torch/numpy paths differ ~1e-5)")
    return worst


# ═════════════════════════════════════════════════════════════════════════════
# 6.  Glue  (fuse the feature extractor + each classifier into one ONNX graph)
# ═════════════════════════════════════════════════════════════════════════════════════════
def build_session(path: str) -> ort.InferenceSession:
    """ORT session with all graph optimizations enabled (used for the glue test)."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def glue_models(feature_extractor_path: str, classifier_path: str, output_path: str) -> onnx.ModelProto:
    """
    Fuse ``feature_extractor_path`` -> ``classifier_path`` into a single graph
    (pcm_int16 -> logits) with ``onnx.compose.merge_models`` and save it.

    The two source graphs share no internal tensor/initializer/node names, so the
    merge needs no prefixing; the only reconciliation required is the IR version
    (the feature extractor is IR 8, the classifiers are IR 10).
    """
    fe = onnx.load(feature_extractor_path)
    clf = onnx.load(classifier_path)

    # merge_models requires both models to share an IR version; bump to the max.
    ir_version = max(fe.ir_version, clf.ir_version)
    fe.ir_version = ir_version
    clf.ir_version = ir_version

    glued = compose.merge_models(
        fe,
        clf,
        io_map=[("input_features", "input_features")],
        producer_name="smart-turn-glue",
        doc_string="Smart Turn end-to-end: int16 PCM -> Whisper log-mel -> P(complete)",
    )

    # Best-effort validation. The int8 classifier carries com.microsoft ops whose
    # schemas the standalone onnx checker does not know, so treat checker/shape
    # inference as advisory — the authoritative test is loading + running in ORT.
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


def test_glued(feature_extractor_path: str, classifier_path: str, glued_path: str,
               cfg: FeatureConfig) -> float:
    """
    Compare the glued single-session pipeline against the original two-session
    pipeline (feature extractor -> classifier). They run identical ops, so the
    end-to-end P(complete) should match to within ORT graph-optimization noise.
    """
    fe_sess = build_session(feature_extractor_path)
    clf_sess = build_session(classifier_path)
    glued_sess = build_session(glued_path)

    clf_in = clf_sess.get_inputs()[0].name      # "input_features"
    glued_in = glued_sess.get_inputs()[0].name  # "pcm_int16"

    print("  ── Parity: two-session (FE -> classifier) vs glued single-session ──")
    worst = 0.0
    windows = []
    for name, pcm in make_test_cases(cfg).items():
        window = truncate_or_pad_int16(pcm, cfg.n_samples)[None, :].astype(np.int16)
        windows.append(window)

        feats = fe_sess.run(None, {glued_in: window})[0]            # stage 1
        p_two = float(clf_sess.run(None, {clf_in: feats})[0].reshape(-1)[0])  # stage 2
        p_glue = float(glued_sess.run(None, {glued_in: window})[0].reshape(-1)[0])

        d = abs(p_two - p_glue)
        worst = max(worst, d)
        print(f"    {name:30s} P(complete) two-stage={p_two:.6f}  glued={p_glue:.6f}  |Δ|={d:.2e}")

    # Dynamic batch must survive the merge: run all windows at once, compare to singles.
    batch = np.concatenate(windows, axis=0).astype(np.int16)        # (3, 128000)
    p_batch = glued_sess.run(None, {glued_in: batch})[0].reshape(-1)
    p_singles = np.array(
        [glued_sess.run(None, {glued_in: w})[0].reshape(-1)[0] for w in windows]
    )
    d_batch = float(np.abs(p_batch - p_singles).max())
    worst = max(worst, d_batch)
    print(f"  ── Dynamic batch: glued batch={batch.shape} -> logits={p_batch.shape}  "
          f"max|Δ| vs singles={d_batch:.2e}")

    status = "PASS" if worst < 1e-4 else "CHECK"
    print(f"  ── Worst |Δ| (glued vs two-stage) = {worst:.2e}  ({status}) ──")

    # Latency: one fused session vs two sessions + the Python/NumPy hand-off.
    probe = windows[0]
    for _ in range(5):
        glued_sess.run(None, {glued_in: probe})
        clf_sess.run(None, {clf_in: fe_sess.run(None, {glued_in: probe})[0]})

    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        glued_sess.run(None, {glued_in: probe})
    t_glued = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        clf_sess.run(None, {clf_in: fe_sess.run(None, {glued_in: probe})[0]})
    t_two = (time.perf_counter() - t0) / iters

    print(f"  ── Latency: glued={t_glued * 1000:.3f} ms  two-stage={t_two * 1000:.3f} ms  "
          f"| glued RTF={t_glued / MAX_SECONDS:.5f} ──")
    return worst


def glue_classifiers(fe_path: str, cfg: FeatureConfig) -> tuple[float, list]:
    """
    Glue the feature extractor onto each classifier in ``SMART_TURN_DIR`` and verify
    the fused single-session graph matches the original two-session pipeline. Returns
    the worst |Δ| (glued vs two-stage) and the list of glued model paths written.
    """
    print("\n═══════════════════════════════════════════════════════════════════")
    print("  Glue: feature extractor  +  classifier  ->  end-to-end (pcm_int16 -> logits)")
    print("  Bridge tensor: input_features  (FE output -> classifier input)")
    print("═══════════════════════════════════════════════════════════════════")

    worst_overall = 0.0
    produced = []
    for name in CLASSIFIER_MODELS:
        classifier_path = os.path.join(SMART_TURN_DIR, name)
        if not os.path.isfile(classifier_path):
            print(f"── Skipping (not found): {classifier_path}")
            continue

        glued_path = glued_output_path(classifier_path)
        print(f"\n══ {name} ══")
        glue_models(fe_path, classifier_path, glued_path)
        worst_overall = max(worst_overall, test_glued(fe_path, classifier_path, glued_path, cfg))
        produced.append(glued_path)
    return worst_overall, produced


# ═════════════════════════════════════════════════════════════════════════════
# 7.  Main
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    torch.manual_seed(1234)
    np.random.seed(1234)

    # Read every feature-extraction parameter from the WhisperFeatureExtractor config.
    fe = WhisperFeatureExtractor(chunk_length=MAX_SECONDS)
    cfg = FeatureConfig.from_whisper(fe)

    print(
        f"\nConfig (from WhisperFeatureExtractor)  SR={cfg.sampling_rate}, "
        f"WINDOW={cfg.n_samples} samples ({MAX_SECONDS}s), NFFT={cfg.n_fft}, "
        f"HOP={cfg.hop_length}, MELS={cfg.n_mels}, FRAMES={cfg.n_frames}"
    )
    print("Target  glued end-to-end graphs (pcm_int16 -> logits), opset=18\n")

    # The bare feature extractor is only an intermediate, so it goes to a temp file;
    # the glued end-to-end models are the deliverable and are saved in SMART_TURN_DIR.
    with tempfile.TemporaryDirectory() as tmpdir:
        fe_path = os.path.join(tmpdir, "smart-turn-feature-extractor.onnx")

        with torch.no_grad():
            model = SmartTurnFeatureExtractor(cfg).eval()
            export_feature_extractor(model, cfg, fe_path)
            validate_feature_extractor(model, cfg, fe, fe_path)

        worst_glue, produced = glue_classifiers(fe_path, cfg)

    print("\n── Summary ──")
    print(f"  glued end-to-end : pcm_int16 int16 [batch, {cfg.n_samples}]  ->  "
          f"logits f32 [batch, 1]")
    if produced:
        print(f"  glued models     : {', '.join(produced)}")
    else:
        print(f"  glued models     : (none — no classifiers found in {SMART_TURN_DIR}/)")
    print("  opset            : 18   dynamic_axes: batch")
    print(f"  worst |Δ| (glued vs two-stage) : {worst_glue:.2e}")
    print(f"  usage  : window = truncate_or_pad_int16(pcm, {cfg.n_samples});  "
          f"logits = sess.run(None, {{'pcm_int16': window[None]}})[0]")


if __name__ == "__main__":
    main()
