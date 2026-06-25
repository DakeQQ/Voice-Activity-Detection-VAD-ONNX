#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Turn — GLUED end-to-end ONNX inference (int16 PCM -> P(complete))
======================================================================

Runs the fused Smart Turn v3 model produced by ``Export_SmartTurn_Glued.py`` in a
single ONNX Runtime session. The glued graph already embeds the Whisper log-mel
feature extractor, so inference is simply: raw 16 kHz mono int16 PCM in,
P(complete) out — no torch / transformers / NumPy feature pre-processing at run
time.

    pcm_int16  [batch, n_samples]  int16   ->   probability  [batch, 1]  float32

``n_samples`` (the fixed context window, 128000 = 8 s for Smart Turn v3) is read
from the model's own input shape, so it always matches whatever glued model is
loaded. A clip is truncated to its last ``n_samples`` (most recent context) or
front-padded with zeros, then scored; ``probability > THRESHOLD`` => turn complete.
"""

import os
import time

import numpy as np
import onnxruntime


# ═════════════════════════════════════════════════════════════════════════════
# Settings
# ═════════════════════════════════════════════════════════════════════════════
# Fused end-to-end model from Export_SmartTurn_Glued.py (pcm_int16 -> P(complete)).
onnx_model_A = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "SmartTurn_Glued_ONNX", "smart-turn-v3.2-cpu-glued.onnx",
)
test_audio = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "vad_sample.wav"
)                                      # 16 kHz mono audio to score; a dummy clip is used if missing.
SAMPLE_RATE = 16000                    # Smart Turn input sample rate (do not edit).
COMPLETE_THRESHOLD = 0.5               # probability > threshold -> turn complete.

ORT_Accelerate_Providers = []          # e.g. ['CUDAExecutionProvider']; empty = CPU only.


# ═════════════════════════════════════════════════════════════════════════════
# ONNX Runtime session
# ═════════════════════════════════════════════════════════════════════════════
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 3
session_opts.inter_op_num_threads = 0                 # 0 = auto.
session_opts.intra_op_num_threads = 0                 # 0 = auto.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")

ort_session_A = onnxruntime.InferenceSession(
    onnx_model_A,
    sess_options=session_opts,
    providers=ORT_Accelerate_Providers if ORT_Accelerate_Providers else ["CPUExecutionProvider"],
)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")

in_name_A0 = ort_session_A.get_inputs()[0].name       # "pcm_int16"
out_name_A0 = ort_session_A.get_outputs()[0].name     # P(complete)

# Fixed context window (samples) read straight from the model's input shape, so it
# can never drift out of sync with the loaded model (Smart Turn v3: 128000 = 8 s).
shape_value_in = ort_session_A.get_inputs()[0].shape[-1]
INPUT_AUDIO_LENGTH = SAMPLE_RATE * 8 if isinstance(shape_value_in, str) else int(shape_value_in)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════
def truncate_or_pad_int16(pcm, n_samples):
    """
    Keep the last ``n_samples`` (most recent context window) of a long clip, or
    zero-pad a short clip at the *front* — so the result is always ``n_samples``.
    """
    pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
    if len(pcm) > n_samples:
        return pcm[-n_samples:]
    if len(pcm) < n_samples:
        return np.pad(pcm, (n_samples - len(pcm), 0), mode="constant", constant_values=0)
    return pcm


def load_audio_int16(path):
    """Load any audio file as 16 kHz mono int16 PCM (what the glued model expects)."""
    from pydub import AudioSegment  # optional dependency; only needed to decode files

    audio = (
        AudioSegment.from_file(path)
        .set_channels(1)
        .set_frame_rate(SAMPLE_RATE)
        .set_sample_width(2)  # 16-bit
    )
    return np.array(audio.get_array_of_samples(), dtype=np.int16)


def predict_endpoint(pcm_int16):
    """
    Score one clip with the glued Smart Turn model.

    Args:
        pcm_int16: 16 kHz mono int16 PCM samples (any length).

    Returns:
        dict with ``prediction`` (1 complete / 0 incomplete) and ``probability``.
    """
    window = truncate_or_pad_int16(pcm_int16, INPUT_AUDIO_LENGTH)[None, :]   # (1, n_samples) int16
    probability = float(ort_session_A.run([out_name_A0], {in_name_A0: window})[0].reshape(-1)[0])
    return {
        "prediction": 1 if probability > COMPLETE_THRESHOLD else 0,
        "probability": probability,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Run
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if os.path.isfile(test_audio):
        print(f"\nTest Input Audio: {test_audio}")
        pcm = load_audio_int16(test_audio)
    else:
        print(f"\n(test audio '{test_audio}' not found — scoring a 2 s dummy clip)")
        pcm = (np.random.randn(SAMPLE_RATE * 2) * 4000.0).astype(np.int16)

    start_time = time.time()
    result = predict_endpoint(pcm)
    elapsed_ms = (time.time() - start_time) * 1000.0

    print(f"\nProbability (complete) : {result['probability']:.4f}")
    print(f"Prediction             : {'Complete' if result['prediction'] else 'Incomplete'}")
    print(f"Inference time         : {elapsed_ms:.2f} ms")
