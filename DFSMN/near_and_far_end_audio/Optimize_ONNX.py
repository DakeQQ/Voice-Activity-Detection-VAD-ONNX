"""Optimize the near/far-input DFSMN VAD model."""

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
    if (_candidate / "Optimize_ONNX_Common.py").exists() and (
        _candidate / "vad_onnx_metadata.py"
    ).exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

from Optimize_ONNX_Common import OptimizerConfig, Plan, run_optimizer


ORIGINAL_FOLDER_PATH = str(SCRIPT_DIR / "DFSMN_VAD_ONNX")
OPTIMIZED_FOLDER_PATH = str(SCRIPT_DIR / "DFSMN_VAD_Optimized")

ENABLE_FP16 = False 

MODEL_PLANS = {
    "DFSMN_VAD": Plan(
        method="F16" if ENABLE_FP16 else "F32",
        opt_level=1,
    ),
}

CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
)


if __name__ == "__main__":
    run_optimizer(CONFIG)
