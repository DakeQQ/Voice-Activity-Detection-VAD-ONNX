#!/usr/bin/env python
"""Simplify the standalone NVIDIA Streaming Sortformer deployment graph."""

import sys
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in _SCRIPT_DIR.parents:
    if (_candidate / "Optimize_ONNX_Common.py").is_file():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

from Optimize_ONNX_Sortformer_Common import (  # pyright: ignore[reportMissingImports]
    OptimizerConfig,
    Plan,
    run_optimizer,
)


SOURCE_FOLDER = _SCRIPT_DIR / "NVIDIA_Streaming_Sortformer_4spk_ONNX"
OPTIMIZED_FOLDER = _SCRIPT_DIR / "NVIDIA_Streaming_Sortformer_4spk_Optimized"
MODEL_NAME = "NVIDIA_Streaming_Sortformer_4spk"


# ============================== USER CONFIG ==============================
# Q8 is the direct-script default. Set QUANT_METHOD to "F16", "Q4", or
# "DYNAMIC" to build another deployment profile.
QUANT_METHOD = "Q8"  # Q4 | Q8 | DYNAMIC | F16 | F32
WEIGHT_ONLY_ALGORITHM = "AFFINE_REFINE_V2"
WEIGHT_ONLY_BLOCK_SIZE = 64
WEIGHT_ONLY_ACCURACY_LEVEL = 4
WEIGHT_ONLY_SYMMETRIC = True
WEIGHT_ONLY_QUANT_FORMAT = "QOperator"

DYNAMIC_WEIGHT_TYPE = "QInt8"
DYNAMIC_PER_CHANNEL = True
DYNAMIC_REDUCE_RANGE = False


CONFIG = OptimizerConfig(
    original_folder_path=str(SOURCE_FOLDER),
    optimized_folder_path=str(OPTIMIZED_FOLDER),
    model_plans={
        MODEL_NAME: Plan(
            method=QUANT_METHOD,
            algo=WEIGHT_ONLY_ALGORITHM,
            op_types=("MatMul",),
            axes=(0,),
            optimize=True,
            transformer=False,
            external=False,
            first_slim_no_shape_infer=False,
        )
    },
    weight_only_algorithm=WEIGHT_ONLY_ALGORITHM,
    block_size=WEIGHT_ONLY_BLOCK_SIZE,
    accuracy_level=WEIGHT_ONLY_ACCURACY_LEVEL,
    quant_symmetric=WEIGHT_ONLY_SYMMETRIC,
    quant_format=WEIGHT_ONLY_QUANT_FORMAT,
    dynamic_weight_type=DYNAMIC_WEIGHT_TYPE,
    dynamic_per_channel=DYNAMIC_PER_CHANNEL,
    dynamic_reduce_range=DYNAMIC_REDUCE_RANGE,
)

# ============================ END USER CONFIG ============================


def main() -> None:
    run_optimizer(CONFIG)
    print(
        f"Optimized {QUANT_METHOD} model written: "
        f"{OPTIMIZED_FOLDER / (MODEL_NAME + '.onnx')}"
    )


if __name__ == "__main__":
    main()