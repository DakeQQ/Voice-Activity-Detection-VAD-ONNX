import argparse
from pathlib import Path

import onnxruntime

from vad_onnx_metadata import (
    REQUIRED_COMMUNITY1_METADATA_KEYS,
    REQUIRED_SMART_TURN_METADATA_KEYS,
    REQUIRED_VAD_METADATA_KEYS,
    community1_metadata_path,
    metadata_path_for_model,
    parse_community1_metadata,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Print and validate VAD, SmartTurn, or Community-1 ONNX metadata keys."
        )
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Main ONNX model path or metadata carrier path.",
    )
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    if model_path.stem.endswith("_Metadata"):
        metadata_path = model_path
    else:
        metadata_path = metadata_path_for_model(model_path)
        package_metadata_path = community1_metadata_path(model_path.parent)
        if not metadata_path.is_file() and package_metadata_path.is_file():
            metadata_path = package_metadata_path
    if not metadata_path.is_file():
        raise SystemExit(f"Metadata carrier not found: {metadata_path}")

    session = onnxruntime.InferenceSession(
        str(metadata_path),
        providers=["CPUExecutionProvider"],
    )
    metadata = session.get_modelmeta().custom_metadata_map or {}
    if "vad_metadata_version" in metadata:
        required_keys = REQUIRED_VAD_METADATA_KEYS
        schema_name = "VAD"
    elif "smart_turn_metadata_version" in metadata:
        required_keys = REQUIRED_SMART_TURN_METADATA_KEYS
        schema_name = "SmartTurn"
    elif "community1_metadata_version" in metadata:
        required_keys = REQUIRED_COMMUNITY1_METADATA_KEYS
        schema_name = "Pyannote Community-1"
        try:
            parse_community1_metadata(metadata)
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"Invalid Community-1 metadata: {error}") from error
    else:
        raise SystemExit(
            "Metadata carrier has no recognized VAD, SmartTurn, or "
            "Community-1 metadata version."
        )

    missing = [key for key in required_keys if key not in metadata]
    print(f"Schema: {schema_name}")
    for key in sorted(metadata):
        print(f"{key}={metadata[key]}")
    if missing:
        raise SystemExit(f"Missing required metadata keys: {', '.join(missing)}")


if __name__ == "__main__":
    main()
