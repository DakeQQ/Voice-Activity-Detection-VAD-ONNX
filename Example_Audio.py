from pathlib import Path


EXAMPLE_AUDIO_ROOT = Path(__file__).resolve().parent / "Test_Examples"

_MODEL_AUDIO_FILES = {
    "dfsmn": (
        ("near_end", "near_and_far_end_audio/nearend_mic1.wav"),
        ("far_end", "near_and_far_end_audio/farend_speech1.wav"),
    ),
    "firered_aed": (("audio", "vad/vad_sample.wav"),),
    "firered_stream_vad": (("audio", "vad/vad_sample.wav"),),
    "firered_vad": (("audio", "vad/vad_sample.wav"),),
    "fsmn": (("audio", "vad/vad_sample.wav"),),
    "nvidia_marblenet": (("audio", "vad/vad_sample.wav"),),
    "pyannote_community1": (("audio", "vad/vad_sample.wav"),),
    "silero": (("audio", "vad/vad_sample.wav"),),
    "smart_turn_v3": (("audio", "vad/vad_sample.wav"),),
}


def example_audio_path(relative_path):
    return str(EXAMPLE_AUDIO_ROOT / relative_path)


def model_audio_cases(model_name):
    try:
        audio_files = _MODEL_AUDIO_FILES[model_name]
    except KeyError as exc:
        names = ", ".join(sorted(_MODEL_AUDIO_FILES))
        raise ValueError(f"Unknown demo audio model {model_name!r}. Available models: {names}") from exc
    return [(example_audio_path(relative_path), case_name) for case_name, relative_path in audio_files]


def model_audio_paths(model_name):
    return [path for path, _case_name in model_audio_cases(model_name)]


def model_audio_path(model_name, case_name=None):
    cases = model_audio_cases(model_name)
    if case_name is None:
        if len(cases) != 1:
            names = ", ".join(case for _path, case in cases)
            raise ValueError(f"Model {model_name!r} has multiple demo audio cases: {names}")
        return cases[0][0]

    for path, name in cases:
        if name == case_name:
            return path

    names = ", ".join(name for _path, name in cases)
    raise ValueError(f"Unknown demo audio case {case_name!r} for model {model_name!r}. Available cases: {names}")
