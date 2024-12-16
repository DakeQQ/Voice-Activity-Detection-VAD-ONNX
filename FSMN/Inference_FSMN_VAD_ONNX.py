import time

import numpy as np
import onnxruntime
from pydub import AudioSegment

onnx_model_A = "/home/DakeQQ/Downloads/FSMN_VAD_Optimized/FSMN_VAD.ort"         # The exported onnx model path.
test_vad_audio = "./vad_sample.wav"                                             # The test audio path.
save_timestamps_second = "./timestamps_second.txt"                              # The saved path.
save_timestamps_indices = "./timestamps_indices.txt"                            # The saved path.

ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
SAMPLE_RATE = 16000                     # The FSMN_VAD parameter, do not edit the value.
ONE_MINUS_SPEECH_THRESHOLD = 0.15       # The judge factor for the VAD model edit it carefully. A higher value increases sensitivity but may mistakenly classify noise as speech. When using denoised audio, this value could be approximately 0.02
SNR_THRESHOLD = 10.0                    # The judge factor for VAD model. Unit: dB.
BACKGROUND_NOISE_dB_INIT = 30.0         # An initial value for the background. More smaller values indicate a quieter environment. Unit: dB. When using denoised audio, set this value to be smaller.
FUSION_THRESHOLD = 1.5                  # A judgment factor used to merge timestamps: if two speech segments are too close, they are combined into one. Unit: second.
MIN_SPEECH_DURATION = 0.5               # A judgment factor used to filter the vad results. Unit: second.
ACTIVATE_SCORE = 0.5                    # A judgment factor used to judge the state is active or not.


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 3         # error level, it an adjustable value.
session_opts.inter_op_num_threads = 0       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True    # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
model_type = ort_session_A._inputs_meta[0].type
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
in_name_A1 = in_name_A[1].name
in_name_A2 = in_name_A[2].name
in_name_A3 = in_name_A[3].name
in_name_A4 = in_name_A[4].name
in_name_A5 = in_name_A[5].name
in_name_A6 = in_name_A[6].name
out_name_A0 = out_name_A[0].name
out_name_A1 = out_name_A[1].name
out_name_A2 = out_name_A[2].name
out_name_A3 = out_name_A[3].name
out_name_A4 = out_name_A[4].name
out_name_A5 = out_name_A[5].name


# # Load the input audio
print(f"\nTest Input Audio: {test_vad_audio}")
audio = np.array(AudioSegment.from_file(test_vad_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples())
audio_len = len(audio)
inv_audio_len = float(100.0 / audio_len)
if "int16" not in model_type:
    audio = audio.astype(np.float32) / 32768.0
    if "float16" in model_type:
        audio = audio.astype(np.float16)
audio = audio.reshape(1, 1, -1)
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = min(4000, audio_len)  # Default is 250ms. You can adjust it.
else:
    INPUT_AUDIO_LENGTH = shape_value_in
stride_step = INPUT_AUDIO_LENGTH
if audio_len > INPUT_AUDIO_LENGTH:
    final_slice = audio[:, :, audio_len // stride_step * stride_step:]
    white_noise = (np.sqrt(np.mean(final_slice * final_slice)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, stride_step - final_slice.shape[-1]))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
elif audio_len < INPUT_AUDIO_LENGTH:
    white_noise = (np.sqrt(np.mean(audio * audio)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
aligned_len = audio.shape[-1]


def vad_to_timestamps(vad_output, frame_duration, fusion_threshold=1.0, min_duration=0.5):
    """
    Convert VAD output to timestamps with filtering for short durations.

    Parameters:
    vad_output (list of bool): Voice activity detection output per frame.
    frame_duration (float): Duration of each frame in seconds.
    fusion_threshold (float): Threshold to merge consecutive segments in seconds.
    min_duration (float): Minimum duration of speech to keep in seconds.

    Returns:
    list of tuple: Filtered and fused timestamps [(start, end), ...].
    """
    timestamps = []
    start = None

    # Extract raw timestamps
    for i, is_speaking in enumerate(vad_output):
        if is_speaking:
            if start is None:  # Start of a new speaking segment
                start = i * frame_duration
        else:
            if start is not None:  # End of the current speaking segment
                end = i * frame_duration + frame_duration
                timestamps.append((start, end))
                start = None

    # Handle the case where speech continues until the end
    if start is not None:
        timestamps.append((start, len(vad_output) * frame_duration))

    # Fuse and filter timestamps
    fused_timestamps = []
    for start, end in timestamps:
        # Merge with the previous segment if within the fusion threshold
        if fused_timestamps and (start - fused_timestamps[-1][1] <= fusion_threshold):
            fused_timestamps[-1] = (fused_timestamps[-1][0], end)
        else:
            fused_timestamps.append((start, end))

    # Filter out short durations
    filtered_timestamps = [
        (start, end) for start, end in fused_timestamps if (end - start) >= min_duration
    ]

    return filtered_timestamps


# Start to run FSMN_VAD
if "float16" in model_type:
    cache_0 = np.zeros((1, 128, 19, 1), dtype=np.float16)
    noise_average_dB = np.array([BACKGROUND_NOISE_dB_INIT + SNR_THRESHOLD], dtype=np.float16)
    one_minus_speech_threshold = np.array([ONE_MINUS_SPEECH_THRESHOLD], dtype=np.float16)
else:
    cache_0 = np.zeros((1, 128, 19, 1), dtype=np.float32)  # FSMN_VAD model fixed cache shape. Do not edit it.
    noise_average_dB = np.array([BACKGROUND_NOISE_dB_INIT + SNR_THRESHOLD], dtype=np.float32)
    one_minus_speech_threshold = np.array([ONE_MINUS_SPEECH_THRESHOLD], dtype=np.float32)

cache_1 = cache_0
cache_2 = cache_0
cache_3 = cache_0
slice_start = 0
SNR_THRESHOLD = SNR_THRESHOLD * 0.5
saved = []
print("\nRunning the FSMN_VAD by ONNX Runtime.")
start_time = time.time()
while slice_start + stride_step <= aligned_len:
    score, cache_0, cache_1, cache_2, cache_3, noisy_dB = ort_session_A.run(
        [out_name_A0, out_name_A1, out_name_A2, out_name_A3, out_name_A4, out_name_A5],
        {
            in_name_A0: audio[:, :, slice_start: slice_start + INPUT_AUDIO_LENGTH],
            in_name_A1: cache_0,
            in_name_A2: cache_1,
            in_name_A3: cache_2,
            in_name_A4: cache_3,
            in_name_A5: one_minus_speech_threshold,
            in_name_A6: noise_average_dB
        })
    if score > ACTIVATE_SCORE:
        saved.append(True)
    else:
        saved.append(False)
    noise_average_dB = 0.5 * (noise_average_dB + noisy_dB) + SNR_THRESHOLD
    print(f"Complete: {slice_start * inv_audio_len:.2f}%")
    slice_start += stride_step

# Generate timestamps.
end_time = time.time()
timestamps = vad_to_timestamps(saved, INPUT_AUDIO_LENGTH / SAMPLE_RATE, FUSION_THRESHOLD, MIN_SPEECH_DURATION)
print(f"Complete: 100.00%")

# Save the timestamps.
with open(save_timestamps_second, "w", encoding='UTF-8') as file:
    print("\nTimestamps in Second:")
    for start, end in timestamps:
        line = f"[{start:.2f} --> {end:.2f}]\n"
        file.write(line)
        print(line.replace("\n", ""))

with open(save_timestamps_indices, "w", encoding='UTF-8') as file:
    print("\nTimestamps in Indices:")
    for start, end in timestamps:
        line = f"[{int(start * SAMPLE_RATE)} --> {int(end * SAMPLE_RATE)}]\n"
        file.write(line)
        print(line.replace("\n", ""))
print(f"\nVAD Process Complete.\n\nTime Cost: {end_time - start_time:.3f} Seconds")
