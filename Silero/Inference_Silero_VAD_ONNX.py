import gc
import time
import torch
import numpy as np
import onnxruntime
from datetime import timedelta
from pydub import AudioSegment
from silero_vad import load_silero_vad, get_speech_timestamps


test_vad_audio = "./vad_sample.wav"                            # The test audio path.
save_timestamps_second = "./timestamps_second.txt"             # The saved path.
save_timestamps_indices = "./timestamps_indices.txt"           # The saved path.

use_gpu_fp16 = False                                # Must first run Export_Silero_VAD.py with use_gpu_fp16=True to generate silero_vad.onnx in float16 format.
ORT_Accelerate_Providers = []                       # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                                    # else keep empty.
provider_options = None
ACTIVATE_THRESHOLD = 0.5                            # Set for silero_vad, none-silence state threshold.
FUSION_THRESHOLD = 0.3                              # A judgment factor used to merge timestamps: if two speech segments are too close, they are combined into one. Unit: second.
MIN_SPEECH_DURATION = 0.25                          # A judgment factor used to filter the vad results. Unit: second.
MAX_SPEECH_DURATION = 20                            # Set for silero_vad, maximum silence duration time. Unit: second.
MIN_SILENCE_DURATION = 250                          # Set for silero_vad, minimum silence duration time. Unit: ms.
SAMPLE_RATE = 16000                                 # Silero VAD accept the audio with 8kHz or 16kHz.


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 3                 # error level, it an adjustable value.
session_opts.inter_op_num_threads = 0               # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0               # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True            # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
if use_gpu_fp16:
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
else:
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")


def process_timestamps(timestamps, fusion_threshold=1.0, min_duration=0.5):
    # Filter out short durations
    filtered_timestamps = [(start, end) for start, end in timestamps if (end - start) >= min_duration]
    del timestamps
    # Fuse and filter timestamps
    fused_timestamps_1st = []
    for start, end in filtered_timestamps:
        # Merge with the previous segment if within the fusion threshold
        if fused_timestamps_1st and (start - fused_timestamps_1st[-1][1] <= fusion_threshold):
            fused_timestamps_1st[-1] = (fused_timestamps_1st[-1][0], end)
        else:
            fused_timestamps_1st.append((start, end))
    del filtered_timestamps
    fused_timestamps_2nd = []
    for start, end in fused_timestamps_1st:
        # Merge with the previous segment if within the fusion threshold
        if fused_timestamps_2nd and (start - fused_timestamps_2nd[-1][1] <= fusion_threshold):
            fused_timestamps_2nd[-1] = (fused_timestamps_2nd[-1][0], end)
        else:
            fused_timestamps_2nd.append((start, end))
    return fused_timestamps_2nd


def vad_to_timestamps(vad_output, frame_duration):
    timestamps = []
    start = None
    # Extract raw timestamps
    for i, silence in enumerate(vad_output):
        if silence:
            if start is not None:  # End of the current speaking segment
                end = i * frame_duration + frame_duration
                timestamps.append((start, end))
                start = None
        else:
            if start is None:  # Start of a new speaking segment
                start = i * frame_duration
    # Handle the case where speech continues until the end
    if start is not None:
        timestamps.append((start, len(vad_output) * frame_duration))
    return timestamps


def format_time(seconds):
    """Convert seconds to VTT time format 'hh:mm:ss.mmm'."""
    td = timedelta(seconds=seconds)
    td_sec = td.total_seconds()
    total_seconds = int(td_sec)
    milliseconds = int((td_sec - total_seconds) * 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


# Load the Silero
silero_vad = load_silero_vad(session_opts=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)

# Load the audio
audio = np.array(AudioSegment.from_file(test_vad_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float16 if use_gpu_fp16 else np.float32) * 0.000030517578  # 1/32768

# Start VAD
print("\nStart to run the VAD process.")
start_time = time.time()
timestamps = get_speech_timestamps(
    torch.from_numpy(audio),
    model=silero_vad,
    threshold=ACTIVATE_THRESHOLD,
    max_speech_duration_s=MAX_SPEECH_DURATION,
    min_speech_duration_ms=int(MIN_SPEECH_DURATION * 1000),
    min_silence_duration_ms=MIN_SILENCE_DURATION,
    return_seconds=True
)
print(f"\nVAD Complete. Time Cost: {(time.time() - start_time):.3f} seconds.")
del audio
gc.collect()

# Generate the timestamps.
timestamps = [(item['start'], item['end']) for item in timestamps]
timestamps = process_timestamps(timestamps, FUSION_THRESHOLD, MIN_SPEECH_DURATION)

# Save the timestamps.
with open(save_timestamps_second, "w", encoding='UTF-8') as file:
    print("\nTimestamps in Second:")
    for start, end in timestamps:
        s_time = format_time(start)
        e_time = format_time(end)
        line = f"{s_time} --> {e_time}\n"
        file.write(line)
        print(line.replace("\n", ""))

with open(save_timestamps_indices, "w", encoding='UTF-8') as file:
    print("\nTimestamps in Indices:")
    for start, end in timestamps:
        line = f"{int(start * SAMPLE_RATE)} --> {int(end * SAMPLE_RATE)}\n"
        file.write(line)
        print(line.replace("\n", ""))
