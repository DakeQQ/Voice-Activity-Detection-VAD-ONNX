import time

import numpy as np
import onnxruntime
from datetime import timedelta
from pydub import AudioSegment

onnx_model_A = "/home/DakeQQ/Downloads/DFSMN_VAD_Optimized/DFSMN_VAD.onnx"     # The exported onnx model path.
test_vad_audio = "./vad_sample.wav"                                            # The test audio path.
save_timestamps_second = "./timestamps_second.txt"                             # The saved path.
save_timestamps_indices = "./timestamps_indices.txt"                           # The saved path.


ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
MAX_THREADS = 4                         # Number of parallel threads for audio denoising.
DEVICE_ID = 0                           # The GPU id, default to 0.
SAMPLE_RATE = 16000                     # The SDAEC parameter, do not edit the value.

# VAD Settings
FUSION_THRESHOLD = 0.3                  # A judgment factor used to merge timestamps: if two speech segments are too close, they are combined into one. Unit: second.
MIN_SPEECH_DURATION = 0.2               # A judgment factor used to filter the vad results. Unit: second.
SPEAKING_SCORE = 0.5                    # A judgment factor used to determine whether the state is speaking or not. A larger value makes activation more difficult.
SILENCE_SCORE = 0.5                     # A judgment factor used to determine whether the state is silent or not. A smaller value makes it easier to cut off speaking.
LOOK_BACKWARD = 0.3                     # Utilize future Voice Activity Detection (VAD) results to assess whether the current index indicates silence. Unit: second. Must be an integer multiple of 0.02.
OUTPUT_FRAME_LENGTH = 320               # The DFSMN_VAD parameter, do not edit the value.


if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_type': 'CPU',                         # [CPU, NPU, GPU, GPU.0, GPU.1]]
            'precision': 'ACCURACY',                      # [FP32, FP16, ACCURACY]
            'num_of_threads': MAX_THREADS,
            'num_streams': 1,
            'enable_opencl_throttling': True,
            'enable_qdq_optimizer': False,                # Enable it carefully
            'disable_dynamic_shapes': True
        }
    ]
    device_type = 'cpu'
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'gpu_mem_limit': 24 * 1024 * 1024 * 1024,     # 24 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',   # ["kNextPowerOfTwo", "kSameAsRequested"]
            'cudnn_conv_algo_search': 'EXHAUSTIVE',       # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]   The MossFormer_SE must using DEFAULT.
            'sdpa_kernel': '2',                           # ["0", "1", "2"]
            'use_tf32': '1',
            'fuse_conv_bias': '1',                        # Set to '0' to avoid potential errors when enabled.
            'cudnn_conv_use_max_workspace': '1',
            'cudnn_conv1d_pad_to_nc1d': '1',
            'tunable_op_enable': '1',
            'tunable_op_tuning_enable': '1',
            'tunable_op_max_tuning_duration_ms': 10000,
            'do_copy_in_default_stream': '0',
            'enable_cuda_graph': '0',                     # Set to '0' to avoid potential errors when enabled.
            'prefer_nhwc': '0',
            'enable_skip_layer_norm_strict_mode': '0',
            'use_ep_level_unified_stream': '0',
        }
    ]
    device_type = 'cuda'
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'performance_preference': 'high_performance',  # [high_performance, default, minimum_power]
            'device_filter': 'npu'                         # [any, npu, gpu]
        }
    ]
    device_type = 'dml'
else:
    # Please config by yourself for others providers.
    device_type = 'cpu'
    provider_options = None


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4                   # Fatal level, it an adjustable value.
session_opts.inter_op_num_threads = MAX_THREADS       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = MAX_THREADS       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True              # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
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


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A0 = out_name_A[0].name


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


# # Load the input audio
print(f"\nTest Input Audio: {test_vad_audio}")
audio = np.array(AudioSegment.from_file(test_vad_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
audio = normalize_to_int16(audio)
audio_len = len(audio)
audio = audio.reshape(1, 1, -1)

shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = min(SAMPLE_RATE * 360, audio_len)  # You can adjust it.
else:
    INPUT_AUDIO_LENGTH = shape_value_in

look_backward = int(LOOK_BACKWARD * SAMPLE_RATE // OUTPUT_FRAME_LENGTH)
stride_step = INPUT_AUDIO_LENGTH - (look_backward + 1) * OUTPUT_FRAME_LENGTH
if audio_len > INPUT_AUDIO_LENGTH:
    num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
    total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
    pad_amount = total_length_needed - audio_len
    final_slice = audio[:, :, -pad_amount:].astype(np.float32)
    white_noise = (np.sqrt(np.mean(final_slice * final_slice)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
elif audio_len < INPUT_AUDIO_LENGTH:
    audio_float = audio.astype(np.float32)
    white_noise = (np.sqrt(np.mean(audio_float * audio_float)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
aligned_len = audio.shape[-1]
inv_audio_len = float(100.0 / aligned_len)


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
            if start is not None:   # End of the current speaking segment
                end = i * frame_duration + frame_duration
                timestamps.append((start, end))
                start = None
        else:
            if start is None:       # Start of a new speaking segment
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


# Start to run DFSMN_VAD

inv_look_backward = float(1.0 / look_backward)
slice_start = 0
slice_end = INPUT_AUDIO_LENGTH
silence = True
saved = []
print("\nRunning the DFSMN_VAD by ONNX Runtime.")
start_time = time.time()
while slice_end <= aligned_len:
    vad_results = ort_session_A.run([out_name_A0], {in_name_A0: audio[:, :, slice_start: slice_end]})[0]
    for i in range(len(vad_results) - look_backward + 1):
        if silence:
            if vad_results[i] >= SPEAKING_SCORE:
                activate = 1
                for j in range(1, look_backward):
                    if vad_results[i + j] >= SPEAKING_SCORE:
                        activate += 1
                activate = activate * inv_look_backward
                if activate >= SPEAKING_SCORE:
                    silence = False
                else:
                    silence = True
            else:
                silence = True
        else:
            if vad_results[i] <= SILENCE_SCORE:
                activate = 1
                for j in range(1, look_backward):
                    if vad_results[i + j] <= SILENCE_SCORE:
                        activate += 1
                activate = activate * inv_look_backward
                if activate <= SILENCE_SCORE:
                    silence = False
                else:
                    silence = True
            else:
                silence = False
        saved.append(silence)
    slice_start += stride_step
    slice_end = slice_start + INPUT_AUDIO_LENGTH
    print(f"Complete: {slice_start * inv_audio_len:.3f}%")
for i in range(len(vad_results) - look_backward, len(vad_results)):
    if silence:
        if vad_results[i] >= SPEAKING_SCORE:
            silence = False
        else:
            silence = True
    else:
        if vad_results[i] <= SILENCE_SCORE:
            silence = True
        else:
            silence = False
    saved.append(silence)
end_time = time.time()

# Generate timestamps.
timestamps = vad_to_timestamps(saved, OUTPUT_FRAME_LENGTH / SAMPLE_RATE)
timestamps = process_timestamps(timestamps, FUSION_THRESHOLD, MIN_SPEECH_DURATION)
print(f"Complete: 100.00%")

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
      
print(f"\nVAD Process Complete.\n\nTime Cost: {end_time - start_time:.3f} Seconds")
