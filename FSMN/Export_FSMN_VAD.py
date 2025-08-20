import gc
import math
import time
import torch
import site
import shutil
import numpy as np
import onnxruntime
import torchaudio
from datetime import timedelta
from pydub import AudioSegment
from STFT_Process import STFT_Process                                                 # The custom STFT/ISTFT can be exported in ONNX format.

model_path = "/home/DakeQQ/Downloads/speech_fsmn_vad_zh-cn-16k-common-pytorch"        # The FSMN_VAD download path.
onnx_model_A = "/home/DakeQQ/Downloads/FSMN_VAD_ONNX/FSMN.onnx"                       # The exported onnx model path.
test_vad_audio = "./vad_sample.wav"                                                   # The test audio path.
save_timestamps_second = "./timestamps_second.txt"                                    # The saved path.
save_timestamps_indices = "./timestamps_indices.txt"                                  # The saved path.


ORT_Accelerate_Providers = []                               # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                                            # else keep empty.
DYNAMIC_AXES = False                                        # Only support static axes.
INPUT_AUDIO_LENGTH = 16000                                  # The input audio segment length.
WINDOW_TYPE = 'hamming'                                     # Type of window function used in the STFT
N_MELS = 80                                                 # Number of Mel bands to generate in the Mel-spectrogram. Do not edit it.
NFFT_STFT = 512                                             # Number of FFT components for the STFT process, edit it carefully.
WINDOW_LENGTH = 400                                         # Length of windowing, edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
SAMPLE_RATE = 16000                                         # The FSMN_VAD parameter, do not edit the value.
LFR_M = 5                                                   # The FSMN_VAD parameter, do not edit the value.
LFR_N = 1                                                   # The FSMN_VAD parameter, do not edit the value.
PRE_EMPHASIZE = 0.97                                        # For audio preprocessing.
SPEECH_2_NOISE_RATIO = 1.0                                  # The judge factor for VAD model, edit it carefully.
ONE_MINUS_SPEECH_THRESHOLD = 1.0                            # The judge factor for the VAD model, edit it carefully. A higher value increases sensitivity but may mistakenly classify noise as speech.
SNR_THRESHOLD = 10.0                                        # The judge factor for VAD model. Unit: dB.
BACKGROUND_NOISE_dB_INIT = 30.0                             # An initial value for the background. More smaller values indicate a quieter environment. Unit: dB. When using denoised audio, set this value to be smaller.
FUSION_THRESHOLD = 0.3                                      # A judgment factor used to merge timestamps: if two speech segments are too close, they are combined into one. Unit: second.
MIN_SPEECH_DURATION = 0.2                                   # A judgment factor used to filter the vad results. Unit: second.
SPEAKING_SCORE = 0.5                                        # A judgment factor used to determine whether the state is speaking or not. A larger value makes activation more difficult.
SILENCE_SCORE = 0.5                                         # A judgment factor used to determine whether the state is silent or not. A smaller value makes it easier to cut off speaking.
LOOK_BACKWARD = 0.3                                         # Utilize future Voice Activity Detection (VAD) results to assess whether the current index indicates silence. Unit: second. Must be an integer multiple of 0.02.
OUTPUT_FRAME_LENGTH = 160                                   # The FSMN_VAD parameter, do not edit the value.


STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1   # The length after STFT processed
LFR_LENGTH = (STFT_SIGNAL_LENGTH + LFR_N - 1) // LFR_N
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH


shutil.copyfile('./modeling_modified/encoder.py', site.getsitepackages()[-1] + "/funasr/models/fsmn_vad_streaming/encoder.py")
from funasr import AutoModel


class FSMN_VAD(torch.nn.Module):
    def __init__(self, fsmn_vad, stft_model, nfft_stft, stft_signal_len, n_mels, sample_rate, pre_emphasis, lfr_m, lfr_n, lfr_len, speech_2_noise_ratio, input_audio_len, hop_len, cmvn_means, cmvn_vars):
        super(FSMN_VAD, self).__init__()
        self.fsmn_vad = fsmn_vad
        self.stft_model = stft_model
        self.speech_2_noise_ratio = float(speech_2_noise_ratio)
        self.pre_emphasis = float(pre_emphasis)
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft_stft // 2 + 1, 20, sample_rate // 2, n_mels, sample_rate, None,'htk')).transpose(0, 1).unsqueeze(0)
        self.nfft_stft = nfft_stft
        self.lfr_m_factor = (lfr_m - 1) // 2
        self.T_lfr = lfr_len
        self.cmvn_means = cmvn_means
        self.cmvn_vars = cmvn_vars
        self.inv_int16 = float(1.0 / 32768.0)
        indices = torch.arange(0, self.T_lfr * lfr_n, lfr_n, dtype=torch.int32).unsqueeze(1) + torch.arange(lfr_m, dtype=torch.int32)
        self.indices_mel = indices.clamp(max=stft_signal_len + self.lfr_m_factor - 1)  # Ensure no out-of-bounds access
        self.indices_audio = torch.arange(nfft_stft, dtype=torch.int32) + torch.arange(0, input_audio_len - nfft_stft + 1, hop_len, dtype=torch.int32).unsqueeze(-1)
        self.inv_reference_air_factor = float(1.0 / (math.sqrt(input_audio_len) * 2e-5))
        # 2e-5 is reference air_pressure value

    def forward(self, audio, cache_0, cache_1, cache_2, cache_3, one_minus_speech_threshold, noise_average_dB):
        audio = audio.float() * self.inv_int16
        audio = audio - torch.mean(audio)  # Remove DC Offset
        if self.pre_emphasis > 0:
            audio = torch.cat([audio[:, :, :1], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]], dim=-1)
        real_part, imag_part = self.stft_model(audio, 'constant')
        mel_features = torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part).transpose(1, 2).clamp(min=1e-5).log()
        left_padding = mel_features[:, [0], :]
        left_padding = torch.cat([left_padding for _ in range(self.lfr_m_factor)], dim=1)
        padded_inputs = torch.cat((left_padding, mel_features), dim=1)
        mel_features = padded_inputs[:, self.indices_mel].reshape(1, self.T_lfr, -1)
        score, cache_0, cache_1, cache_2, cache_3 = self.fsmn_vad((mel_features - self.cmvn_means) * self.cmvn_vars, cache_0, cache_1, cache_2, cache_3)
        if self.speech_2_noise_ratio > 1.0:
            score += torch.pow(score, self.speech_2_noise_ratio)
        elif self.speech_2_noise_ratio < 1.0:
            score += 1.0
        else:
            score += score
        audio = (audio * self.inv_reference_air_factor).squeeze()[self.indices_audio]
        power_dB = torch.log10(torch.sum(audio * audio, dim=-1) + 0.00002)  # Original: 10 * log10(x) -> noise_average_dB * 0.1; SNR_THRESHOLD * 0.1
        total_frames = score.shape[-1]
        padding = power_dB[-1:].expand((total_frames - power_dB.shape[-1]))
        power_dB = torch.cat((power_dB, padding), dim=-1)
        condition = (score <= one_minus_speech_threshold) & (power_dB >= noise_average_dB)
        noisy_dB = power_dB[~condition].mean()
        score = condition.to(torch.uint8)[:-1]
        return score, cache_0, cache_1, cache_2, cache_3, noisy_dB


print('Export start ...')
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()  # The max_frames is not the key parameter for STFT, but it is for ISTFT.
    fsmn_vad = AutoModel(
        model=model_path,
        disable_update=True,
        device="cpu"
    )
    CMVN_MEANS = fsmn_vad.kwargs['frontend'].cmvn[0].repeat(1, 1, 1)
    CMVN_VARS = fsmn_vad.kwargs['frontend'].cmvn[1].repeat(1, 1, 1)
    fsmn_vad = FSMN_VAD(fsmn_vad.model.encoder.eval(), custom_stft, NFFT_STFT, STFT_SIGNAL_LENGTH, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, LFR_M, LFR_N, LFR_LENGTH, SPEECH_2_NOISE_RATIO, INPUT_AUDIO_LENGTH, HOP_LENGTH, CMVN_MEANS, CMVN_VARS)
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    cache_0 = torch.zeros((1, 128, 19, 1), dtype=torch.float32)  # FSMN_VAD model fixed cache shape. Do not edit it.
    cache_1 = cache_0
    cache_2 = cache_0
    cache_3 = cache_0
    one_minus_speech_threshold = torch.ones(1, dtype=torch.float32)
    noise_average_dB = torch.ones(1, dtype=torch.float32)
    torch.onnx.export(
        fsmn_vad,
        (audio, cache_0, cache_1, cache_2, cache_3, one_minus_speech_threshold, noise_average_dB),
        onnx_model_A,
        input_names=['audio', 'cache_0', 'cache_1', 'cache_2', 'cache_3', 'one_minus_speech_threshold', 'noise_average_dB'],
        output_names=['score', 'cache_0', 'cache_1', 'cache_2', 'cache_3', 'noisy_dB'],
        do_constant_folding=True,
        dynamic_axes={
            'audio': {2: 'audio_len'},
            'score': {0: 'signal_len'},
        } if DYNAMIC_AXES else None,
        opset_version=17
    )
    del fsmn_vad
    del cache_0
    del cache_1
    del cache_2
    del cache_3
    del audio
    del custom_stft
    del one_minus_speech_threshold
    del noise_average_dB
    del CMVN_MEANS
    del CMVN_VARS
    gc.collect()
print('\nExport done!\n\nStart to run FSMN_VAD by ONNX Runtime.\n\nNow, loading the model...')


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
model_type = ort_session_A._inputs_meta[1].type
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


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


# Load the input audio
print(f"\nTest Input Audio: {test_vad_audio}")
audio = np.array(AudioSegment.from_file(test_vad_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
audio = normalize_to_int16(audio)
audio_len = len(audio)
inv_audio_len = float(100.0 / audio_len)
audio = audio.reshape(1, 1, -1)
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = max(16000, audio_len)  # You can adjust it.
else:
    INPUT_AUDIO_LENGTH = shape_value_in
look_backward = int(LOOK_BACKWARD * SAMPLE_RATE // OUTPUT_FRAME_LENGTH)
stride_step = INPUT_AUDIO_LENGTH - (look_backward + 1) * OUTPUT_FRAME_LENGTH
if look_backward != 0.0:
    inv_look_backward = float(1.0 / look_backward)
else:
    look_backward = 1
    inv_look_backward = 1.0

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
slice_end = INPUT_AUDIO_LENGTH
noise_average_dB *= 0.1
SNR_THRESHOLD *= 0.1
silence = True
saved = []
print("\nRunning the FSMN_VAD by ONNX Runtime.")
start_time = time.time()
while slice_end <= aligned_len:
    score, cache_0, cache_1, cache_2, cache_3, noisy_dB = ort_session_A.run(
        [out_name_A0, out_name_A1, out_name_A2, out_name_A3, out_name_A4, out_name_A5],
        {
            in_name_A0: audio[:, :, slice_start: slice_end],
            in_name_A1: cache_0,
            in_name_A2: cache_1,
            in_name_A3: cache_2,
            in_name_A4: cache_3,
            in_name_A5: one_minus_speech_threshold,
            in_name_A6: noise_average_dB
        })
    for i in range(len(score) - look_backward + 1):
        if silence:
            if score[i] != 0:
                activate = 1
                for j in range(1, look_backward):
                    if score[i + j] != 0:
                        activate += 1
                activate = activate * inv_look_backward
                if activate >= SPEAKING_SCORE:
                    silence = False
                else:
                    silence = True
            else:
                silence = True
        else:
            if score[i] != 1:
                activate = 1
                for j in range(1, look_backward):
                    if score[i + j] != 1:
                        activate += 1
                activate = activate * inv_look_backward
                if activate <= SILENCE_SCORE:
                    silence = False
                else:
                    silence = True
            else:
                silence = False
        saved.append(silence)

    if noisy_dB > 0.0:
        noise_average_dB = 0.5 * (noise_average_dB + noisy_dB + SNR_THRESHOLD)
    print(f"Complete: {slice_start * inv_audio_len:.3f}%")
    slice_start += stride_step
    slice_end = slice_start + INPUT_AUDIO_LENGTH

for i in range(len(score) - look_backward, len(score)):
    if silence:
        if score[i] != 0:
            silence = False
        else:
            silence = True
    else:
        if score[i] != 1:
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
