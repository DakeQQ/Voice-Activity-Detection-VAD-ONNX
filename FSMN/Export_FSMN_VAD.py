import gc
import shutil
import time

import numpy as np
import onnxruntime
import torch
import torchaudio
from pydub import AudioSegment

from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.

model_path = "/home/DakeQQ/Downloads/speech_fsmn_vad_zh-cn-16k-common-pytorch"                                                            # The ZipEnhancer download path.
onnx_model_A = "/home/DakeQQ/Downloads/FSMN_VAD_ONNX/FSMN_VAD.onnx"                                                                       # The exported onnx model path.
python_fsmn_vad_path = '/home/DakeQQ/anaconda3/envs/python_312/lib/python3.12/site-packages/funasr/models/fsmn_vad_streaming/encoder.py'  # The FSMN_VAD python script path.
modified_path = './modeling_modified/'
test_vad_audio = "./vad_sample.wav"                         # The test audio path.
save_timestamps_second = "./timestamps_second.txt"          # The saved path.
save_timestamps_indices = "./timestamps_indices.txt"        # The saved path.


ORT_Accelerate_Providers = []                               # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                                            # else keep empty.
DYNAMIC_AXES = False                                        # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
USE_PCM_INT16 = False                                       # Enable it, if the audio input is PCM wav data with dtype int16 (short).
INPUT_AUDIO_LENGTH = 1600                                   # Set for static axis export: the length of the audio input signal (in samples) is recommended to be greater than 400 and less than 8000. Smaller values yield fine timestamps.
WINDOW_TYPE = 'kaiser'                                      # Type of window function used in the STFT
MAX_THREADS = 4                                             # Number of parallel threads for test audio denoising.
N_MELS = 80                                                 # Number of Mel bands to generate in the Mel-spectrogram. Do not edit it, the FSMN_VAD model accept the 80 only.
NFFT = 400                                                  # Number of FFT components for the STFT process, edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
SAMPLE_RATE = 16000                                         # The FSMN_VAD parameter, do not edit the value.
LFR_M = 5                                                   # The FSMN_VAD parameter, do not edit the value.
LFR_N = 1                                                   # The FSMN_VAD parameter, do not edit the value.
PRE_EMPHASIZE = 0.95                                        # For audio preprocessing.
SPEECH_2_NOISE_RATIO = 1.0                                  # The judge factor for VAD model, edit it carefully.
ONE_MINUS_SPEECH_THRESHOLD = 0.15                           # The judge factor for the VAD model, edit it carefully. A higher value increases sensitivity but may mistakenly classify noise as speech. When using denoised audio, this value could be approximately 0.02
SNR_THRESHOLD = 10.0                                        # The judge factor for VAD model. Unit: dB.
BACKGROUND_NOISE_dB_INIT = 30.0                             # An initial value for the background. More smaller values indicate a quieter environment. Unit: dB. When using denoised audio, set this value to be smaller.
FUSION_THRESHOLD = 1.5                                      # A judgment factor used to merge timestamps: if two speech segments are too close, they are combined into one. Unit: second.
MIN_SPEECH_DURATION = 0.5                                   # A judgment factor used to filter the vad results. Unit: second.
ACTIVATE_SCORE = 0.5                                        # A judgment factor used to judge the state is active or not.


STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1   # Reference audio length after STFT processed
LFR_LENGTH = (STFT_SIGNAL_LENGTH + LFR_N - 1) // LFR_N
if NFFT > INPUT_AUDIO_LENGTH:
    NFFT = INPUT_AUDIO_LENGTH
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH


shutil.copyfile(modified_path + "encoder.py", python_fsmn_vad_path)
from funasr import AutoModel


class FSMN_VAD(torch.nn.Module):
    def __init__(self, fsmn_vad, stft_model, nfft, n_mels, sample_rate, pre_emphasis, lfr_m, lfr_n, lfr_len, ref_len, speech_2_noise_ratio, input_audio_len, hpp_len, use_pcm_int16):
        super(FSMN_VAD, self).__init__()
        self.fsmn_vad = fsmn_vad
        self.stft_model = stft_model
        self.use_pcm_int16 = use_pcm_int16
        self.speech_2_noise_ratio = speech_2_noise_ratio
        self.pre_emphasis = torch.tensor(pre_emphasis, dtype=torch.float32)
        self.inv_int16 = 1.0 / 32768.0
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft // 2 + 1, 20, 20000, n_mels, sample_rate, None,'htk')).transpose(0, 1).unsqueeze(0)
        self.lfr_m_factor = (lfr_m - 1) // 2
        self.T_lfr = lfr_len
        indices = torch.arange(0, self.T_lfr * lfr_n, lfr_n, dtype=torch.int32).unsqueeze(1) + torch.arange(lfr_m, dtype=torch.int32)
        self.indices_mel = indices.clamp(max=ref_len + self.lfr_m_factor - 1)  # Ensure no out-of-bounds access
        self.indices_audio = torch.arange(nfft, dtype=torch.int32) + torch.arange(0, input_audio_len - nfft + 1, hpp_len, dtype=torch.int32).unsqueeze(-1)
        self.inv_reference_air_pressure_square = 2500000000.0 / input_audio_len

    def forward(self, audio, cache_0, cache_1, cache_2, cache_3, one_minus_speech_threshold, noise_average_dB):
        if self.use_pcm_int16:
            audio = self.inv_int16 * audio.float()
        audio = torch.cat((audio[:, :, :1], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]), dim=-1)  # Pre Emphasize
        audio -= torch.mean(audio)  # Remove DC Offset
        real_part, imag_part = self.stft_model(audio, 'constant')
        mel_features = torch.matmul(self.fbank, torch.sqrt(real_part * real_part + imag_part * imag_part)).transpose(1, 2).clamp(min=1e-5).log()
        left_padding = mel_features[:, [0], :].repeat(1, self.lfr_m_factor, 1)
        padded_inputs = torch.cat((left_padding, mel_features), dim=1)
        mel_features = padded_inputs[:, self.indices_mel].reshape(1, self.T_lfr, -1)  # Merge time and feature dims
        score, cache_0, cache_1, cache_2, cache_3 = self.fsmn_vad.forward(mel_features, cache_0, cache_1, cache_2, cache_3)
        if self.speech_2_noise_ratio > 1.0:
            score += torch.pow(score, self.speech_2_noise_ratio)
        elif self.speech_2_noise_ratio < 1.0:
            score += 1.0
        else:
            score += score
        audio = audio.squeeze()[self.indices_audio]
        power_db = 10.0 * torch.log10(torch.sum(audio * audio, dim=-1) * self.inv_reference_air_pressure_square + 0.00002)
        padding = power_db[-1:].expand((score.shape[-1] - power_db.shape[-1]))
        power_db = torch.cat((power_db, padding), dim=-1)
        condition = (score <= one_minus_speech_threshold)
        speaking_db = power_db[torch.where((condition & (power_db >= noise_average_dB)))]
        noisy_dB = power_db[torch.where(~condition)].mean()
        score = speaking_db.shape[-1] / score.shape[-1]
        return score, cache_0, cache_1, cache_2, cache_3, noisy_dB


print('Export start ...')
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, n_mels=N_MELS, hop_len=HOP_LENGTH, max_frames=1, window_type=WINDOW_TYPE).eval()  # The max_frames is not the key parameter for STFT, but it is for ISTFT.
    fsmn_vad = FSMN_VAD(AutoModel.build_model(model=model_path, device='cpu')[0].encoder.eval(), custom_stft, NFFT,
                        N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, LFR_M, LFR_N, LFR_LENGTH, STFT_SIGNAL_LENGTH, SPEECH_2_NOISE_RATIO, INPUT_AUDIO_LENGTH, HOP_LENGTH, USE_PCM_INT16)
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16 if USE_PCM_INT16 else torch.float32)
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
model_type = ort_session_A._inputs_meta[4].type
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
while slice_start + stride_step < aligned_len:
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
        speaking = True
    else:
        speaking = False
        noise_average_dB = 0.5 * (noise_average_dB + noisy_dB) + SNR_THRESHOLD
    saved.append(speaking)
    print(f"Complete: {slice_start * inv_audio_len:.2f}%")
    slice_start += stride_step
end_time = time.time()

# Generate timestamps.
timestamps = vad_to_timestamps(saved, INPUT_AUDIO_LENGTH / SAMPLE_RATE, FUSION_THRESHOLD, MIN_SPEECH_DURATION)
print(f"Complete: 100.00%\n\nTimestamps in Second:")

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
