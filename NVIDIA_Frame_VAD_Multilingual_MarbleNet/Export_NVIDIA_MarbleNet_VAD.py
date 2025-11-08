import gc
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

model_path = "/home/DakeQQ/Downloads/frame_vad_multilingual_marblenet_v2.0.nemo"      # The NVIDIA VAD download path.
onnx_model_A = "/home/DakeQQ/Downloads/NVIDIA_MarbleNet_ONNX/NVIDIA_MarbleNet.onnx"   # The exported onnx model path.
test_vad_audio = "./vad_sample.wav"                                                   # The test audio path.
save_timestamps_second = "./timestamps_second.txt"                                    # The saved path.
save_timestamps_indices = "./timestamps_indices.txt"                                  # The saved path.


ORT_Accelerate_Providers = []                               # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                                            # else keep empty.
DYNAMIC_AXES = True                                         # The default dynamic axis is input audio length.
SAMPLE_RATE = 16000                                         # The model parameter, do not edit the value.
OUTPUT_FRAME_LENGTH = 320                                   # The model parameter, do not edit it.
INPUT_AUDIO_LENGTH = 160000                                 # The max length of input audio segment. For DYNAMIC_AXES=False.
WINDOW_TYPE = 'hann'                                        # Type of window function used in the STFT
N_MELS = 80                                                 # Number of Mel bands to generate in the Mel-spectrogram. Do not edit it.
NFFT_STFT = 512                                             # Number of FFT components for the STFT process, edit it carefully.
WINDOW_LENGTH = 400                                         # Length of windowing, edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
PRE_EMPHASIZE = 0.97                                        # For audio preprocessing.
FUSION_THRESHOLD = 0.1                                      # A judgment factor used to merge timestamps: if two speech segments are too close, they are combined into one. Unit: second.
MIN_SPEECH_DURATION = 0.05                                  # A judgment factor used to filter the vad results. Unit: second.
SPEAKING_SCORE = 0.5                                        # A judgment factor used to determine whether the state is speaking or not. A larger value makes activation more difficult.
SILENCE_SCORE = 0.5                                         # A judgment factor used to determine whether the state is silent or not. A smaller value makes it easier to cut off speaking.


shutil.copyfile('./modeling_modified/common.py', site.getsitepackages()[-1] + "/nemo/core/classes/common.py")
import nemo.collections.asr as nemo_asr


class NVIDIA_VAD(torch.nn.Module):
    def __init__(self, nvidia_vad, stft_model, nfft_stft, n_mels, sample_rate, pre_emphasis):
        super(NVIDIA_VAD, self).__init__()
        self.nvidia_vad = nvidia_vad
        self.stft_model = stft_model
        self.pre_emphasis = float(pre_emphasis)
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft_stft // 2 + 1, 0, sample_rate // 2, n_mels, sample_rate, 'slaney','slaney')).transpose(0, 1).unsqueeze(0)
        self.inv_int16 = float(1.0 / 32768.0)
        for i in self.nvidia_vad.encoder.encoder:
            for j in i.mconv:
                j.use_mask = False
            if i.res:
                for j in i.res:
                    for k in j:
                        k.use_mask = False

    def forward(self, audio):
        audio = audio.float() * self.inv_int16
        audio = audio - torch.mean(audio)  # Remove DC Offset
        if self.pre_emphasis > 0:
            audio = torch.cat([audio[:, :, :1], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]], dim=-1)
        real_part, imag_part = self.stft_model(audio, 'constant')
        mel_features = (torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part) + 1e-7).log()
        encoded, signal_len = self.nvidia_vad.encoder(([mel_features], mel_features.shape[-1].unsqueeze(0)))
        score = torch.softmax(self.nvidia_vad.decoder(encoded.transpose(1, 2)), dim=-1)
        score_silence, score_active = torch.split(score, [1, 1], dim=-1)
        return score_silence, score_active, signal_len.int() - 1


print('Export start ...')
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()  # The max_frames is not the key parameter for STFT, but it is for ISTFT.
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=model_path, strict=False)
    model = model.to('cpu').float().eval()
    model = NVIDIA_VAD(model, custom_stft, NFFT_STFT, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE)
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    torch.onnx.export(
        model,
        (audio,),
        onnx_model_A,
        input_names=['audio'],
        output_names=['score_silence', 'score_active', 'signal_len'],
        do_constant_folding=True,
        dynamic_axes={
            'audio': {2: 'audio_len'},
            'score_active': {1: 'signal_len'},
            'score_silence': {1: 'signal_len'}
        } if DYNAMIC_AXES else None,
        opset_version=17,
        dynamo=False
    )
    del model
    del audio
    del custom_stft
    gc.collect()
print('\nExport done!\n\nStart to run NVIDIA_VAD by ONNX Runtime.\n\nNow, loading the model...')


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4         # Fatal level, it an adjustable value.
session_opts.log_verbosity_level = 4        # Fatal level, it an adjustable value.
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
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A0 = out_name_A[0].name
out_name_A1 = out_name_A[1].name
out_name_A2 = out_name_A[2].name


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


# # Load the input audio
print(f"\nTest Input Audio: {test_vad_audio}")
audio = np.array(AudioSegment.from_file(test_vad_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
audio = normalize_to_int16(audio)
audio_len = len(audio)
inv_audio_len = float(100.0 / audio_len)
audio = audio.reshape(1, 1, -1)

shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = min(SAMPLE_RATE * 3600, audio_len)  # You can adjust it.
else:
    INPUT_AUDIO_LENGTH = shape_value_in

stride_step = INPUT_AUDIO_LENGTH
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


# Start to run NVIDIA_VAD
slice_start = 0
slice_end = INPUT_AUDIO_LENGTH
silence = True
saved = []
print("\nRunning the NVIDIA_VAD by ONNX Runtime.")
start_time = time.time()
while slice_end <= aligned_len:
    score_silence, score_active, signal_len = ort_session_A.run(
        [out_name_A0, out_name_A1, out_name_A2], {in_name_A0: audio[:, :, slice_start: slice_end]})
    for i in range(signal_len[0]):
        if silence:
            if score_active[:, i] >= SPEAKING_SCORE:
                silence = False
        else:
            if score_silence[:, i] >= SILENCE_SCORE:
                silence = True
        saved.append(silence)
    slice_start += stride_step
    slice_end = slice_start + INPUT_AUDIO_LENGTH
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
