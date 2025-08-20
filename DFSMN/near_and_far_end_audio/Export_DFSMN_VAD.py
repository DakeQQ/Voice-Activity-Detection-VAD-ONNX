import gc
import site
import shutil
import time
import numpy as np
import onnxruntime
import torch
import torchaudio
from datetime import timedelta
from pydub import AudioSegment
from STFT_Process import STFT_Process                                    # The custom STFT/ISTFT can be exported in ONNX format.


project_path_A = "/home/DakeQQ/Downloads/speech_dfsmn_aec_psm_16k"         # The DFSMN AEC download path.
project_path_B = "/home/DakeQQ/Downloads/SDAEC-main"                       # The SDAEC github project download path. The modified DFSMN AEC need it. https://github.com/ZhaoF-i/SDAEC
onnx_model_A = "/home/DakeQQ/Downloads/DFSMN_VAD_ONNX/DFSMN_VAD.onnx"      # The exported onnx model path.
test_near_end_audio = "./examples/nearend_mic1.wav"                        # The near end audio path.
test_far_end_audio = "./examples/farend_speech1.wav"                       # The far end audio path.
save_timestamps_second = "./timestamps_second.txt"                         # The saved path.
save_timestamps_indices = "./timestamps_indices.txt"                       # The saved path.


DYNAMIC_AXES = False                    # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
MAX_SIGNAL_LENGTH = 36864 if DYNAMIC_AXES else 200  # Max frames for audio length after STFT processed. Set a appropriate larger value for long audio input, such as 36864.
INPUT_AUDIO_LENGTH = 16001              # Maximum input audio length: the length of the audio input signal (in samples) is recommended to be greater than 31841. Higher values yield better quality. It is better to set an integer multiple of the NFFT value.
SAMPLE_RATE = 16000                     # The models parameter, do not edit the value.
MAX_THREADS = 4                         # Number of parallel threads for test audio denoising.

# DFSMN_VAD
WINDOW_TYPE = 'hamming'                 # Type of window function used in the STFT
NFFT_A = 1024                           # Number of FFT components for the STFT process, edit it carefully.
NFFT_A2 = 640                           # Number of FFT components for the STFT process, edit it carefully.
WINDOW_LENGTH_A = 640                   # Length of windowing, edit it carefully.
HOP_LENGTH_A = 320                      # Number of samples between successive frames in the STFT
N_MELS = 80                             # Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
PRE_EMPHASIZE = 0.97                    # For audio preprocessing.
OUTPUT_FRAME_LENGTH = 320               # The DFSMN_VAD parameter, do not edit the value.

# SDAEC
WINDOW_TYPE_B = 'hamming'               # Type of window function used in the STFT
NFFT_B = 319                            # Number of FFT components for the STFT process
WINDOW_LENGTH_B = 319                   # Length of windowing, edit it carefully.
HOP_LENGTH_B = 160                      # Number of samples between successive frames in the STFT
ALPHA_K = 10                            # The SDAEC parameter, do not edit the value.

# VAD Settings
FUSION_THRESHOLD = 0.3                  # A judgment factor used to merge timestamps: if two speech segments are too close, they are combined into one. Unit: second.
MIN_SPEECH_DURATION = 0.2               # A judgment factor used to filter the vad results. Unit: second.
SPEAKING_SCORE = 0.5                    # A judgment factor used to determine whether the state is speaking or not. A larger value makes activation more difficult.
SILENCE_SCORE = 0.5                     # A judgment factor used to determine whether the state is silent or not. A smaller value makes it easier to cut off speaking.
LOOK_BACKWARD = 0.3                     # Utilize future Voice Activity Detection (VAD) results to assess whether the current index indicates silence. Unit: second. Must be an integer multiple of 0.02.


shutil.copyfile('./modeling_modified/uni_deep_fsmn.py', site.getsitepackages()[-1] + '/modelscope/models/audio/aec/layers/uni_deep_fsmn.py')
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


class AlphaPredictor(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 1)
        self.linear2 = torch.nn.Linear(k, 1)
        # self.ReLU = nn.ReLU()

    def forward(self, mix_comp, far_comp, k):
        pass


class CFB(torch.nn.Module):
    def __init__(self, in_channels=None, out_channels=None):
        super(CFB, self).__init__()
        self.conv_gate = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1, padding=(0, 0), dilation=1, groups=1, bias=True)
        self.conv_input = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1, padding=(0, 0), dilation=1, groups=1, bias=True)
        self.conv = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0), dilation=1, groups=1, bias=True)
        self.ceps_unit = CepsUnit(ch=out_channels)
        self.LN0 = LayerNorm(in_channels, f=160)
        self.LN1 = LayerNorm(out_channels, f=160)
        self.LN2 = LayerNorm(out_channels, f=160)

    def forward(self, x):
        g = torch.sigmoid(self.conv_gate(self.LN0(x)))
        x = self.conv_input(x)
        gx = g * x
        y = self.conv(self.LN1(gx))
        y = y + self.ceps_unit(self.LN2(x - gx))
        return y


class CepsUnit(torch.nn.Module):
    def __init__(self, ch):
        super(CepsUnit, self).__init__()
        self.ch = ch
        self.ch_lstm_f = CH_LSTM_F(ch * 2, ch, ch * 2, f=81)
        self.LN = LayerNorm(ch * 2, f=81)
        self.f = 81

        # --- Pre-computation and buffer registration ---
        n_fft = NFFT_B // 2 + 1
        self.n_fft = n_fft
        self.hop_len = n_fft
        win_length = n_fft
        half_n_fft = n_fft // 2

        # STFT Kernels
        window = torch.ones(win_length)
        time_idx = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
        freq_idx = torch.arange(half_n_fft + 1, dtype=torch.float32).unsqueeze(1)
        omega = 2 * torch.pi * freq_idx * time_idx / n_fft
        cos_kernel = (torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1)
        sin_kernel = (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1)

        self.register_buffer('cos_kernel', cos_kernel)
        self.register_buffer('sin_kernel', sin_kernel)

        # ISTFT Basis
        fourier_basis = torch.fft.fft(torch.eye(n_fft, dtype=torch.float32))
        fourier_basis_real_imag = torch.vstack([
            torch.real(fourier_basis[:half_n_fft + 1]),
            torch.imag(fourier_basis[:half_n_fft + 1])
        ]).float()
        inverse_basis_pinv = torch.linalg.pinv(fourier_basis_real_imag).T
        inverse_basis = window.unsqueeze(0) * inverse_basis_pinv.unsqueeze(1)

        self.register_buffer('inverse_basis', inverse_basis)

    def forward(self, x0):
        # --- STFT using pre-computed kernels ---
        x_reshaped = x0.permute(0, 1, 3, 2).contiguous().view(-1, 1, self.n_fft)
        real_part_stft = torch.nn.functional.conv1d(x_reshaped, self.cos_kernel, stride=self.hop_len)
        imag_part_stft = torch.nn.functional.conv1d(x_reshaped, self.sin_kernel, stride=self.hop_len)
        stft_real = real_part_stft.view(1, self.ch, -1, self.f).permute(0, 1, 3, 2).contiguous()
        stft_imag = imag_part_stft.view(1, self.ch, -1, self.f).permute(0, 1, 3, 2).contiguous()

        # --- CepsUnit Logic ---
        lstm_input = torch.cat([stft_real, stft_imag], 1)
        lstm_output = self.ch_lstm_f(self.LN(lstm_input))
        processed_real, processed_imag = lstm_output[:, :self.ch], lstm_output[:, self.ch:]
        out_real = processed_real * stft_real - processed_imag * stft_imag
        out_imag = processed_real * stft_imag + processed_imag * stft_real

        # --- ISTFT using pre-computed basis ---
        real_part_istft = out_real.permute(0, 1, 3, 2).contiguous().view(-1, self.f, 1)
        imag_part_istft = out_imag.permute(0, 1, 3, 2).contiguous().view(-1, self.f, 1)
        inp = torch.cat((real_part_istft, imag_part_istft), dim=1)
        inv = torch.nn.functional.conv_transpose1d(inp, self.inverse_basis, stride=self.hop_len)
        x_out = inv.view(1, self.ch, -1, self.n_fft).permute(0, 1, 3, 2).contiguous()
        return x_out


class LayerNorm(torch.nn.Module):
    def __init__(self, c, f):
        super(LayerNorm, self).__init__()
        self.w = torch.nn.Parameter(torch.ones(1, c, f, 1))
        self.b = torch.nn.Parameter(torch.rand(1, c, f, 1) * 1e-4)

    def forward(self, x):
        mean = x.mean([1, 2], keepdim=True)
        std = x.std([1, 2], keepdim=True)
        x = (x - mean) / (std + 1e-6) * self.w + self.b
        return x


class NET(torch.nn.Module):
    def __init__(self, order=10, channels=20, max_frames=2048):
        super().__init__()
        self.act = torch.nn.ELU()
        self.order = order

        # STFT/ISTFT parameters
        self.n_fft = NFFT_B
        self.half_n_fft = self.n_fft // 2
        self.hop_length = HOP_LENGTH_B
        self.win_length = WINDOW_LENGTH_B
        self.hop_length_2 = self.hop_length + self.hop_length
        self.window = torch.hamming_window(self.win_length)

        # --- ISTFT basis pre-computation ---

        fourier_basis_eye = torch.fft.fft(torch.eye(self.n_fft, dtype=torch.float32))
        fourier_basis = torch.vstack([
            torch.real(fourier_basis_eye[:self.half_n_fft + 1]),
            torch.imag(fourier_basis_eye[:self.half_n_fft + 1])
        ]).float()
        pinv_transposed = torch.linalg.pinv((fourier_basis * self.n_fft) / self.hop_length).T
        inverse_basis_kernel = pinv_transposed.unsqueeze(1)
        inverse_basis = inverse_basis_kernel * self.window.view(1, 1, -1)
        self.register_buffer('inverse_basis', inverse_basis)

        # --- Pre-computation of overlap-add normalization buffer ---
        max_output_len = (max_frames - 1) * self.hop_length + self.n_fft
        window_sum = torch.zeros(max_output_len, dtype=torch.float32)
        win_sq = self.window ** 2
        for i in range(max_frames):
            s = i * self.hop_length
            win_len = min(self.n_fft, max_output_len - s)
            if win_len <= 0: break
            window_sum[s:s + win_len] += win_sq[:win_len]
        window_sum_inv = self.n_fft / (window_sum * self.hop_length + 1e-6)
        self.register_buffer('window_sum_inv', window_sum_inv)

        # --- Model Layers ---
        self.in_ch_lstm = CH_LSTM_F(4, channels, channels)
        self.in_conv = torch.nn.Conv2d(in_channels=4 + channels, out_channels=channels, kernel_size=(1, 1))
        self.cfb_e1 = CFB(channels, channels)
        self.cfb_e2 = CFB(channels, channels)
        self.cfb_e3 = CFB(channels, channels)
        self.cfb_e4 = CFB(channels, channels)
        self.cfb_e5 = CFB(channels, channels)
        self.ln = LayerNorm(channels, 160)
        self.ch_lstm = CH_LSTM_T(in_ch=channels, feat_ch=channels * 2, out_ch=channels, num_layers=2)
        self.cfb_d5 = CFB(1 * channels, channels)
        self.cfb_d4 = CFB(2 * channels, channels)
        self.cfb_d3 = CFB(2 * channels, channels)
        self.cfb_d2 = CFB(2 * channels, channels)
        self.cfb_d1 = CFB(2 * channels, channels)
        self.out_ch_lstm = CH_LSTM_T(2 * channels, channels, channels * 2)
        self.out_conv = torch.nn.Conv2d(in_channels=channels * 3, out_channels=2, kernel_size=(1, 1), padding=(0, 0), bias=True)

    def istft(self, Y):
        inv = torch.nn.functional.conv_transpose1d(Y.reshape(1, self.hop_length_2, -1), self.inverse_basis, stride=self.hop_length)
        e = inv.size(-1) - self.half_n_fft
        y_out = inv[..., self.half_n_fft:e] * self.window_sum_inv[self.half_n_fft:e]
        return y_out, e - self.half_n_fft

    def forward(self, x):
        e0 = self.in_ch_lstm(x)
        e0 = self.in_conv(torch.cat([e0, x], 1))
        e1 = self.cfb_e1(e0)
        e2 = self.cfb_e2(e1)
        e3 = self.cfb_e3(e2)
        e4 = self.cfb_e4(e3)
        e5 = self.cfb_e5(e4)
        lstm_out = self.ch_lstm(self.ln(e5))
        d5 = self.cfb_d5(torch.cat([e5 * lstm_out], dim=1))
        d4 = self.cfb_d4(torch.cat([e4, d5], dim=1))
        d3 = self.cfb_d3(torch.cat([e3, d4], dim=1))
        d2 = self.cfb_d2(torch.cat([e2, d3], dim=1))
        d1 = self.cfb_d1(torch.cat([e1, d2], dim=1))
        d0 = self.out_ch_lstm(torch.cat([e0, d1], dim=1))
        out = self.out_conv(torch.cat([d0, d1], dim=1))
        y = self.istft(out)
        return y


class CH_LSTM_T(torch.nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch, bi=False, num_layers=1, f=160):
        super().__init__()
        self.lstm2 = torch.nn.LSTM(in_ch, feat_ch, num_layers=num_layers, batch_first=True, bidirectional=bi)
        self.bi = 1 if not bi else 2
        self.linear = torch.nn.Linear(self.bi * feat_ch, out_ch)
        self.out_ch = out_ch
        self.f = f

    def forward(self, x):
        self.lstm2.flatten_parameters()
        x = x.permute(0, 2, 3, 1).contiguous().view(self.f, -1, self.lstm2.input_size)
        x, _ = self.lstm2(x)
        x = self.linear(x)
        x = x.view(1, self.f, -1, self.out_ch).permute(0, 3, 1, 2).contiguous()
        return x


class CH_LSTM_F(torch.nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch, bi=True, num_layers=1, f=160):
        super().__init__()
        self.lstm2 = torch.nn.LSTM(in_ch, feat_ch, num_layers=num_layers, batch_first=True, bidirectional=bi)
        self.linear = torch.nn.Linear(2 * feat_ch, out_ch)
        self.out_ch = out_ch
        self.f = f

    def forward(self, x):
        self.lstm2.flatten_parameters()
        x = x.permute(0, 3, 2, 1).contiguous().view(-1, self.f, self.lstm2.input_size)
        x, _ = self.lstm2(x)
        x = self.linear(x)
        x = x.view(1, -1, self.f, self.out_ch).permute(0, 3, 2, 1).contiguous()
        return x


class DFSMN_VAD(torch.nn.Module):
    def __init__(self, dfsmn_vad, iccrn, alpha_predictor, custom_stft_A, custom_stft_A2, custom_stft_B, nfft_A, nfft_B, k, max_len, pre_emphasis, sample_rate, n_mels):
        super(DFSMN_VAD, self).__init__()
        self.dfsmn_vad = dfsmn_vad.model.to('cpu').float()
        self.shift = (dfsmn_vad.preprocessor.feature.shift + torch.log(torch.tensor(32768 ** 2, dtype=torch.float32))).view(1, 1, -1)
        self.scale = dfsmn_vad.preprocessor.feature.scale.view(1, 1, -1)
        self.iccrn = iccrn
        self.alpha_predictor = alpha_predictor
        self.custom_stft_A = custom_stft_A
        self.custom_stft_A2 = custom_stft_A2
        self.custom_stft_B = custom_stft_B
        self.inv_int16 = float(1.0 / 32768.0)
        self.k = k
        self.register_buffer('sdaec_pad_zero', torch.zeros((1, 2, nfft_B // 2 + 1, k - 1), dtype=torch.float32))
        if max_len < 256:
            dtype = torch.uint8
        else:
            dtype = torch.int16
        self.frame_starts = torch.arange(max_len, dtype=dtype).unsqueeze(1) + torch.arange(k, dtype=dtype).unsqueeze(0)
        self.k_minus = k - 1
        self.pre_emphasis = torch.tensor(pre_emphasis, dtype=torch.float32)
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft_A // 2 + 1, 20, sample_rate // 2, n_mels, sample_rate, None,'htk')).transpose(0, 1).unsqueeze(0)
        self.factor = float(1.15)  # Experiment value

    def onnx_friendly_unfold(self, input_tensor):
        num_frames = input_tensor.shape[-1] - self.k_minus
        indices = self.frame_starts[:num_frames].to(torch.int64)
        unfolded = input_tensor[..., indices]
        return unfolded

    def forward(self, near_end_audio, far_end_audio):
        near_end_audio = near_end_audio.float() * self.inv_int16
        far_end_audio = far_end_audio.float() * self.inv_int16
        near_end_audio = near_end_audio - torch.mean(near_end_audio)
        far_end_audio = far_end_audio - torch.mean(far_end_audio)
        near_real_part_B, near_imag_part_B = self.custom_stft_B(near_end_audio, 'constant')
        far_real_part, far_imag_part = self.custom_stft_B(far_end_audio, 'constant')
        mix_comp = torch.cat([near_real_part_B, near_imag_part_B], dim=0).unsqueeze(0)
        far_comp = torch.cat([far_real_part, far_imag_part], dim=0).unsqueeze(0)
        mix_complex_padded = torch.cat([self.sdaec_pad_zero, mix_comp], dim=-1)
        mix_complex_unfolded = self.onnx_friendly_unfold(mix_complex_padded)
        far_complex_padded = torch.cat([self.sdaec_pad_zero, far_comp], dim=-1)
        far_complex_unfolded = self.onnx_friendly_unfold(far_complex_padded)
        pow_mix = (mix_complex_unfolded * mix_complex_unfolded).sum(dim=1, keepdim=True)
        pow_far = (far_complex_unfolded * far_complex_unfolded).sum(dim=1, keepdim=True)
        concat_input = torch.stack([pow_far, pow_mix], dim=-1).unsqueeze(dim=1)
        alpha = self.alpha_predictor.linear1(torch.sum(concat_input, dim=2, keepdim=True)).squeeze(dim=-1)
        alpha = self.alpha_predictor.linear2(alpha).squeeze(dim=-1)
        far_comp = far_comp * torch.abs(alpha)
        temp_aec, min_len = self.iccrn(torch.cat([mix_comp, far_comp.squeeze(1)], dim=1))
        near_end_audio = near_end_audio[..., :min_len]
        if self.pre_emphasis > 0:
            near_end_audio = torch.cat([near_end_audio[:, :, :1], near_end_audio[:, :, 1:] - self.pre_emphasis * near_end_audio[:, :, :-1]], dim=-1)
            temp_aec = torch.cat([temp_aec[:, :, :1], temp_aec[:, :, 1:] - self.pre_emphasis * temp_aec[:, :, :-1]], dim=-1)
        echo = near_end_audio - self.factor * temp_aec
        near_real_part_A, near_imag_part_A = self.custom_stft_A(near_end_audio, 'constant')
        temp_aec_real_part, temp_aec_imag_part = self.custom_stft_A(temp_aec, 'constant')
        echo_real_part, echo_imag_part = self.custom_stft_A(echo, 'constant')
        near_features = torch.matmul(self.fbank, near_real_part_A * near_real_part_A + near_imag_part_A * near_imag_part_A)
        temp_aec_features = torch.matmul(self.fbank, temp_aec_real_part * temp_aec_real_part + temp_aec_imag_part * temp_aec_imag_part)
        echo_features = torch.matmul(self.fbank, echo_real_part * echo_real_part + echo_imag_part * echo_imag_part)
        feat = torch.cat([near_features, temp_aec_features, echo_features], dim=1).transpose(1, 2).clamp(1e-6).log()
        feat = (feat + self.shift) * self.scale
        x1 = self.dfsmn_vad.linear1(feat)
        x2 = self.dfsmn_vad.relu(x1)
        x3 = self.dfsmn_vad.deepfsmn(x2)
        vad_results = torch.sigmoid(self.dfsmn_vad.linear3(x3)).squeeze()
        return vad_results


print('Export start ...')
with torch.inference_mode():
    custom_stft_A = STFT_Process(model_type='stft_B', n_fft=NFFT_A, hop_len=HOP_LENGTH_A, win_length=WINDOW_LENGTH_A, max_frames=0, window_type=WINDOW_TYPE).eval()
    custom_stft_A2 = STFT_Process(model_type='stft_B', n_fft=NFFT_A2, hop_len=HOP_LENGTH_A, win_length=WINDOW_LENGTH_A, max_frames=0, window_type=WINDOW_TYPE).eval()
    custom_stft_B = STFT_Process(model_type='stft_B', n_fft=NFFT_B, hop_len=HOP_LENGTH_B, win_length=WINDOW_LENGTH_B, max_frames=0, window_type=WINDOW_TYPE).eval()
    iccrn = NET(max_frames=MAX_SIGNAL_LENGTH)
    iccrn.load_state_dict(torch.load(project_path_B + '/Model/ICCRN.ckpt', map_location='cpu'), strict=False)
    iccrn = iccrn.float().eval()
    alpha_predictor = AlphaPredictor(ALPHA_K)
    alpha_predictor.load_state_dict(torch.load(project_path_B + '/Model/alpha.ckpt', map_location='cpu'), strict=False)
    alpha_predictor = alpha_predictor.float().eval()

    dfsmn_vad = pipeline(
        Tasks.acoustic_echo_cancellation,
        model=project_path_A,
        device='cpu',
        trust_remote_code=True
    )

    dfsmn_vad = DFSMN_VAD(dfsmn_vad, iccrn, alpha_predictor, custom_stft_A, custom_stft_A2, custom_stft_B, NFFT_A, NFFT_B, ALPHA_K, MAX_SIGNAL_LENGTH, PRE_EMPHASIZE, SAMPLE_RATE, N_MELS)
    near_end_audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    far_end_audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)

    torch.onnx.export(
        dfsmn_vad,
        (near_end_audio, far_end_audio),
        onnx_model_A,
        input_names=['near_end_audio', 'far_end_audio'],
        output_names=['vad_results'],
        do_constant_folding=True,
        dynamic_axes={
            'near_end_audio': {2: 'audio_len'},
            'far_end_audio': {2: 'audio_len'},
            'vad_results': {0: 'signal_len'}
        } if DYNAMIC_AXES else None,
        opset_version=17
    )
    del dfsmn_vad
    del iccrn
    del alpha_predictor
    del near_end_audio
    del far_end_audio
    gc.collect()
print('\nExport done!\n\nStart to run DFSMN_VAD by ONNX Runtime.\n\nNow, loading the model...')


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


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=["CPUExecutionProvider"], provider_options=None)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
in_name_A1 = in_name_A[1].name
out_name_A0 = out_name_A[0].name


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


# # Load the input audio
# Load the input audio
print(f"\nTest Input Near_End Audio: {test_near_end_audio}\nTest Input Far_End Audio: {test_far_end_audio}")
near_end_audio = np.array(AudioSegment.from_file(test_near_end_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
far_end_audio = np.array(AudioSegment.from_file(test_far_end_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
near_end_audio_len = len(near_end_audio)
far_nd_audio_len = len(far_end_audio)
min_len = min(near_end_audio_len, far_nd_audio_len)
near_end_audio = normalize_to_int16(near_end_audio[:min_len])
far_end_audio = normalize_to_int16(far_end_audio[:min_len])
near_end_audio = near_end_audio.reshape(1, 1, -1)
far_end_audio = far_end_audio.reshape(1, 1, -1)

shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
shape_value_out = ort_session_A._outputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = min(SAMPLE_RATE * 360, min_len)  # You can adjust it.
else:
    INPUT_AUDIO_LENGTH = shape_value_in


look_backward = int(LOOK_BACKWARD * SAMPLE_RATE // OUTPUT_FRAME_LENGTH)


def align_audio(audio, audio_len):
    stride_step = INPUT_AUDIO_LENGTH - (look_backward + 1) * OUTPUT_FRAME_LENGTH
    if audio_len > INPUT_AUDIO_LENGTH:
        num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
        total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
        pad_amount = total_length_needed - audio_len
        final_slice = audio[:, :, -pad_amount:].astype(np.float32)
        white_noise = (np.sqrt(np.mean(final_slice * final_slice, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio.dtype)
        audio = np.concatenate((audio, white_noise), axis=-1)
    elif audio_len < INPUT_AUDIO_LENGTH:
        audio_float = audio.astype(np.float32)
        white_noise = (np.sqrt(np.mean(audio_float * audio_float, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
        audio = np.concatenate((audio, white_noise), axis=-1)
    aligned_len = audio.shape[-1]
    return audio, aligned_len, stride_step


near_end_audio, _, _ = align_audio(near_end_audio, min_len)
far_end_audio, aligned_len, stride_step = align_audio(far_end_audio, min_len)
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
print("\nRunning the DFSMN_VAD by ONNX Runtime.\n\nThe VAD result is focused on the Near-End microphone recording.\n")
start_time = time.time()
while slice_end <= aligned_len:
    vad_results = ort_session_A.run([out_name_A0], {in_name_A0: near_end_audio[:, :, slice_start: slice_end], in_name_A1: far_end_audio[:, :, slice_start: slice_end]})[0]
    for i in range(len(vad_results) - look_backward):
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
