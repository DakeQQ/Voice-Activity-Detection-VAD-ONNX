# Voice-Activity-Detection-VAD-ONNX
Utilizes ONNX Runtime for speech activity detection.
1. Now support:
   - [FSMN](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)
2. It is recommended to use FSMN with [denoised model](https://github.com/DakeQQ/Audio-Denoiser-ONNX).
3. The FSMN model is essentially a silence detection model, so it activates in response to any sound, not just human voices.
4. This is an end-to-end version that includes the STFT process. Simply pass the audio and get the timestamps as output.
5. [Download](https://drive.google.com/drive/folders/1htM4FYpxEQcouHiR2Wyb407EhD1t_0HB?usp=sharing)
6. See more -> https://dakeqq.github.io/overview/
  


# Audio-Denoiser-ONNX
1. 现在支持:
   - [FSMN](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)
2. 建议将FSMN与[降噪模型](https://github.com/DakeQQ/Audio-Denoiser-ONNX)一起使用。
3. FSMN模型本质上是一种静音检测模型，因此它会对任何声音作出反应，而不仅仅是人声。
4. 这是一个包含 STFT 过程的端到端版本。只需传入音频，即可获得时间戳输出。
5. [下载](https://drive.google.com/drive/folders/1htM4FYpxEQcouHiR2Wyb407EhD1t_0HB?usp=sharing)
6. See more -> https://dakeqq.github.io/overview/


# 性能 Performance
| OS | Device | Backend | Model | Real-Time Factor<br>( Chunk_Size: 1600 or 100ms ) |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Ubuntu-24.04 | Laptop | CPU<br>i5-7300HQ | FSMN<br>f32 | 0.0047 |
| Ubuntu-24.04 | Desktop | CPU<br>i3-12300 | FSMN<br>f32 | 0.0018 |
