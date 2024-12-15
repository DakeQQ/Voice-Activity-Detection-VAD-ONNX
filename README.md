# Voice-Activity-Detection-VAD-ONNX
Utilizes ONNX Runtime for speech activity detection.
1. Now support:
   - [FSMN](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)
2. It is recommended to use FSMN with denoised model.
3. [Download](https://drive.google.com/drive/folders/1htM4FYpxEQcouHiR2Wyb407EhD1t_0HB?usp=sharing)
4. See more -> https://dakeqq.github.io/overview/
  


# Audio-Denoiser-ONNX
1. 现在支持:
   - [FSMN](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)
2. 建议将FSMN与降噪模型一起使用。
3. [下载](https://drive.google.com/drive/folders/1htM4FYpxEQcouHiR2Wyb407EhD1t_0HB?usp=sharing)
4. See more -> https://dakeqq.github.io/overview/


# 性能 Performance
| OS | Device | Backend | Model | Real-Time Factor<br>( Chunk_Size: 1600 or 100ms ) |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Ubuntu-24.04 | Desktop | CPU<br>i5-7300HQ | FSMN<br>f32 | 0.0047 |
