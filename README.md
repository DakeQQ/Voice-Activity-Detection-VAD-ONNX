# Voice-Activity-Detection-VAD-ONNX
Utilizes ONNX Runtime for speech activity detection.
1. Now support:
   - [FSMN](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)
2. It is recommended to use FSMN with denoised model.
  


# Audio-Denoiser-ONNX
1. 现在支持:
   - [FSMN](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)
2. 建议将FSMN与降噪模型一起使用。


# 性能 Performance
| OS | Device | Backend | Model | Real-Time Factor<br>( Chunk_Size: 1600 or 100ms ) |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Ubuntu-24.04 | Desktop | CPU<br>i5-7300HQ | FSMN<br>f32 | 0.0047 |
