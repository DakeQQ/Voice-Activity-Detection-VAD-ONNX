---

## Voice-Activity-Detection-VAD-ONNX  
Speech activity detection powered by ONNX Runtime for high-performance applications.  

### Features  
1. **Supported Model**:  
   - [FSMN](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)
   - [Silero](https://github.com/snakers4/silero-vad)  (Optimized for enhanced parallel computing performance)

2. **Recommendation and Note**:  
   - It is recommended to use the [Audio Denoiser](https://github.com/DakeQQ/Audio-Denoiser-ONNX) for optimal performance in noisy environments.
   - FSMN inference relies on previous time steps, so longer audio will take more time than the real-time factor suggests.

3. **End-to-End Processing**:  
   - This model includes internal `STFT` processing.  
   - Input: Raw audio  
   - Output: Detected speech timestamps  

4. **Resources**:  
   - [Download Models](https://drive.google.com/drive/folders/1htM4FYpxEQcouHiR2Wyb407EhD1t_0HB?usp=sharing)  
   - [Explore More Projects](https://dakeqq.github.io/overview/)  

---

### 性能 Performance  
| OS           | Device       | Backend           | Model        | Real-Time Factor <br> (Chunk Size: 512 or 32ms) |
|:------------:|:------------:|:-----------------:|:------------:|:------------------------------------------------:|
| Ubuntu-24.04 | Laptop       | CPU <br> i5-7300HQ | FSMN <br> f32 | 0.0047                                            |
| Ubuntu-24.04 | Desktop      | CPU <br> i3-12300 | FSMN <br> f32  | 0.0018                                            |  
| Ubuntu-24.04 | Desktop      | CPU <br> i3-12300 | Silero <br> f32  | 0.0026                                          |  

---

## Voice-Activity-Detection-VAD-ONNX  
通过 ONNX Runtime 实现高性能的语音活动检测。  

### 功能  
1. **支持的模型**：  
   - [FSMN](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)
   - [Silero](https://github.com/snakers4/silero-vad)  (优化了并行计算性能)

2. **推荐与注意**：  
   - 建议与 [音频降噪器](https://github.com/DakeQQ/Audio-Denoiser-ONNX) 搭配使用，以在嘈杂环境中获得最佳性能。
   - FSMN推理依赖于前一个时间步，因此较长的音频将比实时因子所显示的耗时更多。

3. **端到端处理**：  
   - 模型包含内部 `STFT` 处理。  
   - 输入：原始音频  
   - 输出：检测到的语音时间戳  

4. **资源**：  
   - [下载模型](https://drive.google.com/drive/folders/1htM4FYpxEQcouHiR2Wyb407EhD1t_0HB?usp=sharing)  
   - [探索更多项目](https://dakeqq.github.io/overview/)  

---
