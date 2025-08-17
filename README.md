---

## Voice-Activity-Detection-VAD-ONNX  
Speech activity detection powered by ONNX Runtime for high-performance applications.  

### Features  
1. **Supported Model**:
   - [NVIDIA Frame-VAD Multilingual MarbleNet v2.0](https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0)
   - [FSMN](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)
   - [Silero](https://github.com/snakers4/silero-vad)  (Optimized for enhanced parallel computing performance)
   - [DFSMN](https://modelscope.cn/models/iic/speech_dfsmn_aec_psm_16k) (Reuse the Acoustic Echo Cancellation model. Accepts Near-End and Far-End audio as inputs. Output the Near-End VAD result only.)

3. **Recommendation and Note**:  
   - It is recommended to use the [Audio Denoiser](https://github.com/DakeQQ/Audio-Denoiser-ONNX) for optimal performance in noisy environments.

4. **End-to-End Processing**:  
   - This model includes internal `STFT` processing.  
   - Input: Raw audio  
   - Output: Detected speech timestamps  

5. **Resources**:  
   - [Explore More Projects](https://github.com/DakeQQ?tab=repositories)  

---
### 输出示例 Example Output
```python
Timestamps in Second:
00:00:00.500 --> 00:00:02.500
00:00:04.200 --> 00:00:05.599

Timestamps in Indices:
8000 --> 40000
67200 --> 89600
```

### 性能 Performance  
| OS           | Device       | Backend           | Model        | Real-Time Factor <br> (Chunk Size: 512 or 32ms) |
|:------------:|:------------:|:-----------------:|:------------:|:------------------------------------------------:|
| Ubuntu-24.04 | Desktop      | CPU <br> i3-12300 | FSMN <br> f32  | 0.0047                                         |  
| Ubuntu-24.04 | Desktop      | CPU <br> i3-12300 | Silero <br> f32  | 0.0026                                       |  
| Ubuntu-24.04 | Desktop      | CPU <br> i7-1165G7 | NVIDIA-VAD <br> f32  | 0.0005 (Chunk Size: 89000)              |  
| Ubuntu-24.04 | Desktop      | CPU <br> i7-1165G7 | DFSMN-VAD <br> f32  | 0.27 (Chunk Size: 31841)                 |  

---

## To Do List
- [Monotonic-Aligner](https://modelscope.cn/models/iic/speech_timestamp_prediction-v1-16k-offline/summary)
- [Pyannote-Segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

## Voice-Activity-Detection-VAD-ONNX  
通过 ONNX Runtime 实现高性能的语音活动检测。  

### 功能  
1. **支持的模型**：
   - [NVIDIA Frame-VAD Multilingual MarbleNet v2.0](https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0)
   - [FSMN](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)
   - [Silero](https://github.com/snakers4/silero-vad)  (优化了并行计算性能)
   - [DFSMN](https://modelscope.cn/models/iic/speech_dfsmn_aec_psm_16k) (复用回声消除模型。接收近端和远端音频作为输入。仅输出近端 VAD 结果。)


3. **推荐与注意**：  
   - 建议与 [音频降噪器](https://github.com/DakeQQ/Audio-Denoiser-ONNX) 搭配使用，以在嘈杂环境中获得最佳性能。

4. **端到端处理**：  
   - 模型包含内部 `STFT` 处理。  
   - 输入：原始音频  
   - 输出：检测到的语音时间戳  

5. **资源**：  
   - [探索更多项目](https://github.com/DakeQQ?tab=repositories)  

---
