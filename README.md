<div align="center">

# Voice-Activity-Detection-VAD-ONNX

[![License: MIT](https://img.shields.io/badge/License-MIT-2563EB.svg)](LICENSE)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-005CED.svg)](https://onnxruntime.ai)
![Pipelines](https://img.shields.io/badge/Pipelines-8-0F766E.svg)
![Input](https://img.shields.io/badge/Input-Raw_Waveform-16A34A.svg)

</div>

<p align="center">
  <a href="#english"><strong>English</strong></a>
  &nbsp;·&nbsp;
  <a href="#zh-cn"><strong>简体中文</strong></a>
</p>

---

<a id="english"></a>

## English

### Raw waveform in, speech events out

End-to-end ONNX Runtime pipelines for voice activity detection, audio event detection, streaming VAD, turn-completion detection, and speaker diarization. Audio preprocessing and feature extraction live inside the deployment graphs, so applications can work directly with waveform samples.

- **8 pipelines** covering offline VAD, streaming VAD, AED, turn completion, and speaker diarization.
- **Raw-audio interfaces** with model-specific fixed, dynamic, cache, or recurrent-state contracts.
- **Portable deployment facts** stored in required metadata carriers and checked before execution.
- **Optimized runtime paths** with execution-provider controls and persistent ONNX Runtime I/O binding.

> [!IMPORTANT]
> A main ONNX model is not a complete deployment package. Keep it beside its matching metadata carrier.

---

### Supported pipelines

| Task | Pipeline | Runtime form | Result | Links |
|---|---|---|---|---|
| VAD | DFSMN: near + far end | Offline, dual input | Near-end speech timestamps | [Code](DFSMN/near_and_far_end_audio/) · [Source](https://modelscope.cn/models/iic/speech_dfsmn_aec_psm_16k) |
| VAD | DFSMN: near end only | Offline, single input | Speech timestamps | [Code](DFSMN/only_near_end_audio/) |
| VAD | FSMN | Cache-based, fixed or dynamic | Timestamps and frame scores | [Code](FSMN/) · [Source](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) |
| VAD · AED | FireRedVAD | Offline and streaming | Timestamps or task scores | [Code](FireRedVAD/) · [Source](https://github.com/FireRedTeam/FireRedVAD) |
| VAD | NVIDIA MarbleNet | Multilingual, frame level | Silence/active scores and timestamps | [Code](NVIDIA_Frame_VAD_Multilingual_MarbleNet/) · [Source](https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0) |
| VAD | Silero | Stateful chunks | Speech probability and timestamps | [Code](Silero/) · [Source](https://github.com/snakers4/silero-vad) |
| Turn completion | SmartTurn v3 | Fixed 8-second mono context | `P(complete)` | [Code](SmartTurn/v3/) · [Source](https://modelscope.cn/models/pipecat-ai/smart-turn-v3) |
| Speaker diarization | Pyannote Community-1 | Two deployable graphs | Speaker turns and embeddings | [Code](Pyannote-Diarization-Community-1/) |

Pyannote Community-1 is a related diarization pipeline, not a VAD model.

---

### Input and runtime contract

| Area | Contract |
|---|---|
| Audio | Raw waveform input; the DFSMN near/far pipeline accepts two synchronized streams. |
| Frontend | Resampling and feature extraction are embedded in the graph. |
| Sample rate | VAD and SmartTurn can use a configurable external rate and process internally at 16 kHz; Community-1 uses 16 kHz input. |
| Input dtype | F16, F32, or INT16 where supported; defaults are model-specific. Float PCM uses `[-1, 1]`. |
| Shape and state | Fixed or dynamic audio axes, with explicit cache or recurrent state where required. |
| Runtime | ONNX Runtime execution-provider settings and persistent I/O binding are implemented per pipeline. |
| Dependencies | Export dependencies vary by upstream model, so the repository does not provide one global lock file. |

For noisy recordings, place the [Audio Denoiser](https://github.com/DakeQQ/Audio-Denoiser-ONNX) before the VAD pipeline.

---

### Deployment package

| Pipeline | Required files | Purpose |
|---|---|---|
| VAD and SmartTurn | `<model>.onnx` + `<model>_Metadata.onnx` | Executable graph plus its validated static runtime contract. |
| Pyannote Community-1 | `segmentation_embedding.onnx` + `vbx_clustering_diarization.onnx` + `community1_Metadata.onnx` | Two executable graphs plus the package manifest. |

Metadata carriers record static facts such as task identity, sample rates, input dtype, axes, I/O counts, and state/cache layout. Runtime thresholds, smoothing, duration limits, paths, and provider choices are intentionally excluded. Community-1 additionally validates graph identities and SHA-256 hashes.

When a VAD or SmartTurn model is copied or renamed, its carrier must be copied with the same filename stem.

**Project files:** [metadata contract](vad_onnx_metadata.py) · [metadata inspector](inspect_vad_metadata.py) · [shared optimizer](Optimize_ONNX_Common.py) · [demo audio registry](Example_Audio.py) · [test audio](Test_Examples/)

---

### Performance

These are existing project measurements. Real-time factor is processing time divided by audio duration; lower is better. Results vary with hardware, graph configuration, execution provider, and thread settings.

| System | Device | Backend | Model | Precision | Chunk | RTF |
|---|---|:---:|---|:---:|---:|---:|
| Ubuntu 24.04 | Desktop, i3-12300 | CPU | FSMN | F32 | 512 samples / 32 ms | 0.0047 |
| Ubuntu 24.04 | Desktop, i3-12300 | CPU | Silero | F32 | 512 samples / 32 ms | 0.0026 |
| Ubuntu 24.04 | Laptop, i7-1165G7 | CPU | NVIDIA VAD | F32 | 89,000 samples | 0.0005 |
| Ubuntu 24.04 | Laptop, i7-1165G7 | CPU | DFSMN VAD | F32 | 32,000 samples | 0.27 |
| Ubuntu 24.04 | Laptop, i7-1165G7 | CPU | FireRedVAD | F32 | 16,000 samples | 0.0013 |

**Related:** [Audio-Denoiser-ONNX](https://github.com/DakeQQ/Audio-Denoiser-ONNX) · [More projects by DakeQQ](https://github.com/DakeQQ?tab=repositories) · [MIT License](LICENSE)

[Back to language selector](#voice-activity-detection-vad-onnx)

---

<a id="zh-cn"></a>

## 简体中文

### 输入原始波形，输出语音事件

本仓库提供基于 ONNX Runtime 的端到端语音处理流程，涵盖语音活动检测、音频事件检测、流式 VAD、话轮结束检测和说话人分段标注。音频预处理与特征提取已集成到部署计算图中，应用端可以直接传入波形采样点。

- **8 套处理流程**，覆盖离线 VAD、流式 VAD、音频事件检测、话轮结束检测和说话人分段标注。
- **原始音频接口**，按模型提供固定形状、动态形状、缓存或循环状态约定。
- **可移植的部署信息**，保存于必需的元数据载体中，并在执行前进行校验。
- **优化的运行路径**，支持执行提供程序配置和持久化 ONNX Runtime 输入输出绑定。

> [!IMPORTANT]
> 单独的 ONNX 主模型并不是完整部署包，必须与匹配的元数据载体放在一起。

---

### 支持的流程

| 任务 | 流程 | 运行方式 | 输出结果 | 链接 |
|---|---|---|---|---|
| VAD | DFSMN：近端 + 远端 | 离线双路输入 | 近端语音时间戳 | [代码](DFSMN/near_and_far_end_audio/) · [来源](https://modelscope.cn/models/iic/speech_dfsmn_aec_psm_16k) |
| VAD | DFSMN：仅近端 | 离线单路输入 | 语音时间戳 | [代码](DFSMN/only_near_end_audio/) |
| VAD | FSMN | 带缓存，可使用固定或动态形状 | 时间戳与帧得分 | [代码](FSMN/) · [来源](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) |
| VAD · AED | FireRedVAD | 离线与流式 | 时间戳或任务得分 | [代码](FireRedVAD/) · [来源](https://github.com/FireRedTeam/FireRedVAD) |
| VAD | NVIDIA MarbleNet | 多语言帧级检测 | 静音/活动得分与时间戳 | [代码](NVIDIA_Frame_VAD_Multilingual_MarbleNet/) · [来源](https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0) |
| VAD | Silero | 有状态分块处理 | 语音概率与时间戳 | [代码](Silero/) · [来源](https://github.com/snakers4/silero-vad) |
| 话轮结束检测 | SmartTurn v3 | 固定 8 秒单声道上下文 | `P(complete)` | [代码](SmartTurn/v3/) · [来源](https://modelscope.cn/models/pipecat-ai/smart-turn-v3) |
| 说话人分段标注 | Pyannote Community-1 | 两个部署计算图 | 说话人时段与嵌入向量 | [代码](Pyannote-Diarization-Community-1/) |

Pyannote Community-1 是相关的说话人分段标注流程，并不是 VAD 模型。

---

### 输入与运行约定

| 项目 | 约定 |
|---|---|
| 音频 | 输入原始波形；DFSMN 近端 + 远端流程接收两路同步音频。 |
| 音频前端 | 重采样与特征提取已集成到计算图中。 |
| 采样率 | VAD 与 SmartTurn 可配置外部采样率，并在图内按 16 kHz 处理；Community-1 使用 16 kHz 输入。 |
| 输入类型 | 支持的流程可使用 F16、F32 或 INT16，默认值因模型而异；浮点 PCM 范围为 `[-1, 1]`。 |
| 形状与状态 | 按模型使用固定或动态音频轴，并在需要时显式传递缓存或循环状态。 |
| 运行方式 | 各流程提供 ONNX Runtime 执行提供程序配置和持久化输入输出绑定。 |
| 依赖 | 不同上游模型的导出依赖各不相同，因此本仓库不提供统一的依赖锁定文件。 |

如果录音噪声较多，可以先使用[音频降噪器](https://github.com/DakeQQ/Audio-Denoiser-ONNX)处理，再送入 VAD 流程。

---

### 部署文件

| 流程 | 必需文件 | 用途 |
|---|---|---|
| VAD 与 SmartTurn | `<model>.onnx` + `<model>_Metadata.onnx` | 可执行计算图及经过校验的固定运行约定。 |
| Pyannote Community-1 | `segmentation_embedding.onnx` + `vbx_clustering_diarization.onnx` + `community1_Metadata.onnx` | 两个可执行计算图及部署包清单。 |

元数据载体记录任务身份、采样率、输入类型、输入轴、输入输出数量以及状态/缓存布局等固定信息。运行阈值、平滑参数、时长限制、路径和执行提供程序选择不会写入载体。Community-1 还会校验计算图身份和 SHA-256 哈希值。

复制或重命名 VAD、SmartTurn 模型时，必须同时复制元数据载体，并保持二者的文件名主干一致。

**项目文件：** [元数据约定](vad_onnx_metadata.py) · [元数据检查工具](inspect_vad_metadata.py) · [公共优化工具](Optimize_ONNX_Common.py) · [示例音频登记表](Example_Audio.py) · [测试音频](Test_Examples/)

---

### 性能

以下数据为项目现有实测值。实时因子等于处理耗时除以音频时长，数值越低越好。实际结果会受到硬件、计算图配置、执行提供程序和线程设置影响。

| 系统 | 设备 | 后端 | 模型 | 精度 | 分块长度 | 实时因子 |
|---|---|:---:|---|:---:|---:|---:|
| Ubuntu 24.04 | 台式机，i3-12300 | CPU | FSMN | F32 | 512 采样点 / 32 毫秒 | 0.0047 |
| Ubuntu 24.04 | 台式机，i3-12300 | CPU | Silero | F32 | 512 采样点 / 32 毫秒 | 0.0026 |
| Ubuntu 24.04 | 笔记本，i7-1165G7 | CPU | NVIDIA VAD | F32 | 89,000 采样点 | 0.0005 |
| Ubuntu 24.04 | 笔记本，i7-1165G7 | CPU | DFSMN VAD | F32 | 32,000 采样点 | 0.27 |
| Ubuntu 24.04 | 笔记本，i7-1165G7 | CPU | FireRedVAD | F32 | 16,000 采样点 | 0.0013 |

**相关链接：** [Audio-Denoiser-ONNX](https://github.com/DakeQQ/Audio-Denoiser-ONNX) · [DakeQQ 的更多项目](https://github.com/DakeQQ?tab=repositories) · [MIT 许可证](LICENSE)

[返回语言选择](#voice-activity-detection-vad-onnx)

---

<div align="center">
<sub><b>Voice-Activity-Detection-VAD-ONNX</b> · ONNX Runtime · <a href="https://github.com/DakeQQ?tab=repositories">DakeQQ</a></sub>
</div>
