#!/usr/bin/env python
"""Export NVIDIA Streaming Sortformer 4spk v2.1 as one static ONNX step."""

import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_ROOT = _SCRIPT_DIR.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

import torch
import torch.nn.functional as F
from torch import nn

import lightning.pytorch.loggers as lightning_loggers

if not hasattr(lightning_loggers, "NeptuneLogger"):

    class _UnavailableNeptuneLogger:
        pass

    lightning_loggers.NeptuneLogger = _UnavailableNeptuneLogger

from lightning.pytorch import Trainer as LightningTrainer

if LightningTrainer.save_checkpoint.__annotations__.get("weights_only") == bool | None:
    LightningTrainer.save_checkpoint.__annotations__["weights_only"] = bool

from nemo.collections.asr.models import SortformerEncLabelModel

from STFT_Process import STFT_Process


for _candidate in _SCRIPT_DIR.parents:
    if (_candidate / "vad_onnx_metadata.py").is_file():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

from vad_onnx_metadata import stamp_export_metadata


# Export settings
HOME_PATH = Path.home()  # Home directory used to locate the downloaded checkpoint.
model_path = (
    HOME_PATH
    / "Downloads"
    / "diar_streaming_sortformer_4spk-v2.1"
    / "diar_streaming_sortformer_4spk-v2.1.nemo"
)

export_folder = _SCRIPT_DIR / "NVIDIA_Streaming_Sortformer_4spk_ONNX"

onnx_model_A = export_folder / "NVIDIA_Streaming_Sortformer_4spk.onnx"

OPSET = 20              # ONNX operator-set version emitted by torch.onnx.export.

IN_AUDIO_DTYPE = "F32"  #  ["F16", "F32", "INT16"]


_INPUT_AUDIO_DTYPES = {
    "F16": (torch.float16, "tensor(float16)"),
    "F32": (torch.float32, "tensor(float)"),
    "INT16": (torch.int16, "tensor(int16)"),
}
IN_AUDIO_DTYPE = IN_AUDIO_DTYPE.upper()
if IN_AUDIO_DTYPE not in _INPUT_AUDIO_DTYPES:
    raise ValueError(
        f"Unsupported IN_AUDIO_DTYPE={IN_AUDIO_DTYPE!r}; "
        f"expected one of {tuple(_INPUT_AUDIO_DTYPES)}."
    )
IN_TORCH_DTYPE, IN_ONNX_DTYPE = _INPUT_AUDIO_DTYPES[IN_AUDIO_DTYPE]
LENGTH_DTYPE = torch.int32
INDEX_DTYPE = torch.int32
FLAG_DTYPE = torch.int8


def _load_model(path: Path) -> SortformerEncLabelModel:
    return SortformerEncLabelModel.restore_from(
        str(path),
        map_location="cpu",
        strict=False,
    ).float().eval()


class StreamingSortformerAudioFrontend(nn.Module):
    """Checkpoint-matched log-mel frontend using the shared STFT_Process."""

    def __init__(self, preprocessor, output_feature_frames: int) -> None:
        super().__init__()
        featurizer = preprocessor.featurizer
        self.stft = STFT_Process(
            model_type="stft_power",
            n_fft=int(featurizer.n_fft),
            win_length=int(featurizer.win_length),
            hop_len=int(featurizer.hop_length),
            max_frames=0,
            window_type="hann_sym",
            center_pad=True,
            pad_mode="constant",
        )
        self.register_buffer(
            "mel_kernel",
            featurizer.fb.detach().squeeze(0).float().unsqueeze(2),
            persistent=True,
        )
        self.register_buffer(
            "preemphasis",
            torch.tensor(float(featurizer.preemph), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "log_guard",
            torch.tensor(
                float(featurizer.log_zero_guard_value),
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.register_buffer(
            "input_scale",
            torch.tensor(
                1.0 / 32768.0 if IN_AUDIO_DTYPE == "INT16" else 1.0,
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.hop_length = int(featurizer.hop_length)
        self.output_feature_frames = output_feature_frames
        self.register_buffer(
            "frame_positions",
            torch.arange(output_feature_frames, dtype=INDEX_DTYPE).unsqueeze(0),
            persistent=True,
        )

    def forward(
        self,
        audio_with_context: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio_with_context = (
            audio_with_context.to(torch.float32) * self.input_scale
        )
        previous_sample = audio_with_context[:, :, :1]
        audio = audio_with_context[:, :, 1:]
        preemphasized = torch.cat(
            (
                audio[:, :, :1] - self.preemphasis * previous_sample,
                audio[:, :, 1:] - self.preemphasis * audio[:, :, :-1],
            ),
            dim=2,
        )
        power = self.stft(preemphasized)
        features = torch.log(
            F.conv1d(power, self.mel_kernel) + self.log_guard
        )[:, :, : self.output_feature_frames]
        feature_lengths = torch.div(
            audio_lengths,
            self.hop_length,
            rounding_mode="floor",
        ).to(LENGTH_DTYPE).clamp(max=self.output_feature_frames)
        valid = self.frame_positions < feature_lengths.unsqueeze(1)
        features = features * valid.unsqueeze(1).to(features.dtype)
        return features, feature_lengths


class PackedStreamingSortformerNeural(nn.Module):
    """Neural Sortformer step with fixed-capacity cache packing."""

    def __init__(self, model: SortformerEncLabelModel) -> None:
        super().__init__()
        self.model = model
        modules = model.sortformer_modules
        self.speaker_cache_capacity = int(modules.spkcache_len)
        self.fifo_capacity = int(modules.fifo_len)
        self.output_activity_frames = int(modules.chunk_len)
        feature_cache_frames = int(modules.subsampling_factor) + 1
        self.pre_encode_drop_frames = (
            feature_cache_frames + int(modules.subsampling_factor) - 1
        ) // int(modules.subsampling_factor)
        self.register_buffer(
            "packed_positions",
            torch.arange(
                self.speaker_cache_capacity
                + self.fifo_capacity
                + self.output_activity_frames,
                dtype=INDEX_DTYPE,
            ).unsqueeze(0),
            persistent=True,
        )

    @staticmethod
    def _gather_rows(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return values[0].index_select(0, indices[0]).unsqueeze(0)

    def forward(
        self,
        chunk: torch.Tensor,
        chunk_lengths: torch.Tensor,
        spkcache: torch.Tensor,
        spkcache_lengths: torch.Tensor,
        fifo: torch.Tensor,
        fifo_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk_embs, chunk_lengths = self.model.encoder.pre_encode(
            x=chunk,
            lengths=chunk_lengths,
        )
        chunk_embs = chunk_embs[:, self.pre_encode_drop_frames :, :]
        chunk_lengths = (
            chunk_lengths - self.pre_encode_drop_frames
        ).clamp(min=0).to(LENGTH_DTYPE)

        source = torch.cat((spkcache, fifo, chunk_embs), dim=1)
        positions = self.packed_positions
        cache_lengths = spkcache_lengths.unsqueeze(1)
        fifo_lengths_expanded = fifo_lengths.unsqueeze(1)
        source_indices = torch.where(
            positions < cache_lengths,
            positions,
            torch.where(
                positions < cache_lengths + fifo_lengths_expanded,
                self.speaker_cache_capacity + positions - cache_lengths,
                self.speaker_cache_capacity
                + self.fifo_capacity
                + positions
                - cache_lengths
                - fifo_lengths_expanded,
            ),
        ).clamp(min=0, max=source.shape[1] - 1)
        packed = self._gather_rows(source, source_indices)
        packed_lengths = spkcache_lengths + fifo_lengths + chunk_lengths
        packed = packed * (
            positions < packed_lengths.unsqueeze(1)
        ).unsqueeze(2).to(packed.dtype)

        frontend_embs, frontend_lengths = self.model.frontend_encoder(
            processed_signal=packed,
            processed_signal_length=packed_lengths,
            bypass_pre_encode=True,
        )
        probabilities = self.model.forward_infer(frontend_embs, frontend_lengths)
        return probabilities, chunk_embs, chunk_lengths


class FixedStreamingState(nn.Module):
    """Tensor-only fixed-capacity equivalent of Sortformer's cache update."""

    def __init__(self, modules) -> None:
        super().__init__()
        self.speaker_cache_capacity = int(modules.spkcache_len)
        self.fifo_capacity = int(modules.fifo_len)
        self.output_activity_frames = int(modules.chunk_len)
        self.speaker_cache_update_period = int(modules.spkcache_update_period)
        self.speaker_cache_silence_frames = int(
            modules.spkcache_sil_frames_per_spk
        )
        self.speaker_count = int(modules.n_spk)
        self.export_batch_size = 1
        self.prediction_score_threshold = float(modules.pred_score_threshold)
        self.silence_threshold = float(modules.sil_threshold)
        self.fifo_candidate_capacity = (
            self.fifo_capacity + self.output_activity_frames
        )
        self.cache_candidate_capacity = (
            self.speaker_cache_capacity + self.speaker_cache_update_period
        )
        self.cache_score_capacity = (
            self.cache_candidate_capacity + self.speaker_cache_silence_frames
        )
        cache_rows_per_speaker = (
            self.speaker_cache_capacity // self.speaker_count
            - self.speaker_cache_silence_frames
        )
        self.strong_boost_frames = int(
            cache_rows_per_speaker * float(modules.strong_boost_rate)
        )
        self.weak_boost_frames = int(
            cache_rows_per_speaker * float(modules.weak_boost_rate)
        )
        self.min_positive_score_frames = int(
            cache_rows_per_speaker * float(modules.min_pos_scores_rate)
        )
        self.register_buffer(
            "activity_positions",
            torch.arange(
                self.output_activity_frames,
                dtype=INDEX_DTYPE,
            ).unsqueeze(0),
            persistent=True,
        )
        self.register_buffer(
            "fifo_positions",
            torch.arange(
                self.fifo_candidate_capacity,
                dtype=INDEX_DTYPE,
            ).unsqueeze(0),
            persistent=True,
        )
        self.register_buffer(
            "fifo_state_positions",
            torch.arange(
                self.fifo_capacity,
                dtype=INDEX_DTYPE,
            ).unsqueeze(0),
            persistent=True,
        )
        self.register_buffer(
            "cache_positions",
            torch.arange(
                self.cache_candidate_capacity,
                dtype=INDEX_DTYPE,
            ).unsqueeze(0),
            persistent=True,
        )
        self.register_buffer(
            "speaker_cache_positions",
            torch.arange(
                self.speaker_cache_capacity,
                dtype=INDEX_DTYPE,
            ).unsqueeze(0),
            persistent=True,
        )
        self.register_buffer(
            "latest_cache_positions",
            (
                torch.arange(self.cache_candidate_capacity, dtype=INDEX_DTYPE)
                >= self.speaker_cache_capacity
            ).view(1, -1, 1),
            persistent=True,
        )
        self.register_buffer(
            "silence_scores",
            torch.full(
                (
                    1,
                    self.speaker_cache_silence_frames,
                    self.speaker_count,
                ),
                torch.inf,
            ),
            persistent=True,
        )
        self.register_buffer(
            "log_two",
            torch.tensor(0.6931471805599453, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "latest_score_boost",
            torch.tensor(float(modules.scores_boost_latest), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "strong_boost_values",
            torch.full(
                (
                    self.export_batch_size,
                    self.strong_boost_frames,
                    self.speaker_count,
                ),
                1.3862943611198906,
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.register_buffer(
            "weak_boost_values",
            torch.full(
                (
                    self.export_batch_size,
                    self.weak_boost_frames,
                    self.speaker_count,
                ),
                0.6931471805599453,
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.register_buffer(
            "placeholder_indices",
            torch.full(
                (self.export_batch_size, self.speaker_cache_capacity),
                99_999,
                dtype=INDEX_DTYPE,
            ),
            persistent=True,
        )
        self.register_buffer(
            "zero_indices",
            torch.zeros(
                self.export_batch_size,
                self.speaker_cache_capacity,
                dtype=INDEX_DTYPE,
            ),
            persistent=True,
        )
        self.register_buffer(
            "zero_speaker_predictions",
            torch.zeros(
                self.export_batch_size,
                self.speaker_cache_capacity,
                self.speaker_count,
            ),
            persistent=True,
        )
        self.register_buffer(
            "scheduled_pop_lengths",
            torch.full(
                (self.export_batch_size,),
                self.speaker_cache_update_period,
                dtype=LENGTH_DTYPE,
            ),
            persistent=True,
        )
        self.register_buffer(
            "zero_lengths",
            torch.zeros(self.export_batch_size, dtype=LENGTH_DTYPE),
            persistent=True,
        )
        self.register_buffer(
            "one_has_preds",
            torch.ones(self.export_batch_size, dtype=LENGTH_DTYPE),
            persistent=True,
        )

    @staticmethod
    def _gather_rows(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return values[0].index_select(0, indices[0]).unsqueeze(0)

    def _boost_topk(
        self,
        scores: torch.Tensor,
        frame_count: int,
        boost_values: torch.Tensor,
    ) -> torch.Tensor:
        _, indices = torch.topk(
            scores,
            frame_count,
            dim=1,
            largest=True,
            sorted=False,
        )
        indices = indices.to(INDEX_DTYPE)
        return scores.scatter_add(
            1,
            indices,
            boost_values,
        )

    def _compress_cache(
        self,
        embeddings: torch.Tensor,
        predictions: torch.Tensor,
        cache_lengths: torch.Tensor,
        mean_sil_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_probs = torch.log(
            torch.clamp(predictions, min=self.prediction_score_threshold)
        )
        log_inverse_probs = torch.log(
            torch.clamp(
                1.0 - predictions,
                min=self.prediction_score_threshold,
            )
        )
        scores = (
            log_probs
            - log_inverse_probs
            + log_inverse_probs.sum(dim=2, keepdim=True)
            + self.log_two
        )
        speech = predictions > 0.5
        scores = torch.where(speech, scores, torch.full_like(scores, -torch.inf))
        positive = scores > 0.0
        replace_nonpositive = (
            (~positive)
            & speech
            & (
                positive.sum(dim=1, keepdim=True)
                >= self.min_positive_score_frames
            )
        )
        scores = torch.where(
            replace_nonpositive,
            torch.full_like(scores, -torch.inf),
            scores,
        )
        valid_cache = self.cache_positions < cache_lengths.unsqueeze(1)
        scores = torch.where(
            valid_cache.unsqueeze(2),
            scores,
            torch.full_like(scores, -torch.inf),
        )
        scores = torch.where(
            self.latest_cache_positions,
            scores + self.latest_score_boost,
            scores,
        )
        scores = self._boost_topk(
            scores,
            self.strong_boost_frames,
            self.strong_boost_values,
        )
        scores = self._boost_topk(
            scores,
            self.weak_boost_frames,
            self.weak_boost_values,
        )
        scores = torch.cat(
            (scores, self.silence_scores),
            dim=1,
        )
        flattened = scores.permute(0, 2, 1).reshape(self.export_batch_size, -1)
        values, indices = torch.topk(
            flattened,
            self.speaker_cache_capacity,
            dim=1,
            largest=True,
            sorted=False,
        )
        indices = indices.to(INDEX_DTYPE)
        indices = torch.where(
            values == -torch.inf,
            self.placeholder_indices,
            indices,
        )
        indices, _ = torch.sort(indices, dim=1)
        is_disabled = indices == 99_999
        indices = torch.remainder(indices, self.cache_score_capacity)
        is_disabled = is_disabled | (indices >= self.cache_candidate_capacity)
        indices = torch.where(is_disabled, self.zero_indices, indices)

        compressed_embeddings = self._gather_rows(embeddings, indices)
        compressed_embeddings = torch.where(
            is_disabled.unsqueeze(2),
            mean_sil_emb.unsqueeze(1),
            compressed_embeddings,
        )
        compressed_predictions = self._gather_rows(predictions, indices)
        compressed_predictions = torch.where(
            is_disabled.unsqueeze(2),
            self.zero_speaker_predictions,
            compressed_predictions,
        )
        return compressed_embeddings, compressed_predictions

    def forward(
        self,
        probabilities: torch.Tensor,
        chunk_embeddings: torch.Tensor,
        chunk_lengths: torch.Tensor,
        spkcache: torch.Tensor,
        spkcache_lengths: torch.Tensor,
        spkcache_preds: torch.Tensor,
        spkcache_has_preds: torch.Tensor,
        fifo: torch.Tensor,
        fifo_lengths: torch.Tensor,
        mean_sil_emb: torch.Tensor,
        n_sil_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        activity_indices = (
            spkcache_lengths.unsqueeze(1)
            + fifo_lengths.unsqueeze(1)
            + self.activity_positions
        ).clamp(
            max=(
                self.speaker_cache_capacity
                + self.fifo_capacity
                + self.output_activity_frames
                - 1
            )
        )
        activity = self._gather_rows(probabilities, activity_indices)
        activity = activity * (
            self.activity_positions < chunk_lengths.unsqueeze(1)
        ).unsqueeze(2).to(activity.dtype)

        fifo_indices = torch.where(
            self.fifo_positions < fifo_lengths.unsqueeze(1),
            self.fifo_positions,
            self.fifo_capacity
            + self.fifo_positions
            - fifo_lengths.unsqueeze(1),
        ).clamp(max=self.fifo_candidate_capacity - 1)
        fifo_candidate = self._gather_rows(
            torch.cat((fifo, chunk_embeddings), dim=1),
            fifo_indices,
        )
        total_fifo_lengths = fifo_lengths + chunk_lengths
        fifo_candidate = fifo_candidate * (
            self.fifo_positions < total_fifo_lengths.unsqueeze(1)
        ).unsqueeze(2).to(fifo_candidate.dtype)
        fifo_prediction_indices = (
            spkcache_lengths.unsqueeze(1) + self.fifo_positions
        ).clamp(
            max=(
                self.speaker_cache_capacity
                + self.fifo_capacity
                + self.output_activity_frames
                - 1
            )
        )
        fifo_predictions = self._gather_rows(
            probabilities,
            fifo_prediction_indices,
        )
        fifo_predictions = fifo_predictions * (
            self.fifo_positions < total_fifo_lengths.unsqueeze(1)
        ).unsqueeze(2).to(fifo_predictions.dtype)

        cache_predictions_from_model = self._gather_rows(
            probabilities,
            self.speaker_cache_positions,
        )
        cache_predictions_from_model = cache_predictions_from_model * (
            self.speaker_cache_positions < spkcache_lengths.unsqueeze(1)
        ).unsqueeze(2).to(cache_predictions_from_model.dtype)
        cache_predictions = torch.where(
            (spkcache_has_preds > 0).reshape(1, 1, 1),
            spkcache_preds,
            cache_predictions_from_model,
        )

        overflow = total_fifo_lengths - self.fifo_capacity
        scheduled_pop_lengths = torch.maximum(
            self.scheduled_pop_lengths,
            overflow,
        )
        pop_lengths = torch.minimum(scheduled_pop_lengths, total_fifo_lengths)
        pop_lengths = torch.where(
            total_fifo_lengths > self.fifo_capacity,
            pop_lengths,
            self.zero_lengths,
        )
        next_fifo_lengths = total_fifo_lengths - pop_lengths
        next_fifo_indices = (
            self.fifo_state_positions + pop_lengths.unsqueeze(1)
        ).clamp(max=self.fifo_candidate_capacity - 1)
        next_fifo = self._gather_rows(fifo_candidate, next_fifo_indices)
        next_fifo = next_fifo * (
            self.fifo_state_positions < next_fifo_lengths.unsqueeze(1)
        ).unsqueeze(2).to(next_fifo.dtype)

        cache_candidate_indices = torch.where(
            self.cache_positions < spkcache_lengths.unsqueeze(1),
            self.cache_positions,
            self.speaker_cache_capacity
            + self.cache_positions
            - spkcache_lengths.unsqueeze(1),
        ).clamp(
            max=self.speaker_cache_capacity + self.fifo_candidate_capacity - 1
        )
        cache_candidate = self._gather_rows(
            torch.cat((spkcache, fifo_candidate), dim=1),
            cache_candidate_indices,
        )
        cache_prediction_candidate = self._gather_rows(
            torch.cat((cache_predictions, fifo_predictions), dim=1),
            cache_candidate_indices,
        )
        cache_candidate_lengths = spkcache_lengths + pop_lengths
        cache_candidate = cache_candidate * (
            self.cache_positions < cache_candidate_lengths.unsqueeze(1)
        ).unsqueeze(2).to(cache_candidate.dtype)
        cache_prediction_candidate = cache_prediction_candidate * (
            self.cache_positions < cache_candidate_lengths.unsqueeze(1)
        ).unsqueeze(2).to(cache_prediction_candidate.dtype)

        silence = (
            fifo_predictions.sum(dim=2) < self.silence_threshold
        ) & (self.fifo_positions < pop_lengths.unsqueeze(1))
        silence_count = silence.sum(dim=1).to(LENGTH_DTYPE)
        silence_embedding_sum = (
            fifo_candidate * silence.unsqueeze(2).to(fifo_candidate.dtype)
        ).sum(dim=1)
        next_n_sil_frames = n_sil_frames + silence_count
        updated_mean_sil_emb = (
            mean_sil_emb * n_sil_frames.unsqueeze(1) + silence_embedding_sum
        ) / torch.clamp(next_n_sil_frames.unsqueeze(1), min=1)
        next_mean_sil_emb = torch.where(
            (silence_count > 0).unsqueeze(1),
            updated_mean_sil_emb,
            mean_sil_emb,
        )

        needs_compression = (
            cache_candidate_lengths > self.speaker_cache_capacity
        )
        compressed_cache, compressed_cache_preds = self._compress_cache(
            cache_candidate,
            cache_prediction_candidate,
            cache_candidate_lengths,
            next_mean_sil_emb,
        )
        next_spkcache = torch.where(
            needs_compression.reshape(1, 1, 1),
            compressed_cache,
            cache_candidate[:, : self.speaker_cache_capacity, :],
        )
        next_spkcache_preds = torch.where(
            needs_compression.reshape(1, 1, 1),
            compressed_cache_preds,
            cache_prediction_candidate[:, : self.speaker_cache_capacity, :],
        )
        next_spkcache_lengths = cache_candidate_lengths.clamp(
            max=self.speaker_cache_capacity
        )
        next_spkcache_has_preds = torch.where(
            needs_compression,
            self.one_has_preds,
            spkcache_has_preds.to(LENGTH_DTYPE),
        ).to(FLAG_DTYPE)
        return (
            activity,
            next_spkcache,
            next_spkcache_lengths,
            next_spkcache_preds,
            next_spkcache_has_preds,
            next_fifo,
            next_fifo_lengths,
            next_mean_sil_emb,
            next_n_sil_frames,
        )


class StreamingSortformerPipeline(nn.Module):
    """Static waveform-to-activity streaming step with ONNX-visible state."""

    def __init__(self, model: SortformerEncLabelModel) -> None:
        super().__init__()
        modules = model.sortformer_modules
        self.sample_rate = int(model.preprocessor.featurizer.sample_rate)
        self.feature_cache_frames = int(modules.subsampling_factor) + 1
        self.new_feature_frames = (
            int(modules.chunk_len) * int(modules.subsampling_factor)
        )
        self.input_feature_frames = (
            self.feature_cache_frames + self.new_feature_frames
        )
        hop_length = int(model.preprocessor.featurizer.hop_length)
        self.audio_cache_samples = (
            self.feature_cache_frames * hop_length
        )
        self.new_audio_samples = (
            self.new_feature_frames * hop_length
        )
        self.audio_chunk_samples = (
            self.audio_cache_samples + self.new_audio_samples
        )
        self.audio_input_samples = self.audio_chunk_samples + 1
        self.speaker_cache_capacity = int(modules.spkcache_len)
        self.fifo_capacity = int(modules.fifo_len)
        self.embedding_dimension = int(modules.fc_d_model)
        self.speaker_count = int(modules.n_spk)
        self.output_activity_frames = int(modules.chunk_len)
        self.frontend = StreamingSortformerAudioFrontend(
            model.preprocessor,
            self.input_feature_frames,
        )
        self.neural = PackedStreamingSortformerNeural(model)
        self.state = FixedStreamingState(modules)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        activity_threshold: torch.Tensor,
        spkcache: torch.Tensor,
        spkcache_lengths: torch.Tensor,
        spkcache_preds: torch.Tensor,
        spkcache_has_preds: torch.Tensor,
        fifo: torch.Tensor,
        fifo_lengths: torch.Tensor,
        mean_sil_emb: torch.Tensor,
        n_sil_frames: torch.Tensor,
        active_frame_counts: torch.Tensor,
        overlap_frame_count: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        features, chunk_lengths = self.frontend(audio, audio_lengths)
        chunk = features.transpose(1, 2)
        probabilities, chunk_embeddings, chunk_lengths = self.neural(
            chunk,
            chunk_lengths,
            spkcache,
            spkcache_lengths,
            fifo,
            fifo_lengths,
        )
        (
            activity,
            next_spkcache,
            next_spkcache_lengths,
            next_spkcache_preds,
            next_spkcache_has_preds,
            next_fifo,
            next_fifo_lengths,
            next_mean_sil_emb,
            next_n_sil_frames,
        ) = self.state(
            probabilities,
            chunk_embeddings,
            chunk_lengths,
            spkcache,
            spkcache_lengths,
            spkcache_preds,
            spkcache_has_preds,
            fifo,
            fifo_lengths,
            mean_sil_emb,
            n_sil_frames,
        )
        active_frames = activity > activity_threshold
        next_active_frame_counts = active_frame_counts + active_frames.to(
            LENGTH_DTYPE
        ).sum(dim=1, dtype=LENGTH_DTYPE)
        overlap_frames = active_frames.to(LENGTH_DTYPE).sum(
            dim=2,
            dtype=LENGTH_DTYPE,
        ) > 1
        next_overlap_frame_count = overlap_frame_count + overlap_frames.to(
            LENGTH_DTYPE
        ).sum(dim=1, dtype=LENGTH_DTYPE)
        return (
            activity,
            next_spkcache,
            next_spkcache_lengths,
            next_spkcache_preds,
            next_spkcache_has_preds,
            next_fifo,
            next_fifo_lengths,
            next_mean_sil_emb,
            next_n_sil_frames,
            next_active_frame_counts,
            next_overlap_frame_count,
        )


def main() -> None:
    print("=" * 70)
    print("NVIDIA Streaming Sortformer 4spk v2.1 ONNX Export")
    print("=" * 70)
    model = _load_model(model_path)
    modules = model.sortformer_modules
    # Match the static 14-frame MultiTalker streaming deployment profile.
    modules.spkcache_len = 188
    modules.fifo_len = 188
    modules.chunk_len = 14
    modules.chunk_left_context = 0
    modules.chunk_right_context = 0
    modules.spkcache_update_period = 144

    pipeline = StreamingSortformerPipeline(model).eval()
    export_folder.mkdir(parents=True, exist_ok=True)
    for stale_model in (
        export_folder / "NVIDIA_Streaming_Sortformer_4spk_Neural.onnx",
        export_folder / "NVIDIA_Streaming_Sortformer_4spk_Frontend.onnx",
    ):
        stale_model.unlink(missing_ok=True)
        stale_model.with_name(stale_model.stem + "_Metadata.onnx").unlink(
            missing_ok=True
        )
    for stale_model in export_folder.glob("*_raw.onnx"):
        stale_model.unlink(missing_ok=True)
        stale_model.with_name(stale_model.stem + "_Metadata.onnx").unlink(
            missing_ok=True
        )

    print(f"Checkpoint: {model_path}")
    print(f"Export folder: {export_folder}")
    print(f"Audio input dtype: {IN_AUDIO_DTYPE}")
    print("Exporting static waveform-to-summary step ...")
    # torch.onnx.export requires tracing tensors; these local tensors are not inference configuration.
    torch.onnx.export(
        pipeline,
        (
            torch.zeros(
                1,
                1,
                pipeline.audio_input_samples,
                dtype=IN_TORCH_DTYPE,
            ),
            torch.full((1,), pipeline.audio_chunk_samples, dtype=LENGTH_DTYPE),
            torch.full((1,), 0.5, dtype=torch.float32),
            torch.zeros(
                1,
                pipeline.speaker_cache_capacity,
                pipeline.embedding_dimension,
            ),
            torch.zeros(1, dtype=LENGTH_DTYPE),
            torch.zeros(
                1,
                pipeline.speaker_cache_capacity,
                pipeline.speaker_count,
            ),
            torch.zeros(1, dtype=FLAG_DTYPE),
            torch.zeros(
                1,
                pipeline.fifo_capacity,
                pipeline.embedding_dimension,
            ),
            torch.zeros(1, dtype=LENGTH_DTYPE),
            torch.zeros(1, pipeline.embedding_dimension),
            torch.zeros(1, dtype=LENGTH_DTYPE),
            torch.zeros(1, pipeline.speaker_count, dtype=LENGTH_DTYPE),
            torch.zeros(1, dtype=LENGTH_DTYPE),
        ),
        onnx_model_A,
        export_params=True,
        input_names=[
            "audio",
            "audio_lengths",
            "activity_threshold",
            "spkcache",
            "spkcache_lengths",
            "spkcache_preds",
            "spkcache_has_preds",
            "fifo",
            "fifo_lengths",
            "mean_sil_emb",
            "n_sil_frames",
            "active_frame_counts",
            "overlap_frame_count",
        ],
        output_names=[
            "activity",
            "next_spkcache",
            "next_spkcache_lengths",
            "next_spkcache_preds",
            "next_spkcache_has_preds",
            "next_fifo",
            "next_fifo_lengths",
            "next_mean_sil_emb",
            "next_n_sil_frames",
            "next_active_frame_counts",
            "next_overlap_frame_count",
        ],
        opset_version=OPSET,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False,
    )

    metadata_path = stamp_export_metadata(
        onnx_model_A,
        {
            "sample_rate": pipeline.sample_rate,
            "audio_cache_samples": pipeline.audio_cache_samples,
            "new_audio_samples": pipeline.new_audio_samples,
            "output_activity_frames": pipeline.output_activity_frames,
            "num_speakers": pipeline.speaker_count,
        },
        OPSET,
    )
    print(f"Published model: {onnx_model_A}")
    print(f"Metadata: {metadata_path}")
    print("Starting standalone ONNX Runtime multitalker demo ...")
    subprocess.run(
        [
            sys.executable,
            str(
                _SCRIPT_DIR
                / "Inference_NVIDIA_Streaming_Sortformer_4spk_ONNX.py"
            ),
            "--model-folder",
            str(export_folder),
        ],
        cwd=_SCRIPT_DIR,
        check=True,
    )


if __name__ == "__main__":
    main()