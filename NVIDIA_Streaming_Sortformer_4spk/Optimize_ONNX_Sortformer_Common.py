#!/usr/bin/env python
"""ASR-style AFFINE_REFINE_V2 quantization for the Sortformer deployment graph."""

from __future__ import annotations

import copy
import gc
import os
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in _SCRIPT_DIR.parents:
    if (_candidate / "Optimize_ONNX_Common.py").is_file():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate the VAD Optimize_ONNX_Common.py helper.")

from Optimize_ONNX_Common import (  # noqa: E402
    OptimizerConfig as _BaseOptimizerConfig,
    Plan as _BasePlan,
    get_model_paths as _get_model_paths,
    run_optimizer as _run_base_optimizer,
)


_WEIGHT_ONLY_METHODS = {"Q4": 4, "Q8": 8}
_VALID_METHODS = frozenset({"F32", "F16", "DYNAMIC", *_WEIGHT_ONLY_METHODS})
_DYNAMIC_WEIGHT_TYPES = frozenset({"QINT8", "QUINT8"})


@dataclass
class Plan(_BasePlan):
    """One ASR-style quantization recipe for the standalone Sortformer graph."""

    method: str = "Q8"
    algo: str | None = None
    op_types: tuple[str, ...] | None = None
    axes: tuple[int, ...] | None = None
    block_size: int | None = None
    accuracy_level: int | None = None
    symmetric: bool | None = None
    quant_format: str | None = None


@dataclass
class OptimizerConfig(_BaseOptimizerConfig):
    """VAD optimizer defaults plus the validated ASR AFFINE_REFINE_V2 controls."""

    weight_only_algorithm: str = "AFFINE_REFINE_V2"
    block_size: int = 64
    accuracy_level: int = 4
    quant_symmetric: bool = True
    quant_format: str = "QOperator"
    affine_v2_seed_iterations: int = 4
    affine_v2_seed_zp_radius: int = 2
    affine_v2_iterations: int = 6
    affine_v2_clip_ratios: tuple[float, ...] = (1.0, 0.94, 0.82, 0.70, 0.55)
    affine_v2_chunk_blocks: int = 8192
    affine_v2_weighted_tolerance: float = 0.15
    affine_v2_asym_zp_sweep_limit: int = 32
    affine_v2_numba_threads: int = 4


@dataclass(frozen=True)
class _ResolvedQuantPlan:
    method: str
    algorithm: str
    op_types: tuple[str, ...]
    axes: tuple[int, ...]
    block_size: int
    accuracy_level: int
    symmetric: bool
    quant_format: str
    dynamic_weight_type: str
    per_channel: bool
    reduce_range: bool
    external: bool
    nodes_to_exclude: object
    nodes_to_include: object


@dataclass
class AffineRefineStats:
    """Aggregate reconstruction error before and after local refinement."""

    blocks: int = 0
    improved_blocks: int = 0
    seed_error: float = 0.0
    refined_error: float = 0.0

    def add(self, other: "AffineRefineStats") -> None:
        self.blocks += other.blocks
        self.improved_blocks += other.improved_blocks
        self.seed_error += other.seed_error
        self.refined_error += other.refined_error


def _pick(value, default):
    return default if value is None else value


def _resolve_quant_plan(plan: Plan, config: OptimizerConfig) -> _ResolvedQuantPlan:
    algorithm = _pick(plan.algo, config.weight_only_algorithm).strip().upper()
    method = plan.method.strip().upper()
    return _ResolvedQuantPlan(
        method=method,
        algorithm=algorithm,
        op_types=tuple(_pick(plan.op_types, ("MatMul",))),
        axes=tuple(_pick(plan.axes, (0,))),
        block_size=int(_pick(plan.block_size, config.block_size)),
        accuracy_level=int(_pick(plan.accuracy_level, config.accuracy_level)),
        symmetric=bool(_pick(plan.symmetric, config.quant_symmetric)),
        quant_format=_pick(plan.quant_format, config.quant_format).strip().upper(),
        dynamic_weight_type=_pick(
            plan.dynamic_weight_type,
            config.dynamic_weight_type,
        ).strip().upper(),
        per_channel=bool(_pick(plan.per_channel, config.dynamic_per_channel)),
        reduce_range=bool(_pick(plan.reduce_range, config.dynamic_reduce_range)),
        external=bool(_pick(plan.external, config.force_external_data)),
        nodes_to_exclude=plan.nodes_to_exclude,
        nodes_to_include=plan.nodes_to_include,
    )


def _validate_quant_plan(name: str, plan: _ResolvedQuantPlan) -> None:
    if plan.method not in _VALID_METHODS:
        raise ValueError(
            f"[{name}] unknown method {plan.method!r}; use 'F32', 'F16', 'Q4', 'Q8', or 'DYNAMIC'."
        )
    if plan.method in {"F32", "F16"}:
        return
    if plan.algorithm != "AFFINE_REFINE_V2":
        raise ValueError(
            f"[{name}] Sortformer supports only algo='AFFINE_REFINE_V2', got {plan.algorithm!r}."
        )
    if plan.op_types != ("MatMul",) or plan.axes != (0,):
        raise ValueError(
            f"[{name}] the standalone Sortformer profile quantizes only MatMul with axes=(0,)."
        )
    if plan.quant_format != "QOPERATOR":
        raise ValueError(f"[{name}] AFFINE_REFINE_V2 requires quant_format='QOperator'.")
    if plan.accuracy_level < 0:
        raise ValueError(f"[{name}] accuracy_level must be nonnegative.")
    if plan.method in _WEIGHT_ONLY_METHODS and (
        plan.block_size < 16
        or plan.block_size > 256
        or plan.block_size & (plan.block_size - 1)
    ):
        raise ValueError(f"[{name}] block_size must be a power of two in [16, 256].")
    if plan.method == "DYNAMIC":
        if plan.dynamic_weight_type not in _DYNAMIC_WEIGHT_TYPES:
            raise ValueError(
                f"[{name}] dynamic_weight_type must be 'QInt8' or 'QUInt8'."
            )
        if plan.reduce_range:
            raise ValueError(
                f"[{name}] AFFINE_REFINE_V2 dynamic quantization does not support reduce_range."
            )


def _resolve_nodes(selector, model_path: Path) -> tuple[str, ...]:
    nodes = selector(str(model_path)) if callable(selector) else selector
    return tuple(nodes or ())


def _iter_affine_row_chunks(values: np.ndarray, block_size: int, max_blocks: int):
    rows, columns = values.shape
    block_count = (columns + block_size - 1) // block_size
    padded_columns = block_count * block_size
    rows_per_chunk = max(1, max_blocks // block_count)
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        chunk = np.ascontiguousarray(values[row_start:row_end], dtype=np.float32)
        if padded_columns != columns:
            chunk = np.pad(chunk, ((0, 0), (0, padded_columns - columns)))
        yield row_start * block_count, row_end * block_count, chunk.reshape(-1, block_size)


_K_QUANT_SEARCH_OFFSETS = np.asarray(
    tuple(-1.0 + 0.1 * index for index in range(20)), dtype=np.float32
)
_K_QUANT_FINAL_CHUNK_VALUES = 262144


def _quant_tensor_k_quant_cpu(
    data: np.ndarray,
    num_bits: int,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the ASR refinement seed for asymmetric Q4/Q8 blocks."""
    values = np.ascontiguousarray(np.asarray(data).reshape(-1, group_size), dtype=np.float32)
    block_count = values.shape[0]
    maxq = (1 << num_bits) - 1
    maxq_float = np.float32(maxq)
    quantized = np.empty_like(values)
    scratch = np.empty_like(values)
    weighted_quantized = np.empty_like(values)
    np.multiply(values, values, out=scratch)
    rms = np.sqrt(np.sum(scratch, axis=1, dtype=np.float32) / np.float32(group_size))
    weights = np.abs(values)
    weights += rms[:, None]
    minimum = np.min(values, axis=1)
    maximum = np.max(values, axis=1)
    span = maximum - minimum
    constant = span == 0.0
    sum_weights = np.sum(weights, axis=1, dtype=np.float32)
    sum_weighted_values = np.sum(weights * values, axis=1, dtype=np.float32)
    inverse_scale = np.ones(block_count, dtype=np.float32)
    np.divide(maxq_float, span, out=inverse_scale, where=span != 0.0)
    best_scale = np.reciprocal(inverse_scale)
    best_minimum = minimum.copy()
    quantized[:] = np.clip(
        np.rint((values - best_minimum[:, None]) * inverse_scale[:, None]),
        0,
        maxq_float,
    )
    best_error = np.sum(
        weights * (quantized * best_scale[:, None] + best_minimum[:, None] - values) ** 2,
        axis=1,
    )
    for offset in _K_QUANT_SEARCH_OFFSETS:
        span = maximum - best_minimum
        candidate_inverse = np.ones(block_count, dtype=np.float32)
        np.divide(maxq_float + offset, span, out=candidate_inverse, where=span != 0.0)
        quantized[:] = np.clip(
            np.rint((values - best_minimum[:, None]) * candidate_inverse[:, None]),
            0,
            maxq_float,
        )
        np.multiply(weights, quantized, out=weighted_quantized)
        sum_l = np.sum(weighted_quantized, axis=1, dtype=np.float32)
        sum_l2 = np.sum(weighted_quantized * quantized, axis=1, dtype=np.float32)
        sum_xl = np.sum(weighted_quantized * values, axis=1, dtype=np.float32)
        determinant = sum_weights * sum_l2 - sum_l * sum_l
        valid = (determinant != 0.0) & np.isfinite(determinant)
        candidate_scale = np.divide(
            sum_weights * sum_xl - sum_weighted_values * sum_l,
            determinant,
            out=np.zeros_like(determinant),
            where=valid,
        )
        candidate_minimum = np.divide(
            sum_l2 * sum_weighted_values - sum_l * sum_xl,
            determinant,
            out=np.zeros_like(determinant),
            where=valid,
        )
        valid &= (
            (candidate_scale > 0.0)
            & np.isfinite(candidate_scale)
            & np.isfinite(candidate_minimum)
        )
        candidate_error = np.sum(
            weights
            * (quantized * candidate_scale[:, None] + candidate_minimum[:, None] - values) ** 2,
            axis=1,
        )
        improved = valid & (candidate_error < best_error)
        best_error[improved] = candidate_error[improved]
        best_scale[improved] = candidate_scale[improved]
        best_minimum[improved] = candidate_minimum[improved]
    if np.any(constant):
        constant_value = values[constant, 0]
        positive = constant_value > 0.0
        negative = constant_value < 0.0
        best_scale[constant] = np.where(
            positive,
            constant_value / maxq_float,
            np.where(negative, -constant_value / maxq_float, np.float32(1.0)),
        )
        best_minimum[constant] = np.where(negative, constant_value, np.float32(0.0))
    zero_point = np.clip(np.rint(-best_minimum / best_scale), 0, maxq).astype(np.uint8)
    rows_per_chunk = max(1, _K_QUANT_FINAL_CHUNK_VALUES // group_size)
    for start in range(0, block_count, rows_per_chunk):
        end = min(start + rows_per_chunk, block_count)
        quantized[start:end] = np.clip(
            np.rint(values[start:end] / best_scale[start:end, None] + zero_point[start:end, None]),
            0,
            maxq,
        )
    return quantized, best_scale.reshape(-1, 1), zero_point.reshape(-1, 1)


@lru_cache(maxsize=None)
def _affine_v2_numba_kernel(thread_count: int):
    """Build the optional ASR parallel refinement kernel lazily."""
    try:
        from numba import njit, prange, set_num_threads
    except ImportError:
        return None
    if thread_count < 1:
        raise ValueError("AFFINE_REFINE_V2 Numba thread count must be positive.")
    set_num_threads(thread_count)

    @njit(parallel=True, nogil=True, cache=True)
    def refine(
        weight,
        quantized,
        scales,
        zero_points,
        clip_ratios,
        iterations,
        tolerance,
        tiny,
        symmetric,
        maxq,
        midpoint,
        sweep_limit,
    ):
        block_count, width = weight.shape
        baseline_errors = np.empty(block_count, dtype=np.float32)
        refined_errors = np.empty(block_count, dtype=np.float32)
        improved = np.zeros(block_count, dtype=np.bool_)
        candidate_codes = np.empty((block_count, width), dtype=np.uint8)
        max_code = int(maxq)
        for block in prange(block_count):
            sum_squares = np.float32(0.0)
            positive_max = np.float32(0.0)
            negative_max = np.float32(0.0)
            for column in range(width):
                value = np.float32(weight[block, column])
                sum_squares += value * value
                positive_max = max(positive_max, value)
                negative_max = max(negative_max, -value)
            rms = np.float32(np.sqrt(sum_squares / np.float32(width)))
            seed_scale = np.float32(scales[block])
            seed_zp = int(zero_points[block])
            baseline_plain = np.float32(0.0)
            baseline_weighted = np.float32(0.0)
            for column in range(width):
                value = np.float32(weight[block, column])
                residual = value - seed_scale * (
                    np.float32(quantized[block, column]) - np.float32(seed_zp)
                )
                squared = residual * residual
                baseline_plain += squared
                baseline_weighted += (rms + np.abs(value)) * squared
            local_plain = baseline_plain
            weighted_bound = tolerance * baseline_weighted
            if symmetric:
                zp_low = midpoint
                zp_high = midpoint
            elif max_code + 1 <= sweep_limit:
                zp_low = 0
                zp_high = max_code
            else:
                zp_low = max(0, seed_zp - sweep_limit // 2)
                zp_high = zp_low + sweep_limit - 1
                if zp_high > max_code:
                    zp_high = max_code
                    zp_low = max(0, zp_high - sweep_limit + 1)
            for zero_point_int in range(zp_low, zp_high + 1):
                positive_scale = np.float32(0.0)
                negative_scale = np.float32(0.0)
                if zero_point_int < max_code:
                    positive_scale = positive_max / np.float32(max_code - zero_point_int)
                if zero_point_int > 0:
                    negative_scale = negative_max / np.float32(zero_point_int)
                coverage_scale = max(positive_scale, negative_scale)
                if coverage_scale <= tiny:
                    coverage_scale = np.float32(1.0)
                for start_index in range(clip_ratios.size + 1):
                    candidate_scale = (
                        seed_scale
                        if start_index == 0
                        else coverage_scale * clip_ratios[start_index - 1]
                    )
                    zero_point = np.float32(zero_point_int)
                    for _ in range(iterations):
                        denominator = np.float32(0.0)
                        numerator = np.float32(0.0)
                        for column in range(width):
                            value = np.float32(weight[block, column])
                            code = np.rint(value / candidate_scale + zero_point)
                            code = min(np.float32(maxq), max(np.float32(0.0), code))
                            centered = code - zero_point
                            denominator += centered * centered
                            numerator += centered * value
                        if denominator <= tiny:
                            break
                        fitted = numerator / denominator
                        if not np.isfinite(fitted) or fitted <= tiny or fitted == candidate_scale:
                            break
                        candidate_scale = fitted
                    candidate_plain = np.float32(0.0)
                    candidate_weighted = np.float32(0.0)
                    for column in range(width):
                        value = np.float32(weight[block, column])
                        code = np.rint(value / candidate_scale + zero_point)
                        code = min(np.float32(maxq), max(np.float32(0.0), code))
                        candidate_codes[block, column] = np.uint8(code)
                        residual = value - candidate_scale * (code - zero_point)
                        squared = residual * residual
                        candidate_plain += squared
                        candidate_weighted += (rms + np.abs(value)) * squared
                    if candidate_plain < local_plain and candidate_weighted <= weighted_bound:
                        local_plain = candidate_plain
                        scales[block] = candidate_scale
                        zero_points[block] = np.uint8(zero_point_int)
                        for column in range(width):
                            quantized[block, column] = candidate_codes[block, column]
            baseline_errors[block] = baseline_plain
            refined_errors[block] = local_plain
            improved[block] = local_plain < baseline_plain
        return baseline_errors, refined_errors, improved

    return refine


def _affine_refine_v2_rows(
    data: np.ndarray,
    block_size: int,
    symmetric: bool,
    bits: int,
    config: OptimizerConfig,
    *,
    allow_arbitrary_block_size: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, AffineRefineStats]:
    """Use the ASR V2 objective: plain-MSE refinement with a weighted guard."""
    values = np.asarray(data)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("AFFINE_REFINE_V2 expects a finite 2-D row matrix.")
    if not allow_arbitrary_block_size and (
        block_size < 16 or block_size > 256 or block_size & (block_size - 1)
    ):
        raise ValueError("AFFINE_REFINE_V2 block_size must be a power of two in [16, 256].")
    if bits not in (4, 8):
        raise ValueError("AFFINE_REFINE_V2 supports 4- or 8-bit weights.")
    if (
        config.affine_v2_seed_iterations < 1
        or config.affine_v2_seed_zp_radius < 0
        or config.affine_v2_iterations < 1
        or config.affine_v2_chunk_blocks < 1
        or config.affine_v2_weighted_tolerance < 0.0
    ):
        raise ValueError("AFFINE_REFINE_V2 iteration/chunk/tolerance settings are invalid.")
    ratios = np.asarray(config.affine_v2_clip_ratios, dtype=np.float32)
    if ratios.ndim != 1 or not ratios.size or np.any((ratios <= 0.0) | (ratios > 1.0)):
        raise ValueError("AFFINE_REFINE_V2 clip ratios must be in (0, 1].")
    maxq = (1 << bits) - 1
    midpoint = 1 << (bits - 1)
    rows, columns = values.shape
    block_count = (columns + block_size - 1) // block_size
    total_blocks = rows * block_count
    all_q = np.empty((total_blocks, block_size), dtype=np.uint8)
    all_scales = np.empty(total_blocks, dtype=np.float32)
    all_zp = np.empty(total_blocks, dtype=np.uint8)
    stats = AffineRefineStats(blocks=total_blocks)
    tiny = np.finfo(np.float32).tiny
    numba_kernel = _affine_v2_numba_kernel(config.affine_v2_numba_threads)

    for start, end, weight in _iter_affine_row_chunks(
        values,
        block_size,
        config.affine_v2_chunk_blocks,
    ):
        if symmetric:
            positive = np.maximum(weight.max(axis=1, keepdims=True), 0.0)
            negative = np.maximum(-weight.min(axis=1, keepdims=True), 0.0)
            seed_scale = np.maximum(
                positive / (maxq - midpoint),
                negative / midpoint,
            )
            seed_scale = np.where(seed_scale > tiny, seed_scale, 1.0).astype(np.float32)
            seed_zp = np.full((weight.shape[0], 1), midpoint, dtype=np.uint8)
            seed_q = np.clip(np.rint(weight / seed_scale + midpoint), 0, maxq)
        else:
            seed_q, seed_scale, seed_zp = _quant_tensor_k_quant_cpu(
                weight,
                bits,
                block_size,
            )
        seed_q = np.asarray(seed_q, dtype=np.float32)
        seed_scale = np.asarray(seed_scale, dtype=np.float32).reshape(-1, 1)
        seed_zp = np.asarray(seed_zp, dtype=np.uint8).reshape(-1, 1)
        importance = np.sqrt(np.mean(weight * weight, axis=1, keepdims=True)) + np.abs(weight)
        importance = np.where(
            np.sum(importance, axis=1, keepdims=True) > 0.0,
            importance,
            1.0,
        )
        seed_dequant = seed_scale * (seed_q - seed_zp.astype(np.float32))
        seed_weighted = np.sum(importance * (weight - seed_dequant) ** 2, axis=1)
        for delta in (
            (0,)
            if symmetric
            else range(
                -config.affine_v2_seed_zp_radius,
                config.affine_v2_seed_zp_radius + 1,
            )
        ):
            candidate_zp = np.clip(
                seed_zp.astype(np.int16) + delta,
                0,
                maxq,
            ).astype(np.float32)
            candidate_scale = seed_scale.copy()
            for _ in range(config.affine_v2_seed_iterations):
                candidate_q = np.clip(
                    np.rint(weight / candidate_scale + candidate_zp),
                    0,
                    maxq,
                )
                centered = candidate_q - candidate_zp
                denominator = np.sum(
                    importance * centered * centered,
                    axis=1,
                    keepdims=True,
                )
                fitted = np.divide(
                    np.sum(
                        importance * centered * weight,
                        axis=1,
                        keepdims=True,
                    ),
                    denominator,
                    out=candidate_scale.copy(),
                    where=denominator > tiny,
                )
                candidate_scale = np.where(
                    np.isfinite(fitted) & (fitted > tiny),
                    fitted,
                    candidate_scale,
                )
            candidate_q = np.clip(
                np.rint(weight / candidate_scale + candidate_zp),
                0,
                maxq,
            )
            candidate_weighted = np.sum(
                importance * (weight - candidate_scale * (candidate_q - candidate_zp)) ** 2,
                axis=1,
            )
            take = candidate_weighted < seed_weighted
            seed_q[take] = candidate_q[take]
            seed_scale[take] = candidate_scale[take]
            seed_zp[take] = candidate_zp[take].astype(np.uint8)
            seed_weighted[take] = candidate_weighted[take]
        seed_q = seed_q.astype(np.uint8)
        seed_residual = weight - seed_scale * (seed_q.astype(np.float32) - seed_zp)
        seed_plain = np.sum(seed_residual * seed_residual, axis=1)
        weighted_bound = (1.0 + config.affine_v2_weighted_tolerance) * np.sum(
            importance * seed_residual * seed_residual,
            axis=1,
        )
        best_q = seed_q.copy()
        best_scale = seed_scale[:, 0].copy()
        best_zp = seed_zp[:, 0].copy()
        best_plain = seed_plain.copy()

        if numba_kernel is not None:
            baseline_plain, refined_plain, improved = numba_kernel(
                weight,
                best_q,
                best_scale,
                best_zp,
                ratios,
                config.affine_v2_iterations,
                np.float32(1.0 + config.affine_v2_weighted_tolerance),
                tiny,
                symmetric,
                np.float32(maxq),
                np.int64(midpoint),
                np.int64(config.affine_v2_asym_zp_sweep_limit),
            )
            all_q[start:end] = best_q
            all_scales[start:end] = best_scale
            all_zp[start:end] = best_zp
            stats.improved_blocks += int(np.count_nonzero(improved))
            stats.seed_error += float(baseline_plain.sum(dtype=np.float64))
            stats.refined_error += float(refined_plain.sum(dtype=np.float64))
            continue

        positive = np.maximum(weight.max(axis=1, keepdims=True), 0.0)
        negative = np.maximum(-weight.min(axis=1, keepdims=True), 0.0)
        if bits == 8 and not symmetric:
            limit = config.affine_v2_asym_zp_sweep_limit
            if limit < 16:
                raise ValueError("AFFINE_REFINE_V2 asymmetric sweep limit must be at least 16.")
            window_low = np.clip(seed_zp.astype(np.int16) - limit // 2, 0, maxq)
            window_low = np.clip(
                window_low - np.maximum(window_low + limit - 1 - maxq, 0),
                0,
                maxq,
            )
            zero_points = (window_low + offset for offset in range(limit))
        elif symmetric:
            zero_points = (np.full_like(seed_zp, midpoint, dtype=np.float32),)
        else:
            zero_points = (
                np.full_like(seed_zp, value, dtype=np.float32)
                for value in range(maxq + 1)
            )
        for zero_point_value in zero_points:
            zero_point = np.asarray(zero_point_value, dtype=np.float32)
            positive_denominator = np.float32(maxq) - zero_point
            pos_scale = np.where(
                positive_denominator > 0.0,
                positive
                / np.where(positive_denominator > 0.0, positive_denominator, 1.0),
                0.0,
            )
            neg_scale = np.where(
                zero_point > 0.0,
                negative / np.where(zero_point > 0.0, zero_point, 1.0),
                0.0,
            )
            coverage = np.where(
                np.maximum(pos_scale, neg_scale) > tiny,
                np.maximum(pos_scale, neg_scale),
                1.0,
            )
            for initial in (seed_scale, *(coverage * ratio for ratio in ratios)):
                candidate_scale = initial.copy()
                for _ in range(config.affine_v2_iterations):
                    candidate_q = np.clip(
                        np.rint(weight / candidate_scale + zero_point),
                        0,
                        maxq,
                    )
                    centered = candidate_q - zero_point
                    denominator = np.sum(centered * centered, axis=1, keepdims=True)
                    fitted = np.divide(
                        np.sum(centered * weight, axis=1, keepdims=True),
                        denominator,
                        out=candidate_scale.copy(),
                        where=denominator > tiny,
                    )
                    candidate_scale = np.where(
                        np.isfinite(fitted) & (fitted > tiny),
                        fitted,
                        candidate_scale,
                    )
                candidate_q = np.clip(
                    np.rint(weight / candidate_scale + zero_point),
                    0,
                    maxq,
                )
                residual = weight - candidate_scale * (candidate_q - zero_point)
                plain = np.sum(residual * residual, axis=1)
                weighted = np.sum(importance * residual * residual, axis=1)
                take = (plain < best_plain) & (weighted <= weighted_bound)
                best_q[take] = candidate_q[take].astype(np.uint8)
                best_scale[take] = candidate_scale[take, 0]
                best_zp[take] = zero_point[take, 0].astype(np.uint8)
                best_plain[take] = plain[take]
        all_q[start:end] = best_q
        all_scales[start:end] = best_scale
        all_zp[start:end] = best_zp
        stats.improved_blocks += int(np.count_nonzero(best_plain < seed_plain))
        stats.seed_error += float(seed_plain.sum(dtype=np.float64))
        stats.refined_error += float(best_plain.sum(dtype=np.float64))
    return (
        all_q.reshape(rows, block_count, block_size),
        all_scales.reshape(rows, block_count),
        all_zp.reshape(rows, block_count),
        stats,
    )


def _pack_codes(values: np.ndarray, bits: int, pad_value: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint8)
    if bits == 8:
        return np.ascontiguousarray(values)
    if values.shape[-1] & 1:
        values = np.pad(
            values,
            [(0, 0)] * (values.ndim - 1) + [(0, 1)],
            constant_values=pad_value,
        )
    return (values[..., 0::2] | (values[..., 1::2] << 4)).astype(np.uint8)


def _unique_quant_name(graph: onnx.GraphProto, prefix: str):
    used = {initializer.name for initializer in graph.initializer}
    used.update(
        name
        for node in graph.node
        for name in (*node.input, *node.output)
        if name
    )
    counter = 0

    def make(suffix: str) -> str:
        nonlocal counter
        while True:
            candidate = f"{prefix}{counter}_{suffix}"
            counter += 1
            if candidate not in used:
                used.add(candidate)
                return candidate

    return make


def _drop_unused_graph_initializers(graph: onnx.GraphProto) -> None:
    used = {name for node in graph.node for name in node.input if name}
    used.update(value.name for value in graph.output)
    kept = [initializer for initializer in graph.initializer if initializer.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept)


def _ensure_ms_opset(model: onnx.ModelProto) -> None:
    for opset in model.opset_import:
        if opset.domain == "com.microsoft":
            opset.version = max(opset.version, 1)
            return
    model.opset_import.append(helper.make_opsetid("com.microsoft", 1))


def _lower_eligible_gemm_to_matmul_add(graph: onnx.GraphProto) -> int:
    """Expose exact affine Gemm projections to the MatMul-only quantizer."""
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    eligible = []
    for node in graph.node:
        if node.domain not in ("", "ai.onnx") or node.op_type != "Gemm" or len(node.input) < 2:
            continue
        attributes = {attribute.name: attribute for attribute in node.attribute}
        alpha = float(attributes["alpha"].f) if "alpha" in attributes else 1.0
        beta = float(attributes["beta"].f) if "beta" in attributes else 1.0
        trans_a = int(attributes["transA"].i) if "transA" in attributes else 0
        trans_b = int(attributes["transB"].i) if "transB" in attributes else 0
        weight = initializers.get(node.input[1])
        if (
            alpha == 1.0
            and beta == 1.0
            and trans_a == 0
            and trans_b == 1
            and weight is not None
            and len(weight.dims) == 2
        ):
            eligible.append(node)
    if not eligible:
        return 0

    eligible_ids = {id(node) for node in eligible}
    existing_tensor_names = {
        name
        for node in graph.node
        for name in (*node.input, *node.output)
        if name
    } | set(initializers)
    transposed_weights: dict[str, str] = {}

    def unique_name(base: str) -> str:
        candidate = base
        suffix = 1
        while candidate in existing_tensor_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        existing_tensor_names.add(candidate)
        return candidate

    for node in eligible:
        source_name = node.input[1]
        if source_name in transposed_weights:
            continue
        source = initializers[source_name]
        exclusive = all(
            id(consumer) in eligible_ids
            for consumer in consumers.get(source_name, ())
        )
        target_name = (
            source_name
            if exclusive
            else unique_name(source_name + "_gemm_transposed")
        )
        transposed = numpy_helper.from_array(
            np.ascontiguousarray(numpy_helper.to_array(source).T),
            name=target_name,
        )
        if exclusive:
            source.CopyFrom(transposed)
        else:
            graph.initializer.append(transposed)
            initializers[target_name] = transposed
        transposed_weights[source_name] = target_name

    rewritten = []
    for index, node in enumerate(graph.node):
        if id(node) not in eligible_ids:
            rewritten.append(node)
            continue
        output_name = node.output[0]
        has_bias = len(node.input) >= 3 and bool(node.input[2])
        matmul_output = (
            unique_name(output_name + "_gemm_matmul") if has_bias else output_name
        )
        node_name = node.name or f"Gemm_{index}"
        rewritten.append(
            helper.make_node(
                "MatMul",
                (node.input[0], transposed_weights[node.input[1]]),
                (matmul_output,),
                name=node_name,
            )
        )
        if has_bias:
            rewritten.append(
                helper.make_node(
                    "Add",
                    (matmul_output, node.input[2]),
                    (output_name,),
                    name=node_name + "_Bias",
                )
            )
    del graph.node[:]
    graph.node.extend(rewritten)
    return len(eligible)


def _rewrite_affine_weight_only(
    model: onnx.ModelProto,
    plan: _ResolvedQuantPlan,
    bits: int,
    config: OptimizerConfig,
    model_path: Path,
) -> int:
    includes = set(_resolve_nodes(plan.nodes_to_include, model_path))
    excludes = set(_resolve_nodes(plan.nodes_to_exclude, model_path))
    counts = 0
    total = AffineRefineStats()

    def rewrite(graph: onnx.GraphProto) -> None:
        nonlocal counts
        initializers = {initializer.name: initializer for initializer in graph.initializer}
        make = _unique_quant_name(graph, f"affine_refine_v2_q{bits}_")
        rewritten = []
        replaced = set()
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    rewrite(attribute.g)
                for subgraph in attribute.graphs:
                    rewrite(subgraph)
            selected = (
                node.op_type == "MatMul"
                and (not includes or node.name in includes)
                and node.name not in excludes
            )
            weight = initializers.get(node.input[1]) if selected and len(node.input) >= 2 else None
            if weight is None:
                rewritten.append(node)
                continue
            array = numpy_helper.to_array(weight)
            if array.ndim != 2 or array.dtype.kind != "f":
                rewritten.append(node)
                continue
            input_features, output_features = array.shape
            quantized, scales, zero_points, stats = _affine_refine_v2_rows(
                array.T,
                plan.block_size,
                plan.symmetric,
                bits,
                config,
            )
            weight_name = make("weight")
            scale_name = make("scales")
            zero_point_name = make("zero_points")
            graph.initializer.extend(
                [
                    numpy_helper.from_array(_pack_codes(quantized, bits), weight_name),
                    numpy_helper.from_array(scales.astype(array.dtype), scale_name),
                    numpy_helper.from_array(
                        _pack_codes(zero_points, bits, 1 << (bits - 1)),
                        zero_point_name,
                    ),
                ]
            )
            attributes = {
                "K": input_features,
                "N": output_features,
                "bits": bits,
                "block_size": plan.block_size,
            }
            if plan.accuracy_level:
                attributes["accuracy_level"] = plan.accuracy_level
            rewritten.append(
                helper.make_node(
                    "MatMulNBits",
                    (node.input[0], weight_name, scale_name, zero_point_name),
                    node.output,
                    name=(node.name or make("matmul")) + f"_AFFINE_REFINE_V2_Q{bits}",
                    domain="com.microsoft",
                    **attributes,
                )
            )
            replaced.add(weight.name)
            counts += 1
            total.add(stats)
        del graph.node[:]
        graph.node.extend(rewritten)
        _drop_unused_graph_initializers(graph)
        removed_inputs = replaced - {initializer.name for initializer in graph.initializer}
        if removed_inputs:
            kept_inputs = [item for item in graph.input if item.name not in removed_inputs]
            del graph.input[:]
            graph.input.extend(kept_inputs)

    rewrite(model.graph)
    if counts:
        _ensure_ms_opset(model)
    ratio = total.refined_error / total.seed_error if total.seed_error else 1.0
    print(
        f"  AFFINE_REFINE_V2 Q{bits}: {counts} MatMul -> MatMulNBits; "
        f"improved {total.improved_blocks}/{total.blocks} blocks, MSE ratio={ratio:.6f}."
    )
    return counts


def _rewrite_affine_dynamic(
    model: onnx.ModelProto,
    plan: _ResolvedQuantPlan,
    config: OptimizerConfig,
    model_path: Path,
) -> int:
    includes = set(_resolve_nodes(plan.nodes_to_include, model_path))
    excludes = set(_resolve_nodes(plan.nodes_to_exclude, model_path))
    count = 0
    total = AffineRefineStats()

    def rewrite(graph: onnx.GraphProto) -> None:
        nonlocal count
        initializers = {initializer.name: initializer for initializer in graph.initializer}
        make = _unique_quant_name(graph, "affine_refine_v2_dynamic_")
        rewritten = []
        replaced = set()
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    rewrite(attribute.g)
                for subgraph in attribute.graphs:
                    rewrite(subgraph)
            selected = (
                node.op_type == "MatMul"
                and (not includes or node.name in includes)
                and node.name not in excludes
            )
            weight = initializers.get(node.input[1]) if selected and len(node.input) >= 2 else None
            if weight is None:
                rewritten.append(node)
                continue
            array = numpy_helper.to_array(weight)
            if array.ndim != 2 or array.dtype.kind != "f":
                rewritten.append(node)
                continue
            rows = np.ascontiguousarray(
                array.T if plan.per_channel else array.reshape(1, -1),
                dtype=np.float32,
            )
            quantized, scales, zero_points, stats = _affine_refine_v2_rows(
                rows,
                rows.shape[1],
                plan.symmetric,
                8,
                config,
                allow_arbitrary_block_size=True,
            )
            quantized = quantized.reshape(rows.shape)
            quantized = quantized.T if plan.per_channel else quantized.reshape(array.shape)
            scales = scales.reshape(-1)
            zero_points = zero_points.reshape(-1)
            if plan.dynamic_weight_type == "QINT8":
                quantized = (quantized.astype(np.int16) - 128).astype(np.int8)
                zero_points = (zero_points.astype(np.int16) - 128).astype(np.int8)
            else:
                quantized = quantized.astype(np.uint8)
                zero_points = zero_points.astype(np.uint8)
            if not plan.per_channel:
                scales = scales[0]
                zero_points = zero_points[0]
            weight_name = make("weight")
            scale_name = make("scale")
            zero_point_name = make("zero_point")
            graph.initializer.extend(
                [
                    numpy_helper.from_array(quantized, weight_name),
                    numpy_helper.from_array(np.asarray(scales, dtype=np.float32), scale_name),
                    numpy_helper.from_array(np.asarray(zero_points), zero_point_name),
                ]
            )
            rewritten.append(
                helper.make_node(
                    "DynamicQuantizeMatMul",
                    (node.input[0], weight_name, scale_name, zero_point_name),
                    node.output,
                    name=(node.name or make("matmul")) + "_AFFINE_REFINE_V2_DYNAMIC",
                    domain="com.microsoft",
                )
            )
            replaced.add(weight.name)
            total.add(stats)
            count += 1
        del graph.node[:]
        graph.node.extend(rewritten)
        _drop_unused_graph_initializers(graph)
        removed_inputs = replaced - {initializer.name for initializer in graph.initializer}
        if removed_inputs:
            kept_inputs = [item for item in graph.input if item.name not in removed_inputs]
            del graph.input[:]
            graph.input.extend(kept_inputs)

    rewrite(model.graph)
    if count:
        _ensure_ms_opset(model)
    ratio = total.refined_error / total.seed_error if total.seed_error else 1.0
    print(
        f"  AFFINE_REFINE_V2 dynamic: {count} MatMul -> DynamicQuantizeMatMul; "
        f"MSE ratio={ratio:.6f}."
    )
    return count


def _save_model(model: onnx.ModelProto, path: Path, external: bool) -> None:
    """Publish a fully checked quantized model without leaving partial output files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{path.stem}.quant-", dir=path.parent) as directory:
        staged_path = Path(directory) / path.name
        if external:
            onnx.save(
                model,
                staged_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=path.name + ".data",
                size_threshold=1024,
                convert_attribute=True,
            )
        else:
            onnx.save(model, staged_path)
        onnx.checker.check_model(staged_path)
        staged_data_path = staged_path.with_name(staged_path.name + ".data")
        final_data_path = path.with_name(path.name + ".data")
        if external:
            if not staged_data_path.is_file():
                raise RuntimeError(f"Quantization did not create {staged_data_path}")
            os.replace(staged_data_path, final_data_path)
        else:
            final_data_path.unlink(missing_ok=True)
        os.replace(staged_path, path)


def _quantize_model(
    model_path: Path,
    plan: _ResolvedQuantPlan,
    config: OptimizerConfig,
) -> None:
    model = onnx.load(model_path, load_external_data=True)
    try:
        lowered_gemms = _lower_eligible_gemm_to_matmul_add(model.graph)
        if lowered_gemms:
            print(
                f"  Lowered {lowered_gemms} affine Gemm node(s) to MatMul + Add "
                "for AFFINE_REFINE_V2."
            )
        if plan.method in _WEIGHT_ONLY_METHODS:
            count = _rewrite_affine_weight_only(
                model,
                plan,
                _WEIGHT_ONLY_METHODS[plan.method],
                config,
                model_path,
            )
        else:
            count = _rewrite_affine_dynamic(model, plan, config, model_path)
        if not count:
            raise RuntimeError(
                f"{model_path.name} has no selected constant MatMul weights to quantize."
            )
        _save_model(model, model_path, plan.external)
    finally:
        del model
        gc.collect()


def run_optimizer(config: OptimizerConfig) -> None:
    """Run VAD graph cleanup, then apply ASR-style quantization in-place."""
    resolved = {
        name: _resolve_quant_plan(plan, config)
        for name, plan in config.model_plans.items()
    }
    for name, plan in resolved.items():
        _validate_quant_plan(name, plan)

    base_config = copy.copy(config)
    base_config.model_plans = {}
    for name, plan in config.model_plans.items():
        base_plan = copy.copy(plan)
        base_plan.method = "F16" if resolved[name].method == "F16" else "F32"
        base_config.model_plans[name] = base_plan
    _run_base_optimizer(base_config)

    for name, plan in resolved.items():
        if plan.method in {"F32", "F16"}:
            continue
        _, output_path = _get_model_paths(config, name)
        output = Path(output_path)
        if not output.is_file():
            continue
        print(
            f"\n{'=' * 60}\nQuantizing: {name} "
            f"[{plan.method}/{plan.algorithm}]\n{'=' * 60}"
        )
        _quantize_model(output, plan, config)