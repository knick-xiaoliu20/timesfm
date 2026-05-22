"""Utility functions for preparing time series data for TimesFM."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def pad_or_truncate(
    time_series: np.ndarray,
    context_length: int,
    pad_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad or truncate a 1-D time series to a fixed context length.

    Args:
        time_series: 1-D array of observed values.
        context_length: Desired output length.
        pad_value: Value used for left-padding when the series is shorter.

    Returns:
        A tuple of (padded_or_truncated_series, padding_mask) where the mask
        is 1 for real observations and 0 for padded positions.
    """
    if time_series.ndim != 1:
        raise ValueError("time_series must be a 1-D array.")

    n = len(time_series)
    if n >= context_length:
        out = time_series[-context_length:]
        mask = np.ones(context_length, dtype=np.int32)
    else:
        pad_width = context_length - n
        out = np.concatenate([np.full(pad_width, pad_value), time_series])
        mask = np.concatenate([np.zeros(pad_width, dtype=np.int32),
                               np.ones(n, dtype=np.int32)])
    return out, mask


def normalize_series(
    time_series: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, float, float]:
    """Z-score normalise a 1-D time series.

    Args:
        time_series: 1-D array of observed values.
        eps: Small constant to avoid division by zero.

    Returns:
        Tuple of (normalised_series, mean, std).
    """
    mean = float(np.mean(time_series))
    std = float(np.std(time_series)) + eps
    return (time_series - mean) / std, mean, std


def batch_time_series(
    series_list: Sequence[np.ndarray],
    context_length: int,
    pad_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare a batch of variable-length time series for model input.

    Args:
        series_list: Sequence of 1-D arrays with possibly different lengths.
        context_length: Fixed context length for the model.
        pad_value: Value used for padding.

    Returns:
        Tuple of (batch_array, mask_array) each with shape
        (len(series_list), context_length).
    """
    batch, masks = [], []
    for ts in series_list:
        padded, mask = pad_or_truncate(np.asarray(ts, dtype=np.float32),
                                       context_length, pad_value)
        batch.append(padded)
        masks.append(mask)
    return np.stack(batch, axis=0), np.stack(masks, axis=0)
