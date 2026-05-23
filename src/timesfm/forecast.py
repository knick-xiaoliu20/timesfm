"""Core forecasting utilities for TimesFM.

Provides functions to generate forecasts from a pre-trained TimesFM model,
including horizon slicing, quantile extraction, and batch inference helpers.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .data_utils import batch_time_series, normalize_series, pad_or_truncate

# Default context length used when none is specified by the caller.
DEFAULT_CONTEXT_LEN: int = 512

# Quantile levels returned alongside the point forecast.
DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.5, 0.8, 0.9)


def prepare_context(
    series: np.ndarray,
    context_len: int = DEFAULT_CONTEXT_LEN,
) -> np.ndarray:
    """Prepare a single time series for model input.

    The series is normalised and then either left-padded (if shorter than
    *context_len*) or right-truncated (if longer) so that the returned array
    has exactly *context_len* elements.

    Args:
        series: 1-D array of observed values.
        context_len: Desired context window length.

    Returns:
        Normalised, padded/truncated 1-D array of length *context_len*.
    """
    normed, _, _ = normalize_series(series)
    return pad_or_truncate(normed, context_len)


def prepare_batch(
    series_list: Sequence[np.ndarray],
    context_len: int = DEFAULT_CONTEXT_LEN,
    batch_size: int = 32,
) -> list[np.ndarray]:
    """Prepare a collection of time series into model-ready batches.

    Each series is individually normalised and padded/truncated, then the
    resulting contexts are grouped into batches of at most *batch_size*.

    Args:
        series_list: Sequence of 1-D time series arrays.
        context_len: Desired context window length for each series.
        batch_size: Maximum number of series per batch.

    Returns:
        List of 2-D arrays with shape ``(batch, context_len)``.
    """
    contexts = np.stack(
        [prepare_context(s, context_len) for s in series_list], axis=0
    )
    return batch_time_series(contexts, batch_size)


def extract_horizon(
    raw_output: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Slice the first *horizon* steps from raw model output.

    Args:
        raw_output: Array of shape ``(batch, output_len)`` or
            ``(batch, output_len, num_quantiles)``.
        horizon: Number of future steps to retain.

    Returns:
        Array with the time-step dimension trimmed to *horizon*.

    Raises:
        ValueError: If *horizon* exceeds the available output length.
    """
    output_len = raw_output.shape[1]
    if horizon > output_len:
        raise ValueError(
            f"Requested horizon {horizon} exceeds model output length {output_len}."
        )
    return raw_output[:, :horizon, ...]


def get_point_forecast(
    raw_output: np.ndarray,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    point_quantile: float = 0.5,
) -> np.ndarray:
    """Extract the point forecast (median by default) from quantile output.

    Args:
        raw_output: Array of shape ``(batch, horizon, num_quantiles)``.
        quantiles: Ordered quantile levels corresponding to the last axis.
        point_quantile: The quantile level to use as the point forecast.

    Returns:
        2-D array of shape ``(batch, horizon)``.

    Raises:
        ValueError: If *point_quantile* is not present in *quantiles*.
    """
    quantiles = list(quantiles)
    if point_quantile not in quantiles:
        raise ValueError(
            f"point_quantile {point_quantile} not found in quantiles {quantiles}."
        )
    idx = quantiles.index(point_quantile)
    return raw_output[:, :, idx]
