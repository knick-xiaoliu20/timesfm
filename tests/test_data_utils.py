"""Tests for src/timesfm/data_utils.py."""

import numpy as np
import pytest

from timesfm.data_utils import (
    batch_time_series,
    normalize_series,
    pad_or_truncate,
)


class TestPadOrTruncate:
    def test_shorter_series_is_left_padded(self):
        ts = np.array([1.0, 2.0, 3.0])
        out, mask = pad_or_truncate(ts, context_length=5)
        assert out.shape == (5,)
        np.testing.assert_array_equal(out, [0.0, 0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(mask, [0, 0, 1, 1, 1])

    def test_longer_series_is_right_truncated(self):
        ts = np.arange(10, dtype=float)
        out, mask = pad_or_truncate(ts, context_length=4)
        np.testing.assert_array_equal(out, [6.0, 7.0, 8.0, 9.0])
        np.testing.assert_array_equal(mask, [1, 1, 1, 1])

    def test_exact_length_unchanged(self):
        ts = np.array([5.0, 6.0, 7.0])
        out, mask = pad_or_truncate(ts, context_length=3)
        np.testing.assert_array_equal(out, ts)
        np.testing.assert_array_equal(mask, [1, 1, 1])

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            pad_or_truncate(np.ones((3, 3)), context_length=3)

    def test_custom_pad_value(self):
        ts = np.array([1.0])
        out, _ = pad_or_truncate(ts, context_length=3, pad_value=-1.0)
        np.testing.assert_array_equal(out, [-1.0, -1.0, 1.0])

    def test_empty_series_is_all_padding(self):
        # Edge case: empty input should produce all-padding output and zero mask.
        ts = np.array([], dtype=float)
        out, mask = pad_or_truncate(ts, context_length=3)
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(mask, [0, 0, 0])


class TestNormalizeSeries:
    def test_output_has_zero_mean_unit_std(self):
        ts = np.array([2.0, 4.0, 6.0, 8.0])
        normed, mean, std = normalize_series(ts)
        assert abs(np.mean(normed)) < 1e-6
        assert mean == pytest.approx(5.0)
        assert std > 0

    def test_constant_series_does_not_raise(self):
        ts = np.ones(5)
        normed, mean, std = normalize_series(ts)
        assert np.all(np.isfinite(normed))


class TestBatchTimeSeries:
    def test_batch_shape(self):
        series = [np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
        batch, masks = batch_time_series(series, context_length=4)
        assert batch.shape == (2, 4)
        assert masks.shape == (2, 4)

    def test_mask_values(self):
        series = [np.array([1.0, 2.0])]
        _, masks = batch_time_series(series, context_length=4)
        np.testing.assert_array_equal(masks[0], [0, 0, 1, 1])

    def test_dtype_is_float32(self):
        series = [np.array([1, 2, 3])]
        batch, _ = batch_time_series(series, context_length=3)
        assert batch.dtype == np.float32

    def test_single_series_batch(self):
        # Sanity check that a single-element list still produces correct shape.
        series = [np.array([1.0, 2.0, 3.0])]
        batch, masks = batch_time_series(series, context_length=3)
        assert batch.shape == (1, 3)
        np.testing.assert_array_equal(masks[0], [1, 1, 1])
