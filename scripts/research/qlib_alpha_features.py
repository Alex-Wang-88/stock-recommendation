"""Pandas port of Qlib Alpha158 formulas, excluding unverified VWAP0.

Formula reference: Microsoft Qlib, commit recorded in the experiment receipt.
Copyright (c) Microsoft Corporation. Licensed under the MIT License.
Full license: docs/research/third_party/qlib_LICENSE.txt.
This port uses calendar-aligned bars and never forward-fills missing quotes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (5, 10, 20, 30, 60)


def regression(series, window):
    """OLS on observation position; missing observations retain their calendar position."""
    raw = series.to_numpy(dtype=float)
    values = np.lib.stride_tricks.sliding_window_view(
        np.pad(raw, (window - 1, 0), constant_values=np.nan), window
    )
    present = np.isfinite(values)
    x = np.arange(window, dtype=float)
    n = present.sum(axis=1)
    y = np.where(present, values, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = present @ x
        sy = y.sum(axis=1)
        sxx = present @ (x * x) - sx * sx / n
        syy = (y * y).sum(axis=1) - sy * sy / n
        sxy = y @ x - sx * sy / n
        slope = sxy / sxx
        residual = raw - (sy / n + slope * (window - 1 - sx / n))
        rsquare = sxy * sxy / (sxx * syy)
    slope[n < 2] = np.nan
    residual[n < 2] = np.nan
    rsquare[(n < 2) | (syy <= 1e-12)] = np.nan
    return slope, rsquare, residual


def alpha157(frame):
    """All values depend only on the same-day completed bar and earlier bars."""
    o, h, low, c, v = (frame[k].astype(float) for k in ["open", "high", "low", "close", "volume"])
    eps = 1e-12
    top, bottom = np.maximum(o, c), np.minimum(o, c)
    fields = {
        "KMID": (c - o) / o,
        "KLEN": (h - low) / o,
        "KMID2": (c - o) / (h - low + eps),
        "KUP": (h - top) / o,
        "KUP2": (h - top) / (h - low + eps),
        "KLOW": (bottom - low) / o,
        "KLOW2": (bottom - low) / (h - low + eps),
        "KSFT": (2 * c - h - low) / o,
        "KSFT2": (2 * c - h - low) / (h - low + eps),
        "OPEN0": o / c,
        "HIGH0": h / c,
        "LOW0": low / c,
    }
    ret = c / c.shift(1)
    vol_ratio = v / v.shift(1)
    delta = c - c.shift(1)
    vdelta = v - v.shift(1)
    weighted = (ret - 1).abs() * v
    for w in WINDOWS:
        cr, vr = c.rolling(w, min_periods=1), v.rolling(w, min_periods=1)
        hr, lr = h.rolling(w, min_periods=1), low.rolling(w, min_periods=1)
        slope, rsquare, residual = regression(c, w)
        imax = hr.apply(lambda x: x.argmax() + 1, raw=True) / w
        imin = lr.apply(lambda x: x.argmin() + 1, raw=True) / w
        sum_abs = delta.abs().rolling(w, min_periods=1).sum() + eps
        gain = delta.clip(lower=0).rolling(w, min_periods=1).sum()
        loss = (-delta).clip(lower=0).rolling(w, min_periods=1).sum()
        sum_vabs = vdelta.abs().rolling(w, min_periods=1).sum() + eps
        vgain = vdelta.clip(lower=0).rolling(w, min_periods=1).sum()
        vloss = (-vdelta).clip(lower=0).rolling(w, min_periods=1).sum()
        cp = (c > c.shift(1)).astype(float).rolling(w, min_periods=1).mean()
        cn = (c < c.shift(1)).astype(float).rolling(w, min_periods=1).mean()
        values = {
            "ROC": c.shift(w) / c,
            "MA": cr.mean() / c,
            "STD": cr.std(ddof=1) / c,
            "BETA": slope / c,
            "RSQR": rsquare,
            "RESI": residual / c,
            "MAX": hr.max() / c,
            "MIN": lr.min() / c,
            "QTLU": cr.quantile(0.8) / c,
            "QTLD": cr.quantile(0.2) / c,
            "RANK": cr.rank(pct=True),
            "RSV": (c - lr.min()) / (hr.max() - lr.min() + eps),
            "IMAX": imax,
            "IMIN": imin,
            "IMXD": imax - imin,
            "CORR": cr.corr(np.log(v + 1)),
            "CORD": ret.rolling(w, min_periods=1).corr(np.log(vol_ratio + 1)),
            "CNTP": cp,
            "CNTN": cn,
            "CNTD": cp - cn,
            "SUMP": gain / sum_abs,
            "SUMN": loss / sum_abs,
            "SUMD": (gain - loss) / sum_abs,
            "VMA": vr.mean() / (v + eps),
            "VSTD": vr.std(ddof=1) / (v + eps),
            "WVMA": weighted.rolling(w, min_periods=1).std(ddof=1)
            / (weighted.rolling(w, min_periods=1).mean() + eps),
            "VSUMP": vgain / sum_vabs,
            "VSUMN": vloss / sum_vabs,
            "VSUMD": (vgain - vloss) / sum_vabs,
        }
        fields.update({f"{name}{w}": value for name, value in values.items()})
    result = pd.DataFrame(fields, index=frame.index).replace([np.inf, -np.inf], np.nan)
    assert result.shape[1] == 157 and result.columns.is_unique
    rolling_names = (
        "ROC",
        "MA",
        "STD",
        "BETA",
        "RSQR",
        "RESI",
        "MAX",
        "MIN",
        "QTLU",
        "QTLD",
        "RANK",
        "RSV",
        "IMAX",
        "IMIN",
        "IMXD",
        "CORR",
        "CORD",
        "CNTP",
        "CNTN",
        "CNTD",
        "SUMP",
        "SUMN",
        "SUMD",
        "VMA",
        "VSTD",
        "WVMA",
        "VSUMP",
        "VSUMN",
        "VSUMD",
    )
    return result[[*list(fields)[:12], *[f"{name}{w}" for name in rolling_names for w in WINDOWS]]]


def purged_fold_masks(days, label_end, fold):
    """Only labels completely observed before the following segment are eligible."""
    train = (
        days.ge(fold["train_start"])
        & days.le(fold["train_end"])
        & label_end.lt(fold["valid_start"])
    )
    valid = (
        days.ge(fold["valid_start"])
        & days.le(fold["valid_end"])
        & label_end.lt(fold["test_first_signal"])
    )
    return train, valid


def fit_transformer(x):
    medians = np.nanmedian(x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    scale = np.nanmedian(np.abs(x - medians), axis=0) * 1.4826
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return medians, scale


def transform(x, medians, scale):
    return np.nan_to_num(np.clip((x - medians) / scale, -3, 3), nan=0.0).astype(np.float32)
