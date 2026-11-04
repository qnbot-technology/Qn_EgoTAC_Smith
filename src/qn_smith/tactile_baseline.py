"""Fixed, per-channel tactile baselines; raw packets are never modified.

An offline lower-tail estimate is a display candidate, not a measured no-load
calibration. Never update the baseline every frame: that erases sustained touch.
"""
import numpy as np

from .tactile_protocol import finger_channels, palm_channels


def _packets(values, valid):
    raw, valid = np.asarray(values), np.asarray(valid, bool)
    if raw.ndim != 2 or raw.shape[1] != 460 or raw.dtype != np.uint8:
        raise ValueError('expected uint8 (packets,460) counts')
    if valid.shape != (len(raw),):
        raise ValueError('valid must have packet shape')
    return raw, valid


def fit_baseline(values, valid, *, method='estimated_lower_tail', reference_mask=None,
                 quantile=.1, lower_tail_quantile=.2, min_deadband=3.,
                 noise_multiplier=3., min_samples=30):
    """Return fixed baseline and noise proxy in count units, with provenance.

    confirmed_unloaded requires an explicitly selected reference mask. Its
    confirmation must come from the operator, never from low signal alone.
    estimated_lower_tail uses the episode's lower quantile independently per
    channel. Low-tail MAD is only a spread proxy, not a calibrated noise sigma.
    """
    raw, valid = _packets(values, valid)
    if (not 0 <= quantile < lower_tail_quantile <= .5
            or not np.isfinite([min_deadband, noise_multiplier]).all()
            or min_deadband < 0 or noise_multiplier < 0 or min_samples < 2):
        raise ValueError('invalid baseline/deadband parameters')
    if method not in ('estimated_lower_tail', 'confirmed_unloaded'):
        raise ValueError('explicit baseline method required')
    selected = valid.copy()
    if reference_mask is not None:
        mask = np.asarray(reference_mask, bool)
        if mask.shape != valid.shape:
            raise ValueError('reference mask must have packet shape')
        selected &= mask
    if method == 'confirmed_unloaded' and reference_mask is None:
        raise ValueError('confirmed unloaded baseline needs an explicit reference interval')
    x = raw[selected].astype(np.float64)
    if len(x) < min_samples:
        raise ValueError('insufficient valid reference packets; do not invent a baseline')
    if method == 'estimated_lower_tail':
        baseline = np.quantile(x, quantile, axis=0)
        limit = np.quantile(x, lower_tail_quantile, axis=0)
        mad = np.array([np.median(np.abs(x[x[:, c] <= limit[c], c]-baseline[c]))
                        for c in range(460)])
    else:
        baseline = np.median(x, axis=0)
        mad = np.median(np.abs(x-baseline), axis=0)
    noise_proxy = 1.4826 * mad
    deadband = np.maximum(min_deadband, noise_multiplier * noise_proxy)
    return dict(baseline_count=baseline, noise_proxy_count=noise_proxy,
                deadband_count=deadband, reference_packet_mask=selected,
                reference_packet_count=int(selected.sum()), method=method,
                baseline_estimated=method == 'estimated_lower_tail',
                no_load_confirmed=method == 'confirmed_unloaded',
                quantile=quantile, lower_tail_quantile=lower_tail_quantile,
                min_deadband_count=min_deadband, noise_multiplier=noise_multiplier,
                fixed_over_time=True, force_calibrated=False)


def apply_baseline(values, valid, baseline_count, deadband_count):
    """Signed raw-minus-baseline and positive residual after a fixed deadband.

    Casting before subtraction prevents uint8 underflow. Negative values are
    retained in signed output; missing packets are NaN, never zero or held.
    """
    raw, valid = _packets(values, valid)
    baseline, deadband = np.asarray(baseline_count, float), np.asarray(deadband_count, float)
    if (baseline.shape != (460,) or deadband.shape != (460,)
            or not np.isfinite(baseline).all() or not np.isfinite(deadband).all()
            or (baseline < 0).any() or (baseline > 255).any() or (deadband < 0).any()):
        raise ValueError('finite channel-wise baseline [0,255] and nonnegative deadband required')
    signed = raw.astype(np.float64)-baseline
    positive = np.maximum(signed-deadband, 0.)
    signed[~valid] = np.nan
    positive[~valid] = np.nan
    return signed, positive


def residual_regions(values, valid, side):
    """Aggregate only after per-channel correction; no unassigned-cell leak."""
    x, valid = np.asarray(values, float), np.asarray(valid, bool)
    if x.ndim != 2 or x.shape[1] != 460 or valid.shape != (len(x),):
        raise ValueError('expected (packets,460) residual with matching valid mask')
    if not np.isfinite(x[valid]).all() or (x[valid] < 0).any() or (x[valid] > 255).any():
        raise ValueError('valid positive residuals must be finite counts [0,255]')
    groups = list(finger_channels(side).values()) + [palm_channels(side)]
    peak = np.stack([x[:, cells].max(-1) for cells in groups], axis=-1)
    mean = np.stack([x[:, cells].mean(-1) for cells in groups], axis=-1)
    peak[~valid] = np.nan
    mean[~valid] = np.nan
    return peak, mean
