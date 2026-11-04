"""Fixed tactile display experiments, independent of contact/force labels.

Only raise an existing channel deadband: never refit the baseline, normalize
per frame, or silently replace values used by downstream learning.
"""
from dataclasses import asdict, dataclass

import numpy as np

from .tactile_baseline import apply_baseline, residual_regions


@dataclass(frozen=True)
class TactileDisplayStyle:
    name: str = 'A · 当前'
    min_deadband: float = 3.
    gamma: float = .5
    neutral_fade_count: float = 0.

    def __post_init__(self):
        if (not np.isfinite([self.min_deadband, self.gamma, self.neutral_fade_count]).all()
                or self.min_deadband < 0 or self.gamma <= 0 or self.neutral_fade_count < 0):
            raise ValueError('finite nonnegative thresholds and positive gamma required')

    def to_dict(self):
        return asdict(self)


PROFILES = {
    'A': TactileDisplayStyle(),
    'B': TactileDisplayStyle('B · 死区≥8', 8., .5, 0.),
    'C': TactileDisplayStyle('C · 柔和色彩', 3., .8, 12.),
    'D': TactileDisplayStyle('D · 死区≥8 + 柔和色彩', 8., .8, 12.),
}


def display_level(count, style=None):
    """Fixed 0..255-count transfer, preserving missing values as NaN."""
    gamma = .5 if style is None else style.gamma
    return np.clip(np.asarray(count, float) / 255., 0, 1) ** gamma


def neutral_weight(count, style=None):
    """Smooth neutral-to-palette onset; does not alter saved count values."""
    x = np.asarray(count, float)
    fade = 0. if style is None else style.neutral_fade_count
    if fade == 0:
        return np.where(np.isnan(x), np.nan, (x > 0).astype(float))
    t = np.clip(x / fade, 0, 1)
    return t * t * (3 - 2 * t)


def prepare_display(source, style):
    """Copy frame data; apply max(existing channel deadband, chosen floor).

    Source raw, signed residual, clocks, indices and validity are not modified.
    Aggregation follows per-channel subtraction, including the palm.
    """
    out = {k: np.array(source[k], copy=True) for k in source.keys()}
    for side in ('left', 'right'):
        valid = out[side + '_tactile_valid'].astype(bool)
        deadband = np.maximum(out[side + '_deadband_count'], style.min_deadband)
        signed, positive = apply_baseline(out[side + '_tactile_values'], valid,
                                          out[side + '_baseline_count'], deadband)
        np.testing.assert_array_equal(signed, out[side + '_signed_residual_count'])
        peaks, means = residual_regions(positive, valid, side)
        out[side + '_deadband_count'] = deadband
        out[side + '_positive_residual_count'] = positive
        for kind, values in (('peak', peaks), ('mean', means)):
            out[side + '_finger_' + kind + '_residual_count'] = values[:, :5]
            out[side + '_palm_' + kind + '_residual_count'] = values[:, 5]
        out[side + '_display_level'] = display_level(peaks[:, :5], style)
        out[side + '_palm_display_level'] = display_level(peaks[:, 5], style)
    return out
