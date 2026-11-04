"""460-channel glove regions from vendor protocol pp.6–7 (zero-based).

Counts are not calibrated forces. White/unassigned channels are retained in
raw heatmaps but never attributed to a finger. Left/right are device roles.
"""
import numpy as np

FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
PROTOCOL_SHA256 = '176378672d48081ca346253f73f1038bb0136727642ceae4b850b2f913e4d242'


def finger_channels(side):
    if side not in ('left', 'right'):
        raise ValueError('explicit left/right device role required')
    groups = range(5) if side == 'left' else reversed(range(5))
    return {name: [r*20+c for r in range(17, 23) for c in range(g*4, g*4+4)]
            for name, g in zip(FINGERS, groups)}


def palm_channels(side):
    """Blue palm cells in vendor pp.6–7; excludes white rows 15–16.

    Right: rows 0..14, columns 0..15. Left: rows 0..14, columns 4..19.
    The protocol defines a palm region, not anatomical subregions within it.
    """
    if side not in ('left', 'right'):
        raise ValueError('explicit left/right device role required')
    columns = range(4, 20) if side == 'left' else range(16)
    return [r*20+c for r in range(15) for c in columns]


def palm_strength(values, valid, side):
    """Palm peak/mean counts from the 240 assigned cells; missing is NaN."""
    raw = np.asarray(values)
    if raw.shape[-1] != 460 or raw.dtype != np.uint8:
        raise ValueError('expected uint8 (...,460) tactile counts')
    valid = np.asarray(valid, bool)
    if valid.shape != raw.shape[:-1]:
        raise ValueError('valid must have the packet shape')
    cells = raw[..., palm_channels(side)]
    peak, mean = cells.max(-1).astype(float), cells.mean(-1)
    peak[~valid] = np.nan
    mean[~valid] = np.nan
    return peak, mean


def finger_strength(values, valid, side):
    """Per-finger peak/mean uint8 counts; missing packets remain NaN."""
    raw = np.asarray(values)
    if raw.shape[-1] != 460 or raw.dtype != np.uint8:
        raise ValueError('expected uint8 (...,460) tactile counts')
    valid = np.asarray(valid, bool)
    if valid.shape != raw.shape[:-1]:
        raise ValueError('valid must have the packet shape')
    cells = np.stack([raw[..., idx] for idx in finger_channels(side).values()], axis=-2)
    peak, mean = cells.max(-1).astype(float), cells.mean(-1)
    peak[~valid] = np.nan
    mean[~valid] = np.nan
    return peak, mean


def display_grid(values):
    """Origin 0 at bottom left, exactly as printed in the protocol."""
    return np.asarray(values).reshape(23, 20)[::-1]
