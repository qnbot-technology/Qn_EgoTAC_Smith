"""Verbatim function subset; provenance/extraction.json records origin."""
import json
import numpy as np


def text_json(dataset):
    value = dataset[()]
    return json.loads(value.decode() if isinstance(value, bytes) else str(value))

def decode_native(raw, valid, protocol):
    """Vendor raw xyz[1e-5m], wxyz -> canonical five sensors, xyzw."""
    order = protocol['finger_order']
    canonical = ('thumb', 'index', 'middle', 'ring', 'pinky')
    if len(order) != 5 or set(order) != set(canonical):
        raise ValueError('Unknown vendor finger order')
    ix = [order.index(x) for x in canonical]
    raw = np.asarray(raw).reshape(-1, 5, 7)[:, ix].astype(float)
    valid = np.asarray(valid, bool)[:, ix].copy()
    q = raw[..., [4, 5, 6, 3]]
    norm = np.linalg.norm(q, axis=-1)
    valid &= (norm > 1e-9) & np.isfinite(raw).all(axis=-1)
    pose = np.full(raw.shape, np.nan)
    pose[..., :3] = raw[..., :3] * 1e-5
    pose[..., 3:] = q / np.maximum(norm[..., None], 1e-12)
    pose[~valid] = np.nan
    return pose, valid

def nearest_indices(source_ns, query_ns):
    """Offline nearest host-receive match; this does not calibrate latency."""
    source_ns, query_ns = np.asarray(source_ns), np.asarray(query_ns)
    if not len(source_ns) or np.any(np.diff(source_ns) <= 0):
        raise ValueError('Native EMF receive times must strictly increase')
    right = np.clip(np.searchsorted(source_ns, query_ns), 0, len(source_ns)-1)
    left = np.maximum(right-1, 0)
    return np.where(abs(source_ns[left]-query_ns) <= abs(source_ns[right]-query_ns), left, right)
