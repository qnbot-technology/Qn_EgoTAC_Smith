"""Offline, source-clock hand stabilization; no visual fitting or world anchoring.

The centered fit uses future samples and is not a real-time controller. A smooth
trajectory is not proof of physical accuracy or removal of low-frequency drift.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def source_groups(ticks_ns, values, valid, epoch, reset=None, max_gap_s=.2):
    """Split invalid/reset/gapped clocks; deduplicate identical source snapshots.

    Reused timestamps must contain identical values. Conflicts are excluded and
    cut both neighboring supports rather than being averaged into new evidence.
    """
    ticks = np.asarray(ticks_ns, np.int64)
    values = np.asarray(values, float)
    valid = np.asarray(valid, bool).copy()
    epoch = np.asarray(epoch)
    n = len(ticks)
    reset = np.zeros(n, bool) if reset is None else np.asarray(reset, bool)
    if values.shape[0] != n or any(x.shape != (n,) for x in (valid, epoch, reset)):
        raise ValueError('Source arrays must have equal row counts')
    valid &= (ticks > 0) & np.isfinite(values).reshape(n, -1).all(1)
    conflict = np.zeros(n, bool)
    for i in range(1, n):
        if valid[i-1] and valid[i] and ticks[i] == ticks[i-1] and epoch[i] == epoch[i-1]:
            if not np.array_equal(values[i], values[i-1]):
                conflict[i-1:i+1] = True
    valid &= ~conflict
    segments = np.full(n, -1, int)
    groups = []
    current = []
    for i in range(n):
        cut = (i == 0 or not valid[i-1] or not valid[i] or reset[i]
               or epoch[i] != epoch[i-1] or ticks[i] < ticks[i-1]
               or (int(ticks[i])-int(ticks[i-1]))/1e9 > max_gap_s)
        if cut and current:
            groups.append(np.array(current, int)); current = []
        if valid[i]:
            current.append(i)
    if current:
        groups.append(np.array(current, int))
    for sid, rows in enumerate(groups):
        segments[rows] = sid
    return groups, segments, conflict


def _center_fit(delta_s, values, radius_s, degree=2):
    # Gaussian time weights with a hard support radius. Centered quadratic fits
    # exactly retain constant velocity/acceleration signals with sufficient rank.
    u = np.asarray(delta_s)/radius_s
    A = np.column_stack([u**k for k in range(degree+1)])
    weights = np.exp(-2*u*u)
    if np.linalg.matrix_rank(A*weights[:, None]**.5) < degree+1:
        return None
    return np.linalg.lstsq(A*weights[:, None]**.5,
                          values*weights[:, None]**.5, rcond=None)[0][0]


def local_polynomial(values, ticks_ns, valid, epoch, *, reset=None,
                     radius_s=.14, max_gap_s=.2, min_unique=5, rotation=False):
    """Centered source-time polynomial; rotations fit a local SO(3) logarithm.

    Short and one-sided segment boundaries are retained unchanged. There is no
    extrapolation, missing-frame filling, pose prediction or persistent lock.
    """
    if not radius_s > 0 or min_unique < 3:
        raise ValueError('Positive support radius and >=3 unique samples required')
    x = np.asarray(values, float)
    if rotation and (x.ndim != 3 or x.shape[1:] != (3, 3)):
        raise ValueError('Rotations must be Nx3x3 matrices')
    groups, segments, conflict = source_groups(ticks_ns, x, valid, epoch, reset, max_gap_s)
    result = x.copy(); support = np.zeros(len(x), int); ticks = np.asarray(ticks_ns, np.int64)
    for rows in groups:
        _, first, inverse = np.unique(ticks[rows], return_index=True, return_inverse=True)
        unique = rows[first]
        seconds = (ticks[unique]-ticks[unique[0]]).astype(float)/1e9
        for k, row in enumerate(unique):
            dt = seconds-seconds[k]
            used = np.flatnonzero(np.abs(dt) <= radius_s+1e-9)
            copies = rows[inverse == k]
            support[copies] = len(used)
            if len(used) < min_unique or (dt[used] < -1e-6).sum() < 2 or (dt[used] > 1e-6).sum() < 2:
                continue
            if rotation:
                y = Rotation.from_matrix(x[row].T @ x[unique[used]]).as_rotvec()
            else:
                y = x[unique[used]].reshape(len(used), -1)
            fitted = _center_fit(dt[used], y, radius_s)
            if fitted is not None:
                result[copies] = (x[row] @ Rotation.from_rotvec(fitted).as_matrix()
                                  if rotation else fitted.reshape(x.shape[1:]))
    return result, dict(segments=segments, support_unique=support, identity_conflict=conflict)


def bound_joint_proposal(q, proposal, names, limits, fk, *, max_delta_rad,
                         max_tip_shift_m, tips=(4, 9, 14, 19, 24)):
    """Limit joint residual, never worsen inherited limits; use rigid FK only."""
    q = np.asarray(q, float); proposal = np.asarray(proposal, float)
    lower = np.array([min(q[i], limits[name][0]) for i, name in enumerate(names)])
    upper = np.array([max(q[i], limits[name][1]) for i, name in enumerate(names)])
    target = np.clip(q+np.clip(proposal-q, -max_delta_rad, max_delta_rad), lower, upper)
    original = fk(q)
    fraction = 1.
    for _ in range(20):
        final = q+fraction*(target-q); points = fk(final)
        shift = float(np.linalg.norm((points-original)[list(tips)], axis=1).max())
        if shift <= max_tip_shift_m+1e-12:
            return final, dict(strength_fraction=fraction, tip_shift_m=shift,
                               correction_limited=bool(fraction < 1 or not np.allclose(target, proposal)))
        fraction *= .5
    return q.copy(), dict(strength_fraction=0., tip_shift_m=0., correction_limited=True)


def stabilize_poses(T, ticks_ns, valid, epoch, *, radius_s=.1, max_gap_s=.2,
                    max_translation_m=.006, max_rotation_rad=np.deg2rad(1.5)):
    """Smooth a tracker HEAD in world, then caller composes the unchanged mount."""
    T = np.asarray(T, float); result = T.copy()
    if T.ndim != 3 or T.shape[1:] != (4, 4):
        raise ValueError('Tracker poses must be Nx4x4')
    _, full_segments, full_conflict = source_groups(ticks_ns, T, valid, epoch, max_gap_s=max_gap_s)
    valid = np.asarray(valid, bool) & (full_segments >= 0)
    trans, td = local_polynomial(T[:, :3, 3], ticks_ns, valid, full_segments,
                                 radius_s=radius_s, max_gap_s=max_gap_s)
    rot, rd = local_polynomial(T[:, :3, :3], ticks_ns, valid, full_segments,
                               radius_s=radius_s, max_gap_s=max_gap_s, rotation=True)
    good = np.asarray(valid, bool) & np.isfinite(T).all((1, 2))
    if not good.any():
        return result, dict(segments=td['segments'], support_unique=td['support_unique'],
                            identity_conflict=full_conflict | td['identity_conflict'] | rd['identity_conflict'])
    move = trans[good]-T[good, :3, 3]
    move *= np.minimum(1., max_translation_m/np.maximum(np.linalg.norm(move, axis=1), 1e-15))[:, None]
    delta = Rotation.from_matrix(np.swapaxes(T[good, :3, :3], 1, 2) @ rot[good]).as_rotvec()
    delta *= np.minimum(1., max_rotation_rad/np.maximum(np.linalg.norm(delta, axis=1), 1e-15))[:, None]
    result[good, :3, 3] += move
    result[good, :3, :3] = T[good, :3, :3] @ Rotation.from_rotvec(delta).as_matrix()
    return result, dict(segments=td['segments'], support_unique=td['support_unique'],
                       identity_conflict=full_conflict | td['identity_conflict'] | rd['identity_conflict'])


def short_period_residual(x, seconds, valid, segments):
    """Deviation from adjacent-time linear interpolation: a motion proxy, not noise truth."""
    x = np.asarray(x, float); t = np.asarray(seconds, float)
    out = np.full_like(x, np.nan)
    for row in range(1, len(x)-1):
        if (not np.asarray(valid)[row-1:row+2].all() or segments[row] < 0
                or not np.all(np.asarray(segments)[row-1:row+2] == segments[row])
                or not t[row-1] < t[row] < t[row+1]):
            continue
        alpha = (t[row]-t[row-1])/(t[row+1]-t[row-1])
        out[row] = x[row]-((1-alpha)*x[row-1]+alpha*x[row+1])
    return out


def resample_joint_trajectory(q, ticks_ns, target_ns, valid, epoch, *, max_gap_s=.2):
    """Host-receive common timeline: interpolate angles, never Cartesian bones.

    This does not estimate camera exposure latency or device clock offsets.
    Unsupported targets stay NaN, with explicit bracketing source row indices.
    """
    from .reliability import receive_support
    q=np.asarray(q,float);valid=np.asarray(valid,bool)&np.isfinite(q).all(1)
    support=receive_support(target_ns,ticks_ns,valid,epoch,max_gap_ms=max_gap_s*1000)
    result=np.full_like(q,np.nan);alpha=np.full(len(q),np.nan)
    rows=np.flatnonzero(support['bracketed'])
    lo=support['before_row'][rows];hi=support['after_row'][rows]
    ticks=np.asarray(ticks_ns,np.int64);target=np.asarray(target_ns,np.int64)
    a=np.divide(target[rows]-ticks[lo],ticks[hi]-ticks[lo],
        out=np.zeros(len(rows)),where=ticks[hi]>ticks[lo])
    result[rows]=(1-a[:,None])*q[lo]+a[:,None]*q[hi];alpha[rows]=a
    return result,dict(**support,alpha=alpha)
