"""Quality-gated visual root estimates, retaining native local hand geometry.

No output from this module makes a missing Tag observation valid. Offline
backward evidence is explicitly acausal and is never advertised as live tracking.
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class RootFusionConfig:
    max_flow_age_s: float = .6
    fb_error_px: float = .8
    max_lk_error: float = 30.
    min_features: int = 10
    min_inlier_fraction: float = .7
    reprojection_rms_px: float = 1.5
    max_step_m: float = .065
    max_step_deg: float = 35.
    bidirectional_agreement_m: float = .018
    bidirectional_agreement_deg: float = 12.
    tag_jump_m: float = .03
    tag_jump_deg: float = 20.
    mano_fit_rms_m: float = .025
    mano_reprojection_rms_px: float = 20.
    mano_anchor_max_age_s: float = .25
    mano_transition_m: float = .035
    mano_transition_deg: float = 20.
    min_isolated_mano_rows: int = 3

    def to_dict(self):
        return asdict(self)


def pose_distance(a, b):
    return (float(np.linalg.norm(a[:3, 3]-b[:3, 3])),
            float(np.rad2deg(Rotation.from_matrix(a[:3, :3].T@b[:3, :3]).magnitude())))


def is_pose(value):
    """Reject malformed, scaled or reflected estimates instead of repairing them."""
    if value is None:
        return False
    a = np.asarray(value)
    return bool(a.shape == (4, 4) and np.isfinite(a).all()
                and np.allclose(a[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
                and np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-5, rtol=0)
                and abs(np.linalg.det(a[:3, :3])-1) < 1e-5)


def blend_pose(a, b, fraction):
    """Convex translation and shortest-arc SO(3); no scaling or bone blending."""
    if not is_pose(a) or not is_pose(b) or not 0 <= fraction <= 1:
        raise ValueError('Expected two SE3 poses and a fraction in [0,1]')
    T = np.eye(4)
    T[:3, 3] = (1-fraction)*a[:3, 3] + fraction*b[:3, 3]
    T[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]])))(fraction).as_matrix()
    return T


def consistent(a, b, metres, degrees):
    d, r = pose_distance(a, b)
    return d <= metres and r <= degrees


def track_pose(previous_gray, gray, xy, points, K, previous_pose, cfg):
    """Forward/backward image evidence + metric CAD PnP; never constant hold."""
    if len(xy) < cfg.min_features:
        return None, dict(reason='too_few_seed_features')
    options=dict(winSize=(21,21),maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    query=np.asarray(xy,np.float32).reshape(-1,1,2)
    nxt, st, err=cv2.calcOpticalFlowPyrLK(previous_gray,gray,query,None,**options)
    if nxt is None:return None,dict(reason='forward_flow_failed')
    back, bs, _=cv2.calcOpticalFlowPyrLK(gray,previous_gray,nxt,None,**options)
    if back is None:return None,dict(reason='backward_flow_failed')
    fb=np.linalg.norm((back-query).reshape(-1,2),axis=1)
    dest=nxt.reshape(-1,2);h,w=gray.shape
    keep=(st.ravel()>0)&(bs.ravel()>0)&(fb<=cfg.fb_error_px)&(err.ravel()<=cfg.max_lk_error)
    keep&=np.isfinite(dest).all(1)&(dest[:,0]>=2)&(dest[:,0]<w-2)&(dest[:,1]>=2)&(dest[:,1]<h-2)
    pts=np.asarray(points,float)[keep];pix=dest[keep].astype(float)
    if len(pts)<cfg.min_features:return None,dict(reason='flow_feature_gate',tracked=int(len(pts)))
    cv2.setRNGSeed(917)
    ok,r,t,inliers=cv2.solvePnPRansac(pts,pix,K,None,cv2.Rodrigues(previous_pose[:3,:3])[0],
        previous_pose[:3,3].copy().reshape(3,1),True,iterationsCount=100,
        reprojectionError=2.5,confidence=.999,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inliers is None or len(inliers)<cfg.min_features or len(inliers)/len(pts)<cfg.min_inlier_fraction:
        return None,dict(reason='pnp_consensus_gate',tracked=int(len(pts)))
    ix=inliers.ravel();pts=pts[ix];pix=pix[ix]
    r,t=cv2.solvePnPRefineLM(pts,pix,K,None,r,t)
    T=np.eye(4);T[:3,:3]=cv2.Rodrigues(r)[0];T[:3,3]=t.ravel()
    proj=cv2.projectPoints(pts,r,t,K,None)[0].reshape(-1,2)
    rms=float(np.sqrt(np.mean(np.sum((proj-pix)**2,axis=1))))
    quality=dict(reason='accepted',features=len(pts),rms_px=rms,fb_p95_px=float(np.percentile(fb[keep][ix],95)))
    if (not np.isfinite(T).all() or np.any((pts@T[:3,:3].T+T[:3,3])[:,2]<.1)
        or not .1<T[2,3]<2. or rms>cfg.reprojection_rms_px
        or not consistent(previous_pose,T,cfg.max_step_m,cfg.max_step_deg)):
        quality['reason']='geometry_or_motion_gate';return None,quality
    return (T,pix,pts),quality


def choose_root(tag, tag_valid, forward, backward, emf_valid, cfg, mano=None,
                tag_temporal_suspect=False):
    """Tag wins unless two-sided visual evidence corroborates its jump.

    An uncorroborated suspect is preserved and flagged. Real rapid movement is
    not discarded merely because it exceeds a configured velocity threshold.
    """
    missing=np.full((4,4),np.nan)
    if not emf_valid:return missing,'MISSING_EMF',False
    if tag_valid and not is_pose(tag):
        raise ValueError('Tag marked valid must be SE3')
    fgood=is_pose(forward)
    bgood=is_pose(backward)
    two=fgood and bgood and consistent(forward,backward,cfg.bidirectional_agreement_m,cfg.bidirectional_agreement_deg)
    if tag_valid:
        disagree=two and not consistent(tag,forward,cfg.tag_jump_m,cfg.tag_jump_deg) and not consistent(tag,backward,cfg.tag_jump_m,cfg.tag_jump_deg)
        if disagree and tag_temporal_suspect:return forward.copy(),'VISUAL_TAG_OUTLIER',True
        return tag.copy(),'TAG',bool(tag_temporal_suspect or disagree)
    if fgood and bgood and not two:return missing,'MISSING_VISUAL_CONFLICT',False
    if fgood:return forward.copy(),'VISUAL_FLOW_GAP',False
    if bgood:return backward.copy(),'VISUAL_FLOW_GAP',False
    if is_pose(mano):return mano.copy(),'VISUAL_MANO_GAP',False
    return missing,'MISSING_ROOT',False


def stabilize_display_roots(roots, seconds, valid):
    """Bounded offline presentation filter, separately retained from selected roots.

    Never fills NaNs or smooths across a large displacement/clock discontinuity.
    It changes the hand root only. Native local geometry is not an input.
    """
    from .hand_stability import stabilize_poses
    roots=np.asarray(roots,float);t=np.asarray(seconds,float)
    good=np.asarray(valid,bool)
    if roots.shape != (len(t),4,4) or good.shape != t.shape or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Expected increasing seconds and aligned pose arrays')
    if any(not is_pose(roots[i]) for i in np.flatnonzero(good)):
        raise ValueError('Invalid SE3 in valid rows')
    epochs=np.zeros(len(t),int)
    cuts=[]
    for i in range(1,len(t)):
        if t[i]-t[i-1]>.1 or (good[i-1] and good[i] and not consistent(roots[i-1],roots[i],.065,35)):
            cuts.append(i)
    for i in cuts:epochs[i:]+=1
    out,diag=stabilize_poses(roots,np.rint((t-t[0]+1)*1e9).astype(np.int64),good,epochs,
        radius_s=.1,max_gap_s=.1,max_translation_m=.012,max_rotation_rad=np.deg2rad(2.5))
    out[~good]=np.nan
    diag.update(offline_acausal=True,radius_s=.1,max_translation_m=.012,max_rotation_deg=2.5,
                large_motion_cuts=cuts,missing_filled=False,local_finger_geometry_changed=False)
    return out,diag


def gate_mano_transitions(roots, sources, cfg):
    """Reject unsupported corrected-MANO segments before presentation filtering.

    A locally smooth MANO fit can still change its Tag installation correction.
    Check the selected, corrected roots against both neighbouring sources. Only
    auxiliary segments are rejected; original Tag/flow rows remain untouched.
    Short islands bounded by missing roots have no continuous handover support.
    """
    out = np.asarray(roots, float).copy()
    labels = np.asarray(sources, dtype='U40').copy()
    indices = np.flatnonzero(labels == 'VISUAL_MANO_GAP')
    audit = []
    for rows in np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1):
        if not len(rows):
            continue
        lo, hi = int(rows[0]), int(rows[-1])
        neighbours = [i for i in (lo-1, hi+1) if 0 <= i < len(out) and is_pose(out[i])]
        reasons = []
        boundaries = []
        for i in neighbours:
            j = lo if i < lo else hi
            metres, degrees = pose_distance(out[i], out[j])
            boundaries.append(dict(source_row=i, translation_m=metres, rotation_deg=degrees))
            if metres > cfg.mano_transition_m or degrees > cfg.mano_transition_deg:
                reasons.append('corrected_root_boundary_disagrees')
        for a, b in zip(rows[:-1], rows[1:]):
            if not consistent(out[a], out[b], cfg.mano_transition_m, cfg.mano_transition_deg):
                reasons.append('corrected_root_internal_disagrees')
        if not neighbours and len(rows) < cfg.min_isolated_mano_rows:
            reasons.append('short_island_without_handover_support')
        rejected = bool(reasons)
        if rejected:
            out[rows] = np.nan
            labels[rows] = 'MISSING_VISUAL_TRANSITION'
        audit.append(dict(first_row=lo, last_row=hi, rejected=rejected,
                          reasons=sorted(set(reasons)), boundaries=boundaries))
    return out, labels, audit
