"""Verbatim function subset; provenance/extraction.json records origin."""
import numpy as np
COMMON21 = np.array([0,1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19,21,22,23,24])


def proper_rigid(source,target,weights=None):
    """Fit row-vector target = source @ R.T + t; no scale, no reflection."""
    a=np.asarray(source,float); b=np.asarray(target,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or len(a)<3:
        raise ValueError('Expected matching point sets')
    if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('Nonfinite alignment points')
    w=np.ones(len(a)) if weights is None else np.asarray(weights,float)
    if w.shape!=(len(a),) or np.any(w<0) or w.sum()<=0: raise ValueError('Invalid weights')
    w=w/w.sum(); ca=w@a; cb=w@b
    u,s,vt=np.linalg.svd((a-ca).T@(w[:,None]*(b-cb)))
    if s[1]<1e-12: raise ValueError('Collinear/degenerate alignment')
    fix=np.eye(3); fix[2,2]=np.linalg.det(vt.T@u.T)
    rotation=vt.T@fix@u.T
    return rotation, cb-rotation@ca, s
