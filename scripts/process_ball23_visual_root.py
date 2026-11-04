#!/usr/bin/env python3
"""Source-bound offline visual-root experiment; never overwrites Tag/native data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from qn_smith.artifacts import write_json,sha256,now
from qn_smith.apriltag_pose import tag_points
from qn_smith.customer_visual import proper_rigid,COMMON21
from qn_smith.hand_root_fusion import (RootFusionConfig,track_pose,choose_root,consistent,
    pose_distance,blend_pose,stabilize_display_roots,is_pose,gate_mano_transitions)
from qn_smith.tactile_hand_panel import render_panel
from qn_smith.tactile_display import PROFILES, prepare_display
from render_human_v2 import project,draw,EDGES,Encoder,verify_video

BASE=PROJECT/'reports/ball23-humanv2-tactile-20261022'
SIDES=('left','right')
COLORS=((70,210,255),(255,160,80))


def load(args,episode):
    old=getattr(args,'baseline_root',BASE)/episode; prior=old/'tag3-tag16-handmap-residual-v4'
    info=json.loads((args.run/'input'/episode/'input.json').read_text())
    trajectory=np.load(old/'trajectory-and-tactile.npz')
    applied=np.load(prior/'applied-camera-trajectory.npz')
    records=[json.loads(v) for v in (old/'detections-and-poses.jsonl').read_text().splitlines()]
    config=json.loads((old/'configuration.json').read_text())
    return old,prior,info,trajectory,applied,records,config


def seed(gray,row,body,K0,D,K,cfg):
    pts=[];pixels=[]
    for det in row['detections']:
        tid=str(det['id'])
        if det['id'] not in row['bodies'][body]['used_ids'] or tid not in cfg['bodies'][body]['T_root_tag']:continue
        corners=cv2.undistortPoints(np.array(det['corners_px'],float).reshape(-1,1,2),K0,D,P=K).reshape(-1,2)
        mask=np.zeros_like(gray);cv2.fillConvexPoly(mask,np.rint(corners).astype(np.int32),255)
        mask=cv2.erode(mask,np.ones((3,3),np.uint8))
        features=cv2.goodFeaturesToTrack(gray,32,.015,3,mask=mask,blockSize=3)
        if features is None:continue
        p2=features.reshape(-1,2).astype(float)
        plane=tag_points(float(cfg['tag_sizes_m'][tid]))
        H=cv2.getPerspectiveTransform(corners.astype(np.float32),plane[:,:2].astype(np.float32))
        local=cv2.perspectiveTransform(p2.reshape(-1,1,2),H).reshape(-1,2)
        local=np.c_[local,np.zeros(len(local))];T=np.array(cfg['bodies'][body]['T_root_tag'][tid])
        pts.extend(local@T[:3,:3].T+T[:3,3]);pixels.extend(p2)
    return np.asarray(pixels),np.asarray(pts)


def flow(args,episode):
    old,prior,info,z,applied,records,config=load(args,episode)
    n=info['rows'];K=np.array(info['K']);c=info['original_camera'];K0=np.array(c['K']);D=np.array(c['distortion'])
    cfg=RootFusionConfig();out=args.run/'flow'/episode;out.mkdir(parents=True,exist_ok=False)
    data={};logs=[]
    for direction in ('forward','backward'):
        state={};previous_gray=None;previous_i=None
        for side in SIDES:
            data[side+'_'+direction]=np.full((n,4,4),np.nan)
            data[side+'_'+direction+'_age_s']=np.full(n,np.nan)
        for i in (range(n) if direction=='forward' else range(n-1,-1,-1)):
            gray=cv2.imread(str(args.run/'input'/episode/'images'/f'{i:06d}.jpg'),0)
            for side in SIDES:
                result=None;quality=dict(reason='no_seed')
                if side in state:
                    anchor,xy,points,T=state[side]
                    age=abs(float(z['sample_time_s'][i]-z['sample_time_s'][anchor]))
                    if age<=cfg.max_flow_age_s and abs(z['sample_time_s'][i]-z['sample_time_s'][previous_i])<.1:
                        result,quality=track_pose(previous_gray,gray,xy,points,K,T,cfg)
                        if result is not None:
                            T,xy,points=result;data[side+'_'+direction][i]=T
                            data[side+'_'+direction+'_age_s'][i]=age
                            state[side]=(anchor,xy,points,T)
                        else:state.pop(side,None)
                    else:state.pop(side,None);quality=dict(reason='expired_or_time_gap')
                logs.append(dict(source_row=i,side=side,direction=direction,**quality))
                if z[side+'_pose_valid'][i]:
                    xy,points=seed(gray,records[i],side,K0,D,K,config)
                    if len(xy)>=cfg.min_features:
                        state[side]=(i,xy,points,applied[side+'_T_camera_root'][i])
            previous_gray=gray;previous_i=i
            if i%150==0:print(episode,direction,i,n,flush=True)
    np.savez_compressed(out/'flow-camera-roots.npz',**data,source_row=z['source_row'],sample_time_s=z['sample_time_s'])
    write_json(out/'quality.json',dict(at=now(),config=cfg.to_dict(),frames=logs,
        input_sha256=info['input_sha256'],input_manifest_sha256=sha256(args.run/'input'/episode/'manifest.json'),
        original_detection_sha256=sha256(old/'detections-and-poses.jsonl'),
        offline_bidirectional=True,root_frame='original fixture root; CAD EMF transform applied later',training_ready=False))


def mano_candidates(args,episode,side,local,tag,z,info,cfg):
    n=info['rows'];out=np.full((n,4,4),np.nan);cost=np.full(n,np.inf);audit=[]
    folder=args.run/'sam-mano'/episode
    if not (folder/'summary.json').exists():return out,audit
    summary=json.loads((folder/'summary.json').read_text())
    if summary['input_sha256']!=info['input_sha256']:raise ValueError('MANO input SHA mismatch')
    weights=np.ones(21);weights[[0,1,5,9,13,17]]=3
    K=np.array(info['K'])
    for cand in summary['candidates']:
        if cand['side']!=side:continue
        track=np.load(folder/cand['file']);rows=track['source_rows'];pred=track['joints_camera_m']
        fits=np.full((n,4,4),np.nan);rms=np.full(n,np.inf)
        for j,row in enumerate(rows):
            if not np.isfinite(local[row]).all() or not np.isfinite(pred[j]).all():continue
            try:R,t,_=proper_rigid(local[row,COMMON21],pred[j],weights)
            except ValueError:continue
            fit=local[row,COMMON21]@R.T+t
            error=float(np.sqrt(np.mean(np.sum((fit-pred[j])**2,axis=1))))
            u=cv2.projectPoints(fit,np.zeros(3),np.zeros(3),K,None)[0].reshape(-1,2)
            v=cv2.projectPoints(pred[j],np.zeros(3),np.zeros(3),K,None)[0].reshape(-1,2)
            reproj=float(np.sqrt(np.mean(np.sum((u-v)**2,axis=1))))
            if error>cfg.mano_fit_rms_m or reproj>cfg.mano_reprojection_rms_px or not .15<t[2]<1.6:continue
            fits[row]=np.eye(4);fits[row,:3,:3]=R;fits[row,:3,3]=t;rms[row]=error
        anchor_valid=np.isfinite(fits).all((1,2))&z[side+'_pose_valid']&z[side+'_emf_time_valid']
        # A single isolated Tag jump must not calibrate a visual gap next to it.
        for row in np.flatnonzero(anchor_valid):
            neighbours=[q for q in (row-1,row+1) if 0<=q<n and z[side+'_pose_valid'][q]]
            if neighbours and all(not consistent(tag[row],tag[q],cfg.tag_jump_m,cfg.tag_jump_deg) for q in neighbours):
                anchor_valid[row]=False
        anchors=np.flatnonzero(anchor_valid)
        accepted=0
        for row in rows:
            if not np.isfinite(fits[row]).all() or not len(anchors):continue
            nearest=anchors[np.argmin(abs(z['sample_time_s'][anchors]-z['sample_time_s'][row]))]
            age=abs(float(z['sample_time_s'][nearest]-z['sample_time_s'][row]))
            if age>cfg.mano_anchor_max_age_s:continue
            correction=np.linalg.inv(fits[nearest])@tag[nearest]
            # Use both nearby endpoint calibrations where they agree, to avoid
            # changing the installation correction at the nearest-anchor midpoint.
            lo=anchors[anchors<=row];hi=anchors[anchors>=row]
            if len(lo) and len(hi) and lo[-1]!=hi[0]:
                l=int(lo[-1]);h=int(hi[0])
                if max(abs(z['sample_time_s'][row]-z['sample_time_s'][l]),abs(z['sample_time_s'][h]-z['sample_time_s'][row]))<=cfg.mano_anchor_max_age_s:
                    ca=np.linalg.inv(fits[l])@tag[l];cb=np.linalg.inv(fits[h])@tag[h]
                    if not consistent(ca,cb,.025,15):continue
                    correction=blend_pose(ca,cb,float((z['sample_time_s'][row]-z['sample_time_s'][l])/(z['sample_time_s'][h]-z['sample_time_s'][l])))
            d,r=pose_distance(np.eye(4),correction)
            if d>.15 or r>65:continue
            T=fits[row]@correction
            # Consecutive visual frames must support motion; no extrapolation.
            neighbours=[q for q in (row-1,row+1) if 0<=q<n and np.isfinite(fits[q]).all()]
            if not neighbours or any(not consistent(fits[row],fits[q],cfg.max_step_m,cfg.max_step_deg) for q in neighbours):continue
            rank=rms[row]+age*.1
            if rank<cost[row]:out[row]=T;cost[row]=rank;accepted+=1
        audit.append(dict(file=cand['file'],side=side,rows=len(rows),geometric_rows=int(np.isfinite(fits).all((1,2)).sum()),
            anchors=len(anchors),accepted_proposals=accepted,method='proper-SE3 native25 to visual21; nearby Tag right correction <=250ms; two endpoint interpolation only if consistent',
            changed_native_fingers=False,offline_future_anchor_possible=True))
    return out,audit


def fuse(args,episode):
    old,prior,info,z,applied,records,config=load(args,episode)
    cfg=RootFusionConfig();n=info['rows'];out=args.run/args.output_stage/episode;out.mkdir(parents=True,exist_ok=False)
    f=np.load(args.run/'flow'/episode/'flow-camera-roots.npz');data={};stats={};qa={}
    for side in SIDES:
        native=np.load(prior/f'filtered-positions-{side}.npz');ix=z[side+'_emf_index']
        local=native['positions_glove_m'][ix];emf_valid=z[side+'_emf_time_valid']&native['valid'][ix]
        tag=applied[side+'_T_camera_emf'];mount=applied[side+'_T_root_emf_applied']
        if mount.ndim==3:assert np.allclose(mount,mount[0]);mount=mount[0]
        forward=f[side+'_forward']@mount;backward=f[side+'_backward']@mount
        mano,mano_audit=mano_candidates(args,episode,side,local,tag,z,info,cfg)
        roots=np.full((n,4,4),np.nan);sources=[];suspect=np.zeros(n,bool)
        for i in range(n):
            neighbours=[q for q in (i-1,i+1) if 0<=q<n and z[side+'_pose_valid'][q] and abs(z['sample_time_s'][i]-z['sample_time_s'][q])<.1]
            temporal=bool(z[side+'_pose_valid'][i] and neighbours and all(not consistent(tag[i],tag[q],cfg.tag_jump_m,cfg.tag_jump_deg) for q in neighbours))
            roots[i],source,suspect[i]=choose_root(tag[i],bool(z[side+'_pose_valid'][i]),forward[i],backward[i],bool(emf_valid[i]),cfg,mano[i],temporal)
            if source in ('VISUAL_FLOW_GAP','VISUAL_TAG_OUTLIER') and is_pose(forward[i]) and is_pose(backward[i]):
                a=f[side+'_forward_age_s'][i];b=f[side+'_backward_age_s'][i]
                roots[i]=blend_pose(forward[i],backward[i],float(a/(a+b)))
            sources.append(source)
        roots,sources,transition_audit=gate_mano_transitions(roots,sources,cfg)
        sources=sources.tolist()
        valid=np.isfinite(roots).all((1,2))&emf_valid
        display,display_diag=stabilize_display_roots(roots,z['sample_time_s'],valid)
        camera=np.einsum('nij,nkj->nki',display[:,:3,:3],local)+display[:,None,:3,3]
        camera[~valid]=np.nan
        data.update({side+'_T_camera_emf':roots,side+'_display_T_camera_emf':display,
            side+'_valid':valid,side+'_source':np.array(sources),
            side+'_tag_observed_valid':z[side+'_pose_valid'],side+'_tag_jump_suspect':suspect,
            side+'_camera_m':camera,side+'_mano_T_camera_emf':mano,
            side+'_forward_T_camera_emf':forward,side+'_backward_T_camera_emf':backward})
        tag_keep=np.array(sources)=='TAG'
        if not np.array_equal(roots[tag_keep],tag[tag_keep]):raise ValueError('Tag-priority invariant failed')
        good=np.isfinite(forward).all((1,2))&z[side+'_pose_valid']
        comparison=np.array([pose_distance(forward[j],tag[j]) for j in np.flatnonzero(good)])
        changes=np.array([pose_distance(roots[j],display[j]) for j in np.flatnonzero(valid)])
        qa[side]=dict(mano_tracks=mano_audit,mano_transition_gate=transition_audit,display_filter=display_diag,
            display_translation_mm_p95=float(np.percentile(changes[:,0],95)*1000),
            display_rotation_deg_p95=float(np.percentile(changes[:,1],95)),
            flow_vs_observed=dict(rows=int(good.sum()),
            translation_mm_median=float(np.median(comparison[:,0])*1000) if len(comparison) else None,
            translation_mm_p95=float(np.percentile(comparison[:,0],95)*1000) if len(comparison) else None,
            rotation_deg_median=float(np.median(comparison[:,1])) if len(comparison) else None))
        baseline=np.load(prior/'bone-projections.npz')[side+'_valid']
        reasons=Counter(r['bodies'][side].get('invalid_reason') for r in records if not r['bodies'][side]['valid_current_observation'])
        stats[side]=dict(baseline_valid=int(baseline.sum()),fused_valid=int(valid.sum()),
            added=int((valid&~baseline).sum()),remaining_missing=int((~valid).sum()),sources=dict(Counter(sources)),
            tag_jump_suspected=int(suspect.sum()),tag_outliers_replaced=sources.count('VISUAL_TAG_OUTLIER'),original_missing_reasons=dict(reasons))
    np.savez_compressed(out/'fused-humanv2.npz',**data,source_row=z['source_row'],sample_time_s=z['sample_time_s'],camera_receive_ns=z['camera_receive_ns'])
    write_json(out/'fusion-report.json',dict(at=now(),input_sha256=info['input_sha256'],config=cfg.to_dict(),
        sides=stats,diagnostics=qa,training_ready=False,baseline_unchanged=True,
        original_observation_valid_never_promoted=True,native_local_fingers_unchanged=True,
        offline_bidirectional=True,coordinate_frame='original color optical',
        warning='Optical flow can drift. MANO is model evidence, not ground truth. Occlusion/out-of-view/EMF gaps remain NaN.',
        hashes={str(p):sha256(p) for p in (prior/'applied-camera-trajectory.npz',prior/'bone-projections.npz',old/'trajectory-and-tactile.npz',args.run/'flow'/episode/'flow-camera-roots.npz')}))
    print(episode,stats,flush=True)


def render(args,episode):
    old,prior,info,z,applied,records,config=load(args,episode)
    folder=args.run/args.output_stage/episode;f=np.load(folder/'fused-humanv2.npz');base=np.load(prior/'bone-projections.npz')
    tactile=np.load(prior/'tactile-display.npz');cam=info['original_camera'];K=np.array(cam['K']);D=np.array(cam['distortion'])
    profile = getattr(args, 'tactile_profile', None)
    style = PROFILES[profile] if profile else None
    if style:
        tactile = prepare_display(dict(tactile), style)
        np.savez_compressed(folder/'tactile-display-profile.npz', **tactile)
        write_json(folder/'tactile-display-profile.json', dict(profile=profile, style=style.to_dict(),
            source_sha256=sha256(prior/'tactile-display.npz'), analysis_baseline_unchanged=True,
            force_calibrated=False, training_ready=False))
    uv={s:project(f[s+'_camera_m'],K,D) for s in SIDES}
    n=info['rows'];fps=(n-1)/(z['sample_time_s'][-1]-z['sample_time_s'][0]);w=cam['width'];h=cam['height']
    enc=Encoder(folder/'humanv2-visual-assisted-full.mp4',(w+640,h),fps)
    cmp=Encoder(folder/'tag-vs-visual-assisted-full.mp4',(w*2,h),fps)
    tag_encoder = Encoder(folder/'humanv2-tag-only-full.mp4',(w+640,h),fps) if getattr(args,'include_tag_only',False) else None
    selected={0,n//2,n-1}
    for s in SIDES:
        for source in ('VISUAL_FLOW_GAP','VISUAL_MANO_GAP','VISUAL_TAG_OUTLIER'):
            ii=np.flatnonzero(f[s+'_source']==source)
            if len(ii):selected.add(int(ii[len(ii)//2]))
    with h5py.File(info['source_hdf5'],'r') as h5:
        for i,payload in enumerate(h5['raw/cameras/head_rgb/payload']):
            raw=cv2.imdecode(np.asarray(payload,np.uint8),1);before=raw.copy();after=raw.copy()
            for j,s in enumerate(SIDES):
                draw(before,base[s+'_filtered_uv_px'][i],EDGES,COLORS[j]);draw(after,uv[s][i],EDGES,COLORS[j])
            for image,title in ((before,'TAG ONLY'),(after,'HUMAN V2 | TAG + VISUAL ROOT')):
                cv2.rectangle(image,(0,0),(w,92),(24,27,35),-1)
                cv2.putText(image,f'{title} | {episode} | row {i+1}/{n}',(15,29),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),1,cv2.LINE_AA)
            for j,s in enumerate(SIDES):
                source=str(f[s+'_source'][i]);cv2.putText(after,f'{s.upper()}: {source}',(15+j*850,68),cv2.FONT_HERSHEY_SIMPLEX,.65,COLORS[j],2,cv2.LINE_AA)
                baseline_source='TAG' if base[s+'_valid'][i] else 'MISSING_TAG_OR_EMF'
                cv2.putText(before,f'{s.upper()}: {baseline_source}',(15+j*850,68),cv2.FONT_HERSHEY_SIMPLEX,.65,COLORS[j],2,cv2.LINE_AA)
            panel=render_panel(tactile,i,signal='residual',style=style)
            canvas=np.hstack([after,panel]);enc.write(canvas);cmp.write(np.hstack([before,after]))
            if tag_encoder:tag_encoder.write(np.hstack([before,panel]))
            if i in selected:cv2.imwrite(str(folder/f'qa-row-{i:04d}.jpg'),np.hstack([cv2.resize(before,(960,540)),cv2.resize(after,(960,540))]))
            if i%150==0:print(episode,'render',i,n,flush=True)
    enc.close();cmp.close()
    if tag_encoder:tag_encoder.close()
    write_json(folder/'video-verification.json',dict(assisted=verify_video(folder/'humanv2-visual-assisted-full.mp4',n,w+640,h),
        comparison=verify_video(folder/'tag-vs-visual-assisted-full.mp4',n,w*2,h),fps=fps,
        tag_only=verify_video(folder/'humanv2-tag-only-full.mp4',n,w+640,h) if tag_encoder else None,
        matched_tactile_profile=profile or 'A',
        tag_only_root='Observed Tag only; same filtered local HumanV2; no visual-root or root display smoothing',
        assisted_root='Tag priority with gated visual fallback and bounded offline root display smoothing',
        native_hand_model='same HumanV2 25 nodes; MANO never drawn',rgb_mirrored=False,camera_follow=False,
        observation_region_overlay=False,tag_detection_boxes_drawn=False,
        complete_original_timeline=True,visual_review='pending',training_ready=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--episode',nargs='+',required=True);ap.add_argument('--mode',choices=('flow','fuse','render'),required=True)
    ap.add_argument('--output-stage',default='fused-stable-v3')
    ap.add_argument('--baseline-root',type=Path,default=BASE)
    ap.add_argument('--tactile-profile',choices=list(PROFILES),help='Independent display profile; analysis baseline is preserved')
    ap.add_argument('--include-tag-only',action='store_true',help='Render a separate full Tag-only video with the identical tactile panel')
    a=ap.parse_args();cv2.setNumThreads(2)
    for e in a.episode:globals()[a.mode](a,e)


if __name__=='__main__':main()
