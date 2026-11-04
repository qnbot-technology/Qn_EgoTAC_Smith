#!/usr/bin/env python3
"""Reuse verified Longtest arrays; render a clean tactile and wrist-motion pair."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from qn_smith.artifacts import now,sha256,write_json
from qn_smith.hand_motion_dashboard import MotionDashboard,FixedTrajectoryView,SIDES
from qn_smith.tactile_hand_panel import SIDE_COLOURS
from qn_smith.tactile_display import PROFILES
from qn_smith.hand_motion_final import FinalMotionDashboard,compose_final,CANVAS_SIZE,VIEWPORT
from qn_smith.tactile_channel_panel import SCHEMATIC_MAPPING
from render_human_v2 import draw,project,EDGES,Encoder,verify_video


def load(a):
    e=getattr(a,'episode','longtest_20261023-093620');source=a.input_run.resolve()
    base=source/getattr(a,'baseline_subdir','background-isolated-v3/baseline')/e
    folder=source/getattr(a,'visual_subdir','background-isolated-v3/visual/fused-stable-v3')/e
    prefix=getattr(a,'evidence_prefix','')
    receipt_path=source/(prefix+'verification.json');manifest_path=source/(prefix+'manifest.json')
    receipt=json.loads(receipt_path.read_text())
    assert receipt['passed'] and sha256(manifest_path)==receipt['manifest_sha256']
    indexed={r['path']:r for r in json.loads(manifest_path.read_text())}
    paths=dict(fused=folder/'fused-humanv2.npz',bones=base/'tag3-tag16-handmap-residual-v4/bone-projections.npz',
        tactile=folder/'tactile-display-profile.npz',camera=base/'camera.json',
        trajectory=base/'trajectory-and-tactile.npz')
    refs={}
    for name,p in paths.items():
        key=p.relative_to(source).as_posix();digest=sha256(p)
        assert digest==indexed[key]['sha256']
        refs[name]=dict(path=str(p),sha256=digest)
    f=np.load(paths['fused']);b=np.load(paths['bones']);z=np.load(paths['trajectory'])
    t=np.load(paths['tactile']);cam=json.loads(paths['camera'].read_text())
    source_info=json.loads((base/'trajectory-report.json').read_text())
    assert sha256(source_info['source'])==source_info['source_sha256']
    n=len(z['source_row']);start=a.first_frame-1;stop=a.last_frame if a.last_frame else n
    assert 0<=start<stop<=n
    rows=slice(start,stop)
    data=dict(source_row=z['source_row'][rows],sample_time_s=z['sample_time_s'][rows],
              camera_receive_ns=z['camera_receive_ns'][rows])
    for key in t.files:data['tactile_'+key]=t[key][rows] if t[key].ndim and t[key].shape[0]==n else t[key]
    for side in SIDES:
        data['tag_'+side+'_points']=b[side+'_filtered_camera_m'][rows]
        data['assisted_'+side+'_points']=f[side+'_camera_m'][rows]
        data['tag_'+side+'_valid']=b[side+'_valid'][rows]
        data['assisted_'+side+'_valid']=f[side+'_valid'][rows]
    final=getattr(a,'layout','legacy')=='final16x9'
    points=np.concatenate([data[v+'_'+s+'_points'].reshape(-1,3) if final else
                           data[v+'_'+s+'_points'][:,0] for v in ('tag','assisted') for s in SIDES])
    view=FixedTrajectoryView(points,viewport=VIEWPORT) if final else FixedTrajectoryView(points)
    fps=(n-1)/(z['sample_time_s'][-1]-z['sample_time_s'][0])
    meta=dict(at=now(),source_hdf5=source_info['source'],source_sha256=source_info['source_sha256'],
        input_run=str(source),input_receipt_sha256=sha256(receipt_path),input_receipt=str(receipt_path),inputs=refs,
        episode=e,first_frame_1based=a.first_frame,last_frame_1based=stop,frames=stop-start,fps=fps,
        camera=cam,spatial_display=view.metadata(),tactile_profile='D',
        UI='Clean tactile pair above fixed 3D bilateral wrist trajectories; no technical captions or raw image headers',
        original_arrays_unchanged=True,tag_only_root_smoothing=False,
        assisted='Same saved bounded display roots, no additional smoothing or inference',
        camera_follow=False,mirrored=False,observation_region_overlay=False,tag_boxes_drawn=False,
        trajectory_point='HumanV2 node 0 wrist in original camera optical metres',
        force_calibrated=False,training_ready=False)
    meta['layout']='final16x9' if final else 'legacy'
    if getattr(a,'tag_only',False):
        meta.update(visual_assist_enabled=False,
                    assisted='Not run; internal render compatibility arrays equal Tag-only mapping')
    meta['canvas_size']=CANVAS_SIZE if final else (2560,1080)
    if final:
        meta.update(UI='16:9; original RGB centered without resize; tactile above fixed metric hand poses and trails',
            rgb_rect_xyxy=[0,180,1920,1260],skeleton_nodes_per_hand=25,skeleton_edges=EDGES,
            skeleton_pose='Current saved camera-space HumanV2 points, same scale and projection as wrist trails',
            aspect_ratio='16:9',sample_aspect_ratio='1:1',bounds_include_all_hand_nodes=True,
            skeleton_local_magnification=False,tactile_anatomical_subregions=False)
    meta['tactile_detail']=getattr(a,'tactile_detail','whole')
    if meta['tactile_detail']=='channels':
        meta['tactile_schematic_mapping']=SCHEMATIC_MAPPING
        meta['channel_colour_onset']=dict(fade='smoothstep(max(saved_positive_residual-channel_deadband,0)/12)',
            gamma=.8,peak_colour_count_range=[0,255],baseline_refit=False,deadband_changed=False,
            source_counts_modified=False,temporal_filter_added=False,
            note='Onset opacity is relative to the existing channel deadband, so crossing it does not abruptly color the cell')
    return data,meta,view


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-run',type=Path,required=True)
    ap.add_argument('--episode',default='longtest_20261023-093620')
    ap.add_argument('--baseline-subdir',default='background-isolated-v3/baseline')
    ap.add_argument('--visual-subdir',default='background-isolated-v3/visual/fused-stable-v3')
    ap.add_argument('--evidence-prefix',default='',help='Prefix for immutable processing manifest and verification')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--first-frame',type=int,default=200)
    ap.add_argument('--last-frame',type=int,default=1400)
    ap.add_argument('--mode',choices=('preview','render'),required=True)
    ap.add_argument('--variants',choices=('tag','assisted'),nargs='+',default=['assisted'])
    ap.add_argument('--preview-subdir',default='preview')
    ap.add_argument('--layout',choices=('legacy','final16x9'),default='legacy')
    ap.add_argument('--tactile-detail',choices=('whole','channels'),default='whole')
    ap.add_argument('--tag-only',action='store_true',help='Explicit optional-vision-off delivery; only Tag variant permitted')
    a=ap.parse_args();cv2.setNumThreads(2);out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if a.tag_only and a.variants!=['tag']:raise ValueError('Tag-only delivery cannot claim an assisted video')
    data,meta,view=load(a);n=meta['frames'];times=data['sample_time_s']
    tactile={k[len('tactile_'):]:v for k,v in data.items() if k.startswith('tactile_')}
    dashboards={};uv={};cam=meta['camera']
    for variant in ('tag','assisted'):
        if a.layout=='final16x9':
            dashboards[variant]=FinalMotionDashboard(tactile,
                {s:data[variant+'_'+s+'_points'] for s in SIDES},times,
                {s:data[variant+'_'+s+'_valid'] for s in SIDES},view,PROFILES['D'],EDGES,
                detail=a.tactile_detail=='channels')
        else:
            dashboards[variant]=MotionDashboard(tactile,
                {s:data[variant+'_'+s+'_points'][:,0] for s in SIDES},times,
                {s:data[variant+'_'+s+'_valid'] for s in SIDES},view,PROFILES['D'])
        uv[variant]={s:project(data[variant+'_'+s+'_points'],np.asarray(cam['K']),
                             np.asarray(cam['distortion'])) for s in SIDES}
    selected=sorted(set([0,n-1]+[x for x in (300,625,637,775,825,900,1050) if x<n]
                        +np.linspace(0,n-1,7).astype(int).tolist()))
    def frame(h5,i,variant):
        source_row=int(data['source_row'][i])
        rgb=cv2.imdecode(np.asarray(h5['raw/cameras/head_rgb/payload'][source_row],np.uint8),1)
        for j,s in enumerate(SIDES):draw(rgb,uv[variant][s][i],EDGES,SIDE_COLOURS[j])
        panel=dashboards[variant].render(i)
        return compose_final(rgb,panel) if a.layout=='final16x9' else np.hstack([rgb,panel])
    if a.mode=='preview':
        preview=out/a.preview_subdir;preview.mkdir(exist_ok=False)
        with h5py.File(meta['source_hdf5'],'r') as h5:
            for i in np.linspace(0,n-1,3).astype(int):
                cv2.imwrite(str(preview/f'source-frame-{int(data["source_row"][i])+1:04d}.jpg'),frame(h5,i,'assisted'))
        write_json(preview/'recipe.json',meta)
        print('Three full-size layout previews saved',flush=True);return
    delivery=out/'delivery';delivery.mkdir(exist_ok=False)
    np.savez_compressed(out/'display-source-arrays.npz',**data)
    write_json(out/'recipe.json',meta)
    mapping=[dict(output_row=i,source_row_0based=int(row),source_frame_1based=int(row)+1,
                  sample_time_s=float(times[i]),camera_receive_ns=int(data['camera_receive_ns'][i]),
                  encoded_pts_s=i/meta['fps']) for i,row in enumerate(data['source_row'])]
    write_json(delivery/'frame-map.json',mapping)
    qa=delivery/'qa';qa.mkdir()
    names=({'tag':'01_tag_only_final_16x9.mp4','assisted':'02_visual_assisted_final_16x9.mp4'}
           if a.layout=='final16x9' else
           {'tag':'01_tag_only_clean_motion.mp4','assisted':'02_visual_assisted_clean_motion.mp4'})
    size=tuple(meta['canvas_size'])
    encoders={v:Encoder(delivery/names[v],size,meta['fps'],square_pixels=a.layout=='final16x9')
              for v in a.variants}
    strips=[]
    with h5py.File(meta['source_hdf5'],'r') as h5:
        for i in range(n):
            pair=[]
            for variant in a.variants:
                image=frame(h5,i,variant);encoders[variant].write(image)
                if i in selected:
                    cv2.imwrite(str(qa/f'{variant}-source-frame-{int(data["source_row"][i])+1:04d}.jpg'),image)
                    pair.append(cv2.resize(image,(1024,round(1024*size[1]/size[0]))))
            if pair:strips.append(np.hstack(pair))
            if i%150==0:print('render',i,'/',n,flush=True)
    for enc in encoders.values():enc.close()
    videos={}
    for variant in a.variants:
        path=delivery/names[variant]
        videos[variant]=dict(path=str(path),**verify_video(path,n,*size),
                            fps=meta['fps'],duration_s=n/meta['fps'])
    for page,i in enumerate(range(0,len(strips),3),1):
        cv2.imwrite(str(qa/f'contact-sheet-{page:02d}.jpg'),np.vstack(strips[i:i+3]))
    write_json(delivery/'delivery.json',dict(at=now(),videos=videos,frames=n,
        first_frame_1based=a.first_frame,last_frame_1based=meta['last_frame_1based'],
        source_sha256=meta['source_sha256'],complete_decode_passed=True,
        camera_follow=False,mirrored=False,observation_region_overlay=False,
        technical_captions_drawn=False,raw_rgb_headers_drawn=False,
        trajectories='Both wrists; camera-relative metres; fixed shared view; no gap bridging or future lines',
        layout=meta['layout'],tactile_detail=meta['tactile_detail'],
        skeleton_poses_drawn=a.layout=='final16x9',
        qa_frames_1based=[int(data['source_row'][i])+1 for i in selected],
        visual_assist_enabled=not a.tag_only,visual_review='pending',training_ready=False))
    print('Clean motion videos fully decoded:',a.variants,flush=True)


if __name__=='__main__':main()
