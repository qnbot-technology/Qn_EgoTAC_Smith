#!/usr/bin/env python3
"""SAM glove crops with explicit Tag/temporal identity association, not hand truth."""
import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from qn_smith.artifacts import write_json,sha256,now
from qn_smith.repositories import repository


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--backend',choices=['torch','openvino'],default='torch')
    ap.add_argument('--sam-model',choices=['sam3','efficientsam3'],default='sam3')
    ap.add_argument('--openvino-dir',type=Path)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    repo=repository('sam3',PROJECT);sys.path.insert(0,str(repo))
    device=os.environ.get('QN_EGO_DEVICE','cuda')
    torch.set_num_threads(int(os.environ.get('QN_EGO_THREADS','4')));cv2.setNumThreads(2)
    info=json.loads((a.input/'input.json').read_text());n=info['rows'];w=info['width'];h=info['height']
    prior=np.load(a.input/'tag-prior.npz');palm=[0,6,11,16,21]
    accelerator=None
    if a.backend=='openvino':
        if device!='cpu' or a.openvino_dir is None:
            raise ValueError('SAM3 OpenVINO GPU requires CPU tensor glue and explicit model directory')
        from qn_ego.openvino_sam3 import OpenVINOSAM3Processor,OpenVINOEfficientSAM3Processor
        cls=OpenVINOSAM3Processor if a.sam_model=='sam3' else OpenVINOEfficientSAM3Processor
        processor=cls(a.openvino_dir,a.checkpoint)
        accelerator=dict(backend='openvino',device='GPU',directory=str(a.openvino_dir.resolve()),
            validation_sha256=sha256(a.openvino_dir/'validation.json'),
            execution_devices=processor.execution_devices,precision=processor.receipt['precision'],
            fixed_prompt='gloved hand',cpu_neural_inference=False,cpu_fallback=False,
            sam_model=a.sam_model,numeric_validation_passed=processor.receipt['numeric_passed'],
            model_status='experimental' if a.sam_model=='efficientsam3' else 'validated',
            image_activation_scale_factor=processor.receipt.get('image_activation_scale_factor'))
    else:
        if a.sam_model!='sam3':raise ValueError('EfficientSAM3 requires explicit OpenVINO GPU backend')
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        model=build_sam3_image_model(checkpoint_path=str(a.checkpoint),load_from_HF=False,
            bpe_path=str(repo/'sam3/assets/bpe_simple_vocab_16e6.txt.gz'),enable_inst_interactivity=False,device=device)
        processor=Sam3Processor(model,confidence_threshold=.35,device=device)
    out={s+'_boxes':np.full((n,5),np.nan) for s in ('left','right')}
    out.update({s+'_valid':np.zeros(n,bool) for s in ('left','right')})
    previous={};records=[];tic=time.time();inference_times=[]
    for i in range(n):
        image=Image.open(a.input/'images'/f'{i:06d}.jpg').convert('RGB')
        with torch.inference_mode(),(torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda' else nullcontext()):
            state=processor.set_image(image);r=processor.set_text_prompt(prompt='gloved hand',state=state)
            masks=r['masks'].detach().cpu().numpy().reshape(-1,h,w)
            scores=r['scores'].float().detach().cpu().numpy().ravel()
        if accelerator:inference_times.append(dict(processor.inference_seconds))
        candidates=[]
        for j,(m,score) in enumerate(zip(masks,scores)):
            yy,xx=np.nonzero(m)
            if not .0008<len(xx)/(w*h)<.3:continue
            box=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1],float)
            candidates.append((j,box,float(score)))
        proposals=[]
        for s in ('left','right'):
            uv=prior[s+'_uv'][i,palm] if prior[s+'_valid'][i] else None
            for j,box,score in candidates:
                centre=(box[:2]+box[2:])/2
                if uv is not None and np.isfinite(uv).all():
                    inside=((uv[:,0]>=box[0])&(uv[:,0]<box[2])&(uv[:,1]>=box[1])&(uv[:,1]<box[3])).mean()
                    dist=np.linalg.norm(centre-np.mean(uv,axis=0))
                    if inside<.4 or dist>170:continue
                    cost=dist/170+(1-inside)*2;kind='tag_palm_associated'
                elif s in previous and i-previous[s][0]<=10:
                    old=previous[s][1];dist=np.linalg.norm(centre-(old[:2]+old[2:])/2)
                    if dist>90:continue
                    cost=dist/90+.5;kind='temporal_visual_associated'
                else:continue
                proposals.append((cost,s,j,box,score,kind))
        used=set();assigned={}
        for cost,s,j,box,score,kind in sorted(proposals,key=lambda p:p[0]):
            if s in assigned or j in used:continue
            # Do not let two nearly identical SAM proposals become two hands.
            if any(np.linalg.norm((box[:2]+box[2:]-v[0][:2]-v[0][2:])/2)<30 for v in assigned.values()):continue
            assigned[s]=(box,score,kind,j);used.add(j)
        for s,(box,score,kind,j) in assigned.items():
            previous[s]=(i,box.copy());pad=.12*max(box[2]-box[0],box[3]-box[1]);box=box+[-pad,-pad,pad,pad]
            box[[0,2]]=box[[0,2]].clip(0,w);box[[1,3]]=box[[1,3]].clip(0,h)
            out[s+'_boxes'][i]=np.r_[box,score];out[s+'_valid'][i]=True
        records.append(dict(source_row=i,candidates=len(candidates),assigned={s:dict(score=v[1],identity_source=v[2],candidate=v[3]) for s,v in assigned.items()}))
        if i%(10 if device=='cpu' else 100)==0 or i==n-1:
            write_json(a.output/'progress.json',dict(row=i,total=n,valid={s:int(out[s+'_valid'].sum()) for s in ('left','right')},elapsed_s=time.time()-tic))
            print(info['episode'],i,n,flush=True)
    np.savez_compressed(a.output/'sam-hand-crops.npz',**out,source_rows=np.arange(n),input_sha256=info['input_sha256'])
    write_json(a.output/'association.json',dict(at=now(),frames=records,input_sha256=info['input_sha256'],
        prior_sha256=sha256(a.input/'tag-prior.npz'),checkpoint_sha256=sha256(a.checkpoint),
        identity='Tag-valid palm anchors or <=10 frame last SAM association; not independent identity ground truth',
        device='openvino:GPU' if accelerator else device,accelerator=accelerator,sam_model=a.sam_model,
        model_status='experimental' if a.sam_model=='efficientsam3' else 'baseline',
        model_inference_seconds=inference_times,training_ready=False,elapsed_s=time.time()-tic))


if __name__=='__main__':main()
